# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

import atexit
import threading

from vllm.logger import logger

from vllm_ascend.model_loader.rfork.config import RForkConfig
from vllm_ascend.model_loader.rfork.planner_client import RForkPlannerClient
from vllm_ascend.model_loader.rfork.seed_server import RForkSeedServerHandle, start_rfork_server
from vllm_ascend.model_loader.rfork.transfer_backend import RForkTransferBackend
from vllm_ascend.model_loader.rfork.types import (
    RForkIdentity,
    RForkLifecycleState,
    SeedLease,
    SeedTransferInfo,
)


class RForkSession:
    """Sole owner of one worker process's RFork runtime resources."""

    def __init__(self, config: RForkConfig, identity: RForkIdentity) -> None:
        if not config.scheduler_url:
            raise ValueError(
                "rfork_scheduler_url is required; configure it with model_loader_extra_config or "
                "VLLM_ASCEND_RFORK_SCHEDULER_URL"
            )
        if not config.model_url or not config.model_deploy_strategy_name:
            raise ValueError("RFork requires non-empty model_url and model_deploy_strategy_name")

        self.config = config
        self.identity = identity
        self.transfer_backend = RForkTransferBackend(
            request_timeout_sec=config.request_timeout_sec,
            auth_token=config.auth_token or "",
        )
        self.planner = RForkPlannerClient(config, identity)
        self.state = RForkLifecycleState.INITIALIZED
        self.seed_lease: SeedLease | None = None
        self.seed_server: RForkSeedServerHandle | None = None
        self.heartbeat_thread: threading.Thread | None = None
        self.heartbeat_stop_event = threading.Event()
        self._lock = threading.RLock()
        atexit.register(self.shutdown)

    def acquire_seed(self) -> bool:
        with self._lock:
            if self.state is not RForkLifecycleState.INITIALIZED:
                logger.error("RFork seed acquisition requires an initialized session; state=%s", self.state.name)
                return False
            self.seed_lease = self.planner.acquire_seed()
            if self.seed_lease is None:
                return False
            self.state = RForkLifecycleState.LEASED
            return True

    def transfer_from_seed(self, model, processed_layout: bool) -> bool:
        with self._lock:
            if self.state is not RForkLifecycleState.LEASED or self.seed_lease is None:
                logger.error("RFork transfer requires an acquired seed lease.")
                return False
            if not self.transfer_backend.register_memory_region(model, processed_layout):
                return False
            self.state = RForkLifecycleState.REGISTERED
            if not self.transfer_backend.recv_from_source(
                model=model,
                seed_instance_ip=self.seed_lease.seed_ip,
                seed_instance_service_port=self.seed_lease.seed_port,
                local_seed_key=self.planner.local_seed_key,
                processed_layout=processed_layout,
            ):
                return False
            return self._release_seed_locked()

    def _release_seed_locked(self) -> bool:
        if self.seed_lease is None:
            return True
        if not self.planner.release_seed(self.seed_lease):
            logger.warning("RFork seed lease release failed; retaining it for retry.")
            return False
        self.seed_lease = None
        if self.state is RForkLifecycleState.LEASED:
            self.state = RForkLifecycleState.INITIALIZED
        return True

    def release_seed(self) -> bool:
        with self._lock:
            return self._release_seed_locked()

    def _reset_transfer_locked(self) -> bool:
        if self.seed_server is not None:
            logger.warning("RFork refuses to unregister memory while the seed server is owned.")
            return False
        if not self.transfer_backend.unregister_memory_region():
            logger.warning("RFork memory remains registered; retaining transfer state for retry.")
            return False
        if self.state is not RForkLifecycleState.FINALIZED:
            self.state = RForkLifecycleState.INITIALIZED
        return True

    def prepare_for_fallback(self) -> bool:
        """Return to an initialized, reusable session without finalizing TE."""
        with self._lock:
            if self.state is RForkLifecycleState.FINALIZED:
                logger.error("RFork cannot prepare a finalized session for fallback.")
                return False
            service_ok = self._stop_seed_service_locked()
            release_ok = self._release_seed_locked()
            reset_ok = self._reset_transfer_locked() if service_ok else False
            return service_ok and release_ok and reset_ok

    def _seed_transfer_info(self) -> SeedTransferInfo:
        session_id = self.transfer_backend.rfork_transfer_engine_session_id
        weights = self.transfer_backend.rfork_transfer_engine_weights_info_dict
        shapes = self.transfer_backend.rfork_transfer_engine_weights_shape_dict
        if not isinstance(session_id, str) or not session_id or not isinstance(weights, dict):
            raise RuntimeError("RFork transfer metadata is unavailable after memory registration.")
        return SeedTransferInfo(session_id=session_id, weights=weights, shapes=shapes)

    def start_seed_service(self, model, processed_layout: bool) -> bool:
        with self._lock:
            if self.state is RForkLifecycleState.SERVING:
                return True
            if self.state not in (RForkLifecycleState.INITIALIZED, RForkLifecycleState.REGISTERED):
                logger.error("RFork seed service cannot start from state=%s", self.state.name)
                return False
            if self.state is not RForkLifecycleState.REGISTERED:
                if not self.transfer_backend.register_memory_region(model, processed_layout):
                    return False
                self.state = RForkLifecycleState.REGISTERED

            try:
                handle = start_rfork_server(
                    self.planner.local_seed_key,
                    self._seed_transfer_info(),
                    health_timeout_sec=self.config.seed_timeout_sec,
                    bind_host=self.config.seed_bind_host,
                    auth_token=self.planner.auth_token,
                )
                self.seed_server = handle
                if not self.planner.report_seed_once(
                    handle.port,
                    seed_ip=self.config.seed_advertise_host,
                ):
                    raise RuntimeError("planner rejected the initial seed advertisement")

                self.heartbeat_stop_event = threading.Event()
                self.heartbeat_thread = threading.Thread(
                    target=self.planner.report_seed,
                    args=(handle.port,),
                    kwargs={
                        "stop_event": self.heartbeat_stop_event,
                        "seed_ip": self.config.seed_advertise_host,
                        "initial_delay": True,
                    },
                    daemon=True,
                    name="RForkHeartbeat",
                )
                self.heartbeat_thread.start()
                self.state = RForkLifecycleState.SERVING
                logger.info(
                    "RFork seed service started for device_id=%s, port=%s",
                    self.identity.device_id,
                    handle.port,
                )
                return True
            except Exception as exc:
                logger.warning("RFork seed service startup failed for device_id=%s: %s", self.identity.device_id, exc)
                service_ok = self._stop_seed_service_locked()
                if service_ok:
                    self._reset_transfer_locked()
                else:
                    self.state = RForkLifecycleState.SERVING
                    logger.warning("RFork seed cleanup is incomplete; registered memory remains pinned.")
                return False

    def _stop_seed_service_locked(self) -> bool:
        self.heartbeat_stop_event.set()
        heartbeat = self.heartbeat_thread
        heartbeat_ok = True
        if heartbeat is not None and heartbeat is not threading.current_thread():
            heartbeat.join(timeout=max(float(self.config.seed_timeout_sec), 0.0))
            heartbeat_ok = not heartbeat.is_alive()
            if not heartbeat_ok:
                logger.warning("RFork heartbeat thread did not stop in time.")

        self.planner.remove_seed()
        server_ok = True
        if self.seed_server is not None:
            try:
                server_ok = self.seed_server.stop()
            except Exception as exc:
                logger.warning("RFork seed server shutdown failed: %s", exc)
                server_ok = False

        stopped = heartbeat_ok and server_ok
        if stopped:
            self.heartbeat_thread = None
            self.seed_server = None
            if self.state is RForkLifecycleState.SERVING:
                self.state = RForkLifecycleState.REGISTERED
        return stopped

    def stop_seed_service(self) -> bool:
        with self._lock:
            return self._stop_seed_service_locked()

    def shutdown(self) -> bool:
        with self._lock:
            if self.state is RForkLifecycleState.FINALIZED:
                return True
            service_ok = self._stop_seed_service_locked()
            release_ok = self._release_seed_locked()
            finalize_ok = self.transfer_backend.finalize_transfer_engine() if service_ok and release_ok else False
            if finalize_ok:
                self.state = RForkLifecycleState.FINALIZED
            elif not service_ok:
                logger.warning("RFork shutdown retained registered memory because the seed server is still alive.")
            elif not release_ok:
                logger.warning("RFork shutdown retained TransferEngine state until the seed lease can be released.")
            else:
                logger.warning("RFork shutdown retained TransferEngine state because finalization did not complete.")
            return service_ok and release_ok and finalize_ok
