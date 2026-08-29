#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import atexit
import threading

from vllm.logger import logger

from vllm_ascend.model_loader.rfork.seed_protocol import RForkSeedProtocol
from vllm_ascend.model_loader.rfork.seed_server import (
    RForkSeedServerHandle,
    start_rfork_server,
)
from vllm_ascend.model_loader.rfork.transfer_backend import (
    RForkTransferBackend,
)


class RForkWorker:
    def __init__(
        self,
        disaggregation_mode: str,
        node_rank: int,
        tp_rank: int,
        device_id: int,
        scheduler_url: str,
        model_url: str,
        model_deploy_strategy_name: str,
        seed_timeout_sec: float = 30.0,
        seed_key_separator: str = "$",
        is_draft_model: bool = False,
        pp_rank: int | None = None,
        ep_rank: int | None = None,
        compatibility_fingerprint: str | None = None,
        request_timeout_sec: float = 10.0,
        auth_token: str | None = None,
        seed_bind_host: str = "0.0.0.0",
        seed_advertise_host: str | None = None,
    ):
        if not scheduler_url:
            raise ValueError(
                "rfork_scheduler_url is required; configure it with model_loader_extra_config or "
                "VLLM_ASCEND_RFORK_SCHEDULER_URL"
            )
        if not model_url or not model_deploy_strategy_name:
            raise ValueError("RFork requires non-empty model_url and model_deploy_strategy_name")
        self.device_id = device_id
        self.rfork_seed = None
        self.transfer_backend = RForkTransferBackend(
            request_timeout_sec=request_timeout_sec,
            auth_token=auth_token or "",
        )
        self.ready_to_start_seed_service = False
        self.seed_service_started = False
        self.seed_timeout_sec = seed_timeout_sec
        self.seed_bind_host = seed_bind_host
        self.seed_advertise_host = seed_advertise_host
        self.seed_server_handle: RForkSeedServerHandle | object | None = None
        self.rfork_heartbeat_thread: threading.Thread | None = None
        self._heartbeat_stop_event = threading.Event()
        self.seed_protocol = RForkSeedProtocol(
            disaggregation_mode=disaggregation_mode,
            node_rank=node_rank,
            tp_rank=tp_rank,
            scheduler_url=scheduler_url,
            model_url=model_url,
            model_deploy_strategy_name=model_deploy_strategy_name,
            seed_key_separator=seed_key_separator,
            is_draft_worker=is_draft_model,
            pp_rank=pp_rank,
            ep_rank=ep_rank,
            compatibility_fingerprint=compatibility_fingerprint,
            request_timeout_sec=request_timeout_sec,
            auth_token=auth_token,
        )
        # Register cleanup as soon as the worker can own a planner lease or a
        # memory region, not only after seed-service startup. This also covers
        # transfer/fallback paths whose unregister or lease release failed.
        atexit.register(self.shutdown)

    def is_seed_available(self) -> bool:
        # Do not overwrite a lease whose release previously failed. Retry the
        # release first so a subsequent model load cannot orphan the old lease.
        if self.rfork_seed is not None and not self.post_transfer():
            return False
        self.rfork_seed = self.seed_protocol.get_seed()
        return self.rfork_seed is not None

    def pre_transfer(self, model, processed_layout: bool) -> bool:
        try:
            assert self.transfer_backend.is_initialized(), "transfer_backend is not initialized, cannot pre_transfer."
            result = self.transfer_backend.register_memory_region(model, processed_layout)
            self.ready_to_start_seed_service = result
            return result
        except AssertionError as e:
            logger.exception("Pre-transfer failed for device_id=%s: %s", self.device_id, e)
            return False

    def reset_transfer_state(self) -> bool:
        try:
            reset_result = self.transfer_backend.unregister_memory_region()
        except Exception as e:
            logger.warning("Failed to unregister rfork memory region: %s", e)
            return False
        if reset_result is False:
            logger.warning("RFork memory region remains registered; retaining transfer state for retry.")
            return False
        self.ready_to_start_seed_service = False
        return True

    def transfer(self, model, processed_layout: bool) -> bool:
        try:
            assert self.transfer_backend.is_initialized(), "transfer_backend is not initialized, cannot transfer."
            assert self.rfork_seed is not None, "rfork seed is None, cannot transfer."
            return self.transfer_backend.recv_from_source(
                model=model,
                seed_instance_ip=self.rfork_seed["seed_ip"],
                seed_instance_service_port=self.rfork_seed["seed_port"],
                local_seed_key=self.seed_protocol.get_local_seed_key(),
                processed_layout=processed_layout,
            )
        except AssertionError as e:
            logger.exception(
                "Transfer failed for device_id=%s: %s",
                self.device_id,
                e,
            )
            return False

    def post_transfer(self) -> bool:
        if self.rfork_seed is None:
            logger.info("rfork seed is None, no need to release.")
            return True
        released = self.seed_protocol.release_seed(self.rfork_seed)
        if released:
            self.rfork_seed = None
        else:
            logger.warning("RFork seed lease release failed; retaining lease for retry.")
        return released

    def start_seed_service(self, model, processed_layout: bool) -> bool:
        if self.seed_service_started:
            logger.info("Seed service already started, skipping.")
            return True

        if not self.ready_to_start_seed_service:
            if not self.pre_transfer(model, processed_layout):
                logger.warning(
                    "start_seed_service aborted for device_id=%s: pre_transfer failed",
                    self.device_id,
                )
                return False

        server_handle = None
        port: int | None = None
        try:
            server_handle = start_rfork_server(
                self.seed_protocol.get_local_seed_key(),
                (
                    self.transfer_backend.rfork_transfer_engine_session_id,
                    self.transfer_backend.rfork_transfer_engine_weights_info_dict,
                    self.transfer_backend.rfork_transfer_engine_weights_shape_dict,
                ),
                health_timeout_sec=self.seed_timeout_sec,
                bind_host=self.seed_bind_host,
                auth_token=self.seed_protocol.auth_token,
            )
            port = getattr(server_handle, "port", server_handle)
            if not isinstance(port, int) or port <= 0:
                logger.warning("start_seed_service failed for device_id=%s", self.device_id)
                if hasattr(server_handle, "stop"):
                    server_handle.stop()
                self.reset_transfer_state()
                return False

            # Take ownership before planner advertisement so every subsequent
            # failure can retry stopping this exact server handle.
            self.seed_server_handle = server_handle
            if not self.seed_protocol.report_seed_once(port, seed_ip=self.seed_advertise_host):
                logger.warning("RFork planner rejected the initial seed advertisement for port=%s", port)
                server_stop = getattr(server_handle, "stop", None)
                stopped = bool(server_stop()) if callable(server_stop) else False
                if stopped:
                    self.seed_server_handle = None
                    self.reset_transfer_state()
                else:
                    self.seed_service_started = True
                    logger.warning("RFork seed server did not stop; retaining registered memory for safety.")
                return False

            self._heartbeat_stop_event = threading.Event()
            self.rfork_heartbeat_thread = threading.Thread(
                target=self.seed_protocol.report_seed,
                args=(port,),
                kwargs={
                    "stop_event": self._heartbeat_stop_event,
                    "seed_ip": self.seed_advertise_host,
                    "initial_delay": True,
                },
                daemon=True,
                name="RForkHeartbeat",
            )
            self.rfork_heartbeat_thread.start()
            self.seed_service_started = True
            logger.info("Seed service started for device_id=%s, port=%s", self.device_id, port)
            return True
        except Exception as e:
            logger.warning("start_seed_service failed for device_id=%s: %s", self.device_id, e)
            self._heartbeat_stop_event.set()
            heartbeat_thread = self.rfork_heartbeat_thread
            if heartbeat_thread is not None and heartbeat_thread is not threading.current_thread():
                heartbeat_thread.join(timeout=max(float(self.seed_timeout_sec), 0.0))
            if isinstance(port, int) and port > 0:
                self.seed_protocol.remove_seed(port=port, seed_ip=self.seed_advertise_host)
            server_stopped = server_handle is None
            if server_handle is not None and hasattr(server_handle, "stop"):
                try:
                    server_stopped = bool(server_handle.stop())
                except Exception:
                    logger.exception("Failed to stop RFork seed server after startup failure.")
                    server_stopped = False
            heartbeat_stopped = heartbeat_thread is None or not heartbeat_thread.is_alive()
            if server_stopped and heartbeat_stopped:
                self.seed_server_handle = None
                self.seed_service_started = False
                self.rfork_heartbeat_thread = None
                self.reset_transfer_state()
            else:
                self.seed_service_started = True
                logger.warning("RFork startup cleanup is incomplete; retaining registered memory for safety.")
            return False

    def stop_seed_service(self) -> bool:
        """Stop heartbeat and HTTP advertising; safe to call repeatedly."""
        stop_ok = True
        stop_event = self._heartbeat_stop_event
        stop_event.set()
        heartbeat_thread = self.rfork_heartbeat_thread
        if heartbeat_thread is not None and heartbeat_thread is not threading.current_thread():
            heartbeat_thread.join(timeout=max(float(self.seed_timeout_sec), 0.0))
            if heartbeat_thread.is_alive():
                logger.warning("RFork heartbeat thread did not stop in time.")
                stop_ok = False

        server_handle = self.seed_server_handle
        if server_handle is not None:
            port = getattr(server_handle, "port", server_handle if isinstance(server_handle, int) else None)
        else:
            port = None
        if port is not None:
            # Planner removal is idempotent and best-effort. A failed remove
            # is reported but does not prevent local server cleanup attempts.
            removed = self.seed_protocol.remove_seed(port=port, seed_ip=self.seed_advertise_host)
            if not removed:
                logger.warning("RFork seed removal from planner failed for port=%s", port)

        if server_handle is not None:
            try:
                server_stop = getattr(server_handle, "stop", None)
                if callable(server_stop):
                    stop_ok = bool(server_stop()) and stop_ok
                else:
                    logger.warning("RFork seed server handle does not support explicit shutdown.")
                    stop_ok = False
            except Exception as e:
                logger.warning("Failed to stop RFork seed server: %s", e)
                stop_ok = False

        if stop_ok:
            self.seed_service_started = False
            self.seed_server_handle = None
            self.rfork_heartbeat_thread = None
        return stop_ok

    def shutdown(self) -> bool:
        """Release all RFork resources; safe for explicit and atexit calls."""
        service_ok = self.stop_seed_service()
        release_ok = self.post_transfer()
        unregister_ok = self.reset_transfer_state() if service_ok else False
        if not service_ok:
            logger.warning("RFork shutdown retained registered memory because the seed server is still alive.")
        return service_ok and release_ok and unregister_ok
