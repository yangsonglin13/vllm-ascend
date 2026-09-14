# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

import atexit
import threading
import time
from typing import Any

from vllm.logger import logger

from vllm_ascend.model_loader.rfork.config import RForkConfig
from vllm_ascend.model_loader.rfork.load_state import capture_load_derived_state, restore_load_derived_state
from vllm_ascend.model_loader.rfork.planner_client import RForkPlannerClient, lease_log_id
from vllm_ascend.model_loader.rfork.seed_client import build_seed_url, fetch_seed_transfer_info
from vllm_ascend.model_loader.rfork.seed_server import (
    RForkSeedServerHandle,
    RForkSeedServerStartupError,
    start_rfork_server,
)
from vllm_ascend.model_loader.rfork.transfer_backend import RForkTransferBackend
from vllm_ascend.model_loader.rfork.types import (
    LeaseReleaseResult,
    RForkFallbackCleanupResult,
    RForkIdentity,
    RForkLifecycleState,
    RForkSeedServiceStartResult,
    SeedLease,
    SeedTransferInfo,
)

HEARTBEAT_STOP_GRACE_SEC = 1.0
HEARTBEAT_LOG_EVERY_N = 4
LEASE_RENEW_MIN_INTERVAL_SEC = 0.1
LEASE_RENEW_MAX_INTERVAL_SEC = 30.0


class RForkSession:
    """Sole owner of one worker process's RFork runtime resources."""

    def __init__(self, config: RForkConfig, identity: RForkIdentity) -> None:
        if not config.planner_url:
            raise ValueError(
                "rfork_scheduler_url is required; configure it with model_loader_extra_config or RFORK_SCHEDULER_URL"
            )
        if not config.model_url or not config.model_deploy_strategy_name:
            raise ValueError("RFork requires non-empty model_url and model_deploy_strategy_name")

        self.config = config
        self.identity = identity
        self.planner = RForkPlannerClient(config, identity)
        self.transfer_backend = RForkTransferBackend()
        self.state = RForkLifecycleState.INITIALIZED
        self.seed_lease: SeedLease | None = None
        self.seed_server: RForkSeedServerHandle | None = None
        self.heartbeat_thread: threading.Thread | None = None
        self.heartbeat_stop_event = threading.Event()
        self.lease_release_thread: threading.Thread | None = None
        self.lease_release_stop_event = threading.Event()
        self.lease_renew_thread: threading.Thread | None = None
        self.lease_renew_stop_event = threading.Event()
        self._lease_release_attempts = 0
        self._lease_release_exhausted = False
        self._lease_acquired_at: float | None = None
        self._registration_elapsed = 0.0
        self._deferred_seed_start: tuple[Any, bool, list[tuple[int, int]] | None] | None = None
        # Draft seed start waits until target weight sharing is final; lease-release promotion must not overtake it.
        self._deferred_seed_awaiting_sharing = False
        self._lock = threading.RLock()
        # Acquire before _lock; release responses must not wait on removal I/O.
        self._seed_lifecycle_lock = threading.RLock()
        atexit.register(self.shutdown)

    def register_destination(
        self, model, processed_layout: bool, exclude_blocks: list[tuple[int, int]] | None = None
    ) -> bool:
        """Prepare local buffers on the NPU caller thread before acquiring a lease."""
        with self._lock:
            if (
                self.state is not RForkLifecycleState.INITIALIZED
                or self.seed_lease is not None
                or self.lease_release_stop_event.is_set()
            ):
                return False
            self.state = RForkLifecycleState.CLEANUP_REQUIRED
            started_at = time.monotonic()
            if not self.transfer_backend.register_memory_region(model, processed_layout, exclude_blocks):
                return False
            self._registration_elapsed = time.monotonic() - started_at
            self.state = RForkLifecycleState.REGISTERED
            return True

    def acquire_seed(self) -> bool:
        with self._lock:
            if self.lease_release_stop_event.is_set():
                return False
            if self.lease_release_thread is not None and self.lease_release_thread.is_alive():
                return False
            # Retry unresolved leases only for release; acquisition also needs registered buffers.
            if self.seed_lease is not None:
                self._release_seed_locked()
                return False
            if self.state is not RForkLifecycleState.REGISTERED:
                logger.error("RFork seed acquisition requires registered buffers; state=%s", self.state.name)
                return False
            acquisition_started = time.monotonic()
            self.seed_lease = self.planner.acquire_seed()
            if self.seed_lease is None:
                return False
            self._lease_acquired_at = acquisition_started
            self._lease_release_attempts = 0
            self._lease_release_exhausted = False
            if not self._start_lease_renewal_locked(self.seed_lease):
                self.state = RForkLifecycleState.CLEANUP_REQUIRED
                self._release_seed_locked()
                return False
            logger.debug(
                "RFork lease acquired: lease=%s global_rank=%s request_elapsed=%.3fs",
                lease_log_id(self.seed_lease),
                self.identity.global_rank,
                time.monotonic() - acquisition_started,
            )
            self.state = RForkLifecycleState.LEASED
            return True

    def _start_lease_renewal_locked(self, lease: SeedLease) -> bool:
        stop_event = threading.Event()
        thread = threading.Thread(
            target=self._renew_seed_lease,
            args=(lease, stop_event),
            daemon=True,
            name="RForkLeaseRenewal",
        )
        self.lease_renew_stop_event = stop_event
        self.lease_renew_thread = thread
        try:
            thread.start()
        except RuntimeError:
            self.lease_renew_thread = None
            stop_event.set()
            logger.exception("RFork could not start the lease renewal worker.")
            return False
        return True

    def _renew_seed_lease(self, lease: SeedLease, stop_event: threading.Event) -> None:
        interval = min(
            max(float(lease.lease_ttl_sec) / 3, LEASE_RENEW_MIN_INTERVAL_SEC),
            LEASE_RENEW_MAX_INTERVAL_SEC,
        )
        try:
            while not stop_event.wait(interval):
                self.planner.renew_seed_once(lease)
        finally:
            with self._lock:
                if self.lease_renew_thread is threading.current_thread():
                    self.lease_renew_thread = None

    def _stop_lease_renewal_locked(self) -> None:
        self.lease_renew_stop_event.set()

    def can_reuse_shared_weights(self, model, processed_layout: bool, exclude_blocks: list[tuple[int, int]]) -> bool:
        with self._lock:
            if (
                self.state is not RForkLifecycleState.INITIALIZED
                or self.seed_lease is not None
                or self.lease_release_stop_event.is_set()
            ):
                return False
            return self.transfer_backend.can_reuse_shared_weights(model, processed_layout, exclude_blocks)

    def transfer_from_seed(
        self,
        model,
        processed_layout: bool,
    ) -> bool:
        # Hold _lock through metadata/RDMA so cleanup cannot unregister buffers mid-transfer.
        with self._lock:
            if self.state is not RForkLifecycleState.LEASED or self.seed_lease is None:
                logger.error("RFork transfer requires an acquired seed lease.")
                return False
            # Metadata/read failures require cleanup because destination registration already succeeded.
            self.state = RForkLifecycleState.CLEANUP_REQUIRED
            metadata_started = time.monotonic()
            seed_info = fetch_seed_transfer_info(
                build_seed_url(self.seed_lease.seed_ip, self.seed_lease.seed_port),
                self.planner.seed_key,
                self.config.request_timeout_sec,
            )
            if seed_info is None:
                return False
            # Receiver never runs load_weights; restore seed load-state before post-load/spec-decode sharing reads it.
            restore_load_derived_state(model, seed_info.load_state)
            metadata_elapsed = time.monotonic() - metadata_started
            read_started = time.monotonic()
            if not self.transfer_backend.read_weights_from_seed(
                model=model,
                seed_info=seed_info,
                processed_layout=processed_layout,
            ):
                return False
            self.state = RForkLifecycleState.TRANSFERRED
            logger.debug(
                "RFork transfer stages: lease=%s global_rank=%s registration=%.3fs metadata=%.3fs read=%.3fs",
                lease_log_id(self.seed_lease),
                self.identity.global_rank,
                self._registration_elapsed,
                metadata_elapsed,
                time.monotonic() - read_started,
            )
            # Lease bookkeeping must never put planner network latency on the startup thread.
            self._ensure_lease_release_retry_locked()
            return True

    def _ensure_lease_release_retry_locked(self) -> None:
        self._stop_lease_renewal_locked()
        if (
            self.seed_lease is None
            or self._lease_release_exhausted
            or self.lease_release_stop_event.is_set()
            or (self.lease_release_thread is not None and self.lease_release_thread.is_alive())
        ):
            return
        self.lease_release_thread = threading.Thread(
            target=self._retry_seed_lease_release,
            daemon=True,
            name="RForkLeaseRelease",
        )
        try:
            self.lease_release_thread.start()
        except RuntimeError:
            self.lease_release_thread = None
            self._lease_release_exhausted = True
            logger.exception("RFork could not start lease release worker; retaining unresolved lease.")

    def _retry_seed_lease_release(self) -> None:
        try:
            while not self.lease_release_stop_event.is_set():
                # Wait only between attempts, never before the initial release.
                if self._lease_release_attempts and self.lease_release_stop_event.wait(
                    self.config.lease_release_retry_interval_sec
                ):
                    return
                with self._lock:
                    if self.state is RForkLifecycleState.FINALIZED or self.seed_lease is None:
                        return
                    lease = self.seed_lease
                    self._lease_release_attempts += 1
                # Perform release HTTP outside _lock, then reacquire it to update state.
                try:
                    result = self.planner.release_seed_once(lease)
                except Exception:
                    logger.exception("RFork background lease release raised; retaining unresolved lease.")
                    result = LeaseReleaseResult.REJECTED
                with self._lock:
                    if self.seed_lease is not lease:
                        return
                    finished = self._record_lease_release_locked(lease, result)
                if finished:
                    self._promote_deferred_seed()
                    return

        finally:
            with self._lock:
                if self.lease_release_thread is threading.current_thread():
                    self.lease_release_thread = None

    def _record_lease_release_locked(self, lease: SeedLease, result: LeaseReleaseResult) -> bool:
        """Apply one release response; True stops retries."""
        acquired_at = self._lease_acquired_at
        logger.debug(
            "RFork lease release outcome: lease=%s attempt=%d/%d result=%s held_elapsed=%.3fs",
            lease_log_id(lease),
            self._lease_release_attempts,
            self.config.lease_release_max_attempts,
            result.name,
            time.monotonic() - acquired_at if acquired_at is not None else 0.0,
        )
        if result is LeaseReleaseResult.RELEASED:
            self.seed_lease = None
            self._lease_acquired_at = None
            if self.state is RForkLifecycleState.LEASED:
                self.state = RForkLifecycleState.REGISTERED
            return True
        if (
            result is LeaseReleaseResult.REJECTED
            or self._lease_release_attempts >= self.config.lease_release_max_attempts
        ):
            self._lease_release_exhausted = True
            self._deferred_seed_start = None
            self._deferred_seed_awaiting_sharing = False
            logger.error(
                "RFork lease release stopped: lease=%s attempts=%d; lease remains unresolved, "
                "worker will not advertise a seed. Model loading/inference may continue.",
                lease_log_id(lease),
                self._lease_release_attempts,
            )
            return True
        return False

    def _release_seed_locked(self) -> bool:
        """Schedule release without waiting; True only after an acknowledged release."""
        if self.seed_lease is None:
            return True
        self._ensure_lease_release_retry_locked()
        return False

    def _promote_deferred_seed(self) -> None:
        with self._seed_lifecycle_lock:
            with self._lock:
                if (
                    self._deferred_seed_start is None
                    or self._deferred_seed_awaiting_sharing
                    or self.seed_lease is not None
                    or self.state is RForkLifecycleState.FINALIZED
                    or self.lease_release_stop_event.is_set()
                ):
                    return
                model, processed_layout, exclude_blocks = self._deferred_seed_start
                self._deferred_seed_start = None
            try:
                promoted = self._start_seed_service(model, processed_layout, exclude_blocks)
            except Exception:
                logger.exception(
                    "RFork deferred seed promotion raised after the source lease was released; "
                    "the model remains available for inference."
                )
                promoted = False
            if not promoted:
                self._cleanup_failed_seed_start()
                logger.warning(
                    "RFork deferred seed promotion failed after the source lease was released; "
                    "the model remains available for inference."
                )

    def schedule_deferred_seed_start(
        self,
        model,
        processed_layout: bool,
        exclude_blocks: list[tuple[int, int]] | None = None,
    ) -> RForkSeedServiceStartResult:
        """Defer draft registration until target weight sharing finishes."""
        with self._seed_lifecycle_lock:
            with self._lock:
                if self.lease_release_stop_event.is_set():
                    return RForkSeedServiceStartResult.FAILED
                if self._deferred_seed_start is not None:
                    logger.error("RFork cannot schedule a deferred seed start while one is already pending.")
                    return RForkSeedServiceStartResult.FAILED
                if self.state not in (RForkLifecycleState.INITIALIZED, RForkLifecycleState.TRANSFERRED):
                    logger.error(
                        "RFork deferred draft seed start requires a complete model; state=%s",
                        self.state.name,
                    )
                    return RForkSeedServiceStartResult.FAILED
                self._deferred_seed_start = (model, processed_layout, exclude_blocks)
                self._deferred_seed_awaiting_sharing = True
            logger.info("RFork draft seed start is deferred until weight sharing with the target model is complete.")
            return RForkSeedServiceStartResult.DEFERRED

    def has_deferred_seed_start(self) -> bool:
        with self._lock:
            return self._deferred_seed_start is not None

    def get_seed_shared_names(self) -> tuple[str, ...]:
        """Return weights skipped because the seed shared them with its target."""
        with self._lock:
            names = self.transfer_backend.seed_shared_names
        if not isinstance(names, (list, tuple)):
            return ()
        return tuple(names)

    def complete_deferred_seed_start(self) -> RForkSeedServiceStartResult:
        """Register and publish the draft after target sharing finishes."""
        with self._seed_lifecycle_lock:
            with self._lock:
                self._deferred_seed_awaiting_sharing = False
                pending = self._deferred_seed_start
                if pending is None:
                    return RForkSeedServiceStartResult.FAILED
                if self.state is RForkLifecycleState.FINALIZED or self.lease_release_stop_event.is_set():
                    return RForkSeedServiceStartResult.FAILED
                model, processed_layout, exclude_blocks = pending
                if self.seed_lease is None:
                    # No lease to wait for: clear the stash; start_seed_service would otherwise re-stash it.
                    self._deferred_seed_start = None
            # Weight sharing replaced tensors post-transfer, so rebuild registration even for processed-layout models.
            return self.start_seed_service(model, processed_layout, exclude_blocks, refresh_registration=True)

    def _reset_transfer_locked(self) -> bool:
        if self.seed_server is not None:
            self.state = RForkLifecycleState.CLEANUP_REQUIRED
            logger.warning("RFork refuses to unregister memory while the seed server is owned.")
            return False
        try:
            reset = self.transfer_backend.unregister_memory_region()
        except Exception as exc:
            self.state = RForkLifecycleState.CLEANUP_REQUIRED
            logger.warning("RFork memory unregistration raised: %s", exc)
            return False
        if not reset:
            self.state = RForkLifecycleState.CLEANUP_REQUIRED
            logger.warning("RFork memory remains registered; retaining transfer state for retry.")
            return False
        if self.state is not RForkLifecycleState.FINALIZED:
            self.state = RForkLifecycleState.INITIALIZED
        return True

    def prepare_for_fallback(self) -> RForkFallbackCleanupResult:
        """Return to an initialized, reusable session without finalizing TransferEngine."""
        with self._seed_lifecycle_lock:
            with self._lock:
                if self.state is RForkLifecycleState.FINALIZED:
                    logger.error("RFork cannot prepare a finalized session for fallback.")
                    return RForkFallbackCleanupResult(False, False, False)
                self._deferred_seed_start = None
                self._deferred_seed_awaiting_sharing = False
            service_ok = self._stop_seed_service()
            with self._lock:
                release_ok = self._release_seed_locked()
                reset_ok = self._reset_transfer_locked() if service_ok else False
                return RForkFallbackCleanupResult(service_ok, release_ok, reset_ok)

    def _seed_transfer_info(self, load_state: dict[str, Any] | None = None) -> SeedTransferInfo:
        session_id = self.transfer_backend.transfer_session_id
        weights = self.transfer_backend.weight_manifest
        if not isinstance(session_id, str) or not session_id or not isinstance(weights, dict) or not weights:
            raise RuntimeError("RFork transfer metadata is unavailable after memory registration.")
        return SeedTransferInfo(
            session_id=session_id,
            weights=weights,
            shared_names=tuple(self.transfer_backend.shared_with_target_names) or None,
            formats=dict(self.transfer_backend.weight_formats) if self.transfer_backend.weight_formats else None,
            load_state=dict(load_state) if load_state else None,
        )

    def start_seed_service(
        self,
        model,
        processed_layout: bool,
        exclude_blocks: list[tuple[int, int]] | None = None,
        *,
        refresh_registration: bool = False,
    ) -> RForkSeedServiceStartResult:
        with self._seed_lifecycle_lock:
            with self._lock:
                if self.lease_release_stop_event.is_set():
                    return RForkSeedServiceStartResult.FAILED
                if self.state is RForkLifecycleState.SERVING:
                    return RForkSeedServiceStartResult.STARTED
                if self.seed_lease is not None and self._lease_release_exhausted:
                    return RForkSeedServiceStartResult.FAILED
                if self.state not in (
                    RForkLifecycleState.INITIALIZED,
                    RForkLifecycleState.LEASED,
                    RForkLifecycleState.TRANSFERRED,
                    RForkLifecycleState.READY,
                ):
                    logger.error("RFork seed service requires a complete model; state=%s", self.state.name)
                    return RForkSeedServiceStartResult.FAILED
                if self.state is not RForkLifecycleState.READY:
                    # Refresh registration after post-load storage changes; processed layouts already use final buffers.
                    if (
                        refresh_registration
                        or self.state is not RForkLifecycleState.TRANSFERRED
                        or not processed_layout
                    ):
                        self.state = RForkLifecycleState.CLEANUP_REQUIRED
                        try:
                            registered = self.transfer_backend.register_memory_region(
                                model, processed_layout, exclude_blocks
                            )
                        except Exception:
                            logger.exception(
                                "RFork seed memory registration raised; cleaning up before continuing inference."
                            )
                            registered = False
                        if not registered:
                            self._reset_transfer_locked()
                            return RForkSeedServiceStartResult.FAILED
                    self.state = RForkLifecycleState.READY
                if self.seed_lease is not None:
                    if self._lease_release_exhausted or self.lease_release_stop_event.is_set():
                        return RForkSeedServiceStartResult.FAILED
                    self._deferred_seed_start = (model, processed_layout, exclude_blocks)
                    # Deferral waits only for the source lease release; sharing deferral was completed by the caller.
                    self._deferred_seed_awaiting_sharing = False
                    self._ensure_lease_release_retry_locked()
                    logger.debug(
                        "RFork seed promotion is deferred until the source seed lease is released; "
                        "the transferred model remains available for inference."
                    )
                    return RForkSeedServiceStartResult.DEFERRED
            started = self._start_seed_service(model, processed_layout, exclude_blocks)
            if not started:
                self._cleanup_failed_seed_start()
            return RForkSeedServiceStartResult.STARTED if started else RForkSeedServiceStartResult.FAILED

    def _start_seed_service(
        self,
        model,
        processed_layout: bool,
        exclude_blocks: list[tuple[int, int]] | None = None,
    ) -> bool:
        # Keep health/planner I/O outside _lock; transitional state blocks transfer and registration.
        with self._lock:
            if self.state is RForkLifecycleState.SERVING:
                return True
            if self.state is not RForkLifecycleState.READY:
                logger.error("RFork seed service cannot start from state=%s", self.state.name)
                return False
            if self.seed_lease is not None or self.lease_release_stop_event.is_set():
                logger.error("RFork seed service cannot start with an unresolved lease or during shutdown.")
                return False
            self.state = RForkLifecycleState.CLEANUP_REQUIRED
        try:
            info = self._seed_transfer_info(capture_load_derived_state(model))
            handle = start_rfork_server(
                self.planner.seed_key,
                info,
                health_timeout_sec=self.config.seed_timeout_sec,
                bind_host=self.config.seed_bind_host,
            )
            with self._lock:
                self.seed_server = handle
                if self.lease_release_stop_event.is_set():
                    raise RuntimeError("shutdown requested during seed server startup")
            if not handle.is_alive:
                raise RuntimeError("seed HTTP server exited before advertisement")
            if not self.planner.report_seed_once(handle.port, seed_ip=self.config.seed_advertise_host):
                raise RuntimeError("planner rejected the initial seed advertisement")

            with self._lock:
                if self.lease_release_stop_event.is_set():
                    raise RuntimeError("shutdown requested during seed advertisement")
                if not handle.is_alive:
                    raise RuntimeError("seed HTTP server exited during advertisement")
                self.heartbeat_stop_event = threading.Event()
                self.heartbeat_thread = threading.Thread(
                    target=self._run_seed_heartbeat,
                    args=(handle, self.heartbeat_stop_event),
                    daemon=True,
                    name="RForkHeartbeat",
                )
                try:
                    self.heartbeat_thread.start()
                except RuntimeError:
                    self.heartbeat_thread = None
                    raise
                self.state = RForkLifecycleState.SERVING
            logger.debug(
                "RFork seed service started for global_rank=%s, port=%s",
                self.identity.global_rank,
                handle.port,
            )
            return True
        except Exception as exc:
            with self._lock:
                if isinstance(exc, RForkSeedServerStartupError):
                    self.seed_server = exc.handle
                self.state = RForkLifecycleState.CLEANUP_REQUIRED
            logger.warning("RFork seed service startup failed for global_rank=%s: %s", self.identity.global_rank, exc)
            return False

    def _run_seed_heartbeat(self, handle: RForkSeedServerHandle, stop_event: threading.Event) -> None:
        # Do not take _seed_lifecycle_lock: shutdown owns it while joining this thread; native cleanup stays elsewhere.
        heartbeat_index = 0
        while not stop_event.wait(self.config.heartbeat_interval_sec):
            if not handle.is_alive:
                break
            heartbeat_index += 1
            try:
                reported = self.planner.report_seed_once(handle.port, seed_ip=self.config.seed_advertise_host)
            except Exception:
                # report_seed_once usually swallows errors; this guard is for the rest, so heartbeat exits via cleanup.
                logger.exception("RFork heartbeat raised; withdrawing the seed.")
                break
            # Server may exit while add_seed is in flight; revoke that advertisement even if the report succeeded.
            if not handle.is_alive:
                break
            if not reported:
                logger.warning("RFork heartbeat failed for seed_key=%s", self.planner.seed_key)
            elif heartbeat_index % HEARTBEAT_LOG_EVERY_N == 0:
                logger.debug("RFork heartbeat accepted for seed_key=%s", self.planner.seed_key)
        else:
            return

        with self._lock:
            if stop_event.is_set() or self.seed_server is not handle:
                return
            stop_event.set()
            self.state = RForkLifecycleState.CLEANUP_REQUIRED
        logger.error("RFork seed heartbeat stopped after a service failure; inference can continue.")
        try:
            removed = self.planner.remove_seed()
        except Exception:
            logger.exception("RFork failed to withdraw the seed after a service failure.")
            removed = False
        if not removed:
            logger.warning("RFork seed removal remains pending; heartbeats stopped, retaining registered memory.")
        # Retain the handle and registrations until normal cleanup; removal alone does not prove native reads finished.

    def _cleanup_failed_seed_start(self) -> None:
        # The caller owns _seed_lifecycle_lock, but must not hold _lock here.
        service_ok = self._stop_seed_service()
        with self._lock:
            if service_ok:
                self._reset_transfer_locked()
            else:
                self.state = RForkLifecycleState.CLEANUP_REQUIRED
                logger.warning("RFork seed cleanup is incomplete; registered memory remains pinned.")

    def _stop_seed_service(self) -> bool:
        """Caller serializes seed lifecycle; join and removal run outside the state lock."""
        with self._lock:
            self.state = RForkLifecycleState.CLEANUP_REQUIRED
            self.heartbeat_stop_event.set()
            heartbeat = self.heartbeat_thread
            server = self.seed_server
        if heartbeat is not None and heartbeat is not threading.current_thread():
            # Join add_seed before removal; Requests timeout bounds inactivity, not total duration.
            join_timeout_sec = max(
                float(self.config.seed_timeout_sec),
                2 * float(self.config.request_timeout_sec) + HEARTBEAT_STOP_GRACE_SEC,
            )
            heartbeat.join(timeout=join_timeout_sec)
            if heartbeat.is_alive():
                logger.warning("RFork heartbeat thread did not stop in time.")
                return False

        try:
            removed = self.planner.remove_seed()
        except Exception as exc:
            logger.warning("RFork planner seed removal raised: %s", exc)
            removed = False
        if not removed:
            logger.warning("RFork planner seed removal failed; keeping the local seed service available for retry.")
            return False
        if server is not None:
            try:
                if not server.stop():
                    return False
            except Exception as exc:
                logger.warning("RFork seed server shutdown failed: %s", exc)
                return False
        with self._lock:
            self.heartbeat_thread = None
            self.seed_server = None
        return True

    def shutdown(self) -> bool:
        # Avoid release I/O; existing attempts may finish, otherwise planner expiry reclaims the lease.
        self.lease_release_stop_event.set()
        self.lease_renew_stop_event.set()
        with self._seed_lifecycle_lock:
            with self._lock:
                if self.state is RForkLifecycleState.FINALIZED:
                    return True
                self._deferred_seed_start = None
                self._deferred_seed_awaiting_sharing = False
            service_ok = self._stop_seed_service()
            with self._lock:
                release_ok = self.seed_lease is None
                finalize_ok = self.transfer_backend.finalize_transfer_engine() if service_ok and release_ok else False
                if finalize_ok:
                    self.state = RForkLifecycleState.FINALIZED
                elif not service_ok:
                    logger.warning(
                        "RFork shutdown retained registered memory because seed service cleanup is incomplete."
                    )
                elif not release_ok:
                    logger.warning(
                        "RFork shutdown retained TransferEngine state because the source lease is unresolved. "
                        "No new release retries will be started; an in-flight request may still acknowledge. "
                        "Otherwise lease recovery depends on the planner's expiry/reclamation policy."
                    )
                else:
                    logger.warning(
                        "RFork shutdown retained TransferEngine state because finalization did not complete."
                    )
                return service_ok and release_ok and finalize_ok
