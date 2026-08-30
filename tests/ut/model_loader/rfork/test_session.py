# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

import threading
from types import SimpleNamespace

import pytest

from vllm_ascend.model_loader.rfork import session as session_module
from vllm_ascend.model_loader.rfork.config import RForkConfig
from vllm_ascend.model_loader.rfork.types import (
    RForkIdentity,
    RForkLifecycleState,
    SeedAdvertisement,
    SeedLease,
)


class _FakePlanner:
    def __init__(self, config, identity):
        self.config = config
        self.identity = identity
        self.local_seed_key = "key"
        self.lease = SeedLease("127.0.0.1", 1234, "lease", 0, "key")
        self.acquire_result = self.lease
        self.release_result = True
        self.remove_result = True
        self.report_result = True
        self.last_advertisement = None
        self.calls = []
        self.acquire_calls = 0

    def acquire_seed(self):
        self.acquire_calls += 1
        return self.acquire_result

    def release_seed(self, lease):
        self.calls.append(("release", lease))
        return self.release_result

    def remove_seed(self, advertisement=None):
        self.calls.append(("remove", advertisement))
        return self.remove_result

    def report_seed_once(self, port, seed_ip=None):
        self.calls.append(("report", port, seed_ip))
        if self.report_result:
            self.last_advertisement = SeedAdvertisement(seed_ip or "127.0.0.1", port, 0)
        return self.report_result

    def report_seed(self, port, **kwargs):
        self.calls.append(("heartbeat", port))
        kwargs["stop_event"].wait(2)


class _FakeBackend:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.register_result = True
        self.transfer_result = True
        self.unregister_result = True
        self.finalize_result = True
        self.register_calls = 0
        self.transfer_calls = 0
        self.unregister_calls = 0
        self.finalize_calls = 0
        self.exclude_blocks_history: list[list[tuple[int, int]] | None] = []
        self.rfork_transfer_engine_session_id = "session"
        self.rfork_transfer_engine_weights_info_dict = {"weight": (1, 1, 1)}
        self.rfork_transfer_engine_weights_shape_dict = {"weight": (1,)}

    def register_memory_region(self, model, processed_layout, exclude_blocks=None):
        self.register_calls += 1
        self.exclude_blocks_history.append(exclude_blocks)
        return self.register_result

    def recv_from_source(self, **kwargs):
        self.transfer_calls += 1
        return self.transfer_result

    def unregister_memory_region(self):
        self.unregister_calls += 1
        return self.unregister_result

    def finalize_transfer_engine(self):
        self.finalize_calls += 1
        return self.finalize_result


def _make_session(monkeypatch, **config_overrides):
    monkeypatch.setattr(session_module, "RForkPlannerClient", _FakePlanner)
    monkeypatch.setattr(session_module, "RForkTransferBackend", _FakeBackend)
    monkeypatch.setattr(session_module.atexit, "register", lambda callback: None)
    config_values = {
        "model_url": "/model",
        "model_deploy_strategy_name": "decode",
        "scheduler_url": "http://planner",
    }
    config_values.update(config_overrides)
    identity = RForkIdentity(tp_rank=0, device_id=0, compatibility_fingerprint="fp")
    return session_module.RForkSession(RForkConfig(**config_values), identity)


def test_constructor_passes_typed_config_and_identity(monkeypatch):
    session = _make_session(
        monkeypatch,
        request_timeout_sec=1.5,
        seed_bind_host="::1",
        seed_advertise_host="2001:db8::1",
    )
    assert session.planner.config is session.config
    assert session.planner.identity is session.identity
    assert session.planner.identity.compatibility_fingerprint == "fp"
    assert session.transfer_backend.kwargs == {"request_timeout_sec": 1.5}
    assert session.state is RForkLifecycleState.INITIALIZED


def test_constructor_rejects_missing_required_identity(monkeypatch):
    with pytest.raises(ValueError, match="scheduler_url"):
        _make_session(monkeypatch, scheduler_url="")
    with pytest.raises(ValueError, match="model_url"):
        _make_session(monkeypatch, model_url="")


def test_acquire_transfer_release_updates_state(monkeypatch):
    session = _make_session(monkeypatch)
    assert session.acquire_seed()
    assert session.state is RForkLifecycleState.LEASED
    exclude_blocks = [(4096, 128)]
    assert session.transfer_from_seed(object(), False, exclude_blocks)
    assert session.seed_lease is None
    assert session.state is RForkLifecycleState.REGISTERED
    assert session.transfer_backend.register_calls == 1
    assert session.transfer_backend.transfer_calls == 1
    assert session.transfer_backend.exclude_blocks_history == [exclude_blocks]


def test_transfer_requires_a_seed_lease(monkeypatch):
    session = _make_session(monkeypatch)
    assert not session.transfer_from_seed(object(), False)
    assert session.transfer_backend.register_calls == 0


def test_acquire_rejects_non_initialized_session(monkeypatch):
    session = _make_session(monkeypatch)
    session.state = RForkLifecycleState.REGISTERED
    assert not session.acquire_seed()
    assert session.planner.acquire_calls == 0


def test_seed_service_rejects_outstanding_lease(monkeypatch):
    session = _make_session(monkeypatch)
    assert session.acquire_seed()
    assert not session.start_seed_service(object(), False)
    assert session.transfer_backend.register_calls == 0


def test_failed_release_retains_typed_lease_for_retry(monkeypatch):
    session = _make_session(monkeypatch)
    assert session.acquire_seed()
    session.planner.release_result = False
    assert not session.release_seed()
    assert session.seed_lease is session.planner.lease
    session.planner.release_result = True
    assert session.release_seed()
    assert session.seed_lease is None


def test_prepare_for_fallback_resets_registered_memory(monkeypatch):
    session = _make_session(monkeypatch)
    session.state = RForkLifecycleState.REGISTERED
    assert session.prepare_for_fallback()
    assert session.transfer_backend.unregister_calls == 1
    assert session.state is RForkLifecycleState.INITIALIZED


def test_prepare_for_fallback_preserves_failed_unregistration(monkeypatch):
    session = _make_session(monkeypatch)
    session.state = RForkLifecycleState.REGISTERED
    session.transfer_backend.unregister_result = False
    assert not session.prepare_for_fallback()
    assert session.state is RForkLifecycleState.REGISTERED


def test_prepare_for_fallback_rejects_finalized_session(monkeypatch):
    session = _make_session(monkeypatch)
    session.state = RForkLifecycleState.FINALIZED
    assert not session.prepare_for_fallback()
    assert session.transfer_backend.unregister_calls == 0


def test_seed_service_start_and_stop_are_owned_by_session(monkeypatch):
    session = _make_session(monkeypatch, seed_advertise_host="127.0.0.1")
    stopped = []

    class Handle:
        port = 4567

        def stop(self):
            stopped.append(True)
            return True

    monkeypatch.setattr(session_module, "start_rfork_server", lambda *args, **kwargs: Handle())
    assert session.start_seed_service(SimpleNamespace(), False)
    assert session.state is RForkLifecycleState.SERVING
    assert session.heartbeat_thread is not None
    assert session.stop_seed_service()
    assert session.stop_seed_service()
    assert stopped == [True]
    assert session.state is RForkLifecycleState.REGISTERED


def test_seed_start_failure_cleans_registration(monkeypatch):
    session = _make_session(monkeypatch)
    monkeypatch.setattr(
        session_module,
        "start_rfork_server",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("start failed")),
    )
    assert not session.start_seed_service(object(), False)
    assert session.transfer_backend.unregister_calls == 1
    assert session.state is RForkLifecycleState.INITIALIZED


def test_planner_rejection_stops_server_and_cleans_registration(monkeypatch):
    session = _make_session(monkeypatch)
    session.planner.report_result = False
    stopped = []

    class Handle:
        port = 4567

        def stop(self):
            stopped.append(True)
            return True

    monkeypatch.setattr(session_module, "start_rfork_server", lambda *args, **kwargs: Handle())
    assert not session.start_seed_service(object(), False)
    assert stopped == [True]
    assert session.transfer_backend.unregister_calls == 1


def test_shutdown_retains_memory_when_server_cannot_stop(monkeypatch):
    session = _make_session(monkeypatch)
    session.state = RForkLifecycleState.SERVING

    class Handle:
        port = 4567

        def stop(self):
            return False

    session.seed_server = Handle()
    assert not session.shutdown()
    assert session.transfer_backend.finalize_calls == 0
    assert session.state is RForkLifecycleState.SERVING


def test_shutdown_finalizes_without_unregistering_first(monkeypatch):
    session = _make_session(monkeypatch)
    session.state = RForkLifecycleState.REGISTERED
    assert session.shutdown()
    assert session.transfer_backend.finalize_calls == 1
    assert session.transfer_backend.unregister_calls == 0
    assert session.state is RForkLifecycleState.FINALIZED
    assert session.shutdown()
    assert session.transfer_backend.finalize_calls == 1


def test_shutdown_preserves_state_when_finalize_is_not_ready(monkeypatch):
    session = _make_session(monkeypatch)
    session.state = RForkLifecycleState.REGISTERED
    session.transfer_backend.finalize_result = False
    assert not session.shutdown()
    assert session.state is RForkLifecycleState.REGISTERED
    session.transfer_backend.finalize_result = True
    assert session.shutdown()
    assert session.state is RForkLifecycleState.FINALIZED


def test_shutdown_retries_lease_release_before_finalizing(monkeypatch):
    session = _make_session(monkeypatch)
    assert session.acquire_seed()
    session.planner.release_result = False
    assert not session.shutdown()
    assert session.transfer_backend.finalize_calls == 0
    assert session.seed_lease is session.planner.lease

    session.planner.release_result = True
    assert session.shutdown()
    assert session.transfer_backend.finalize_calls == 1
    assert session.seed_lease is None
    assert session.state is RForkLifecycleState.FINALIZED


def test_session_lock_serializes_resource_transitions(monkeypatch):
    session = _make_session(monkeypatch)
    entered = threading.Event()
    release = threading.Event()

    def acquire_seed():
        entered.set()
        assert release.wait(1)
        return session.planner.lease

    session.planner.acquire_seed = acquire_seed
    thread = threading.Thread(target=session.acquire_seed)
    thread.start()
    assert entered.wait(1)
    assert not session._lock.acquire(timeout=0.05)
    release.set()
    thread.join(1)
    assert session._lock.acquire(timeout=1)
    session._lock.release()
