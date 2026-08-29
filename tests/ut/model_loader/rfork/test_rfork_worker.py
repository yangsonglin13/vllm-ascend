# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

from types import SimpleNamespace

import pytest

from vllm_ascend.model_loader.rfork import rfork_worker as worker_module


class _FakeProtocol:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.auth_token = kwargs.get("auth_token")
        self.seed = {"user_id": "lease", "seed_ip": "127.0.0.1", "seed_port": 1234, "seed_rank": 0}
        self.release_result = True
        self.remove_result = True
        self.report_result = True
        self.report_calls = []

    def get_local_seed_key(self):
        return "key"

    def get_seed(self):
        return self.seed

    def release_seed(self, seed):
        return self.release_result

    def remove_seed(self, **kwargs):
        self.report_calls.append(kwargs)
        return self.remove_result

    def report_seed(self, port, **kwargs):
        self.report_calls.append({"port": port, **kwargs})
        kwargs["stop_event"].wait(2)

    def report_seed_once(self, port, **kwargs):
        self.report_calls.append({"port": port, **kwargs})
        return self.report_result


class _FakeBackend:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.unregister_result = True
        self.register_result = True
        self.rfork_transfer_engine_session_id = "session"
        self.rfork_transfer_engine_weights_info_dict = {}
        self.rfork_transfer_engine_weights_shape_dict = {}
        self.unregister_calls = 0

    def is_initialized(self):
        return True

    def register_memory_region(self, model, processed_layout):
        return self.register_result

    def unregister_memory_region(self):
        self.unregister_calls += 1
        return self.unregister_result


def _make_worker(monkeypatch, **kwargs):
    monkeypatch.setattr(worker_module, "RForkSeedProtocol", _FakeProtocol)
    monkeypatch.setattr(worker_module, "RForkTransferBackend", _FakeBackend)
    monkeypatch.setattr(worker_module.atexit, "register", lambda callback: None)
    options = {
        "disaggregation_mode": "kv_both",
        "node_rank": 0,
        "tp_rank": 0,
        "device_id": 0,
        "scheduler_url": "http://planner",
        "model_url": "/model",
        "model_deploy_strategy_name": "decode",
    }
    options.update(kwargs)
    return worker_module.RForkWorker(**options)


def test_constructor_forwards_lifecycle_options(monkeypatch):
    worker = _make_worker(
        monkeypatch,
        compatibility_fingerprint="fp",
        request_timeout_sec=1.5,
        auth_token="secret",
        seed_bind_host="::1",
        seed_advertise_host="2001:db8::1",
    )
    assert worker.seed_protocol.kwargs["compatibility_fingerprint"] == "fp"
    assert worker.transfer_backend.kwargs == {"request_timeout_sec": 1.5, "auth_token": "secret"}
    assert worker.seed_bind_host == "::1"
    assert worker.seed_advertise_host == "2001:db8::1"


def test_constructor_rejects_missing_required_identity(monkeypatch):
    with pytest.raises(ValueError, match="scheduler_url"):
        _make_worker(monkeypatch, scheduler_url="")
    with pytest.raises(ValueError, match="model_url"):
        _make_worker(monkeypatch, model_url="")


def test_reset_transfer_state_propagates_unregister_failure(monkeypatch):
    worker = _make_worker(monkeypatch)
    worker.ready_to_start_seed_service = True
    worker.transfer_backend.unregister_result = False

    assert worker.reset_transfer_state() is False
    assert worker.ready_to_start_seed_service is True

    worker.transfer_backend.unregister_result = True
    assert worker.reset_transfer_state() is True
    assert worker.ready_to_start_seed_service is False


def test_post_transfer_propagates_release_failure_and_keeps_lease(monkeypatch):
    worker = _make_worker(monkeypatch)
    lease = worker.seed_protocol.seed
    worker.rfork_seed = lease
    worker.seed_protocol.release_result = False

    assert worker.post_transfer() is False
    assert worker.rfork_seed is lease

    worker.seed_protocol.release_result = True
    assert worker.post_transfer() is True
    assert worker.rfork_seed is None


def test_seed_availability_does_not_overwrite_unreleased_lease(monkeypatch):
    worker = _make_worker(monkeypatch)
    lease = worker.seed_protocol.seed
    worker.rfork_seed = lease
    worker.seed_protocol.release_result = False
    assert worker.is_seed_available() is False
    assert worker.rfork_seed is lease


def test_seed_service_starts_only_after_health_and_shutdown_is_idempotent(monkeypatch):
    worker = _make_worker(monkeypatch, seed_advertise_host="127.0.0.1")
    worker.ready_to_start_seed_service = True
    stopped = []

    class Handle:
        port = 4567

        def stop(self):
            stopped.append(True)
            return True

    monkeypatch.setattr(worker_module, "start_rfork_server", lambda *args, **kwargs: Handle())
    assert worker.start_seed_service(SimpleNamespace(), False) is True
    assert worker.seed_service_started is True
    assert worker.rfork_heartbeat_thread is not None
    assert worker.stop_seed_service() is True
    assert worker.stop_seed_service() is True
    assert stopped == [True]
    assert worker.seed_service_started is False


def test_seed_service_start_failure_cleans_registration(monkeypatch):
    worker = _make_worker(monkeypatch)
    worker.ready_to_start_seed_service = True
    monkeypatch.setattr(worker_module, "start_rfork_server", lambda *args, **kwargs: -1)
    assert worker.start_seed_service(SimpleNamespace(), False) is False
    assert worker.transfer_backend.unregister_calls == 1
    assert worker.seed_service_started is False


def test_seed_service_stops_when_initial_planner_advertisement_fails(monkeypatch):
    worker = _make_worker(monkeypatch)
    worker.ready_to_start_seed_service = True
    worker.seed_protocol.report_result = False
    stopped = []

    class Handle:
        port = 4567

        def stop(self):
            stopped.append(True)
            return True

    monkeypatch.setattr(worker_module, "start_rfork_server", lambda *args, **kwargs: Handle())
    assert worker.start_seed_service(SimpleNamespace(), False) is False
    assert stopped == [True]
    assert worker.transfer_backend.unregister_calls == 1
    assert worker.seed_service_started is False


def test_shutdown_retains_memory_when_server_cannot_stop(monkeypatch):
    worker = _make_worker(monkeypatch)
    worker.ready_to_start_seed_service = True
    worker.seed_service_started = True

    class Handle:
        port = 4567

        def stop(self):
            return False

    worker.seed_server_handle = Handle()
    assert worker.shutdown() is False
    assert worker.transfer_backend.unregister_calls == 0
    assert worker.ready_to_start_seed_service is True
