# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Huawei Technologies Co., Ltd.

import logging
from types import SimpleNamespace
from unittest.mock import Mock

from .rfork_test_support import _load_module, _stub


def _make_runtime(monkeypatch):
    prefix = "vllm_ascend.model_loader.rfork"
    _stub(monkeypatch, "vllm.logger", logger=logging.getLogger("rfork-deferred-sharing-test"))
    _stub(
        monkeypatch,
        "vllm.utils.network_utils",
        get_ip=lambda: "127.0.0.1",
        join_host_port=lambda host, port: f"{host}:{port}",
    )
    _stub(monkeypatch, f"{prefix}.identity", build_seed_key=lambda **kwargs: "model-key")
    _stub(monkeypatch, f"{prefix}.transfer_backend", RForkTransferBackend=Mock)

    class StartupError(RuntimeError):
        def __init__(self, message, *, handle=None):
            super().__init__(message)
            self.handle = handle

    _stub(
        monkeypatch,
        f"{prefix}.seed_server",
        RForkSeedServerHandle=Mock,
        RForkSeedServerStartupError=StartupError,
        start_rfork_server=Mock(),
    )
    types = _load_module(monkeypatch, f"{prefix}.types", "types.py")
    config = _load_module(monkeypatch, f"{prefix}.config", "config.py")
    _load_module(monkeypatch, f"{prefix}.manifest", "manifest.py")
    _load_module(monkeypatch, f"{prefix}.seed_client", "seed_client.py")
    _load_module(monkeypatch, f"{prefix}.planner_client", "planner_client.py")
    _load_module(monkeypatch, f"{prefix}.load_state", "load_state.py")
    session_module = _load_module(monkeypatch, f"{prefix}.session", "session.py")
    monkeypatch.setattr(session_module.atexit, "register", lambda callback: None)
    config = config.RForkConfig("model", "strategy", "http://planner")
    identity = types.RForkIdentity(0, 0, compatibility_fingerprint="fingerprint")
    return types, session_module, config, identity


def _make_session(runtime):
    types, session_module, config, identity = runtime
    session = session_module.RForkSession(config, identity)
    session.transfer_backend = Mock()
    session.transfer_backend.register_memory_region.return_value = True
    session.transfer_backend.unregister_memory_region.return_value = True
    session.state = types.RForkLifecycleState.TRANSFERRED
    return session


def test_draft_seed_start_waits_for_sharing_and_refreshes_registration(monkeypatch):
    runtime = _make_runtime(monkeypatch)
    session = _make_session(runtime)
    model = object()
    exclude_blocks = [(128, 4096)]

    assert session.schedule_deferred_seed_start(model, True, exclude_blocks) is (
        runtime[0].RForkSeedServiceStartResult.DEFERRED
    )
    blocked = Mock(return_value=True)
    monkeypatch.setattr(session, "_start_seed_service", blocked)
    session._promote_deferred_seed()
    blocked.assert_not_called()

    promoted = Mock(return_value=runtime[0].RForkSeedServiceStartResult.STARTED)
    monkeypatch.setattr(session, "start_seed_service", promoted)
    assert session.complete_deferred_seed_start() is runtime[0].RForkSeedServiceStartResult.STARTED
    promoted.assert_called_once_with(model, True, exclude_blocks, refresh_registration=True)


def test_unreleased_lease_keeps_completed_sharing_deferred(monkeypatch):
    types, _, _, _ = runtime = _make_runtime(monkeypatch)
    session = _make_session(runtime)
    model = object()
    session.seed_lease = types.SeedLease("127.0.0.1", 1234, "user", 0, "model-key")
    session.schedule_deferred_seed_start(model, True)

    promoted = Mock(return_value=types.RForkSeedServiceStartResult.DEFERRED)
    monkeypatch.setattr(session, "start_seed_service", promoted)
    assert session.complete_deferred_seed_start() is types.RForkSeedServiceStartResult.DEFERRED
    assert session.has_deferred_seed_start()
    assert not session._deferred_seed_awaiting_sharing


def test_transfer_restores_load_state_before_read_completion(monkeypatch):
    types, session_module, _, _ = runtime = _make_runtime(monkeypatch)
    session = _make_session(runtime)
    session.state = types.RForkLifecycleState.LEASED
    session.seed_lease = types.SeedLease("127.0.0.1", 1234, "user", 0, "model-key")
    session.transfer_backend.read_weights_from_seed.return_value = True
    monkeypatch.setattr(session.planner, "release_seed_once", Mock(return_value=types.LeaseReleaseResult.RELEASED))
    seed_info = types.SeedTransferInfo(
        session_id="seed-session",
        weights={"weight": (1, 1, 4, [1], "float32")},
        load_state={"has_own_lm_head": False},
    )
    monkeypatch.setattr(session_module, "fetch_seed_transfer_info", lambda *args: seed_info)
    model = SimpleNamespace()

    assert session.transfer_from_seed(model, True)
    assert model.has_own_lm_head is False
