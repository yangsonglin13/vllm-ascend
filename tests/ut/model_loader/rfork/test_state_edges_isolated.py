# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from .session_test_utils import make_session, run_and_join


def test_release_before_transfer_restores_registered_state(runtime):
    session = make_session(runtime)
    session.planner.release_seed_once.return_value = runtime.types.LeaseReleaseResult.RELEASED
    run_and_join(session)
    assert session.state is runtime.types.RForkLifecycleState.REGISTERED
    assert session.seed_lease is None
    session.transfer_backend.unregister_memory_region.assert_not_called()


def test_finalized_fallback_and_serving_start_are_idempotent(runtime):
    session = runtime.session.RForkSession(runtime.config, runtime.identity)
    session.state = runtime.types.RForkLifecycleState.FINALIZED
    assert session.prepare_for_fallback() == runtime.types.RForkFallbackCleanupResult(False, False, False)
    session.transfer_backend.unregister_memory_region.assert_not_called()
    assert session.shutdown()
    session.transfer_backend.finalize_transfer_engine.assert_not_called()
    session = runtime.session.RForkSession(runtime.config, runtime.identity)
    session.state = runtime.types.RForkLifecycleState.SERVING
    assert session.start_seed_service(object(), True) is runtime.types.RForkSeedServiceStartResult.STARTED
    session.transfer_backend.register_memory_region.assert_not_called()
    runtime.session.start_rfork_server.assert_not_called()


@pytest.mark.parametrize("deferred", [False, True])
def test_heartbeat_start_failure_cleans_server_before_memory_reset(runtime, monkeypatch, caplog, deferred):
    session = runtime.session.RForkSession(runtime.config, runtime.identity)
    session.planner = Mock(seed_key="model-key")
    session.transfer_backend.transfer_session_id = "native-session"
    session.transfer_backend.weight_manifest = {"weight": [1234, 4, 4, [4], "float32"]}
    server = runtime.session.start_rfork_server.return_value
    events = []
    session.planner.remove_seed.side_effect = lambda: events.append("remove") or True
    server.stop.side_effect = lambda: events.append("stop") or True
    session.transfer_backend.unregister_memory_region.side_effect = lambda: events.append("unregister") or True
    monkeypatch.setattr(
        runtime.session.threading.Thread, "start", Mock(side_effect=RuntimeError("can't start new thread"))
    )

    if deferred:
        session.state = runtime.types.RForkLifecycleState.READY
        session._deferred_seed_start = (object(), True, [])
        session._promote_deferred_seed()
    else:
        assert session.start_seed_service(object(), True) is runtime.types.RForkSeedServiceStartResult.FAILED

    assert "can't start new thread" in caplog.text
    assert events == ["remove", "stop", "unregister"]
    assert session.heartbeat_thread is None
    assert session.seed_server is None
    assert session.state is runtime.types.RForkLifecycleState.INITIALIZED
    assert session.shutdown()
    session.transfer_backend.finalize_transfer_engine.assert_called_once()


@pytest.mark.parametrize("raises", [False, True])
@pytest.mark.parametrize("removal_ok", [False, True])
def test_failed_deferred_promotion_cleans_before_memory_reset(runtime, monkeypatch, raises, removal_ok):
    session = runtime.session.RForkSession(runtime.config, runtime.identity)
    session.planner = Mock(seed_key="model-key")
    session.planner.remove_seed.return_value = removal_ok
    session.state = runtime.types.RForkLifecycleState.READY
    session._deferred_seed_start = (object(), True, [])
    server = Mock()
    server.stop.return_value = True

    def start(*args):
        session.seed_server = server
        if raises:
            raise RuntimeError("startup failure")
        return False

    monkeypatch.setattr(session, "_start_seed_service", start)
    session._promote_deferred_seed()
    assert session._deferred_seed_start is None
    assert session.state is (
        runtime.types.RForkLifecycleState.INITIALIZED
        if removal_ok
        else runtime.types.RForkLifecycleState.CLEANUP_REQUIRED
    )
    if removal_ok:
        server.stop.assert_called_once()
        session.transfer_backend.unregister_memory_region.assert_called_once()
    else:
        assert session.seed_server is server
        server.stop.assert_not_called()
        session.transfer_backend.unregister_memory_region.assert_not_called()


@pytest.mark.parametrize("registration_ok", [False, True])
def test_native_registration_preserves_owners_until_cleanup(tensor_runtime, monkeypatch, registration_ok):
    r = tensor_runtime
    backend = r.RForkTransferBackend()
    weight = torch.ones(4)
    error = SimpleNamespace(is_error=lambda: True, to_string=lambda: "registration failed")
    ok = SimpleNamespace(is_error=lambda: False)
    backend.transfer_engine = Mock()
    backend.transfer_engine.batch_register_memory_ex.return_value = ok if registration_ok else error
    backend.transfer_engine.batch_unregister_memory.return_value = ok
    backend._memory_registration_cls = lambda *args: args
    monkeypatch.setattr(r.transfer_backend, "find_non_npu_state_tensors", lambda *args: [])
    monkeypatch.setattr(r.transfer_backend, "collect_transferable_tensors", lambda *args: [("weight", weight)])
    monkeypatch.setattr(r.transfer_backend, "is_transferable_tensor", lambda *args: True)
    snapshot = [{"blocks": [{"address": weight.data_ptr(), "size": 16, "state": "active_allocated"}]}]
    monkeypatch.setattr(
        torch, "npu", SimpleNamespace(memory=SimpleNamespace(memory_snapshot=lambda: snapshot)), raising=False
    )
    assert backend.register_memory_region(object(), True) is registration_ok
    if registration_ok:
        assert backend.weight_manifest == {"weight": (weight.data_ptr(), 4, 4, (4,), "float32")}
        assert backend.weight_shapes == {"weight": (4,)}
        assert backend.registered_weight_blocks == [(weight.data_ptr(), 16)]
        assert backend.registered_memory_addresses == [weight.data_ptr()]
        assert backend._registered_transferable_tensors[0][1] is weight
        assert len(backend._registered_transferable_storages) == 1
        backend.transfer_engine.batch_unregister_memory.assert_not_called()
        assert backend.unregister_memory_region()
    backend.transfer_engine.batch_unregister_memory.assert_called_once_with([weight.data_ptr()])
    assert backend.weight_manifest is None
    assert backend.registered_memory_addresses == []
    assert backend._registered_transferable_tensors is None


def test_individual_unregister_retains_unresolved_regions(tensor_runtime):
    backend = tensor_runtime.RForkTransferBackend()
    backend._not_found_error_code = "missing"
    missing = SimpleNamespace(is_error=lambda: True, get_code=lambda: "missing", to_string=lambda: "missing")
    error = SimpleNamespace(is_error=lambda: True, get_code=lambda: "failed", to_string=lambda: "failed")
    engine = Mock()
    engine.batch_unregister_memory.side_effect = [missing, error, RuntimeError("native failure"), None]
    assert backend._unregister_addresses_individually(engine, [1, 2, 3, 4]) == [2, 3, 4]
    backend.registered_memory_addresses = [1, 2]
    backend.registered_weight_blocks = [(1, 8)]
    owner = object()
    backend._registered_transferable_tensors = [owner]
    engine.batch_unregister_memory.side_effect = [missing, missing, error]
    assert not backend._unregister_weight_blocks(engine)
    assert backend.registered_memory_addresses == [2]
    assert backend._registered_transferable_tensors == [owner]
