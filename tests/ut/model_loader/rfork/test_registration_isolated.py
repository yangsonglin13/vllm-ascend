# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

from unittest.mock import Mock

import pytest


def make_session(runtime):
    session = runtime.session.RForkSession(runtime.config, runtime.identity)
    session.planner = Mock(seed_key="model-key")
    session.planner.acquire_seed.return_value = runtime.lease
    session.planner.remove_seed.return_value = True
    return session


def test_registration_precedes_the_only_lease_acquisition(runtime, monkeypatch):
    session = make_session(runtime)
    events = []

    def register(*args):
        assert session.seed_lease is None
        session.planner.acquire_seed.assert_not_called()
        events.append("register")
        return True

    def acquire():
        assert session.state is runtime.types.RForkLifecycleState.REGISTERED
        events.append("acquire")
        return runtime.lease

    def metadata(*args):
        assert session.seed_lease is runtime.lease
        events.append("metadata")
        return object()

    def read(**kwargs):
        events.append("read")
        return True

    def release(lease):
        assert lease is runtime.lease
        events.append("release")
        return runtime.types.LeaseReleaseResult.RELEASED

    session.transfer_backend.register_memory_region.side_effect = register
    session.transfer_backend.read_weights_from_seed.side_effect = read
    session.planner.acquire_seed.side_effect = acquire
    session.planner.release_seed_once.side_effect = release
    monkeypatch.setattr(runtime.session, "fetch_seed_transfer_info", metadata)
    assert not session.acquire_seed()
    model = object()
    assert session.register_destination(model, True)
    assert session.acquire_seed()
    assert session.transfer_from_seed(model, True)
    worker = session.lease_release_thread
    if worker is not None:
        worker.join(2)
        assert not worker.is_alive()
    assert events == ["register", "acquire", "metadata", "read", "release"]
    session.planner.acquire_seed.assert_called_once_with()
    session.planner.release_seed_once.assert_called_once_with(runtime.lease)
    session.transfer_backend.register_memory_region.assert_called_once_with(model, True, None)


@pytest.mark.parametrize("raises", [False, True])
def test_destination_registration_failure_requires_cleanup_without_lease(runtime, raises):
    session = make_session(runtime)
    session.transfer_backend.register_memory_region.return_value = False
    if raises:
        session.transfer_backend.register_memory_region.side_effect = RuntimeError("partial registration")
        with pytest.raises(RuntimeError, match="partial registration"):
            session.register_destination(object(), True)
    else:
        assert not session.register_destination(object(), True)
    assert session.state is runtime.types.RForkLifecycleState.CLEANUP_REQUIRED
    assert not session.acquire_seed()
    session.planner.acquire_seed.assert_not_called()
    assert session.prepare_for_fallback().memory_reset
    session.transfer_backend.unregister_memory_region.assert_called_once()
    assert not session.transfer_from_seed(object(), True)


def test_formal_seed_miss_unregisters_before_reuse(runtime):
    session = make_session(runtime)
    assert session.register_destination(object(), True)
    session.planner.acquire_seed.return_value = None
    assert not session.acquire_seed()
    assert session.state is runtime.types.RForkLifecycleState.REGISTERED
    assert session.prepare_for_fallback().memory_reset
    assert session.state is runtime.types.RForkLifecycleState.INITIALIZED
    session.transfer_backend.unregister_memory_region.assert_called_once()
    session.planner.release_seed_once.assert_not_called()


def test_acquire_with_existing_lease_only_schedules_release(runtime):
    session = make_session(runtime)
    session.planner.release_seed_once.return_value = runtime.types.LeaseReleaseResult.RELEASED
    assert session.register_destination(object(), True)
    assert session.acquire_seed()
    with session._lock:
        assert not session.acquire_seed()
        worker = session.lease_release_thread
        assert worker is not None
        assert session.seed_lease is runtime.lease
    worker.join(2)
    assert not worker.is_alive()
    session.planner.acquire_seed.assert_called_once()
    session.planner.release_seed_once.assert_called_once_with(runtime.lease)
    assert session.seed_lease is None
    assert session.state is runtime.types.RForkLifecycleState.REGISTERED
