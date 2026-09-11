# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

import threading
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from tests.ut.model_loader.rfork.session_test_utils import make_session


@pytest.mark.parametrize("removal_ok", [False, True])
def test_blocked_removal_allows_release_ack_but_retains_seed_memory(runtime, monkeypatch, removal_ok):
    session = make_session(runtime)
    removing, resume_removal = threading.Event(), threading.Event()
    releasing, resume_release, acknowledged = threading.Event(), threading.Event(), threading.Event()
    server = Mock()
    server.stop.return_value = True
    session.seed_server = server
    session.state = runtime.types.RForkLifecycleState.SERVING
    heartbeat = Mock()
    heartbeat.is_alive.return_value = False
    session.heartbeat_thread = heartbeat

    def remove():
        assert session.heartbeat_stop_event.is_set()
        heartbeat.join.assert_called_once()
        removing.set()
        assert resume_removal.wait(3)
        return removal_ok

    def release(lease):
        releasing.set()
        assert resume_release.wait(3)
        return runtime.types.LeaseReleaseResult.RELEASED

    original_record = session._record_lease_release_locked

    def record(*args):
        result = original_record(*args)
        acknowledged.set()
        return result

    monkeypatch.setattr(session, "_record_lease_release_locked", record)
    session.planner.remove_seed.side_effect = remove
    session.planner.release_seed_once.side_effect = release
    outcomes = []
    cleanup = threading.Thread(target=lambda: outcomes.append(session.prepare_for_fallback()), daemon=True)
    with session._lock:
        session._ensure_lease_release_retry_locked()
        worker = session.lease_release_thread
    try:
        assert releasing.wait(1)
        cleanup.start()
        assert removing.wait(1)
        resume_release.set()
        assert acknowledged.wait(1), "removal HTTP held the session lock and blocked the release response"
        assert session.seed_lease is None
        assert session._lock.acquire(timeout=0.5)
        try:
            assert session.state is runtime.types.RForkLifecycleState.CLEANUP_REQUIRED
            assert not session.register_destination(object(), True)
        finally:
            session._lock.release()
        session.transfer_backend.unregister_memory_region.assert_not_called()
        session.transfer_backend.finalize_transfer_engine.assert_not_called()
        server.stop.assert_not_called()
    finally:
        resume_release.set()
        resume_removal.set()
        cleanup.join(2)
        worker.join(2)
    assert not cleanup.is_alive() and not worker.is_alive()
    assert len(outcomes) == 1
    assert outcomes[0].service_stopped is removal_ok
    assert outcomes[0].memory_reset is removal_ok
    if removal_ok:
        server.stop.assert_called_once()
        session.transfer_backend.unregister_memory_region.assert_called_once()
        assert session.seed_server is None
    else:
        assert session.seed_server is server
        server.stop.assert_not_called()
        session.transfer_backend.unregister_memory_region.assert_not_called()


def test_seed_restart_is_serialized_after_removal(runtime, monkeypatch):
    session = runtime.session.RForkSession(runtime.config, runtime.identity)
    session.planner = Mock(seed_key="model-key")
    session.state = runtime.types.RForkLifecycleState.SERVING
    session.seed_server = Mock()
    session.seed_server.stop.return_value = True
    removing, resume, starting, started = (threading.Event() for _ in range(4))
    events = []

    def remove():
        removing.set()
        assert resume.wait(3)
        events.append("remove")
        return True

    def reset():
        events.append("unregister")
        return True

    def publish(*args):
        events.append("publish")
        return True

    def start():
        starting.set()
        assert session.start_seed_service(object(), True) is runtime.types.RForkSeedServiceStartResult.STARTED
        started.set()

    session.planner.remove_seed.side_effect = remove
    session.transfer_backend.unregister_memory_region.side_effect = reset
    monkeypatch.setattr(session, "_start_seed_service", publish)
    cleanup = threading.Thread(target=session.prepare_for_fallback, daemon=True)
    restart = threading.Thread(target=start, daemon=True)
    try:
        cleanup.start()
        assert removing.wait(1)
        restart.start()
        assert starting.wait(1)
        assert not started.wait(0.05)
        assert not events
    finally:
        resume.set()
        cleanup.join(2)
        restart.join(2)
    assert not cleanup.is_alive() and not restart.is_alive()
    assert started.is_set()
    assert events == ["remove", "unregister", "publish"]


def test_unfinished_heartbeat_prevents_removal_and_unregistration(runtime):
    session = runtime.session.RForkSession(runtime.config, runtime.identity)
    session.planner = Mock()
    session.heartbeat_thread = Mock()
    session.heartbeat_thread.is_alive.return_value = True
    result = session.prepare_for_fallback()
    assert not result.service_stopped and not result.memory_reset
    session.planner.remove_seed.assert_not_called()
    session.transfer_backend.unregister_memory_region.assert_not_called()


def test_failed_seed_start_removes_outside_state_lock(runtime, monkeypatch):
    session = runtime.session.RForkSession(runtime.config, runtime.identity)
    session.planner = Mock(seed_key="model-key")
    session.planner.report_seed_once.return_value = False
    removing, resume = threading.Event(), threading.Event()
    server = Mock()
    server.stop.return_value = True
    runtime.session.start_rfork_server.return_value = server
    monkeypatch.setattr(session, "_seed_transfer_info", lambda: object())

    def remove():
        removing.set()
        assert resume.wait(3)
        return True

    session.planner.remove_seed.side_effect = remove
    outcomes = []
    worker = threading.Thread(target=lambda: outcomes.append(session.start_seed_service(object(), True)), daemon=True)
    try:
        worker.start()
        assert removing.wait(1)
        assert session._lock.acquire(timeout=0.5), "startup failure retained the outer session lock"
        try:
            assert session.seed_server is server
            assert session.state is runtime.types.RForkLifecycleState.CLEANUP_REQUIRED
        finally:
            session._lock.release()
        session.transfer_backend.unregister_memory_region.assert_not_called()
    finally:
        resume.set()
        worker.join(2)
    assert not worker.is_alive()
    assert outcomes == [runtime.types.RForkSeedServiceStartResult.FAILED]
    server.stop.assert_called_once()
    session.transfer_backend.unregister_memory_region.assert_called_once()


def test_remove_ack_does_not_clear_a_newer_equal_advertisement(runtime, monkeypatch):
    client = runtime.client.RForkPlannerClient(runtime.config, runtime.identity)
    ad_type = runtime.types.SeedAdvertisement
    old = ad_type("127.0.0.1", 1234, 0)
    newer = ad_type("127.0.0.1", 1234, 0)
    client.last_advertisement = old

    def post(*args, **kwargs):
        # Exercise snapshot application only. Session teardown joins heartbeats
        # to prevent concurrent add/remove on the remote planner in production.
        with client._advertisement_lock:
            client.last_advertisement = newer
        return SimpleNamespace(status_code=200)

    monkeypatch.setattr(runtime.client.requests, "post", post)
    assert client.remove_seed()
    assert client.last_advertisement is newer


def test_advertisement_lock_is_not_held_during_report_http(runtime, monkeypatch):
    client = runtime.client.RForkPlannerClient(runtime.config, runtime.identity)

    def post(*args, **kwargs):
        assert client._advertisement_lock.acquire(blocking=False)
        client._advertisement_lock.release()
        return SimpleNamespace(status_code=200)

    monkeypatch.setattr(runtime.client.requests, "post", post)
    assert client.report_seed_once(1234, "127.0.0.1")
    assert client.last_advertisement == runtime.types.SeedAdvertisement("127.0.0.1", 1234, 0)


@pytest.mark.parametrize("report_failure", ["response", "exception"])
def test_failed_report_keeps_advertisement_for_idempotent_cleanup(runtime, monkeypatch, report_failure):
    client = runtime.client.RForkPlannerClient(runtime.config, runtime.identity)
    planner_seeds = set()

    def post(url, *, headers, **kwargs):
        identity = (headers["SEED_IP"], headers["SEED_PORT"], headers["SEED_RANK"])
        if url.endswith("/add_seed"):
            # Model the planner committing the add before the response is
            # rejected or lost on the client side.
            planner_seeds.add(identity)
            if report_failure == "exception":
                raise runtime.client.requests.Timeout("response lost")
            return SimpleNamespace(status_code=503)

        assert url.endswith("/remove_seed")
        removed = identity in planner_seeds
        planner_seeds.discard(identity)
        return SimpleNamespace(status_code=200 if removed else 404)

    monkeypatch.setattr(runtime.client.requests, "post", post)

    assert not client.report_seed_once(1234, "127.0.0.1")
    assert client.last_advertisement is not None
    assert planner_seeds

    # Cleanup can remove a seed even when add_seed never produced a usable
    # response. A repeated cleanup is a no-op after acknowledgement.
    assert client.remove_seed()
    assert not planner_seeds
    assert client.last_advertisement is None
    assert client.remove_seed()
