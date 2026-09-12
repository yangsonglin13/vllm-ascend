# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

import threading
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from .test_session_server_integration import live_session as live_session


def make_serving_session(runtime):
    session = runtime.session.RForkSession(replace(runtime.config, heartbeat_interval_sec=17), runtime.identity)
    handle = SimpleNamespace(port=1234, is_alive=True, stop=Mock(return_value=True))
    session.seed_server = handle
    session.state = runtime.types.RForkLifecycleState.SERVING
    session.planner = Mock(seed_key="key")
    session.planner.report_seed_once.return_value = True
    session.planner.remove_seed.return_value = True
    session.transfer_backend.finalize_transfer_engine.return_value = True
    return session, handle


def test_healthy_heartbeat_waits_and_reports_at_configured_interval(runtime):
    session, handle = make_serving_session(runtime)
    stop = Mock()
    stop.wait.side_effect = [False, False, True]
    session._run_seed_heartbeat(handle, stop)
    assert [call.args for call in stop.wait.call_args_list] == [(17,), (17,), (17,)]
    assert session.planner.report_seed_once.call_count == 2
    session.planner.remove_seed.assert_not_called()
    assert session.state is runtime.types.RForkLifecycleState.SERVING


@pytest.mark.parametrize("phase", ["before_report", "during_report", "report_exception"])
@pytest.mark.parametrize("removal", ["ok", "failed", "exception"])
def test_service_failure_withdraws_without_releasing_native_memory(runtime, phase, removal):
    session, handle = make_serving_session(runtime)
    stop = Mock()
    stop.wait.return_value = False
    stop.is_set.return_value = False
    if phase == "before_report":
        handle.is_alive = False
    else:

        def report(*args, **kwargs):
            if phase == "report_exception":
                raise RuntimeError("heartbeat worker failed")
            handle.is_alive = False
            return True

        session.planner.report_seed_once.side_effect = report
    if removal == "failed":
        session.planner.remove_seed.return_value = False
    elif removal == "exception":
        session.planner.remove_seed.side_effect = RuntimeError("planner unavailable")
    session._run_seed_heartbeat(handle, stop)
    assert session.planner.report_seed_once.call_count == (0 if phase == "before_report" else 1)
    session.planner.remove_seed.assert_called_once()
    stop.set.assert_called_once()
    assert session.state is runtime.types.RForkLifecycleState.CLEANUP_REQUIRED
    assert session.seed_server is handle
    session.transfer_backend.unregister_memory_region.assert_not_called()
    session.transfer_backend.finalize_transfer_engine.assert_not_called()
    handle.stop.assert_not_called()


def test_heartbeat_shutdown_does_not_deadlock_during_removal(runtime):
    session, handle = make_serving_session(runtime)
    handle.is_alive = False
    session.config = replace(session.config, heartbeat_interval_sec=0.001)
    removal_entered, release_removal = threading.Event(), threading.Event()

    def remove():
        removal_entered.set()
        assert release_removal.wait(3)
        return True

    session.planner.remove_seed.side_effect = remove
    heartbeat = threading.Thread(target=session._run_seed_heartbeat, args=(handle, session.heartbeat_stop_event))
    session.heartbeat_thread = heartbeat
    join_entered = threading.Event()
    original_join = heartbeat.join

    def join(timeout=None):
        join_entered.set()
        original_join(timeout)

    heartbeat.join = join
    results = []
    shutdown = threading.Thread(target=lambda: results.append(session.shutdown()))
    try:
        heartbeat.start()
        assert removal_entered.wait(2)
        # HTTP removal must not hold the state lock needed by lease callbacks.
        assert session._lock.acquire(timeout=0.5)
        session._lock.release()
        shutdown.start()
        assert join_entered.wait(2)
    finally:
        release_removal.set()
        heartbeat.join(3)
        if shutdown.ident is not None:
            shutdown.join(3)
    assert not heartbeat.is_alive() and not shutdown.is_alive()
    assert results == [True]
    assert session.state is runtime.types.RForkLifecycleState.FINALIZED


@pytest.mark.parametrize("phase", ["before_report", "during_report"])
def test_startup_detects_service_exit_and_cleans_up(runtime, monkeypatch, phase):
    session, handle = make_serving_session(runtime)
    session.state = runtime.types.RForkLifecycleState.READY
    session.seed_server = None
    monkeypatch.setattr(session, "_seed_transfer_info", Mock())
    monkeypatch.setattr(runtime.session, "start_rfork_server", lambda *args, **kwargs: handle)
    if phase == "before_report":
        handle.is_alive = False
    else:

        def report(*args, **kwargs):
            handle.is_alive = False
            return True

        session.planner.report_seed_once.side_effect = report
    assert session.start_seed_service(object(), True) is runtime.types.RForkSeedServiceStartResult.FAILED
    assert session.heartbeat_thread is None
    assert session.seed_server is None
    handle.stop.assert_called_once()
    session.transfer_backend.unregister_memory_region.assert_called_once()
    assert session.planner.report_seed_once.call_count == (0 if phase == "before_report" else 1)


def test_live_server_exit_stops_advertising_and_shutdown_releases_resources(request):
    r = request.getfixturevalue("live_session")
    session = r.session
    session.config = replace(session.config, heartbeat_interval_sec=0.02)
    assert session.start_seed_service(object(), True)
    handle, heartbeat = session.seed_server, session.heartbeat_thread
    assert handle.stop()
    heartbeat.join(3)
    assert not heartbeat.is_alive()
    assert session.heartbeat_stop_event.is_set()
    assert session.state is r.runtime.types.RForkLifecycleState.CLEANUP_REQUIRED
    session.planner.remove_seed.assert_called_once()
    session.transfer_backend.unregister_memory_region.assert_not_called()
    session.transfer_backend.finalize_transfer_engine.assert_not_called()
    reports = session.planner.report_seed_once.call_count
    assert session.shutdown()
    assert session.planner.report_seed_once.call_count == reports
    session.transfer_backend.finalize_transfer_engine.assert_called_once()
