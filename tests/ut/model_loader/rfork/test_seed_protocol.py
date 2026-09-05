# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

import threading

import pytest

from vllm_ascend.model_loader.rfork.seed_protocol import (
    RForkSeedProtocol,
    get_local_seed_key,
)


def _protocol(**kwargs):
    options = {
        "tp_rank": 1,
        "scheduler_url": "http://planner",
        "model_url": "/models/model",
        "model_deploy_strategy_name": "decode",
        "compatibility_fingerprint": "fp",
    }
    options.update(kwargs)
    return RForkSeedProtocol(**options)


def test_seed_key_is_opaque_and_collision_free():
    first = get_local_seed_key(
        tp_rank=2,
        model_url="/models/a$b",
        model_deploy_strategy_name="decode",
        compatibility_fingerprint="fp-a",
    )
    second = get_local_seed_key(
        tp_rank=2,
        model_url="/models/a",
        model_deploy_strategy_name="b$decode",
        compatibility_fingerprint="fp-a",
    )

    assert len(first) == 64
    assert all(character in "0123456789abcdef" for character in first)
    assert first != second


def test_seed_key_changes_with_fingerprint_and_requires_one():
    common = {
        "tp_rank": 0,
        "model_url": "/models/model",
        "model_deploy_strategy_name": "decode",
    }
    assert get_local_seed_key(**common, compatibility_fingerprint="fp-a") != get_local_seed_key(
        **common, compatibility_fingerprint="fp-b"
    )
    with pytest.raises(TypeError):
        get_local_seed_key(**common)
    with pytest.raises(RuntimeError):
        get_local_seed_key(**common, compatibility_fingerprint="")


def test_request_timeout_is_applied(monkeypatch):
    protocol = _protocol(request_timeout_sec=1.25)
    calls = []

    class Response:
        status_code = 404
        headers = {}

    def fake_get(*args, **kwargs):
        calls.append((args, kwargs))
        return Response()

    monkeypatch.setattr(
        "vllm_ascend.model_loader.rfork.seed_protocol.requests.get",
        fake_get,
    )
    assert protocol.get_seed() is None
    assert calls[0][1]["timeout"] == pytest.approx(1.25)


def test_release_retries_with_a_bounded_count(monkeypatch):
    protocol = _protocol(request_timeout_sec=0.5, release_max_retries=3, release_retry_backoff_sec=0)
    calls = []

    class Response:
        def __init__(self, status_code):
            self.status_code = status_code

    responses = iter([Response(503), Response(503), Response(200)])

    def fake_post(*args, **kwargs):
        calls.append(kwargs)
        return next(responses)

    monkeypatch.setattr(
        "vllm_ascend.model_loader.rfork.seed_protocol.requests.post",
        fake_post,
    )
    seed = {"seed_ip": "127.0.0.1", "seed_port": 1234, "seed_rank": 0, "user_id": "lease"}
    assert protocol.release_seed(seed) is True
    assert len(calls) == 3
    assert all(call["timeout"] == pytest.approx(0.5) for call in calls)


def test_release_treats_already_expired_lease_as_success(monkeypatch):
    protocol = _protocol(release_max_retries=3, release_retry_backoff_sec=0)

    class Response:
        status_code = 404

    calls = []

    def fake_post(*args, **kwargs):
        calls.append(kwargs)
        return Response()

    monkeypatch.setattr(
        "vllm_ascend.model_loader.rfork.seed_protocol.requests.post",
        fake_post,
    )
    seed = {"seed_ip": "127.0.0.1", "seed_port": 1234, "seed_rank": 0, "user_id": "expired"}
    assert protocol.release_seed(seed) is True
    assert len(calls) == 1


def test_report_seed_stops_without_an_extra_heartbeat(monkeypatch):
    protocol = _protocol()
    stop_event = threading.Event()
    calls = []

    class Response:
        status_code = 200

    def fake_post(*args, **kwargs):
        calls.append(kwargs)
        stop_event.set()
        return Response()

    monkeypatch.setattr(
        "vllm_ascend.model_loader.rfork.seed_protocol.requests.post",
        fake_post,
    )
    monkeypatch.setattr("vllm_ascend.model_loader.rfork.seed_protocol.get_ip", lambda: "127.0.0.1")
    protocol.report_seed(2345, sleep_interval=60, stop_event=stop_event)
    assert len(calls) == 1


def test_report_seed_once_propagates_planner_rejection(monkeypatch):
    protocol = _protocol()

    class Response:
        status_code = 401

    monkeypatch.setattr(
        "vllm_ascend.model_loader.rfork.seed_protocol.requests.post",
        lambda *args, **kwargs: Response(),
    )
    assert protocol.report_seed_once(2345, seed_ip="127.0.0.1") is False
    assert protocol._last_report is None


def test_remove_seed_is_idempotent_and_bounded(monkeypatch):
    protocol = _protocol(request_timeout_sec=0.75)
    calls = []

    class Response:
        status_code = 404

    def fake_post(*args, **kwargs):
        calls.append((args, kwargs))
        return Response()

    monkeypatch.setattr(
        "vllm_ascend.model_loader.rfork.seed_protocol.requests.post",
        fake_post,
    )
    assert protocol.remove_seed(port=2345, seed_ip="127.0.0.1") is True
    assert calls[0][1]["timeout"] == pytest.approx(0.75)
