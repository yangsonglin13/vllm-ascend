# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

import threading

import pytest

from vllm_ascend.model_loader.rfork.config import RForkConfig
from vllm_ascend.model_loader.rfork.planner_client import (
    AUTH_TOKEN_HEADER,
    RForkPlannerClient,
    get_local_seed_key,
)
from vllm_ascend.model_loader.rfork.types import RForkIdentity, SeedAdvertisement, SeedLease


def _planner(**kwargs):
    config_options = {
        "scheduler_url": "http://planner",
        "model_url": "/models/model",
        "model_deploy_strategy_name": "decode",
        "request_timeout_sec": kwargs.pop("request_timeout_sec", 10.0),
        "auth_token": kwargs.pop("auth_token", None),
    }
    identity = RForkIdentity("kv_consumer", 0, 1, 0)
    return RForkPlannerClient(RForkConfig(**config_options), identity, **kwargs)


def test_v2_seed_key_is_opaque_and_collision_free():
    first = get_local_seed_key(
        disaggregation_mode="kv_consumer$0",
        node_rank=1,
        tp_rank=2,
        model_url="/models/a$b",
        model_deploy_strategy_name="decode",
        compatibility_fingerprint="fp-a",
    )
    second = get_local_seed_key(
        disaggregation_mode="kv_consumer",
        node_rank=0,
        tp_rank=2,
        model_url="/models/a",
        model_deploy_strategy_name="b$decode",
        compatibility_fingerprint="fp-a",
    )

    assert first.startswith("rfork-v2:")
    assert second.startswith("rfork-v2:")
    assert first != second


def test_v2_seed_key_changes_with_fingerprint_and_legacy_key_is_preserved():
    common = {
        "disaggregation_mode": "kv_consumer",
        "node_rank": 0,
        "tp_rank": 0,
        "model_url": "/models/model",
        "model_deploy_strategy_name": "decode",
    }
    assert get_local_seed_key(**common) == "/models/model$decode$kv_consumer$0$0"
    assert get_local_seed_key(**common, compatibility_fingerprint="fp-a") != get_local_seed_key(
        **common, compatibility_fingerprint="fp-b"
    )


def test_request_timeout_and_auth_are_applied(monkeypatch):
    protocol = _planner(request_timeout_sec=1.25, auth_token="secret")
    calls = []

    class Response:
        status_code = 404
        headers = {}

    def fake_get(*args, **kwargs):
        calls.append((args, kwargs))
        return Response()

    monkeypatch.setattr(
        "vllm_ascend.model_loader.rfork.planner_client.requests.get",
        fake_get,
    )
    assert protocol.acquire_seed() is None
    assert calls[0][1]["timeout"] == pytest.approx(1.25)
    assert calls[0][1]["headers"][AUTH_TOKEN_HEADER] == "secret"


def test_release_retries_with_a_bounded_count(monkeypatch):
    protocol = _planner(request_timeout_sec=0.5, release_max_retries=3, release_retry_backoff_sec=0)
    calls = []

    class Response:
        def __init__(self, status_code):
            self.status_code = status_code

    responses = iter([Response(503), Response(503), Response(200)])

    def fake_post(*args, **kwargs):
        calls.append(kwargs)
        return next(responses)

    monkeypatch.setattr(
        "vllm_ascend.model_loader.rfork.planner_client.requests.post",
        fake_post,
    )
    seed = SeedLease("127.0.0.1", 1234, "lease", 0, protocol.local_seed_key)
    assert protocol.release_seed(seed) is True
    assert len(calls) == 3
    assert all(call["timeout"] == pytest.approx(0.5) for call in calls)


def test_release_treats_already_expired_lease_as_success(monkeypatch):
    protocol = _planner(release_max_retries=3, release_retry_backoff_sec=0)

    class Response:
        status_code = 404

    calls = []

    def fake_post(*args, **kwargs):
        calls.append(kwargs)
        return Response()

    monkeypatch.setattr(
        "vllm_ascend.model_loader.rfork.planner_client.requests.post",
        fake_post,
    )
    seed = SeedLease("127.0.0.1", 1234, "expired", 0, protocol.local_seed_key)
    assert protocol.release_seed(seed) is True
    assert len(calls) == 1


def test_report_seed_stops_without_an_extra_heartbeat(monkeypatch):
    protocol = _planner(auth_token="secret")
    stop_event = threading.Event()
    calls = []

    class Response:
        status_code = 200

    def fake_post(*args, **kwargs):
        calls.append(kwargs)
        stop_event.set()
        return Response()

    monkeypatch.setattr(
        "vllm_ascend.model_loader.rfork.planner_client.requests.post",
        fake_post,
    )
    monkeypatch.setattr("vllm_ascend.model_loader.rfork.planner_client.get_ip", lambda: "127.0.0.1")
    protocol.report_seed(2345, sleep_interval=60, stop_event=stop_event)
    assert len(calls) == 1
    assert calls[0]["headers"][AUTH_TOKEN_HEADER] == "secret"


def test_report_seed_once_propagates_planner_rejection(monkeypatch):
    protocol = _planner()

    class Response:
        status_code = 401

    monkeypatch.setattr(
        "vllm_ascend.model_loader.rfork.planner_client.requests.post",
        lambda *args, **kwargs: Response(),
    )
    assert protocol.report_seed_once(2345, seed_ip="127.0.0.1") is False
    assert protocol.last_advertisement is None


def test_remove_seed_is_idempotent_and_bounded(monkeypatch):
    protocol = _planner(request_timeout_sec=0.75)
    calls = []

    class Response:
        status_code = 404

    def fake_post(*args, **kwargs):
        calls.append((args, kwargs))
        return Response()

    monkeypatch.setattr(
        "vllm_ascend.model_loader.rfork.planner_client.requests.post",
        fake_post,
    )
    assert protocol.remove_seed(SeedAdvertisement("127.0.0.1", 2345, 1)) is True
    assert calls[0][1]["timeout"] == pytest.approx(0.75)
