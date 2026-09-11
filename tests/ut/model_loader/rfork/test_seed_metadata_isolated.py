# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

import logging
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest


@pytest.fixture
def seed_module(runtime):
    return sys.modules["vllm_ascend.model_loader.rfork.seed_client"]


def _response(status_code, payload):
    return SimpleNamespace(status_code=status_code, json=lambda: payload)


@pytest.mark.parametrize("status_code", [404, 500])
def test_non_ok_shape_response_falls_back_to_complete_weight_metadata(seed_module, monkeypatch, caplog, status_code):
    main_response = _response(
        200,
        {"rfork_transfer_engine_info": ["session", {"weight": (1, 1, 4, [1], "float32")}]},
    )
    shape_response = _response(status_code, {"error": "shape endpoint unavailable"})
    get = Mock(side_effect=[main_response, shape_response])
    monkeypatch.setattr(seed_module.requests, "get", get)

    with caplog.at_level(logging.WARNING):
        result = seed_module.fetch_seed_transfer_info("http://seed:1234", "key", 1.0)

    assert result is not None
    assert result.shapes is None
    assert f"status={status_code}" in caplog.text
    assert "using complete weight metadata" in caplog.text


@pytest.mark.parametrize("status_code", [404, 500])
def test_non_ok_shape_response_rejects_incomplete_weight_metadata(seed_module, monkeypatch, caplog, status_code):
    main_response = _response(200, {"rfork_transfer_engine_info": ["session", {"weight": (1, 1, 4)}]})
    shape_response = _response(status_code, {"error": "shape endpoint unavailable"})
    get = Mock(side_effect=[main_response, shape_response])
    monkeypatch.setattr(seed_module.requests, "get", get)

    with caplog.at_level(logging.ERROR):
        result = seed_module.fetch_seed_transfer_info("http://seed:1234", "key", 1.0)

    assert result is None
    assert f"status={status_code}" in caplog.text
    assert "weight metadata is incomplete" in caplog.text


@pytest.mark.parametrize("complete", [False, True])
def test_ok_shape_response_is_used_when_available(seed_module, monkeypatch, complete):
    weight_info = (1, 1, 4, [1], "float32") if complete else (1, 1, 4)
    main_response = _response(200, {"rfork_transfer_engine_info": ["session", {"weight": weight_info}]})
    shape_response = _response(
        200,
        {"rfork_transfer_engine_shape_info": {"weight": [1]}},
    )
    get = Mock(side_effect=[main_response, shape_response])
    monkeypatch.setattr(seed_module.requests, "get", get)

    result = seed_module.fetch_seed_transfer_info("http://seed:1234", "key", 1.0)

    assert result is not None
    assert result.shapes == {"weight": [1]}


@pytest.mark.parametrize("unavailable", ["http_error", "timeout"])
@pytest.mark.parametrize(
    "weight_info, accepted",
    [
        pytest.param((1, 1, 4, [1], "float32"), True, id="tuple"),
        pytest.param((1, 1, 4, {"shape": [1], "dtype": "float32"}), True, id="four-field-metadata"),
        pytest.param({"ptr": 1, "numel": 1, "element_size": 4, "shape": [1], "dtype": "float32"}, True, id="mapping"),
        pytest.param((1, 1, 4, [], "float32"), True, id="scalar"),
        pytest.param((1, 1, 4, {"shape": [1], "dtype": "float32"}, "float16"), False, id="dtype-conflict"),
        pytest.param((1, 1, 4, [1], 42), False, id="invalid-dtype"),
        pytest.param((1, 1, 4, [1], "torch."), False, id="empty-normalized-dtype"),
        pytest.param((1, 1, 4, [2], "float32"), False, id="numel-mismatch"),
        pytest.param((0, 1, 4, [1], "float32"), False, id="invalid-address"),
        pytest.param((1, 1, 4), False, id="missing-metadata"),
    ],
)
def test_shape_endpoint_degradation_uses_validated_metadata(
    seed_module, monkeypatch, caplog, unavailable, weight_info, accepted
):
    main_response = _response(200, {"rfork_transfer_engine_info": ["session", {"weight": weight_info}]})
    shape_response = (
        seed_module.requests.Timeout("shape endpoint unavailable") if unavailable == "timeout" else _response(503, {})
    )
    get = Mock(side_effect=[main_response, shape_response])
    monkeypatch.setattr(seed_module.requests, "get", get)

    with caplog.at_level(logging.WARNING):
        result = seed_module.fetch_seed_transfer_info("http://seed:1234", "key", 1.0)

    assert (result is not None) is accepted
    assert get.call_count == 2
    assert len(caplog.records) == 1
    assert caplog.records[0].levelno == (logging.WARNING if accepted else logging.ERROR)
