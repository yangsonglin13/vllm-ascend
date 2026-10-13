# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

"""Seed withdrawal stays opt-in and resolves JSON before the centralized environment switch."""

import pytest

from .test_legacy_planner import _advertise, _LegacyPlannerAdapter, _make_client

ENV_NAME = "VLLM_ASCEND_RFORK_SEED_REMOVAL_SUPPORTED"


@pytest.mark.parametrize("env_value, expected", [(None, False), ("0", False), ("1", True)])
def test_environment_switch_controls_seed_withdrawal(runtime, monkeypatch, env_value, expected):
    if env_value is None:
        monkeypatch.delenv(ENV_NAME, raising=False)
    else:
        monkeypatch.setenv(ENV_NAME, env_value)
    config = type(runtime.config).from_extra_config({"rfork_scheduler_url": "http://planner"})
    assert config.seed_removal_supported is expected
    planner = _LegacyPlannerAdapter(remove_supported=True)
    client = _make_client(runtime, monkeypatch, planner, config=config)
    _advertise(client, planner)

    result = client.remove_seed()

    if expected:
        assert result is runtime.types.SeedRemovalResult.REMOVED
        assert client.last_advertisement is None
        assert client.seed_key not in planner.advertisements
    else:
        assert result is runtime.types.SeedRemovalResult.UNSUPPORTED
        assert client.last_advertisement is not None
        assert client.seed_key in planner.advertisements
        assert "/remove_seed" not in [path for _, path in planner.calls]


def test_direct_config_defaults_to_skipping_seed_withdrawal(runtime, monkeypatch):
    config = type(runtime.config)("model", "strategy", "http://planner")
    planner = _LegacyPlannerAdapter()
    client = _make_client(runtime, monkeypatch, planner, config=config)
    _advertise(client, planner)

    assert client.remove_seed() is runtime.types.SeedRemovalResult.UNSUPPORTED
    assert client.last_advertisement is not None
    assert "/remove_seed" not in [path for _, path in planner.calls]


@pytest.mark.parametrize("env_value", ["", "2", "true", "false", " 1 ", "auto"])
def test_invalid_environment_switch_fails_config_initialization(runtime, monkeypatch, env_value):
    monkeypatch.setenv(ENV_NAME, env_value)

    with pytest.raises(ValueError, match=ENV_NAME):
        type(runtime.config).from_extra_config({})


@pytest.mark.parametrize(
    "json_value, env_value",
    [(True, "0"), (False, "1"), (None, "1"), (True, "invalid"), (False, "invalid"), (None, "invalid")],
)
def test_explicit_json_overrides_environment_including_invalid_values(runtime, monkeypatch, json_value, env_value):
    monkeypatch.setenv(ENV_NAME, env_value)

    config = type(runtime.config).from_extra_config({"rfork_seed_removal_supported": json_value})

    assert config.seed_removal_supported is json_value


def test_explicit_json_null_uses_auto_detection_instead_of_environment(runtime, monkeypatch):
    monkeypatch.setenv(ENV_NAME, "1")
    config = type(runtime.config).from_extra_config(
        {"rfork_seed_removal_supported": None, "rfork_scheduler_url": "http://planner"}
    )
    planner = _LegacyPlannerAdapter()
    client = _make_client(runtime, monkeypatch, planner, config=config)
    _advertise(client, planner)

    assert client.remove_seed() is runtime.types.SeedRemovalResult.UNCONFIRMED
    assert client.last_advertisement is not None


def test_environment_is_resolved_once_when_config_is_built(runtime, monkeypatch):
    monkeypatch.setenv(ENV_NAME, "0")
    config = type(runtime.config).from_extra_config({"rfork_scheduler_url": "http://planner"})
    planner = _LegacyPlannerAdapter(remove_supported=True)
    client = _make_client(runtime, monkeypatch, planner, config=config)
    _advertise(client, planner)
    monkeypatch.setenv(ENV_NAME, "1")

    assert client.remove_seed() is runtime.types.SeedRemovalResult.UNSUPPORTED
    assert client.last_advertisement is not None
    assert "/remove_seed" not in [path for _, path in planner.calls]
