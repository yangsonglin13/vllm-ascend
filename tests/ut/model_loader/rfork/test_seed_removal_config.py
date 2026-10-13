# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

"""Seed withdrawal stays opt-in through JSON configuration only."""

import pytest

from .test_legacy_planner import _advertise, _LegacyPlannerAdapter, _make_client


@pytest.mark.parametrize(
    "extra_config, expected",
    [({}, False), ({"rfork_seed_removal_supported": False}, False), ({"rfork_seed_removal_supported": True}, True)],
)
def test_json_switch_controls_seed_withdrawal(runtime, monkeypatch, extra_config, expected):
    config = type(runtime.config).from_extra_config({"rfork_scheduler_url": "http://planner", **extra_config})
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


@pytest.mark.parametrize("env_value", ["1", "invalid"])
def test_seed_withdrawal_ignores_unrecognized_environment_switch(runtime, monkeypatch, env_value):
    monkeypatch.setenv("VLLM_ASCEND_RFORK_SEED_REMOVAL_SUPPORTED", env_value)
    config = type(runtime.config).from_extra_config({"rfork_scheduler_url": "http://planner"})
    planner = _LegacyPlannerAdapter()
    client = _make_client(runtime, monkeypatch, planner, config=config)
    _advertise(client, planner)

    assert client.remove_seed() is runtime.types.SeedRemovalResult.UNSUPPORTED
    assert client.last_advertisement is not None
    assert "/remove_seed" not in [path for _, path in planner.calls]


def test_explicit_json_null_uses_auto_detection(runtime, monkeypatch):
    config = type(runtime.config).from_extra_config(
        {"rfork_seed_removal_supported": None, "rfork_scheduler_url": "http://planner"}
    )
    planner = _LegacyPlannerAdapter()
    client = _make_client(runtime, monkeypatch, planner, config=config)
    _advertise(client, planner)

    assert client.remove_seed() is runtime.types.SeedRemovalResult.UNCONFIRMED
    assert client.last_advertisement is not None
