# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

import sys

import pytest

from .rfork_test_support import _load_module, _stub


def test_native_hardware_profile_is_returned(monkeypatch):
    expected = object()
    _stub(
        monkeypatch,
        "vllm_ascend.device.hardware_profile",
        get_current_hardware_profile=lambda: expected,
    )
    compat = _load_module(monkeypatch, "rfork_test_compat_native", "compat.py")

    assert compat.get_current_hardware_profile() is expected


@pytest.mark.parametrize(("is_310p", "policy"), [(True, "FORCE_NZ"), (False, "CONFIGURABLE")])
def test_missing_hardware_profile_uses_legacy_policy(monkeypatch, is_310p, policy):
    monkeypatch.setitem(sys.modules, "vllm_ascend.device.hardware_profile", None)
    _stub(monkeypatch, "vllm_ascend.utils", is_310p=lambda: is_310p)
    compat = _load_module(monkeypatch, "rfork_test_compat_legacy", "compat.py")

    assert compat.get_current_hardware_profile().weight_layout_policy.name == policy
