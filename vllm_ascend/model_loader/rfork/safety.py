# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

"""Compatibility guards for operations that replace live model memory."""

from typing import Any


def mutable_weights_bypass_reason(vllm_config: Any, model_config: Any) -> str | None:
    if any(
        bool(getattr(config, "enable_sleep_mode", False))
        for config in (model_config, getattr(vllm_config, "model_config", None))
    ):
        return "sleep mode"
    if getattr(vllm_config, "weight_transfer_config", None) is not None:
        return "online weight transfer (weight_transfer_config)"
    return None


def ensure_no_rfork_session(vllm_config: Any, operation: str) -> None:
    """Reject unconfigured mutations too, including a separately loaded draft.

    A session may retain registrations even after failed cleanup or shutdown.
    Until mutation has a drain/re-registration protocol, require a fresh worker
    loaded without RFork instead of treating stopped advertisement as safe.
    """
    speculative_config = getattr(vllm_config, "speculative_config", None)
    load_configs = (
        getattr(vllm_config, "load_config", None),
        getattr(speculative_config, "draft_load_config", None),
    )
    if any(
        getattr(config, attr, None) is not None
        for config in load_configs
        for attr in ("rfork_session", "rfork_draft_session")
    ):
        raise RuntimeError(
            f"{operation} is unavailable for a worker with an RFork session. "
            "Restart with --load-format auto for both target and draft models, "
            "or configure sleep mode/weight_transfer_config before loading."
        )
