# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

"""Compatibility guards for operations that replace live model memory."""

import logging
from typing import Any

logger = logging.getLogger(__name__)


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
    """Reject weight mutation while any RFork session may own memory."""
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


def shutdown_rfork_sessions(vllm_config: Any) -> bool:
    """Shut down RFork sessions before their model storage is released."""
    speculative_config = getattr(vllm_config, "speculative_config", None)
    load_configs = (
        getattr(speculative_config, "draft_load_config", None),
        getattr(vllm_config, "load_config", None),
    )
    references: list[tuple[Any, str, Any]] = []
    sessions: list[Any] = []
    seen_ids: set[int] = set()
    all_ok = True

    for config in load_configs:
        if config is None:
            continue
        for attr in ("rfork_draft_session", "rfork_session"):
            try:
                session = getattr(config, attr, None)
            except Exception:
                logger.exception("Failed to inspect RFork session attribute %s during worker shutdown.", attr)
                all_ok = False
                continue
            if session is None:
                continue
            references.append((config, attr, session))
            session_id = id(session)
            if session_id not in seen_ids:
                seen_ids.add(session_id)
                sessions.append(session)

    shutdown_ids: set[int] = set()
    for session in sessions:
        try:
            result = session.shutdown()
        except Exception:
            logger.exception("RFork session shutdown raised during worker shutdown.")
            all_ok = False
            continue
        if not result:
            logger.warning("RFork session shutdown did not complete during worker shutdown.")
            all_ok = False
            continue
        shutdown_ids.add(id(session))

    for config, attr, session in references:
        if id(session) not in shutdown_ids:
            continue
        try:
            if getattr(config, attr, None) is session:
                setattr(config, attr, None)
        except Exception:
            logger.exception("Failed to clear completed RFork session attribute %s.", attr)
            all_ok = False

    return all_ok
