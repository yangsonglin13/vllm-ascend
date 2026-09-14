# Copyright 2026 Huawei Technologies Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Restore load-derived flags and target-shared draft module aliases."""

from typing import Any

from vllm.logger import logger

from vllm_ascend.model_loader.rfork.types import RForkSeedServiceStartResult

# Load-derived attributes read by later runtime decisions; keep JSON-serializable for transfer-info.
RFORK_LOAD_STATE_ATTRS: tuple[str, ...] = (
    "has_own_lm_head",
    "has_own_embed_tokens",
    # DSpark sets False when no confidence head ships; True default would compute from unloaded weights.
    "enable_confidence_head",
    # DSpark records that load_weights rotated fc; receivers must not rotate the already-rotated bytes.
    "fc_rotation_applied",
)

# Suffixes identifying the draft module for a seed-shared weight; heads bind to the target's lm_head.
EMBEDDING_WEIGHT_SUFFIXES = ("embed_tokens.weight", "embedding.weight")
HEAD_WEIGHT_SUFFIXES = ("lm_head.weight", "shared_head.head.weight")
# Buffers the v1 proposer shares unconditionally; binding mirrors that path for receivers.
SHARED_BUFFER_SUFFIXES = ("topk_indices_buffer",)


def capture_load_derived_state(model: Any) -> dict[str, Any]:
    """Read load-derived attributes from a fully loaded model."""
    state: dict[str, Any] = {}
    for attr in RFORK_LOAD_STATE_ATTRS:
        if hasattr(model, attr):
            state[attr] = getattr(model, attr)
    return state


def restore_load_derived_state(model: Any, state: Any) -> None:
    """Apply validated load-derived attributes and rebuild model aliases."""
    if isinstance(state, dict):
        # Restore only known names; require real booleans and validate all before applying any.
        validated: dict[str, bool] = {}
        for attr in RFORK_LOAD_STATE_ATTRS:
            if attr not in state:
                continue
            value = state[attr]
            if not isinstance(value, bool):
                raise ValueError(f"RFork seed load-state attribute {attr!r} must be a boolean, got {value!r}.")
            validated[attr] = value
        for attr, value in validated.items():
            setattr(model, attr, value)
    # Module aliases (e.g. the MTP lm_head alias) cannot travel as JSON; the model rebuilds them from flags.
    restore_hook = getattr(model, "_restore_load_derived_state", None)
    if callable(restore_hook):
        restore_hook()


def _get_child_module(obj: Any, part: str) -> Any:
    """Resolve one path component across dict/list/attribute containers."""
    get = getattr(obj, "get", None)
    if callable(get) and not isinstance(obj, (list, tuple)):
        child = get(part)
        if child is not None:
            return child
    try:
        index = int(part)
    except ValueError:
        index = None
    # list-like containers (list, tuple, nn.ModuleList) index numerically.
    if index is not None and hasattr(obj, "__getitem__"):
        try:
            return obj[index]
        except (IndexError, TypeError, KeyError):
            pass
    return getattr(obj, part, None)


def _navigate_module(root: Any, parts: list[str]) -> Any:
    """Walk a dotted module path; None when any component is missing."""
    obj = root
    for part in parts:
        if obj is None:
            return None
        obj = _get_child_module(obj, part)
    return obj


def _resolve_target_language_model(target_model: Any) -> Any:
    """Resolve the language model out of multimodal wrappers, if any."""
    if hasattr(target_model, "get_language_model"):
        try:
            language_model = target_model.get_language_model()
        except Exception:
            language_model = None
        if language_model is not None:
            return language_model
    language_model = getattr(target_model, "language_model", None)
    return language_model if language_model is not None else target_model


def _resolve_target_lm_head(target_language_model: Any) -> Any:
    """Resolve the target's lm_head through the same fallbacks as the proposer."""
    target_lm_head = getattr(target_language_model, "lm_head", None)
    if target_lm_head is not None:
        return target_lm_head
    if hasattr(target_language_model, "get_language_model"):
        target_lm_head = getattr(target_language_model.get_language_model(), "lm_head", None)
        if target_lm_head is not None:
            return target_lm_head
    return getattr(getattr(target_language_model, "language_model", None), "lm_head", None)


def _bind_shared_head_name(draft_model: Any, target_language_model: Any, name: str, parts: list[str]) -> None:
    """Bind one seed-declared head weight to the target's lm_head."""
    target_lm_head = _resolve_target_lm_head(target_language_model)
    if target_lm_head is None:
        raise RuntimeError(f"RFork seed shared head weight {name!r} but the target model has no accessible lm_head.")

    if name == "lm_head.weight":
        if getattr(draft_model, "lm_head", None) is None:
            raise RuntimeError(f"RFork seed shared {name!r} but the draft exposes no top-level lm_head.")
        draft_model.lm_head = target_lm_head
        return

    # Replace only that exact layer's head; independently transferred heads elsewhere stay intact.
    parent = _navigate_module(draft_model, parts[:-2])
    old_head = getattr(parent, "head", None) if parent is not None else None
    if old_head is None:
        raise RuntimeError(f"RFork seed shared {name!r} but the draft has no shared_head.head at that path.")
    parent.head = target_lm_head
    # Keep the load-derived alias in sync when it referenced the replaced head.
    if getattr(draft_model, "lm_head", None) is old_head:
        draft_model.lm_head = target_lm_head


def _bind_shared_module_by_path(
    draft_model: Any,
    target_language_model: Any,
    name: str,
    module_parts: list[str],
) -> None:
    """Bind one seed-declared shared weight to the target module at the same path."""
    target_module = _navigate_module(target_language_model, module_parts)
    if target_module is None and name.endswith(EMBEDDING_WEIGHT_SUFFIXES):
        # Fall back to the proposer's embedding resolution for other wrapper layouts.
        target_inner = getattr(target_language_model, "model", None)
        target_module = getattr(target_inner, "embed_tokens", None) or getattr(target_inner, "embedding", None)
    if target_module is None:
        raise RuntimeError(f"RFork seed shared weight {name!r} but the target model has no module at that path.")

    parent = _navigate_module(draft_model, module_parts[:-1]) if len(module_parts) > 1 else draft_model
    leaf = module_parts[-1]
    if parent is None or getattr(parent, leaf, None) is None:
        raise RuntimeError(f"RFork seed shared weight {name!r} but the draft has no module at that path.")
    setattr(parent, leaf, target_module)


def _bind_shared_buffer_name(
    draft_model: Any,
    target_language_model: Any,
    name: str,
    parts: list[str],
) -> None:
    """Bind one seed-declared buffer to the target's model-level buffer."""
    target_inner = getattr(target_language_model, "model", None)
    target_buffer = getattr(target_inner, "topk_indices_buffer", None)
    if target_buffer is None:
        # Fall back to the declared path for targets that keep per-layer buffers.
        target_buffer = _navigate_module(target_language_model, parts)
    if target_buffer is None:
        raise RuntimeError(f"RFork seed shared buffer {name!r} but the target model has no topk_indices_buffer.")

    parent = _navigate_module(draft_model, parts[:-1]) if len(parts) > 1 else draft_model
    leaf = parts[-1]
    if parent is None or getattr(parent, leaf, None) is None:
        raise RuntimeError(f"RFork seed shared buffer {name!r} but the draft has no module at that path.")
    setattr(parent, leaf, target_buffer)


def force_bind_seed_shared_modules(draft_model: Any, target_model: Any, shared_names: Any) -> None:
    """Bind seed-shared modules to the target; reject unknown mappings."""
    names = tuple(shared_names or ())
    if not names:
        return

    target_language_model = _resolve_target_language_model(target_model)
    for name in names:
        parts = name.split(".")
        if name.endswith(HEAD_WEIGHT_SUFFIXES):
            _bind_shared_head_name(draft_model, target_language_model, name, parts)
        elif name.endswith(EMBEDDING_WEIGHT_SUFFIXES):
            _bind_shared_module_by_path(draft_model, target_language_model, name, parts[:-1])
        elif name.endswith(SHARED_BUFFER_SUFFIXES):
            _bind_shared_buffer_name(draft_model, target_language_model, name, parts)
        else:
            raise RuntimeError(
                f"RFork seed shared weight {name!r} has no known module mapping; "
                "the transfer skipped it and the draft would keep allocation-time values."
            )


def _deferred_rfork_session(owner: Any) -> Any:
    """Resolve the draft session from v1 or v2 load configuration."""
    speculative_config = getattr(owner, "speculative_config", None)
    if speculative_config is None:
        speculative_config = getattr(getattr(owner, "vllm_config", None), "speculative_config", None)
    vllm_config = getattr(owner, "vllm_config", None)
    candidates = (
        getattr(getattr(speculative_config, "draft_load_config", None), "rfork_draft_session", None),
        getattr(getattr(vllm_config, "load_config", None), "rfork_draft_session", None),
    )
    for session in candidates:
        if session is not None:
            return session
    return None


def force_bind_rfork_seed_shared_modules(owner: Any, draft_model: Any, target_model: Any) -> None:
    """Bind seed-declared shared draft weights to this process's target."""
    session = _deferred_rfork_session(owner)
    if session is None:
        return
    get_names = getattr(session, "get_seed_shared_names", None)
    if not callable(get_names):
        return
    shared_names = get_names()
    if not isinstance(shared_names, (list, tuple, set, frozenset)):
        # Fresh backends and non-RFork transfer paths carry no recorded names.
        return
    if shared_names:
        force_bind_seed_shared_modules(draft_model, target_model, shared_names)


def complete_deferred_rfork_seed_start(owner: Any) -> None:
    """Promote a deferred draft seed after target sharing is final."""
    session = _deferred_rfork_session(owner)
    if session is None:
        return
    has_pending = getattr(session, "has_deferred_seed_start", None)
    complete = getattr(session, "complete_deferred_seed_start", None)
    if not callable(has_pending) or not callable(complete):
        return
    try:
        if not has_pending():
            return
        result = complete()
    except Exception:
        logger.exception("[spec_decode] RFork deferred seed start promotion raised; inference can continue.")
        return
    if result is RForkSeedServiceStartResult.FAILED:
        logger.warning("[spec_decode] RFork deferred seed start promotion failed; inference can continue.")


def finish_rfork_deferred_seed_start(owner: Any, draft_model: Any, target_model: Any) -> None:
    """Bind skipped shared weights, then publish the draft's final topology."""
    force_bind_rfork_seed_shared_modules(owner, draft_model, target_model)
    complete_deferred_rfork_seed_start(owner)
