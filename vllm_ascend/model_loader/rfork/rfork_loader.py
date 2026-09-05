#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
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
#

import gc
import hashlib
import json
import math
import os
import time
from collections.abc import Iterator
from contextlib import contextmanager
from copy import copy
from enum import Enum
from typing import Any, cast

import torch
import torch.nn as nn
from torch.nn import Module
from vllm.config import ModelConfig, VllmConfig
from vllm.config.load import LoadConfig
from vllm.distributed import get_tensor_model_parallel_rank
from vllm.distributed.parallel_state import get_ep_group, get_pp_group
from vllm.logger import logger
from vllm.model_executor.model_loader import register_model_loader
from vllm.model_executor.model_loader.base_loader import BaseModelLoader
from vllm.model_executor.model_loader.utils import (
    initialize_model,
    process_weights_after_loading,
)
from vllm.utils.torch_utils import set_default_torch_dtype

from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.device.hardware_profile import get_current_hardware_profile
from vllm_ascend.model_loader.rfork.rfork_worker import RForkWorker

DEFAULT_RFORK_SEED_TIMEOUT_SEC = 5.0
DEFAULT_RFORK_REQUEST_TIMEOUT_SEC = 10.0


def _canonicalize_fingerprint_value(value: Any) -> Any:
    """Convert a configuration value to deterministic JSON-compatible data."""

    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, torch.dtype):
        return str(value)
    if isinstance(value, Enum):
        return value.name
    if isinstance(value, dict):
        return {
            str(key): _canonicalize_fingerprint_value(item)
            for key, item in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_canonicalize_fingerprint_value(item) for item in value]
    if isinstance(value, set | frozenset):
        canonical_values = [_canonicalize_fingerprint_value(item) for item in value]
        return sorted(canonical_values, key=lambda item: json.dumps(item, sort_keys=True, default=str))

    # Quantization configs and hardware policy enums generally expose a stable
    # name. Avoid repr(value), which may contain a process-local memory address.
    name = getattr(value, "name", None)
    if isinstance(name, str):
        return name
    get_name = getattr(value, "get_name", None)
    if callable(get_name):
        try:
            named_value = get_name()
        except Exception:  # pragma: no cover - defensive for third-party configs
            named_value = None
        if isinstance(named_value, str):
            return named_value

    public_attributes = getattr(value, "__dict__", None)
    if isinstance(public_attributes, dict):
        attributes = {
            str(key): _canonicalize_fingerprint_value(item)
            for key, item in sorted(public_attributes.items(), key=lambda item: str(item[0]))
            if not str(key).startswith("_") and not callable(item)
        }
        if attributes:
            value_type = type(value)
            return {
                "type": f"{value_type.__module__}.{value_type.__qualname__}",
                "attributes": attributes,
            }

    value_type = type(value)
    return f"{value_type.__module__}.{value_type.__qualname__}"


def _get_hf_config_descriptor(model_config: ModelConfig) -> list[dict[str, Any]]:
    descriptors: list[dict[str, Any]] = []
    for attr in ("hf_config", "hf_text_config"):
        hf_config = getattr(model_config, attr, None)
        if hf_config is None:
            continue
        architectures = getattr(hf_config, "architectures", None)
        if isinstance(architectures, str):
            architectures = [architectures]
        if not isinstance(architectures, (list, tuple)):
            architectures = []
        descriptors.append(
            {
                "field": attr,
                "model_type": _canonicalize_fingerprint_value(getattr(hf_config, "model_type", None)),
                "architectures": sorted(
                    str(architecture) for architecture in architectures if isinstance(architecture, str)
                ),
                "config_type": f"{type(hf_config).__module__}.{type(hf_config).__qualname__}",
            }
        )
    if descriptors:
        return descriptors

    architecture = getattr(model_config, "architecture", None)
    if architecture is None:
        architecture = getattr(model_config, "architectures", None)
    if architecture is not None:
        return [{"field": "model_config", "architectures": _canonicalize_fingerprint_value(architecture)}]

    return [
        {
            "field": "model_config",
            "config_type": f"{type(model_config).__module__}.{type(model_config).__qualname__}",
        }
    ]


def _get_model_revision(model_config: ModelConfig) -> Any:
    revision = getattr(model_config, "revision", None)
    if revision is None:
        revision = getattr(model_config, "model_revision", None)
    if revision is not None:
        return _canonicalize_fingerprint_value(revision)
    for attr in ("hf_config", "hf_text_config"):
        hf_config = getattr(model_config, attr, None)
        for revision_attr in ("_commit_hash", "commit_hash", "revision"):
            revision = getattr(hf_config, revision_attr, None)
            if revision is not None:
                return _canonicalize_fingerprint_value(revision)
    return None


def _get_parallel_world_size(parallel_config: object | None, *attrs: str, default: Any = None) -> Any:
    if parallel_config is None:
        return default
    for attr in attrs:
        value = getattr(parallel_config, attr, None)
        if value is not None:
            return _canonicalize_fingerprint_value(value)
    return default


def _get_quantization_config_digest(model_config: ModelConfig) -> str | None:
    """Digest the effective quantization config so parameter changes re-key the fingerprint."""
    quantization_config = None
    try:
        quantization_config = getattr(model_config, "hf_quant_config", None)
    except Exception:  # pragma: no cover - property may raise before config load
        quantization_config = None
    if quantization_config is None:
        for attr in ("hf_config", "hf_text_config"):
            hf_config = getattr(model_config, attr, None)
            if hf_config is None:
                continue
            quantization_config = getattr(hf_config, "quantization_config", None)
            if quantization_config is not None:
                break
    if quantization_config is None:
        return None
    canonical_json = json.dumps(
        _canonicalize_fingerprint_value(quantization_config),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(canonical_json).hexdigest()


def _build_rfork_compatibility_fingerprint(
    vllm_config: VllmConfig,
    model_config: ModelConfig,
    *,
    model_url: str,
    model_deploy_strategy_name: str,
) -> str:
    """Build the RFork compatibility identity and return its SHA256 digest."""

    parallel_config = getattr(vllm_config, "parallel_config", None)

    try:
        ascend_config = get_ascend_config()
        weight_nz_mode = getattr(ascend_config, "weight_nz_mode", None)
    except Exception:  # pragma: no cover - only reached before Ascend config setup
        weight_nz_mode = None
    try:
        hardware_profile = get_current_hardware_profile()
        hardware_policy = getattr(hardware_profile, "weight_layout_policy", None)
    except Exception:  # pragma: no cover - only reached on unsupported test hardware
        hardware_policy = None

    descriptor = {
        "model_url": model_url,
        "model_deploy_strategy_name": model_deploy_strategy_name,
        "model_revision": _get_model_revision(model_config),
        "dtype": _canonicalize_fingerprint_value(getattr(model_config, "dtype", None)),
        "quantization": _canonicalize_fingerprint_value(getattr(model_config, "quantization", None)),
        "quantization_config": _get_quantization_config_digest(model_config),
        "architecture": _get_hf_config_descriptor(model_config),
        "parallel": {
            "tensor_world_size": _get_parallel_world_size(
                parallel_config, "tensor_parallel_size", "tp_size", default=1
            ),
            "pipeline_world_size": _get_parallel_world_size(
                parallel_config, "pipeline_parallel_size", "pp_size", default=1
            ),
            "expert_world_size": _get_parallel_world_size(
                parallel_config, "expert_parallel_size", "ep_size", default=None
            ),
            "data_world_size": _get_parallel_world_size(parallel_config, "data_parallel_size", "dp_size", default=None),
            "expert_parallel_enabled": bool(getattr(parallel_config, "enable_expert_parallel", False)),
            "is_moe_model": _canonicalize_fingerprint_value(getattr(parallel_config, "is_moe_model", None)),
        },
        "weight_layout": {
            "weight_nz_mode": _canonicalize_fingerprint_value(weight_nz_mode),
            "hardware_policy": _canonicalize_fingerprint_value(hardware_policy),
        },
    }
    canonical_json = json.dumps(
        _canonicalize_fingerprint_value(descriptor),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(canonical_json).hexdigest()


def _is_mtp_hf_config(hf_config: object | None) -> bool:
    if hf_config is None:
        return False

    model_type = getattr(hf_config, "model_type", None)
    if isinstance(model_type, str) and model_type.lower().endswith("_mtp"):
        return True

    architectures = getattr(hf_config, "architectures", None)
    if isinstance(architectures, str):
        architectures = [architectures]
    if not isinstance(architectures, (list, tuple)):
        return False

    return any(isinstance(architecture, str) and architecture.endswith("MTPModel") for architecture in architectures)


def _is_draft_model_config(model_config: object | None) -> bool:
    if model_config is None:
        return False
    if getattr(model_config, "runner_type", None) == "draft":
        return True

    return any(
        _is_mtp_hf_config(getattr(model_config, hf_config_attr, None))
        for hf_config_attr in ("hf_config", "hf_text_config")
    )


def _is_draft_model(vllm_config: VllmConfig, model_config: ModelConfig | None = None) -> bool:
    return (
        _is_draft_model_config(model_config)
        or _is_draft_model_config(getattr(vllm_config, "model_config", None))
        or _is_draft_model_config(getattr(vllm_config, "scheduler_config", None))
    )


def _get_rfork_worker_attr(vllm_config: VllmConfig, model_config: ModelConfig) -> str:
    return "rfork_draft_worker" if _is_draft_model(vllm_config, model_config) else "rfork_worker"


def _get_ep_rank(vllm_config: VllmConfig) -> int | None:
    parallel_config = vllm_config.parallel_config
    if not parallel_config.enable_expert_parallel or getattr(parallel_config, "is_moe_model", None) is False:
        return None

    try:
        return get_ep_group().rank_in_group
    except AssertionError as e:
        raise RuntimeError("Expert parallelism is enabled, but the EP group is not initialized.") from e


def _get_pp_rank(vllm_config: VllmConfig) -> int | None:
    if getattr(vllm_config.parallel_config, "pipeline_parallel_size", 1) <= 1:
        return None

    try:
        return get_pp_group().rank_in_group
    except AssertionError as e:
        raise RuntimeError("Pipeline parallelism is enabled, but the PP group is not initialized.") from e


def _make_fallback_load_config(load_config: LoadConfig) -> LoadConfig:
    fallback_load_config = copy(load_config)
    fallback_load_config.load_format = "auto"
    fallback_load_config.model_loader_extra_config = {}
    return fallback_load_config


def _reset_process_global_model_state(vllm_config: VllmConfig, model: Module | None = None) -> None:
    """Remove process-global layer registries a discarded RFork model left behind."""
    stale_modules = set(model.modules()) if model is not None else None
    removed_names: set[str] = set()
    compilation_config = getattr(vllm_config, "compilation_config", None)
    if compilation_config is not None:
        static_forward_context = getattr(compilation_config, "static_forward_context", None)
        if isinstance(static_forward_context, dict):
            if stale_modules is None:
                removed_names.update(static_forward_context)
                static_forward_context.clear()
            else:
                for name, module in list(static_forward_context.items()):
                    if module in stale_modules:
                        removed_names.add(name)
                        del static_forward_context[name]
        static_all_moe_layers = getattr(compilation_config, "static_all_moe_layers", None)
        if isinstance(static_all_moe_layers, list):
            if stale_modules is None:
                static_all_moe_layers.clear()
            else:
                static_all_moe_layers[:] = [
                    layer
                    for layer in static_all_moe_layers
                    if layer not in stale_modules and layer not in removed_names
                ]

    # ROPE instances are cached globally and keyed by config; rebuild fresh rope.
    try:
        from vllm.model_executor.layers.rotary_embedding import _ROPE_DICT

        if isinstance(_ROPE_DICT, dict):
            _ROPE_DICT.clear()
    except Exception as e:  # pragma: no cover - best-effort across vLLM versions
        logger.debug("RFork fallback: skip clearing _ROPE_DICT: %s", e)


def _iter_ascend_moe_quant_methods(model: Module) -> Iterator[Any]:
    """Yield each quant method owned by an Ascend MoE runner once."""
    from vllm_ascend.ops.fused_moe.fused_moe import AscendMoERunner

    seen_quant_methods: set[int] = set()
    for module in model.modules():
        if not isinstance(module, AscendMoERunner):
            continue

        # AscendMoERunner exposes the routed experts' method via private _quant_method.
        quant_method = getattr(module, "_quant_method", None)
        if quant_method is None or id(quant_method) in seen_quant_methods:
            continue

        seen_quant_methods.add(id(quant_method))
        yield cast(Any, quant_method)


@contextmanager
def _rfork_pre_transfer_weight_processing(model: Module):
    """Use the unwrapped MoE post-load step so RFork pre-transfer skips shared-expert validation."""
    restored: list[tuple[Any, object]] = []
    for quant_method in _iter_ascend_moe_quant_methods(model):
        process_weights = getattr(quant_method, "process_weights_after_loading", None)
        original_process_weights = getattr(process_weights, "__wrapped__", None)
        if original_process_weights is None:
            continue

        restored.append((quant_method, process_weights))
        quant_method.process_weights_after_loading = original_process_weights

    try:
        yield
    finally:
        for quant_method, process_weights in restored:
            quant_method.process_weights_after_loading = process_weights


def _is_dynamic_eplb_enabled(vllm_config: VllmConfig) -> bool:
    parallel_config = getattr(vllm_config, "parallel_config", None)
    if bool(getattr(parallel_config, "enable_eplb", False)):
        return True

    eplb_config = get_ascend_config().eplb_config
    return eplb_config.dynamic_eplb or bool(eplb_config.expert_map_record_path)


@contextmanager
def _rfork_skip_unquantized_moe_post_load_processing(model: Module):
    """Suppress unquantized MoE post-load processing; dense layers still run theirs."""

    from vllm_ascend.ops.fused_moe.routed_experts import AscendUnquantizedFusedMoEMethod

    restored_methods: list[tuple[Any, object]] = []
    for quant_method in _iter_ascend_moe_quant_methods(model):
        if not isinstance(quant_method, AscendUnquantizedFusedMoEMethod):
            continue

        process_weights = getattr(quant_method, "process_weights_after_loading", None)
        if process_weights is None:
            continue

        restored_methods.append((quant_method, process_weights))
        quant_method.process_weights_after_loading = _noop_process_weights_after_loading  # type: ignore[method-assign]

    try:
        yield
    finally:
        for quant_method, process_weights in restored_methods:
            quant_method.process_weights_after_loading = process_weights


def _noop_process_weights_after_loading(*args: Any, **kwargs: Any) -> None:
    pass


@register_model_loader("rfork")
class RForkModelLoader(BaseModelLoader):
    def __init__(self, load_config: LoadConfig):
        super().__init__(load_config)
        config = load_config.model_loader_extra_config
        if config is None:
            config = {}
        elif not isinstance(config, dict):
            err_msg = "RFork requires --model-loader-extra-config to be a JSON object."
            logger.error(err_msg)
            raise RuntimeError(err_msg)

        def _get_env_value(env_name: str) -> Any:
            return os.getenv(env_name)

        def _get_extra_config_string(
            keys: tuple[str, ...],
            env_name: str,
            default: str | None = "",
        ) -> str | None:
            value: Any = None
            for key in keys:
                if key in config:
                    value = config[key]
                    break
            if not isinstance(value, str) or not value:
                value = _get_env_value(env_name)
            return value if isinstance(value, str) and value else default

        def _parse_positive_float(value: Any) -> float | None:
            if isinstance(value, bool) or not isinstance(value, (int, float, str)):
                return None
            try:
                parsed_value = float(value)
            except (TypeError, ValueError):
                return None
            if not math.isfinite(parsed_value) or parsed_value <= 0:
                return None
            return parsed_value

        def _get_extra_config_float(
            keys: tuple[str, ...],
            env_name: str,
            default: float,
        ) -> float:
            config_value: Any = None
            has_config_value = False
            for key in keys:
                if key in config:
                    config_value = config[key]
                    has_config_value = True
                    break
            if has_config_value:
                parsed_value = _parse_positive_float(config_value)
                if parsed_value is not None:
                    return parsed_value

            env_value = _get_env_value(env_name)
            parsed_value = _parse_positive_float(env_value)
            return default if parsed_value is None else parsed_value

        self.model_url = _get_extra_config_string(("model_url",), "MODEL_URL", "") or ""
        self.model_deploy_strategy_name = (
            _get_extra_config_string(
                ("model_deploy_strategy_name",),
                "MODEL_DEPLOY_STRATEGY_NAME",
                "",
            )
            or ""
        )
        self.scheduler_url = _get_extra_config_string(("rfork_scheduler_url",), "RFORK_SCHEDULER_URL", "") or ""
        self.seed_timeout_sec = _get_extra_config_float(
            ("rfork_seed_timeout_sec",),
            "RFORK_SEED_TIMEOUT_SEC",
            DEFAULT_RFORK_SEED_TIMEOUT_SEC,
        )
        self.request_timeout_sec = _get_extra_config_float(
            ("rfork_request_timeout_sec", "request_timeout_sec"),
            "RFORK_REQUEST_TIMEOUT_SEC",
            DEFAULT_RFORK_REQUEST_TIMEOUT_SEC,
        )
        self.seed_bind_host = (
            _get_extra_config_string(
                ("rfork_seed_bind_host", "seed_bind_host", "bind_host"),
                "RFORK_SEED_BIND_HOST",
                "0.0.0.0",
            )
            or "0.0.0.0"
        )
        self.seed_advertise_host = _get_extra_config_string(
            ("rfork_seed_advertise_host", "seed_advertise_host", "advertise_host"),
            "RFORK_SEED_ADVERTISE_HOST",
            None,
        )

        logger.info(
            "Initializing rfork with config: "
            "MODEL_URL=%s, MODEL_DEPLOY_STRATEGY_NAME=%s, "
            "SCHEDULER_URL=%s, SEED_TIMEOUT_SEC=%s, REQUEST_TIMEOUT_SEC=%s, "
            "SEED_BIND_HOST=%s, SEED_ADVERTISE_HOST=%s",
            self.model_url,
            self.model_deploy_strategy_name,
            self.scheduler_url,
            self.seed_timeout_sec,
            self.request_timeout_sec,
            self.seed_bind_host,
            self.seed_advertise_host,
        )

    def download_model(self, model_config: ModelConfig) -> None:
        raise NotImplementedError

    def load_weights(self, model: nn.Module, model_config: ModelConfig) -> None:
        raise NotImplementedError

    def _ensure_rfork_worker(self, vllm_config: VllmConfig, model_config: ModelConfig) -> RForkWorker:
        worker_attr = _get_rfork_worker_attr(vllm_config, model_config)
        rfork_worker = getattr(self.load_config, worker_attr, None)
        if rfork_worker is None:
            is_draft_model = _is_draft_model(vllm_config, model_config)
            device_id = torch.distributed.get_rank()
            pp_rank = _get_pp_rank(vllm_config)
            ep_rank = _get_ep_rank(vllm_config)
            tp_rank = get_tensor_model_parallel_rank()
            compatibility_fingerprint = _build_rfork_compatibility_fingerprint(
                vllm_config,
                model_config,
                model_url=self.model_url,
                model_deploy_strategy_name=self.model_deploy_strategy_name,
            )
            rfork_worker = RForkWorker(
                tp_rank=tp_rank,
                device_id=device_id,
                scheduler_url=self.scheduler_url,
                model_url=self.model_url,
                model_deploy_strategy_name=self.model_deploy_strategy_name,
                seed_timeout_sec=self.seed_timeout_sec,
                request_timeout_sec=self.request_timeout_sec,
                seed_bind_host=self.seed_bind_host,
                seed_advertise_host=self.seed_advertise_host,
                is_draft_model=is_draft_model,
                pp_rank=pp_rank,
                ep_rank=ep_rank,
                compatibility_fingerprint=compatibility_fingerprint,
            )
            setattr(self.load_config, worker_attr, rfork_worker)
            logger.info(
                "RFork worker initialized, load_format=rfork, is_draft_model=%s, worker_attr=%s, fingerprint=%s",
                is_draft_model,
                worker_attr,
                compatibility_fingerprint,
            )
        return rfork_worker

    def _get_target_registered_blocks(
        self,
        vllm_config: VllmConfig,
        model_config: ModelConfig,
    ) -> list[tuple[int, int]]:
        if not _is_draft_model(vllm_config, model_config):
            return []
        target_worker = getattr(self.load_config, "rfork_worker", None)
        target_transfer_backend = getattr(target_worker, "transfer_backend", None)
        return list(getattr(target_transfer_backend, "registered_weight_blocks", None) or [])

    def _requires_processed_layout_transfer(self, model_config: ModelConfig) -> bool:
        if getattr(model_config, "quantization", None) is not None:
            return True

        try:
            weight_nz_mode = getattr(get_ascend_config(), "weight_nz_mode", 0)
            if not isinstance(weight_nz_mode, bool) and int(weight_nz_mode) == 2:
                return True
        except (TypeError, ValueError, RuntimeError):
            # The model loader may be inspected before AscendConfig has been
            # fully initialized (for example by CPU-only unit tests).
            pass

        try:
            hardware_policy = getattr(get_current_hardware_profile(), "weight_layout_policy", None)
            return getattr(hardware_policy, "name", str(hardware_policy).split(".")[-1]) == "FORCE_NZ"
        except (AttributeError, RuntimeError):
            return False

    @staticmethod
    def _stop_rfork_seed_service(rfork_worker: RForkWorker) -> bool:
        """Stop any service owned by a worker before fallback/reinitialization."""

        for method_name in ("stop_seed_service", "shutdown"):
            stop_method = getattr(rfork_worker, method_name, None)
            if not callable(stop_method):
                continue
            try:
                result = stop_method()
                return result is not False
            except Exception as exc:  # pragma: no cover - best-effort cleanup
                logger.warning("Failed to stop RFork seed service during cleanup: %s", exc)
                return False
        return True

    @classmethod
    def _cleanup_rfork_worker(cls, rfork_worker: RForkWorker) -> bool:
        """Best-effort cleanup preserving worker-owned unregister retry state."""

        service_stopped = cls._stop_rfork_seed_service(rfork_worker)
        cleanup_ok = service_stopped
        try:
            released = rfork_worker.post_transfer()
            if released is False:
                logger.warning("RFork seed lease release failed during cleanup.")
                cleanup_ok = False
        except Exception as exc:  # pragma: no cover - best-effort cleanup
            logger.warning("Failed to release RFork seed lease during cleanup: %s", exc)
            cleanup_ok = False
        if service_stopped:
            try:
                reset = rfork_worker.reset_transfer_state()
                if reset is False:
                    logger.warning("RFork transfer state reset failed during cleanup; retaining retry state.")
                    cleanup_ok = False
            except Exception as exc:  # pragma: no cover - best-effort cleanup
                logger.warning("Failed to reset RFork transfer state during cleanup: %s", exc)
                cleanup_ok = False
        else:
            logger.warning("RFork seed service is still alive; retaining its registered memory for safety.")
        return cleanup_ok

    @staticmethod
    def _start_rfork_seed_service(rfork_worker: RForkWorker, model: Module, processed_layout: bool) -> bool:
        """Start seed advertising and normalize old workers' implicit success."""

        try:
            result = rfork_worker.start_seed_service(model, processed_layout)
        except Exception as exc:
            logger.warning("RFork seed service startup failed: %s", exc)
            return False
        # Some older worker implementations returned None on success; treat
        # only an explicit False as failure.
        return result is not False

    def load_model(
        self,
        vllm_config: VllmConfig,
        model_config: ModelConfig,
        prefix: str = "",
    ) -> Module | None:
        device_config = vllm_config.device_config
        load_config = self.load_config
        load_device = device_config.device if load_config.device is None else load_config.device
        target_device = torch.device(load_device)

        with set_default_torch_dtype(model_config.dtype):
            need_del = False
            model: Module | None = None
            rfork_worker: RForkWorker | None = None
            processed_layout_transfer = self._requires_processed_layout_transfer(model_config)
            bypass_reason = None
            if _is_dynamic_eplb_enabled(vllm_config):
                bypass_reason = "dynamic EPLB"

            if bypass_reason is not None:
                logger.warning(
                    "RFork transfer is disabled when %s is enabled; using the default model loader.",
                    bypass_reason,
                )
                fallback_load_config = _make_fallback_load_config(self.load_config)

                from vllm.model_executor.model_loader import get_model

                try:
                    return get_model(
                        vllm_config=vllm_config,
                        model_config=model_config,
                        load_config=fallback_load_config,
                        prefix=prefix,
                    )
                except Exception:
                    logger.exception("RFork disabled for %s, but default loader failed.", bypass_reason)
                    raise

            try:
                # Worker construction and TransferEngine initialization belong
                # to the guarded RFork path. Missing optional dependencies or
                # an uninitialized parallel group must still reach fallback.
                rfork_worker = self._ensure_rfork_worker(vllm_config, model_config)
                # Draft workers must not re-register memory blocks already
                # registered by the target model (shared embedding storage).
                rfork_worker.set_excluded_weight_blocks(self._get_target_registered_blocks(vllm_config, model_config))
                if not rfork_worker.is_seed_available():
                    raise RuntimeError("seed is not available.")

                with target_device:
                    model = initialize_model(
                        vllm_config=vllm_config,
                        model_config=model_config,
                        prefix=prefix,
                    )
                    need_del = True

                if processed_layout_transfer:
                    logger.info("RFork uses post-load tensor layout transfer for this model layout.")
                    with _rfork_pre_transfer_weight_processing(model):
                        process_weights_after_loading(model, model_config, target_device)
                    # Complete async NPU layout conversion before exposing buffers.
                    torch.npu.synchronize()

                weight_load_start_time = time.perf_counter()
                if not rfork_worker.pre_transfer(model, processed_layout_transfer):
                    raise RuntimeError("pre_transfer failed.")
                if not rfork_worker.transfer(model, processed_layout_transfer):
                    raise RuntimeError("transfer failed.")
                if not rfork_worker.post_transfer():
                    raise RuntimeError("post_transfer failed.")
                logger.info(
                    "Loading model weights took %.2f seconds",
                    time.perf_counter() - weight_load_start_time,
                )

                if not processed_layout_transfer:
                    with _rfork_skip_unquantized_moe_post_load_processing(model):
                        process_weights_after_loading(model, model_config, target_device)

                # A seed must only become visible after all post-load work and
                # eval mode are complete. If service startup fails, the model
                # is already valid: unregister its MR and return it directly.
                model = model.eval()
                if not self._start_rfork_seed_service(rfork_worker, model, processed_layout_transfer):
                    if self._stop_rfork_seed_service(rfork_worker):
                        try:
                            reset = rfork_worker.reset_transfer_state()
                            if reset is False:
                                logger.warning("RFork seed startup cleanup could not unregister memory.")
                        except Exception as exc:  # pragma: no cover - best-effort cleanup
                            logger.warning("RFork seed startup cleanup failed: %s", exc)
                    else:
                        logger.warning("RFork seed server is still alive; retaining registered memory for safety.")
                return model
            except Exception as e:
                logger.warning("RFork transfer failed: %s, clean up and fall back to default loader", e)

                cleanup_ok = False
                if rfork_worker is not None:
                    cleanup_ok = self._cleanup_rfork_worker(rfork_worker)

                if need_del and model is not None:
                    _reset_process_global_model_state(vllm_config, model)

                    del model
                    gc.collect()
                    torch.npu.empty_cache()
                    for _ in range(3):
                        gc.collect()
                        torch.npu.empty_cache()

                fallback_load_config = _make_fallback_load_config(self.load_config)

                from vllm.model_executor.model_loader import get_model

                try:
                    model = get_model(
                        vllm_config=vllm_config,
                        model_config=model_config,
                        load_config=fallback_load_config,
                        prefix=prefix,
                    )
                except Exception:
                    logger.exception("RFork fallback default loader failed.")
                    raise

                # A failed RFork transfer may still leave a worker available to
                # seed later instances. Never advertise if construction failed;
                # a seed-start failure only cleans the MR and keeps this valid
                # fallback model.
                if (
                    rfork_worker is not None
                    and cleanup_ok
                    and not self._start_rfork_seed_service(rfork_worker, model, processed_layout_transfer)
                ):
                    if self._stop_rfork_seed_service(rfork_worker):
                        try:
                            reset = rfork_worker.reset_transfer_state()
                            if reset is False:
                                logger.warning("Fallback seed startup cleanup could not unregister memory.")
                        except Exception as exc:  # pragma: no cover - best-effort cleanup
                            logger.warning("Fallback seed startup cleanup failed: %s", exc)
                    else:
                        logger.warning("Fallback seed server is still alive; retaining registered memory for safety.")
                return model
