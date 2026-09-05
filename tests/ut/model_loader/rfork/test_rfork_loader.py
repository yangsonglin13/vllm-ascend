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

from contextlib import nullcontext
from functools import wraps
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from vllm_ascend.model_loader.rfork.rfork_loader import (
    RForkModelLoader,
    _build_rfork_compatibility_fingerprint,
    _get_ep_rank,
    _get_pp_rank,
    _get_rfork_worker_attr,
    _is_draft_model,
    _is_dynamic_eplb_enabled,
    _make_fallback_load_config,
    _reset_process_global_model_state,
    _rfork_pre_transfer_weight_processing,
    _rfork_skip_unquantized_moe_post_load_processing,
)
from vllm_ascend.model_loader.rfork.rfork_worker import RForkWorker
from vllm_ascend.model_loader.rfork.seed_protocol import get_local_seed_key


class DummyLoadConfig:
    device = None
    load_format = "rfork"

    def __init__(self, model_loader_extra_config):
        self.model_loader_extra_config = model_loader_extra_config
        self.rfork_worker: Any = None
        self.rfork_draft_worker: Any = None


@pytest.mark.parametrize("config_value", [True, False])
def test_rfork_seed_timeout_bool_falls_back_to_env(monkeypatch, config_value):
    monkeypatch.setenv("RFORK_SEED_TIMEOUT_SEC", "7.5")

    loader = RForkModelLoader(
        DummyLoadConfig(
            {
                "rfork_seed_timeout_sec": config_value,
            }
        )
    )

    assert loader.seed_timeout_sec == 7.5


@pytest.mark.parametrize("config_value", [True, False])
def test_rfork_seed_timeout_bool_falls_back_to_default(monkeypatch, config_value):
    monkeypatch.delenv("RFORK_SEED_TIMEOUT_SEC", raising=False)

    loader = RForkModelLoader(
        DummyLoadConfig(
            {
                "rfork_seed_timeout_sec": config_value,
            }
        )
    )

    assert loader.seed_timeout_sec == 5.0


def test_rfork_environment_values_are_used_as_fallbacks(monkeypatch):
    monkeypatch.setenv("MODEL_URL", "env-model")
    monkeypatch.setenv("RFORK_REQUEST_TIMEOUT_SEC", "4.0")

    loader = RForkModelLoader(DummyLoadConfig({}))

    assert loader.model_url == "env-model"
    assert loader.request_timeout_sec == 4.0


@pytest.mark.parametrize("invalid_value", [True, False, 0, -1, float("nan"), float("inf"), "nan", "inf"])
def test_rfork_numeric_config_rejects_invalid_values(monkeypatch, invalid_value):
    monkeypatch.delenv("RFORK_SEED_TIMEOUT_SEC", raising=False)
    monkeypatch.delenv("RFORK_REQUEST_TIMEOUT_SEC", raising=False)

    loader = RForkModelLoader(
        DummyLoadConfig(
            {
                "rfork_seed_timeout_sec": invalid_value,
                "rfork_request_timeout_sec": invalid_value,
            }
        )
    )

    assert loader.seed_timeout_sec == 5.0
    assert loader.request_timeout_sec == 10.0


def test_rfork_network_configuration_is_passed(monkeypatch):
    loader = RForkModelLoader(
        DummyLoadConfig(
            {
                "rfork_seed_bind_host": "127.0.0.1",
                "rfork_seed_advertise_host": "10.0.0.2",
            }
        )
    )

    assert loader.seed_bind_host == "127.0.0.1"
    assert loader.seed_advertise_host == "10.0.0.2"


def _parallel_config(
    *,
    enable_eplb=False,
    enable_expert_parallel=False,
    pipeline_parallel_size=1,
    is_moe_model=True,
):
    return SimpleNamespace(
        enable_eplb=enable_eplb,
        enable_expert_parallel=enable_expert_parallel,
        pipeline_parallel_size=pipeline_parallel_size,
        is_moe_model=is_moe_model,
    )


def _vllm_config(model_config=None, scheduler_config=None, parallel_config=None):
    return SimpleNamespace(
        additional_config=None,
        device_config=SimpleNamespace(device="cpu"),
        model_config=model_config or SimpleNamespace(),
        parallel_config=parallel_config or _parallel_config(),
        scheduler_config=scheduler_config or SimpleNamespace(),
    )


def _parallel_vllm_config(
    *,
    enable_expert_parallel=False,
    pipeline_parallel_size=1,
    is_moe_model=True,
):
    return SimpleNamespace(
        parallel_config=_parallel_config(
            enable_expert_parallel=enable_expert_parallel,
            pipeline_parallel_size=pipeline_parallel_size,
            is_moe_model=is_moe_model,
        )
    )


def test_rfork_ep_rank_is_not_added_when_expert_parallel_is_disabled(monkeypatch):
    def fail_if_ep_group_is_accessed():
        pytest.fail("EP group should not be accessed when expert parallelism is disabled.")

    monkeypatch.setattr(
        "vllm_ascend.model_loader.rfork.rfork_loader.get_ep_group",
        fail_if_ep_group_is_accessed,
    )

    assert _get_ep_rank(_parallel_vllm_config()) is None


def test_rfork_ep_rank_comes_from_ep_group(monkeypatch):
    monkeypatch.setattr(
        "vllm_ascend.model_loader.rfork.rfork_loader.get_ep_group",
        lambda: SimpleNamespace(rank_in_group=7),
    )

    assert _get_ep_rank(_parallel_vllm_config(enable_expert_parallel=True)) == 7


def test_rfork_ep_rank_is_not_added_for_dense_model(monkeypatch):
    def fail_if_ep_group_is_accessed():
        pytest.fail("EP group should not be accessed for a dense model.")

    monkeypatch.setattr(
        "vllm_ascend.model_loader.rfork.rfork_loader.get_ep_group",
        fail_if_ep_group_is_accessed,
    )

    assert _get_ep_rank(_parallel_vllm_config(enable_expert_parallel=True, is_moe_model=False)) is None


def test_rfork_requires_initialized_ep_group(monkeypatch):
    def raise_uninitialized_ep_group():
        raise AssertionError("expert parallel group is not initialized")

    monkeypatch.setattr(
        "vllm_ascend.model_loader.rfork.rfork_loader.get_ep_group",
        raise_uninitialized_ep_group,
    )

    with pytest.raises(RuntimeError, match="EP group is not initialized"):
        _get_ep_rank(_parallel_vllm_config(enable_expert_parallel=True))


def test_rfork_pp_rank_is_not_added_when_pipeline_parallelism_is_disabled(monkeypatch):
    def fail_if_pp_group_is_accessed():
        pytest.fail("PP group should not be accessed when pipeline parallelism is disabled.")

    monkeypatch.setattr(
        "vllm_ascend.model_loader.rfork.rfork_loader.get_pp_group",
        fail_if_pp_group_is_accessed,
    )

    assert _get_pp_rank(_parallel_vllm_config()) is None


def test_rfork_pp_rank_comes_from_pp_group(monkeypatch):
    monkeypatch.setattr(
        "vllm_ascend.model_loader.rfork.rfork_loader.get_pp_group",
        lambda: SimpleNamespace(rank_in_group=3),
    )

    assert _get_pp_rank(_parallel_vllm_config(pipeline_parallel_size=2)) == 3


def test_rfork_requires_initialized_pp_group(monkeypatch):
    def raise_uninitialized_pp_group():
        raise AssertionError("pipeline parallel group is not initialized")

    monkeypatch.setattr(
        "vllm_ascend.model_loader.rfork.rfork_loader.get_pp_group",
        raise_uninitialized_pp_group,
    )

    with pytest.raises(RuntimeError, match="PP group is not initialized"):
        _get_pp_rank(_parallel_vllm_config(pipeline_parallel_size=2))


def test_rfork_seed_key_is_deterministic_for_identical_identity():
    common = {
        "tp_rank": 3,
        "model_url": "/models/dsv4",
        "model_deploy_strategy_name": "decode",
        "compatibility_fingerprint": "fp",
    }

    key = get_local_seed_key(**common)

    assert len(key) == 64
    assert get_local_seed_key(**common) == key


def test_rfork_seed_key_isolated_by_ep_rank():
    common_config = {
        "tp_rank": 0,
        "model_url": "/models/dsv4",
        "model_deploy_strategy_name": "decode",
        "compatibility_fingerprint": "fp",
    }

    assert get_local_seed_key(**common_config, ep_rank=0) != get_local_seed_key(**common_config, ep_rank=1)


def test_rfork_seed_key_isolated_by_pp_rank():
    common_config = {
        "tp_rank": 0,
        "ep_rank": 0,
        "model_url": "/models/dsv4",
        "model_deploy_strategy_name": "decode",
        "compatibility_fingerprint": "fp",
    }

    assert get_local_seed_key(**common_config, pp_rank=0) != get_local_seed_key(**common_config, pp_rank=1)


def test_rfork_seed_key_distinguishes_parallel_rank_types():
    common_config = {
        "model_url": "/models/dsv4",
        "model_deploy_strategy_name": "decode",
        "compatibility_fingerprint": "fp",
    }

    pp_key = get_local_seed_key(**common_config, pp_rank=3, tp_rank=1)
    ep_key = get_local_seed_key(**common_config, tp_rank=3, ep_rank=1)

    assert pp_key != ep_key


def test_rfork_draft_seed_key_isolated_by_ep_rank():
    common_config = {
        "tp_rank": 0,
        "model_url": "/models/dsv4",
        "model_deploy_strategy_name": "decode",
        "compatibility_fingerprint": "fp",
    }

    draft_key = get_local_seed_key(**common_config, is_draft_worker=True, ep_rank=5)
    target_key = get_local_seed_key(**common_config, ep_rank=5)

    assert draft_key != target_key


def test_rfork_worker_receives_parallel_ranks(monkeypatch):
    load_config = DummyLoadConfig({"model_url": "model", "model_deploy_strategy_name": "strategy"})
    loader = RForkModelLoader(load_config)
    model_config = SimpleNamespace()
    vllm_config = SimpleNamespace(
        model_config=model_config,
        scheduler_config=SimpleNamespace(),
        parallel_config=SimpleNamespace(),
    )
    captured = {}
    expected_worker = SimpleNamespace()

    def fake_rfork_worker(**kwargs):
        captured.update(kwargs)
        return expected_worker

    monkeypatch.setattr("vllm_ascend.model_loader.rfork.rfork_loader.RForkWorker", fake_rfork_worker)
    monkeypatch.setattr("vllm_ascend.model_loader.rfork.rfork_loader._get_pp_rank", lambda config: 3)
    monkeypatch.setattr("vllm_ascend.model_loader.rfork.rfork_loader._get_ep_rank", lambda config: 7)
    monkeypatch.setattr("vllm_ascend.model_loader.rfork.rfork_loader.get_tensor_model_parallel_rank", lambda: 5)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 11)

    worker = loader._ensure_rfork_worker(vllm_config, model_config)

    assert worker is expected_worker
    assert captured["tp_rank"] == 5
    assert captured["pp_rank"] == 3
    assert captured["ep_rank"] == 7
    assert captured["device_id"] == 11


def test_rfork_worker_set_excluded_weight_blocks_normalizes_input():
    worker = RForkWorker.__new__(RForkWorker)

    worker.set_excluded_weight_blocks([(128, 4096)])
    assert worker._excluded_weight_blocks == [(128, 4096)]

    worker.set_excluded_weight_blocks(None)
    assert worker._excluded_weight_blocks == []


def test_rfork_worker_pre_transfer_forwards_excluded_blocks():
    worker: Any = RForkWorker.__new__(RForkWorker)
    forwarded_blocks = []
    worker.device_id = 0
    worker.ready_to_start_seed_service = False
    worker._excluded_weight_blocks = [(128, 4096)]

    def register_memory_region(model, processed_layout, blocks):
        forwarded_blocks.append(blocks)
        return True

    worker.transfer_backend = SimpleNamespace(
        is_initialized=lambda: True,
        register_memory_region=register_memory_region,
    )

    assert worker.pre_transfer(object(), False)
    assert forwarded_blocks == [[(128, 4096)]]
    assert worker.ready_to_start_seed_service


def test_rfork_target_registered_blocks_not_collected_for_target_model():
    load_config = DummyLoadConfig({"model_url": "model", "model_deploy_strategy_name": "strategy"})
    loader = RForkModelLoader(load_config)
    load_config.rfork_worker = SimpleNamespace(transfer_backend=SimpleNamespace(registered_weight_blocks=[(128, 4096)]))
    target_model_config = SimpleNamespace()
    vllm_config = _vllm_config(model_config=target_model_config)

    assert loader._get_target_registered_blocks(vllm_config, target_model_config) == []


def test_rfork_target_registered_blocks_collected_for_draft_model():
    load_config = DummyLoadConfig({"model_url": "model", "model_deploy_strategy_name": "strategy"})
    loader = RForkModelLoader(load_config)
    draft_model_config = SimpleNamespace(hf_config=SimpleNamespace(model_type="qwen3_5_mtp"))
    target_blocks = [(128, 4096), (8192, 1024)]
    load_config.rfork_worker = SimpleNamespace(transfer_backend=SimpleNamespace(registered_weight_blocks=target_blocks))
    vllm_config = _vllm_config(model_config=draft_model_config)

    blocks = loader._get_target_registered_blocks(vllm_config, draft_model_config)

    assert blocks == target_blocks
    assert blocks is not target_blocks


def test_rfork_target_registered_blocks_empty_without_target_worker():
    load_config = DummyLoadConfig({"model_url": "model", "model_deploy_strategy_name": "strategy"})
    loader = RForkModelLoader(load_config)
    draft_model_config = SimpleNamespace(hf_config=SimpleNamespace(model_type="qwen3_5_mtp"))
    vllm_config = _vllm_config(model_config=draft_model_config)

    assert loader._get_target_registered_blocks(vllm_config, draft_model_config) == []


def _make_reset_transfer_state_worker(unregister_result):
    worker: Any = RForkWorker.__new__(RForkWorker)
    worker.device_id = 0
    worker.ready_to_start_seed_service = True
    worker.transfer_backend = SimpleNamespace(unregister_memory_region=lambda: unregister_result)
    return worker


def test_reset_transfer_state_propagates_unregister_result():
    worker = _make_reset_transfer_state_worker(False)

    # A failed unregister keeps the live registration servable and retains
    # the tracking state for a subsequent retry.
    assert not worker.reset_transfer_state()
    assert worker.ready_to_start_seed_service is True

    worker = _make_reset_transfer_state_worker(True)

    assert worker.reset_transfer_state()
    assert worker.ready_to_start_seed_service is False


def test_reset_transfer_state_survives_backend_exception():
    worker: Any = RForkWorker.__new__(RForkWorker)
    worker.device_id = 0
    worker.ready_to_start_seed_service = True

    def raise_error():
        raise RuntimeError("engine down")

    worker.transfer_backend = SimpleNamespace(unregister_memory_region=raise_error)

    assert not worker.reset_transfer_state()
    assert worker.ready_to_start_seed_service is True


def test_rfork_draft_load_passes_target_registered_blocks_to_worker(monkeypatch):
    import vllm.model_executor.model_loader as model_loader

    load_config = DummyLoadConfig({"model_url": "model", "model_deploy_strategy_name": "strategy"})
    loader = RForkModelLoader(load_config)
    draft_model_config = SimpleNamespace(
        dtype=torch.float32,
        model="/models/test",
        hf_config=SimpleNamespace(model_type="qwen3_5_mtp"),
    )
    vllm_config = _vllm_config(model_config=draft_model_config)
    target_blocks = [(128, 4096)]
    load_config.rfork_worker = SimpleNamespace(transfer_backend=SimpleNamespace(registered_weight_blocks=target_blocks))
    captured_blocks = []
    draft_worker = SimpleNamespace(
        is_seed_available=lambda: False,
        set_excluded_weight_blocks=lambda blocks: captured_blocks.append(list(blocks)),
        post_transfer=lambda: True,
        reset_transfer_state=lambda: True,
        start_seed_service=lambda model, processed_layout: None,
    )
    load_config.rfork_draft_worker = draft_worker

    expected_model = SimpleNamespace()

    def fake_get_model(**kwargs):
        return expected_model

    monkeypatch.setattr(model_loader, "get_model", fake_get_model)
    monkeypatch.setattr(
        "vllm_ascend.model_loader.rfork.rfork_loader.get_ascend_config",
        lambda: SimpleNamespace(eplb_config=SimpleNamespace(dynamic_eplb=False, expert_map_record_path=None)),
    )

    model = loader.load_model(vllm_config=vllm_config, model_config=draft_model_config)

    assert model is expected_model
    assert captured_blocks == [target_blocks]


def test_rfork_worker_receives_hardening_configuration(monkeypatch):
    load_config = DummyLoadConfig(
        {
            "model_url": "model",
            "model_deploy_strategy_name": "strategy",
            "rfork_request_timeout_sec": 2.5,
            "rfork_seed_bind_host": "127.0.0.1",
            "rfork_seed_advertise_host": "10.0.0.9",
        }
    )
    loader = RForkModelLoader(load_config)
    model_config = SimpleNamespace(dtype=torch.float16, quantization="ascend", revision="rev-a")
    vllm_config = SimpleNamespace(
        model_config=model_config,
        scheduler_config=SimpleNamespace(),
        parallel_config=SimpleNamespace(tensor_parallel_size=4, pipeline_parallel_size=1),
    )
    captured = {}
    expected_worker = SimpleNamespace()

    def fake_rfork_worker(**kwargs):
        captured.update(kwargs)
        return expected_worker

    monkeypatch.setattr("vllm_ascend.model_loader.rfork.rfork_loader.RForkWorker", fake_rfork_worker)
    monkeypatch.setattr("vllm_ascend.model_loader.rfork.rfork_loader._get_pp_rank", lambda config: None)
    monkeypatch.setattr("vllm_ascend.model_loader.rfork.rfork_loader._get_ep_rank", lambda config: None)
    monkeypatch.setattr("vllm_ascend.model_loader.rfork.rfork_loader.get_tensor_model_parallel_rank", lambda: 1)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 11)

    assert loader._ensure_rfork_worker(vllm_config, model_config) is expected_worker
    assert captured["request_timeout_sec"] == 2.5
    assert captured["seed_bind_host"] == "127.0.0.1"
    assert captured["seed_advertise_host"] == "10.0.0.9"
    assert len(captured["compatibility_fingerprint"]) == 64


def test_rfork_fingerprint_changes_for_compatibility_inputs(monkeypatch):
    model_config = SimpleNamespace(
        dtype=torch.float16,
        quantization="ascend",
        revision="rev-a",
        hf_config=SimpleNamespace(model_type="qwen", architectures=["QwenForCausalLM"]),
    )
    vllm_config = SimpleNamespace(
        parallel_config=SimpleNamespace(
            tensor_parallel_size=4,
            pipeline_parallel_size=2,
            expert_parallel_size=2,
            data_parallel_size=1,
        ),
    )
    monkeypatch.setattr(
        "vllm_ascend.model_loader.rfork.rfork_loader.get_ascend_config",
        lambda: SimpleNamespace(weight_nz_mode=1),
    )
    monkeypatch.setattr(
        "vllm_ascend.model_loader.rfork.rfork_loader.get_current_hardware_profile",
        lambda: SimpleNamespace(weight_layout_policy=SimpleNamespace(name="CONFIGURABLE")),
    )

    common_args = {
        "vllm_config": vllm_config,
        "model_config": model_config,
        "model_url": "model",
        "model_deploy_strategy_name": "strategy",
    }
    fingerprint = _build_rfork_compatibility_fingerprint(**common_args)
    changed_model_config = SimpleNamespace(**{**vars(model_config), "dtype": torch.bfloat16})
    changed = _build_rfork_compatibility_fingerprint(**{**common_args, "model_config": changed_model_config})

    assert len(fingerprint) == 64
    assert fingerprint != changed
    assert all(character in "0123456789abcdef" for character in fingerprint)


def test_rfork_fingerprint_changes_with_quantization_config(monkeypatch):
    hf_config = SimpleNamespace(
        model_type="qwen",
        architectures=["QwenForCausalLM"],
        quantization_config={"quant_method": "awq", "weight_bits": 4},
    )
    model_config = SimpleNamespace(
        dtype=torch.float16,
        quantization="ascend",
        revision="rev-a",
        hf_config=hf_config,
    )
    vllm_config = SimpleNamespace(
        parallel_config=SimpleNamespace(
            tensor_parallel_size=4,
            pipeline_parallel_size=2,
            expert_parallel_size=2,
            data_parallel_size=1,
        ),
    )
    monkeypatch.setattr(
        "vllm_ascend.model_loader.rfork.rfork_loader.get_ascend_config",
        lambda: SimpleNamespace(weight_nz_mode=1),
    )
    monkeypatch.setattr(
        "vllm_ascend.model_loader.rfork.rfork_loader.get_current_hardware_profile",
        lambda: SimpleNamespace(weight_layout_policy=SimpleNamespace(name="CONFIGURABLE")),
    )

    common_args = {
        "vllm_config": vllm_config,
        "model_config": model_config,
        "model_url": "model",
        "model_deploy_strategy_name": "strategy",
    }
    fingerprint = _build_rfork_compatibility_fingerprint(**common_args)
    changed_hf_config = SimpleNamespace(
        **{
            **vars(hf_config),
            "quantization_config": {"quant_method": "awq", "weight_bits": 8},
        }
    )
    changed_model_config = SimpleNamespace(**{**vars(model_config), "hf_config": changed_hf_config})
    changed = _build_rfork_compatibility_fingerprint(**{**common_args, "model_config": changed_model_config})

    assert fingerprint != changed


@pytest.mark.parametrize(
    ("quantization", "weight_nz_mode", "hardware_policy", "expected"),
    [
        ("ascend", 0, "CONFIGURABLE", True),
        (None, 2, "CONFIGURABLE", True),
        (None, 0, "FORCE_NZ", True),
        (None, 0, "CONFIGURABLE", False),
    ],
)
def test_rfork_processed_layout_covers_quantization_and_nz_modes(
    monkeypatch, quantization, weight_nz_mode, hardware_policy, expected
):
    monkeypatch.setattr(
        "vllm_ascend.model_loader.rfork.rfork_loader.get_ascend_config",
        lambda: SimpleNamespace(weight_nz_mode=weight_nz_mode),
    )
    monkeypatch.setattr(
        "vllm_ascend.model_loader.rfork.rfork_loader.get_current_hardware_profile",
        lambda: SimpleNamespace(weight_layout_policy=SimpleNamespace(name=hardware_policy)),
    )
    loader = RForkModelLoader(DummyLoadConfig({}))

    assert loader._requires_processed_layout_transfer(SimpleNamespace(quantization=quantization)) is expected


@pytest.mark.parametrize(
    "model_config",
    [
        SimpleNamespace(runner_type="draft"),
        SimpleNamespace(hf_config=SimpleNamespace(model_type="deepseek_mtp")),
        SimpleNamespace(hf_config=SimpleNamespace(architectures=["DeepSeekV4MTPModel"])),
        SimpleNamespace(hf_text_config=SimpleNamespace(architectures=["OpenPanguMTPModel"])),
    ],
)
def test_rfork_detects_draft_model(model_config):
    assert _is_draft_model(_vllm_config(model_config=model_config))


def test_rfork_detects_draft_model_from_scheduler_config():
    scheduler_config = SimpleNamespace(runner_type="draft")

    assert _is_draft_model(_vllm_config(scheduler_config=scheduler_config))


def test_rfork_does_not_treat_target_model_as_draft():
    target_model_config = SimpleNamespace(
        hf_config=SimpleNamespace(
            model_type="deepseek_v4",
            architectures=["DeepSeekV4ForCausalLM"],
        )
    )

    assert not _is_draft_model(_vllm_config(model_config=target_model_config))


def test_rfork_detects_explicit_draft_model_config():
    target_vllm_config = _vllm_config(
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(
                model_type="deepseek_v4",
                architectures=["DeepSeekV4ForCausalLM"],
            )
        )
    )
    draft_model_config = SimpleNamespace(
        hf_config=SimpleNamespace(
            model_type="deepseek_mtp",
            architectures=["DeepSeekV4MTPModel"],
        )
    )

    assert _is_draft_model(target_vllm_config, draft_model_config)


def test_rfork_uses_separate_worker_attr_for_explicit_draft_model_config():
    target_vllm_config = _vllm_config(
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(
                model_type="deepseek_v4",
                architectures=["DeepSeekV4ForCausalLM"],
            )
        )
    )
    draft_model_config = SimpleNamespace(
        hf_config=SimpleNamespace(
            model_type="deepseek_mtp",
            architectures=["DeepSeekV4MTPModel"],
        )
    )

    assert _get_rfork_worker_attr(target_vllm_config, target_vllm_config.model_config) == "rfork_worker"
    assert _get_rfork_worker_attr(target_vllm_config, draft_model_config) == "rfork_draft_worker"


def test_rfork_fallback_load_config_copy_does_not_mutate_original():
    original_extra_config = {"model_url": "model", "model_deploy_strategy_name": "tp8"}
    load_config = DummyLoadConfig(original_extra_config)

    fallback_load_config = _make_fallback_load_config(load_config)

    assert fallback_load_config is not load_config
    assert fallback_load_config.load_format == "auto"
    assert fallback_load_config.model_loader_extra_config == {}
    assert load_config.load_format == "rfork"
    assert load_config.model_loader_extra_config == original_extra_config


def test_rfork_detects_dynamic_eplb_config(monkeypatch):
    # Native Model Runner V2 EPLB is represented by ParallelConfig and does
    # not require the AscendConfig singleton.

    def fail_singleton_read():
        raise AssertionError("singleton should not be read")

    monkeypatch.setattr(
        "vllm_ascend.model_loader.rfork.rfork_loader.get_ascend_config",
        fail_singleton_read,
    )
    assert _is_dynamic_eplb_enabled(
        SimpleNamespace(
            parallel_config=SimpleNamespace(enable_eplb=True),
            additional_config=None,
        )
    )

    vllm_config = SimpleNamespace(
        parallel_config=SimpleNamespace(enable_eplb=False),
        # A conflicting raw value verifies that RFork consumes only the typed
        # singleton after AscendConfig initialization.
        additional_config={"eplb_config": {"dynamic_eplb": False}},
    )
    monkeypatch.setattr(
        "vllm_ascend.model_loader.rfork.rfork_loader.get_ascend_config",
        lambda: SimpleNamespace(eplb_config=SimpleNamespace(dynamic_eplb=True, expert_map_record_path=None)),
    )
    assert _is_dynamic_eplb_enabled(vllm_config)

    monkeypatch.setattr(
        "vllm_ascend.model_loader.rfork.rfork_loader.get_ascend_config",
        lambda: SimpleNamespace(
            eplb_config=SimpleNamespace(dynamic_eplb=False, expert_map_record_path="/tmp/expert-map.json")
        ),
    )
    assert _is_dynamic_eplb_enabled(vllm_config)

    monkeypatch.setattr(
        "vllm_ascend.model_loader.rfork.rfork_loader.get_ascend_config",
        lambda: SimpleNamespace(eplb_config=SimpleNamespace(dynamic_eplb=False, expert_map_record_path=None)),
    )
    assert not _is_dynamic_eplb_enabled(vllm_config)


def test_rfork_dynamic_eplb_uses_default_loader(monkeypatch):
    import vllm.model_executor.model_loader as model_loader

    load_config = DummyLoadConfig({"model_url": "model", "model_deploy_strategy_name": "tp8"})
    loader = RForkModelLoader(load_config)
    model_config = SimpleNamespace(dtype=torch.float32, model="/models/test")
    vllm_config = _vllm_config(model_config=model_config)
    vllm_config.additional_config = {"eplb_config": {"dynamic_eplb": True}}

    def fail_if_rfork_worker_is_created(*args, **kwargs):
        raise AssertionError("RFork worker should not be initialized when dynamic EPLB is enabled.")

    expected_model = SimpleNamespace()
    captured = {}

    def fake_get_model(**kwargs):
        captured.update(kwargs)
        return expected_model

    monkeypatch.setattr(loader, "_ensure_rfork_worker", fail_if_rfork_worker_is_created)
    monkeypatch.setattr(model_loader, "get_model", fake_get_model)
    monkeypatch.setattr(
        "vllm_ascend.model_loader.rfork.rfork_loader.get_ascend_config",
        lambda: SimpleNamespace(eplb_config=SimpleNamespace(dynamic_eplb=True, expert_map_record_path=None)),
    )

    model = loader.load_model(vllm_config=vllm_config, model_config=model_config)

    assert model is expected_model
    assert captured["vllm_config"] is vllm_config
    assert captured["model_config"] is model_config
    assert captured["prefix"] == ""
    assert captured["load_config"] is not load_config
    assert captured["load_config"].load_format == "auto"
    assert captured["load_config"].model_loader_extra_config == {}


def test_rfork_native_eplb_uses_default_loader(monkeypatch):
    import vllm.model_executor.model_loader as model_loader

    load_config = DummyLoadConfig({"model_url": "model", "model_deploy_strategy_name": "tp8"})
    loader = RForkModelLoader(load_config)
    model_config = SimpleNamespace(dtype=torch.float32, model="/models/test")
    vllm_config = _vllm_config(
        model_config=model_config,
        parallel_config=_parallel_config(enable_eplb=True),
    )
    vllm_config.additional_config = None

    def fail_if_rfork_worker_is_created(*args, **kwargs):
        raise AssertionError("RFork worker should not be initialized when native EPLB is enabled.")

    expected_model = SimpleNamespace()
    captured = {}

    def fake_get_model(**kwargs):
        captured.update(kwargs)
        return expected_model

    monkeypatch.setattr(loader, "_ensure_rfork_worker", fail_if_rfork_worker_is_created)
    monkeypatch.setattr(model_loader, "get_model", fake_get_model)

    model = loader.load_model(vllm_config=vllm_config, model_config=model_config)

    assert model is expected_model
    assert captured["vllm_config"] is vllm_config
    assert captured["model_config"] is model_config
    assert captured["prefix"] == ""
    assert captured["load_config"] is not load_config
    assert captured["load_config"].load_format == "auto"
    assert captured["load_config"].model_loader_extra_config == {}


def test_rfork_worker_construction_failure_falls_back_without_starting_seed(monkeypatch):
    import vllm.model_executor.model_loader as model_loader

    load_config = DummyLoadConfig({"model_url": "model", "model_deploy_strategy_name": "strategy"})
    loader = RForkModelLoader(load_config)
    model_config = SimpleNamespace(dtype=torch.float32, model="/models/test", quantization=None)
    vllm_config = _vllm_config(model_config=model_config)
    expected_model = SimpleNamespace()
    start_calls = []

    def fail_worker(*args, **kwargs):
        raise RuntimeError("TransferEngine unavailable")

    monkeypatch.setattr(loader, "_ensure_rfork_worker", fail_worker)
    monkeypatch.setattr(model_loader, "get_model", lambda **kwargs: expected_model)
    monkeypatch.setattr(
        "vllm_ascend.model_loader.rfork.rfork_loader.get_ascend_config",
        lambda: SimpleNamespace(eplb_config=SimpleNamespace(dynamic_eplb=False, expert_map_record_path=None)),
    )

    class _UnexpectedWorker:
        def start_seed_service(self, *args, **kwargs):
            start_calls.append((args, kwargs))

    monkeypatch.setattr(
        "vllm_ascend.model_loader.rfork.rfork_loader.RForkWorker",
        lambda **kwargs: _UnexpectedWorker(),
    )

    assert loader.load_model(vllm_config=vllm_config, model_config=model_config) is expected_model
    assert start_calls == []


def test_rfork_seed_start_failure_returns_valid_model_without_disk_reload(monkeypatch):
    load_config = DummyLoadConfig({"model_url": "model", "model_deploy_strategy_name": "strategy"})
    loader = RForkModelLoader(load_config)
    model_config = SimpleNamespace(dtype=torch.float32, model="/models/test", quantization=None)
    vllm_config = _vllm_config(model_config=model_config)
    events = []

    class _Model(torch.nn.Module):
        def eval(self):
            events.append("eval")
            return super().eval()

    model = _Model()

    class _Worker:
        def is_seed_available(self):
            events.append("seed")
            return True

        def pre_transfer(self, model, processed_layout):
            events.append("pre_transfer")
            return True

        def transfer(self, model, processed_layout):
            events.append("transfer")
            return True

        def post_transfer(self):
            events.append("post_transfer")
            return True

        def start_seed_service(self, model, processed_layout):
            events.append("start_seed_service")
            return False

        def reset_transfer_state(self):
            events.append("reset_transfer_state")
            return True

    worker = _Worker()
    loader._ensure_rfork_worker = lambda vc, mc: worker
    loader._requires_processed_layout_transfer = lambda mc: False
    monkeypatch.setattr(
        "vllm_ascend.model_loader.rfork.rfork_loader.get_ascend_config",
        lambda: SimpleNamespace(eplb_config=SimpleNamespace(dynamic_eplb=False, expert_map_record_path=None)),
    )
    monkeypatch.setattr(
        "vllm_ascend.model_loader.rfork.rfork_loader.initialize_model",
        lambda **kwargs: model,
    )
    monkeypatch.setattr(
        "vllm_ascend.model_loader.rfork.rfork_loader.process_weights_after_loading",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        "vllm_ascend.model_loader.rfork.rfork_loader._rfork_skip_unquantized_moe_post_load_processing",
        lambda model: nullcontext(),
    )

    result = loader.load_model(vllm_config=vllm_config, model_config=model_config)

    assert result is model
    assert events.index("eval") < events.index("start_seed_service")
    assert events[-1] == "reset_transfer_state"


def test_rfork_cleanup_retains_memory_while_seed_server_is_alive():
    reset_calls = []
    worker = SimpleNamespace(
        stop_seed_service=lambda: False,
        post_transfer=lambda: True,
        reset_transfer_state=lambda: reset_calls.append(True) or True,
    )

    assert RForkModelLoader._cleanup_rfork_worker(worker) is False
    assert reset_calls == []


def test_rfork_fallback_clears_only_failed_model_state_before_reinit(monkeypatch):
    """Fallback re-init in the same process must first clear stale layer registries."""
    import vllm.model_executor.model_loader as model_loader
    from vllm.model_executor.layers.rotary_embedding import _ROPE_DICT

    load_config = DummyLoadConfig({"model_url": "model", "model_deploy_strategy_name": "tp8"})
    loader = RForkModelLoader(load_config)
    model_config = SimpleNamespace(dtype=torch.float32, model="/models/test", quantization="ascend")
    vllm_config = _vllm_config(model_config=model_config)

    class _FakeModule:
        pass

    stale_attention = _FakeModule()
    stale_moe = _FakeModule()
    unrelated_layer = _FakeModule()
    fallback_down_proj = _FakeModule()
    vllm_config.compilation_config = SimpleNamespace(
        static_forward_context={
            "model.layers.0.self_attn.indexer.k_cache": stale_attention,
            "unrelated.layer": unrelated_layer,
        },
        static_all_moe_layers=[
            stale_moe,
            "model.layers.0.self_attn.indexer.k_cache",
            "unrelated.layer",
        ],
    )
    _ROPE_DICT[("identity", 1.0, 32768)] = object()

    class _DiscardedModel:
        def modules(self):
            return iter([self, stale_attention, stale_moe])

    rfork_model = _DiscardedModel()
    expected_model = SimpleNamespace()
    get_model_calls = []

    def fake_get_model(**kwargs):
        get_model_calls.append(kwargs)
        assert vllm_config.compilation_config.static_forward_context == {
            "unrelated.layer": unrelated_layer,
        }
        assert vllm_config.compilation_config.static_all_moe_layers == ["unrelated.layer"]
        assert _ROPE_DICT == {}
        vllm_config.compilation_config.static_forward_context["model.layers.0.mlp.down_proj"] = fallback_down_proj
        return expected_model

    rfork_worker = SimpleNamespace(
        is_seed_available=lambda: True,
        set_excluded_weight_blocks=lambda blocks: None,
        pre_transfer=lambda model, processed_layout: True,
        transfer=lambda model, processed_layout: False,
        post_transfer=lambda: True,
        reset_transfer_state=lambda: True,
        start_seed_service=lambda model, processed_layout: None,
    )

    monkeypatch.setattr(loader, "_ensure_rfork_worker", lambda vc, mc: rfork_worker)
    monkeypatch.setattr(model_loader, "get_model", fake_get_model)
    monkeypatch.setattr(
        "vllm_ascend.model_loader.rfork.rfork_loader.get_ascend_config",
        lambda: SimpleNamespace(eplb_config=SimpleNamespace(dynamic_eplb=False, expert_map_record_path=None)),
    )
    monkeypatch.setattr(
        "vllm_ascend.model_loader.rfork.rfork_loader.initialize_model",
        lambda **kwargs: rfork_model,
    )
    monkeypatch.setattr(
        "vllm_ascend.model_loader.rfork.rfork_loader.process_weights_after_loading",
        lambda *args, **kwargs: None,
    )

    model = loader.load_model(vllm_config=vllm_config, model_config=model_config)

    assert model is expected_model
    assert len(get_model_calls) == 1
    assert vllm_config.compilation_config.static_forward_context == {
        "unrelated.layer": unrelated_layer,
        "model.layers.0.mlp.down_proj": fallback_down_proj,
    }
    assert vllm_config.compilation_config.static_all_moe_layers == ["unrelated.layer"]
    assert _ROPE_DICT == {}


def test_rfork_seed_miss_fallback_preserves_existing_process_global_state(monkeypatch):
    import vllm.model_executor.model_loader as model_loader
    from vllm.model_executor.layers.rotary_embedding import _ROPE_DICT

    load_config = DummyLoadConfig({"model_url": "model", "model_deploy_strategy_name": "tp8"})
    loader = RForkModelLoader(load_config)
    model_config = SimpleNamespace(dtype=torch.float32, model="/models/test", quantization="ascend")
    vllm_config = _vllm_config(model_config=model_config)
    existing_layer = SimpleNamespace()
    vllm_config.compilation_config = SimpleNamespace(
        static_forward_context={"existing.layer": existing_layer},
        static_all_moe_layers=["existing.layer"],
    )
    rope_key = ("identity", 1.0, 32768)
    rope_value = object()
    _ROPE_DICT[rope_key] = rope_value

    expected_model = SimpleNamespace()

    def fake_get_model(**kwargs):
        assert vllm_config.compilation_config.static_forward_context == {
            "existing.layer": existing_layer,
        }
        assert vllm_config.compilation_config.static_all_moe_layers == ["existing.layer"]
        assert _ROPE_DICT[rope_key] is rope_value
        return expected_model

    rfork_worker = SimpleNamespace(
        is_seed_available=lambda: False,
        set_excluded_weight_blocks=lambda blocks: None,
        post_transfer=lambda: True,
        reset_transfer_state=lambda: True,
        start_seed_service=lambda model, processed_layout: None,
    )

    monkeypatch.setattr(loader, "_ensure_rfork_worker", lambda vc, mc: rfork_worker)
    monkeypatch.setattr(model_loader, "get_model", fake_get_model)
    monkeypatch.setattr(
        "vllm_ascend.model_loader.rfork.rfork_loader.get_ascend_config",
        lambda: SimpleNamespace(eplb_config=SimpleNamespace(dynamic_eplb=False, expert_map_record_path=None)),
    )
    monkeypatch.setattr(
        "vllm_ascend.model_loader.rfork.rfork_loader.initialize_model",
        lambda **kwargs: pytest.fail("seed-miss fallback must not initialize an RFork model"),
    )

    model = loader.load_model(vllm_config=vllm_config, model_config=model_config)

    assert model is expected_model
    assert vllm_config.compilation_config.static_forward_context == {
        "existing.layer": existing_layer,
    }
    assert vllm_config.compilation_config.static_all_moe_layers == ["existing.layer"]
    assert _ROPE_DICT[rope_key] is rope_value


def test_reset_process_global_model_state_is_safe_when_attrs_missing():
    vllm_config = SimpleNamespace(compilation_config=SimpleNamespace())
    _reset_process_global_model_state(vllm_config)


def test_rfork_pre_transfer_weight_processing_unwraps_and_restores_quant_methods(monkeypatch):
    import vllm_ascend.ops.fused_moe.fused_moe as fused_moe_module

    class _FakeAscendMoERunner:
        def __init__(self, quant_method):
            self._quant_method = quant_method

    calls = []

    def original_process_weights(*args, **kwargs):
        calls.append("original")

    @wraps(original_process_weights)
    def wrapped_process_weights(*args, **kwargs):
        calls.append("wrapped")
        original_process_weights(*args, **kwargs)

    quant_method = SimpleNamespace(process_weights_after_loading=wrapped_process_weights)
    fused_moe_layer = _FakeAscendMoERunner(quant_method)
    other_layer = SimpleNamespace()

    class _FakeModule:
        def modules(self):
            return iter([self, fused_moe_layer, other_layer])

    fake_module = _FakeModule()
    monkeypatch.setattr(fused_moe_module, "AscendMoERunner", _FakeAscendMoERunner)

    with _rfork_pre_transfer_weight_processing(fake_module):
        assert quant_method.process_weights_after_loading is original_process_weights
        quant_method.process_weights_after_loading()
    assert quant_method.process_weights_after_loading is wrapped_process_weights
    assert calls == ["original"]

    # Restoration must happen even when the wrapped block raises.
    with pytest.raises(RuntimeError, match="boom"), _rfork_pre_transfer_weight_processing(fake_module):
        assert quant_method.process_weights_after_loading is original_process_weights
        raise RuntimeError("boom")
    assert quant_method.process_weights_after_loading is wrapped_process_weights


def test_rfork_skips_only_unquantized_moe_post_load_processing(monkeypatch):
    import vllm_ascend.ops.fused_moe.fused_moe as fused_moe_module
    import vllm_ascend.ops.fused_moe.routed_experts as routed_experts_module

    class _FakeAscendUnquantizedFusedMoEMethod:
        def __init__(self, process_weights_after_loading):
            self.process_weights_after_loading = process_weights_after_loading

    class _FakeAscendMoERunner:
        def __init__(self, quant_method):
            self._quant_method = quant_method

    calls = []

    def unquantized_process(*args, **kwargs):
        calls.append("unquantized")

    def quantized_process(*args, **kwargs):
        calls.append("quantized")

    unquantized_method = _FakeAscendUnquantizedFusedMoEMethod(unquantized_process)
    quantized_method = SimpleNamespace(process_weights_after_loading=quantized_process)
    unquantized_layer = _FakeAscendMoERunner(unquantized_method)
    quantized_layer = _FakeAscendMoERunner(quantized_method)
    duplicate_unquantized_layer = _FakeAscendMoERunner(unquantized_method)

    class _FakeModule:
        def modules(self):
            return iter(
                [
                    self,
                    unquantized_layer,
                    quantized_layer,
                    duplicate_unquantized_layer,
                ]
            )

    monkeypatch.setattr(
        routed_experts_module,
        "AscendUnquantizedFusedMoEMethod",
        _FakeAscendUnquantizedFusedMoEMethod,
    )
    monkeypatch.setattr(fused_moe_module, "AscendMoERunner", _FakeAscendMoERunner)

    with _rfork_skip_unquantized_moe_post_load_processing(_FakeModule()):
        assert unquantized_method.process_weights_after_loading() is None
        quantized_method.process_weights_after_loading()
        assert calls == ["quantized"]
    assert unquantized_method.process_weights_after_loading is unquantized_process
    assert quantized_method.process_weights_after_loading is quantized_process

    with pytest.raises(RuntimeError, match="boom"), _rfork_skip_unquantized_moe_post_load_processing(_FakeModule()):
        raise RuntimeError("boom")
    assert unquantized_method.process_weights_after_loading is unquantized_process
