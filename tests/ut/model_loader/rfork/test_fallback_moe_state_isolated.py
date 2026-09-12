# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

import importlib.util
import sys
import weakref
from pathlib import Path
from types import ModuleType
from unittest.mock import Mock

import pytest
import torch

from .test_late_lease_isolated import loader_runtime as loader_runtime


@pytest.fixture
def moe_runtime(request, monkeypatch):
    """Use the real Ascend registry with a minimal layer constructor on CPU."""
    r = request.getfixturevalue("loader_runtime")
    root = Path(__file__).resolve().parents[4]
    for name in ("vllm_ascend.quantization.quant_type", "vllm_ascend.eplb.adaptor.vllm_adaptor"):
        spec = importlib.util.spec_from_file_location(name, root.joinpath(*name.split(".")).with_suffix(".py"))
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, name, module)
        spec.loader.exec_module(module)
    adaptor_module = sys.modules["vllm_ascend.eplb.adaptor.vllm_adaptor"]
    adaptor = adaptor_module.VllmEplbAdaptor

    class RoutedExperts(torch.nn.Module):
        moe_counter = -1

        def __init__(self):
            super().__init__()
            RoutedExperts.moe_counter += 1
            self.weight = torch.nn.Parameter(torch.ones(4))
            adaptor.register_layer(self)

    routed_module = ModuleType("vllm_ascend.ops.fused_moe.routed_experts")
    routed_module.AscendRoutedExperts = RoutedExperts
    monkeypatch.setitem(sys.modules, routed_module.__name__, routed_module)
    r.adaptor = adaptor
    r.adaptor_module = adaptor_module
    r.routed_module = routed_module
    r.layer_cls = RoutedExperts
    return r


@pytest.mark.parametrize("runtime_loaded", [False, True])
@pytest.mark.parametrize("failure", ["seed_miss", "initialize", "layout", "transfer"])
def test_moe_layers_and_weights_released_before_fallback(moe_runtime, monkeypatch, runtime_loaded, failure, caplog):
    r = moe_runtime
    existing = r.layer_cls() if runtime_loaded else None
    baseline_counter = r.layer_cls.moe_counter
    baseline_layers = list(r.adaptor._registered_moe_layers)
    registry = r.adaptor._registered_moe_layers
    references = []
    if not runtime_loaded:
        monkeypatch.delitem(sys.modules, r.adaptor_module.__name__)
        monkeypatch.delitem(sys.modules, r.routed_module.__name__)

    def initialize(**kwargs):
        monkeypatch.setitem(sys.modules, r.adaptor_module.__name__, r.adaptor_module)
        monkeypatch.setitem(sys.modules, r.routed_module.__name__, r.routed_module)
        model = torch.nn.Module()
        layer = r.layer_cls()
        model.add_module("experts", layer)
        if existing is not None:
            model.add_module("shared_target", existing)
        # Exercise partial construction and cyclic references as well as the
        # registry's strong references. No mock call history owns these tensors.
        layer.cycle = [layer]
        references.extend([weakref.ref(model), weakref.ref(layer), weakref.ref(layer.weight)])
        if failure == "initialize":
            raise RuntimeError("partial MoE initialization failed")
        return model

    def process(model, *args):
        if failure == "layout":
            raise RuntimeError("MoE layout failed")

    def transfer(model, *args):
        if failure == "transfer":
            raise RuntimeError("MoE transfer failed")
        return True

    def fallback(**kwargs):
        assert references and all(reference() is None for reference in references)
        assert r.adaptor._registered_moe_layers is registry
        assert registry == baseline_layers
        assert r.layer_cls.moe_counter == baseline_counter
        return r.fallback

    monkeypatch.setattr(r.module, "initialize_model", initialize)
    monkeypatch.setattr(r.module, "process_weights_after_loading", process)
    monkeypatch.setattr(r.loader, "_requires_processed_layout_transfer", lambda config: True)
    r.session.register_destination = lambda *args: True
    r.session.acquire_seed = lambda: failure != "seed_miss"
    r.session.transfer_from_seed = transfer
    r.session.start_seed_service = lambda *args: True
    monkeypatch.setattr(sys.modules["vllm.model_executor.model_loader"], "get_model", fallback)
    assert r.loader.load_model(r.vc, r.config) is r.fallback
    if failure != "seed_miss":
        assert "RFork transfer failed" in caplog.text


def test_moe_rollback_restores_replaced_registry_in_place(moe_runtime):
    r = moe_runtime
    target = r.layer_cls()
    registry = r.adaptor._registered_moe_layers
    snapshot = r.module._snapshot_process_global_model_state(r.vc)
    discarded = r.layer_cls()
    r.adaptor._registered_moe_layers = [discarded]
    r.module._reset_process_global_model_state(r.vc, snapshot=snapshot)
    assert r.adaptor._registered_moe_layers is registry
    assert registry == [target]
    assert r.layer_cls.moe_counter == 0


def test_successful_load_keeps_new_moe_registrations(moe_runtime, monkeypatch):
    r = moe_runtime

    def initialize(**kwargs):
        model = r.layer_cls()
        return model

    monkeypatch.setattr(r.module, "initialize_model", initialize)
    model = r.loader.load_model(r.vc, r.config)
    assert r.adaptor._registered_moe_layers == [model]
    assert r.layer_cls.moe_counter == 0
    r.session.prepare_for_fallback.assert_not_called()


@pytest.mark.parametrize("draft", [False, True])
def test_static_expert_map_bypasses_all_rfork_resources(request, monkeypatch, draft):
    r = request.getfixturevalue("loader_runtime")
    r.config.runner_type = "draft" if draft else "generate"
    ascend_config = r.module.get_ascend_config()
    ascend_config.eplb_config.expert_map_path = "placement.json"
    monkeypatch.setattr(r.module, "get_ascend_config", lambda: ascend_config)
    initialize = Mock()
    ensure_session = Mock()
    monkeypatch.setattr(r.module, "initialize_model", initialize)
    monkeypatch.setattr(r.loader, "_ensure_rfork_session", ensure_session)
    fallback = Mock(return_value=r.fallback)
    monkeypatch.setattr(sys.modules["vllm.model_executor.model_loader"], "get_model", fallback)
    assert r.loader.load_model(r.vc, r.config, prefix="draft_prefix") is r.fallback
    initialize.assert_not_called()
    ensure_session.assert_not_called()
    assert r.session.mock_calls == []
    fallback.assert_called_once()
    kwargs = fallback.call_args.kwargs
    assert kwargs["prefix"] == "draft_prefix"
    assert kwargs["model_config"] is r.config
    assert kwargs["vllm_config"] is r.vc
    assert kwargs["load_config"].load_format == "auto"
    assert kwargs["load_config"].model_loader_extra_config == {}
    assert r.loader.load_config.load_format == "rfork"
