# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from .rfork_test_support import _stub
from .test_late_lease_isolated import loader_runtime as loader_runtime


@pytest.fixture
def flatquant_runtime(request, monkeypatch):
    """Load the real schemes and adapter; replace only external runtime dependencies."""
    r = request.getfixturevalue("loader_runtime")

    def stub(name, **attrs):
        return _stub(monkeypatch, name, **attrs)

    stub("torch_npu")
    monkeypatch.setattr(sys.modules["vllm.config"], "get_current_vllm_config", Mock(), raising=False)
    monkeypatch.setattr(
        sys.modules["vllm.distributed"], "get_tensor_model_parallel_world_size", lambda: 1, raising=False
    )
    stub("vllm.model_executor.layers.linear", LinearMethodBase=object, RowParallelLinear=type("Row", (), {}))
    stub("vllm.model_executor.layers.fused_moe", FusedMoEMethodBase=object, FusedMoeWeightScaleSupported=Mock())
    stub("vllm.model_executor.layers.fused_moe.config", FusedMoEConfig=object)
    stub("vllm.model_executor.layers.quantization.kv_cache", BaseKVCacheMethod=object)
    stub("vllm.model_executor.parameter", BlockQuantScaleParameter=object, PerTensorScaleParameter=object)
    stub("vllm.model_executor.utils", set_weight_attrs=Mock())
    stub("vllm_ascend.distributed.parallel_state", get_mlp_tp_group=Mock(), get_otp_group=Mock())
    stub("vllm_ascend.utils", mlp_tp_enable=lambda: False, oproj_tp_enable=lambda: False)
    stub("vllm_ascend.quantization", __path__=[])
    stub(
        "vllm_ascend.quantization.methods",
        __path__=[],
        AscendAttentionScheme=object,
        AscendLinearScheme=object,
        AscendMoEScheme=object,
        is_mx_quant_type=lambda scheme: False,
    )
    stub("vllm_ascend.quantization.methods.w4a4", __path__=[])
    stub("vllm_ascend.quantization.methods.base", AscendLinearScheme=object)
    stub("vllm_ascend.quantization.methods.registry", register_scheme=lambda *args: lambda cls: cls)

    def load(name):
        path = Path(__file__).resolve().parents[4].joinpath(*name.split(".")).with_suffix(".py")
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, name, module)
        spec.loader.exec_module(module)
        return module

    flat = load("vllm_ascend.quantization.methods.w4a4.w4a4_flatquant")
    mxfp = load("vllm_ascend.quantization.methods.w4a4.w4a4_mxfp4_flatquant")
    adapter = load("vllm_ascend.quantization.method_adapters")
    # Packing is a native NPU operation; a clone lets the real hook retain its
    # destructive parameter replacement behavior on CPU.
    monkeypatch.setattr(flat, "pack_int4_weights", lambda weight: weight.clone())
    return SimpleNamespace(loader=r, flat=flat, mxfp=mxfp, adapter=adapter)


@pytest.mark.parametrize("mxfp", [False, True])
def test_flatquant_transfer_refreshes_inference_scalar_without_reprocessing(flatquant_runtime, monkeypatch, mxfp):
    f = flatquant_runtime
    r = f.loader
    cls = f.mxfp.AscendW4A4MXFP4FlatQuantDynamicLinearMethod if mxfp else f.flat.AscendW4A4FlatQuantDynamicLinearMethod
    scheme = cls.__new__(cls)
    scheme.group_size = 32
    layer = torch.nn.Module()
    for name, tensor in {
        "weight": torch.ones(2, 4),
        "weight_scale": torch.ones(2, 2),
        "weight_offset": torch.zeros(2, 1),
        "left_trans": torch.arange(4.0).view(2, 2),
        "right_trans": torch.eye(2),
        "clip_ratio": torch.tensor([0.0]),
    }.items():
        layer.register_parameter(name, torch.nn.Parameter(tensor, requires_grad=False))
    layer.quant_method = f.adapter.AscendLinearMethod(scheme)
    r.model.add_module("linear", layer)
    r.config.quantization = "modelslim"
    post_load = Mock(side_effect=lambda *args: layer.quant_method.process_weights_after_loading(layer))
    monkeypatch.setattr(r.module, "process_weights_after_loading", post_load)
    prepared = {}

    def transfer(*args):
        assert layer.aclnn_clip_ratio == 0.0
        prepared.update(
            {name: (parameter.data_ptr(), parameter.clone()) for name, parameter in layer.named_parameters()}
        )
        with torch.no_grad():
            layer.clip_ratio.fill_(0.5)
        return True

    r.session.transfer_from_seed.side_effect = transfer

    def publish(*args):
        assert layer.aclnn_clip_ratio == 0.5
        return True

    r.session.start_seed_service.side_effect = publish
    assert r.loader.load_model(r.vc, r.config) is r.model
    assert layer.aclnn_clip_ratio == layer.clip_ratio.item() == 0.5
    post_load.assert_called_once()
    for name, parameter in layer.named_parameters():
        assert parameter.data_ptr() == prepared[name][0]
        if name != "clip_ratio":
            torch.testing.assert_close(parameter, prepared[name][1])

    # Verify the value passed by the real inference method, not only the cache.
    quant = Mock(return_value=(torch.ones(1, 2, 2), torch.ones(1)))
    native = sys.modules["torch_npu"]
    monkeypatch.setattr(native, "npu_kronecker_quant", quant, raising=False)
    monkeypatch.setattr(native, "npu_quant_matmul", Mock(return_value=torch.ones(1, 2)), raising=False)
    monkeypatch.setattr(native, "float4_e2m1fn_x2", 0, raising=False)
    monkeypatch.setattr(native, "float8_e8m0fnu", 0, raising=False)
    if mxfp:
        scheme.apply(layer, torch.ones(1, 4))
        assert quant.call_args.args[3] == 0.5
    else:
        # A 4x4 transform supplies the int4 packing width required by apply().
        layer.left_trans = torch.nn.Parameter(torch.eye(4))
        layer.right_trans = torch.nn.Parameter(torch.eye(4))
        quant.return_value = (torch.ones(1, 2, dtype=torch.int32), torch.ones(1))
        scheme.apply(layer, torch.ones(1, 16))
        assert quant.call_args.kwargs["clip_ratio"] == 0.5


def test_non_flatquant_layer_is_unchanged(flatquant_runtime):
    f = flatquant_runtime
    scheme = SimpleNamespace(process_weights_after_loading=Mock())
    layer = torch.nn.Module()
    layer.quant_method = f.adapter.AscendLinearMethod(scheme)
    layer.clip_ratio = torch.nn.Parameter(torch.tensor([0.5]))
    f.loader.module._refresh_rfork_flatquant_state(layer)
    assert not hasattr(layer, "aclnn_clip_ratio")
    scheme.process_weights_after_loading.assert_not_called()


@pytest.mark.parametrize("clip_ratio", [None, torch.empty(0), torch.ones(2)])
def test_refresh_failure_uses_loader_cleanup_and_fallback(request, clip_ratio):
    r = request.getfixturevalue("loader_runtime")
    r.config.quantization = "modelslim"
    r.model.aclnn_clip_ratio = 0.0
    r.model.clip_ratio = clip_ratio
    assert r.loader.load_model(r.vc, r.config) is r.fallback
    assert r.model.aclnn_clip_ratio == 0.0
    assert r.events.index("transfer") < r.events.index("cleanup") < r.events.index("fallback")


def test_checkpoint_layout_does_not_refresh_twice(request):
    r = request.getfixturevalue("loader_runtime")
    # The normal post-load path owns runtime initialization in this mode.
    # A sentinel cache must not trigger the processed-layout refresh path.
    r.model.aclnn_clip_ratio = "sentinel"
    assert r.loader.load_model(r.vc, r.config) is r.model
    assert r.model.aclnn_clip_ratio == "sentinel"
    assert r.events.count("layout") == 1
