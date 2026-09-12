# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

import ast
import ctypes
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from .test_late_lease_isolated import loader_runtime as loader_runtime


def seed_info(runtime, tensors, with_shapes):
    weights = {
        name: (tensor.data_ptr(), tensor.numel(), tensor.element_size(), tuple(tensor.shape), str(tensor.dtype))
        for name, tensor in tensors
    }
    shapes = {name: tuple(tensor.shape) for name, tensor in tensors} if with_shapes else None
    return runtime.SeedTransferInfo("seed", weights, shapes)


def backend_for(runtime, tensors):
    backend = runtime.RForkTransferBackend()
    backend._registered_transferable_tensors = tensors
    reads = []

    def read(session_id, destinations, sources, lengths):
        for destination, source, length in zip(destinations, sources, lengths, strict=True):
            reads.append((source, length))
            ctypes.memmove(destination, source, length)
        return SimpleNamespace(is_error=lambda: False)

    backend.transfer_engine = SimpleNamespace(batch_transfer_sync_read=read)
    return backend, reads


@pytest.mark.parametrize("invalid", [None, "count", "bytes", "missing", "dtype", "shape", "numel", "element_size"])
def test_checkpoint_superset_validates_legacy_optional_manifest(tensor_runtime, monkeypatch, invalid):
    r = tensor_runtime
    monkeypatch.setattr(r.tensor_layout, "is_tensor_on_transfer_device", lambda tensor: True)
    model = torch.nn.Module()
    model.weight = torch.nn.Parameter(torch.zeros(4))
    source = torch.arange(4, dtype=torch.float32)
    derived = source.square()
    tensors = [("weight", source), ("derived", derived)]
    info = seed_info(r, tensors, True)
    metadata = {
        "tensor_count": 2,
        "total_bytes": 32,
        "weights": {name: {"numel": 4, "element_size": 4, "shape": [4], "dtype": "float32"} for name, _ in tensors},
    }
    if invalid == "count":
        metadata["tensor_count"] = 1
    elif invalid == "bytes":
        metadata["total_bytes"] = 16
    elif invalid == "missing":
        del metadata["weights"]["derived"]
    elif invalid is not None:
        metadata["weights"]["derived"][invalid] = {"dtype": "int32", "shape": [2, 2], "numel": 2, "element_size": 2}[
            invalid
        ]
    backend, reads = backend_for(r, [("weight", model.weight)])
    assert backend.read_weights_from_seed(model, info, False, metadata) is (invalid is None)
    assert reads == ([(source.data_ptr(), 16)] if invalid is None else [])


@pytest.mark.parametrize("with_shapes", [False, True])
@pytest.mark.parametrize("processed_layout", [False, True])
def test_only_checkpoint_transfer_accepts_seed_only_tensors(tensor_runtime, monkeypatch, with_shapes, processed_layout):
    r = tensor_runtime
    monkeypatch.setattr(r.tensor_layout, "is_tensor_on_transfer_device", lambda tensor: True)
    model = torch.nn.Module()
    model.weight = torch.nn.Parameter(torch.zeros(4))
    source = torch.arange(4, dtype=torch.float32)
    derived = source.square()
    info = seed_info(r, [("weight", source), ("impl.derived", derived)], with_shapes)
    backend, reads = backend_for(r, [("weight", model.weight)])
    assert backend.read_weights_from_seed(model, info, processed_layout) is (not processed_layout)
    if processed_layout:
        assert reads == []
        assert torch.equal(model.weight, torch.zeros(4))
    else:
        assert reads == [(source.data_ptr(), source.numel() * source.element_size())]
        torch.testing.assert_close(model.weight, source)


@pytest.mark.parametrize("invalid", ["missing_weight", "dtype", "size", "extra_shape", "extra_metadata", "shape_names"])
def test_checkpoint_superset_keeps_validation_before_native_reads(tensor_runtime, monkeypatch, invalid):
    r = tensor_runtime
    monkeypatch.setattr(r.tensor_layout, "is_tensor_on_transfer_device", lambda tensor: True)
    model = torch.nn.Module()
    model.weight = torch.nn.Parameter(torch.zeros(4))
    source = torch.arange(4, dtype=torch.float32)
    derived = source.square()
    info = seed_info(r, [("weight", source), ("impl.derived", derived)], True)
    # Keep a real owner for every address in successful reads. All these cases
    # must fail validation, without invoking the native read at all.
    if invalid == "missing_weight":
        del info.weights["weight"]
        del info.shapes["weight"]
    elif invalid == "dtype":
        info.weights["weight"] = (*info.weights["weight"][:4], "int32")
    elif invalid == "size":
        info.weights["weight"] = (source.data_ptr(), 4, 2, (4,), "float32")
    elif invalid == "extra_shape":
        info.shapes["impl.derived"] = (2, 2)
    elif invalid == "extra_metadata":
        info.weights["impl.derived"] = (-1, 4, 4, (4,), "float32")
    else:
        del info.shapes["impl.derived"]
    backend, reads = backend_for(r, [("weight", model.weight)])
    assert not backend.read_weights_from_seed(model, info, False)
    assert reads == []
    torch.testing.assert_close(model.weight, torch.zeros(4))


def mla_post_load():
    # Execute the real MLA hook while replacing only external NPU operators.
    path = Path(__file__).resolve().parents[4] / "vllm_ascend/attention/mla_v1.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    hook = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "process_weights_after_loading"
    )
    linear_method = type("UnquantizedLinearMethod", (), {})
    namespace = {
        "torch": torch,
        "torch_npu": SimpleNamespace(npu_format_cast=lambda tensor, fmt: tensor),
        "ACL_FORMAT_FRACTAL_ND": 2,
        "UnquantizedLinearMethod": linear_method,
        "maybe_trans_nz": lambda tensor: tensor,
    }
    exec(compile(ast.Module(body=[hook], type_ignores=[]), str(path), "exec"), namespace)
    return namespace["process_weights_after_loading"], linear_method


def make_mla_model(linear_method, fill):
    model = torch.nn.Module()
    model.kv_b_proj = torch.nn.Module()
    model.kv_b_proj.weight = torch.nn.Parameter(torch.arange(8, dtype=torch.float32).reshape(4, 2) * fill)
    model.kv_b_proj.quant_method = linear_method()
    model.impl = SimpleNamespace(
        kv_b_proj=model.kv_b_proj,
        kv_lora_rank=2,
        num_heads=1,
        qk_nope_head_dim=2,
        v_head_dim=2,
        enable_mlapo=False,
        fa_quant_layer=False,
    )
    return model


def test_real_loader_rebuilds_mla_derived_tensors_after_checkpoint_transfer(request, monkeypatch):
    loader = request.getfixturevalue("loader_runtime")
    r = request.getfixturevalue("tensor_runtime")
    monkeypatch.setattr(r.tensor_layout, "is_tensor_on_transfer_device", lambda tensor: True)
    hook, linear_method = mla_post_load()
    source = make_mla_model(linear_method, 1)
    destination = make_mla_model(linear_method, 0)
    hook(source.impl, torch.float32)
    info = seed_info(r, r.tensor_layout.collect_checkpoint_layout_tensors(source), True)
    backend, reads = backend_for(r, r.tensor_layout.collect_checkpoint_layout_tensors(destination))
    monkeypatch.setattr(loader.module, "initialize_model", lambda **kwargs: destination)
    monkeypatch.setattr(
        loader.module, "process_weights_after_loading", lambda model, *args: hook(model.impl, torch.float32)
    )
    loader.session.transfer_from_seed.side_effect = (
        lambda model, processed, exclude_blocks=None: backend.read_weights_from_seed(model, info, processed)
    )
    assert loader.loader.load_model(loader.vc, loader.config) is destination
    assert "fallback" not in loader.events
    loader.session.prepare_for_fallback.assert_not_called()
    assert reads == [(source.kv_b_proj.weight.data_ptr(), 8 * 4)]
    torch.testing.assert_close(destination.kv_b_proj.weight, source.kv_b_proj.weight)
    torch.testing.assert_close(destination.impl.W_UV, source.impl.W_UV)
    torch.testing.assert_close(destination.impl.W_UK_T, source.impl.W_UK_T)
    # v0.26.0rc1 builds W_UV/W_UK_T; the later MLA hook also adds mlapo_W_UK_T.
    assert not hasattr(source.impl, "mlapo_W_UK_T")
    assert not hasattr(destination.impl, "mlapo_W_UK_T")
