# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

import sys
from types import SimpleNamespace
from unittest.mock import Mock

import torch


def test_read_rejects_missing_registration_cache_without_rescanning_or_reading(tensor_runtime, monkeypatch):
    transfer_backend = tensor_runtime.transfer_backend
    backend = tensor_runtime.RForkTransferBackend()
    read = Mock()
    rescan = Mock(side_effect=AssertionError("destination tensors were not registered"))
    backend.transfer_engine = SimpleNamespace(batch_transfer_sync_read=read)
    monkeypatch.setattr(transfer_backend, "collect_transferable_tensors", rescan)
    seed_info = tensor_runtime.SeedTransferInfo("seed-session", {"weight": (1234, 4, 4, [4], "float32")})

    assert not backend.read_weights_from_seed(object(), seed_info, True)
    rescan.assert_not_called()
    read.assert_not_called()


def test_registered_seed_shared_name_still_skips_local_weight(tensor_runtime, monkeypatch):
    transfer_backend = tensor_runtime.transfer_backend
    manifest = sys.modules["vllm_ascend.model_loader.rfork.manifest"]
    monkeypatch.setattr(manifest, "read_npu_format", lambda tensor: 0)
    monkeypatch.setattr(transfer_backend, "read_npu_format", lambda tensor: 0)
    shared_weight = torch.arange(10, dtype=torch.float32)
    own_weight = torch.arange(12, dtype=torch.float32)
    backend = tensor_runtime.RForkTransferBackend()
    read = Mock(return_value=SimpleNamespace(is_error=lambda: False))
    backend.transfer_engine = SimpleNamespace(batch_transfer_sync_read=read)
    backend._registered_transferable_tensors = [
        ("model.embed_tokens.weight", shared_weight),
        ("layers.0.fc.weight", own_weight),
    ]
    seed_info = tensor_runtime.SeedTransferInfo(
        "seed-session",
        {"layers.0.fc.weight": (1234, own_weight.numel(), own_weight.element_size(), [12], "float32")},
        shared_names=("model.embed_tokens.weight",),
        formats={"layers.0.fc.weight": 0},
    )

    assert backend.read_weights_from_seed(object(), seed_info, True)
    assert backend.seed_shared_names == ["model.embed_tokens.weight"]
    read.assert_called_once()
