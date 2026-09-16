# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

import logging
import re
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


def test_layout_summary_is_one_bounded_info_record_with_fixed_digests(tensor_runtime, caplog):
    tensors = [(f"weight_{index}", torch.arange(4, dtype=torch.float32)) for index in range(6)]
    formats = {name: 29 for name, _ in tensors}

    with caplog.at_level(logging.INFO, logger=tensor_runtime.tensor_layout.logger.name):
        tensor_runtime.tensor_layout.log_tensor_layout_summary(
            tensors,
            stage="receiver_before_read",
            session_id="receiver-session",
            peer_session_id="seed-session",
            processed_layout=True,
            known_formats=formats,
        )

    records = [record.getMessage() for record in caplog.records if "RFork tensor layout summary" in record.getMessage()]
    assert len(records) == 1
    message = records[0]
    assert "tensors=6" in message
    assert "session=receiver-session peer_session=seed-session" in message
    assert len(re.findall(r"(?:semantic|physical)_digest=[0-9a-f]{64}", message)) == 2
    assert "weight_0" in message and "weight_2" in message
    assert "weight_3" not in message and "weight_5" not in message


def test_layout_summary_includes_npu_format_and_physical_size(tensor_runtime, monkeypatch, caplog):
    class _NPUTensorProxy:
        device = SimpleNamespace(type="npu")

        def __init__(self, tensor):
            self._tensor = tensor

        def __getattr__(self, name):
            return getattr(self._tensor, name)

    monkeypatch.setitem(
        sys.modules,
        "torch_npu",
        SimpleNamespace(
            get_npu_format=lambda tensor: 29,
            get_storage_size=lambda tensor: tensor.numel() + 8,
        ),
    )
    tensor = _NPUTensorProxy(torch.arange(4, dtype=torch.float32))

    with caplog.at_level(logging.INFO, logger=tensor_runtime.tensor_layout.logger.name):
        tensor_runtime.tensor_layout.log_tensor_layout_summary(
            [("weight", tensor)],
            stage="registered",
            session_id="seed-session",
            processed_layout=True,
        )

    message = next(
        record.getMessage() for record in caplog.records if "RFork tensor layout summary" in record.getMessage()
    )
    assert "physical_nonlogical_tensors=1" in message
    assert "formats={'29': 1}" in message
    assert "'npu_format': 29" in message
    assert "'npu_storage_numel': 12" in message


def test_post_load_layout_summary_is_observational(tensor_runtime, monkeypatch, caplog):
    tensor_layout = tensor_runtime.tensor_layout
    monkeypatch.setattr(tensor_layout, "is_tensor_on_transfer_device", lambda tensor: True)
    model = torch.nn.Module()
    model.weight = torch.nn.Parameter(torch.arange(4, dtype=torch.float32))
    original = model.weight.detach().clone()
    backend = tensor_runtime.RForkTransferBackend()
    backend.transfer_session_id = "receiver-session"

    with caplog.at_level(logging.INFO, logger=tensor_layout.logger.name):
        backend.log_model_layout_summary(
            model,
            False,
            stage="receiver_after_post_load",
            peer_session_id="seed-session",
        )

    message = next(record.getMessage() for record in caplog.records if "RFork tensor layout summary" in record.message)
    assert "stage=receiver_after_post_load" in message
    assert "session=receiver-session peer_session=seed-session" in message
    assert "tensors=1" in message
    torch.testing.assert_close(model.weight, original)


def test_post_load_layout_diagnostic_failure_does_not_escape(tensor_runtime, monkeypatch, caplog):
    transfer_backend = tensor_runtime.transfer_backend
    monkeypatch.setattr(
        transfer_backend,
        "collect_transferable_tensors",
        Mock(side_effect=RuntimeError("inspection failed")),
    )
    backend = tensor_runtime.RForkTransferBackend()

    with caplog.at_level(logging.INFO, logger=transfer_backend.logger.name):
        backend.log_model_layout_summary(object(), False, stage="receiver_after_post_load")

    assert "unavailable=RuntimeError:inspection failed" in caplog.text
