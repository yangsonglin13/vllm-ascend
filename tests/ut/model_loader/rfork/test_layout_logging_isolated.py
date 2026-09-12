# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

import ast
import logging
import sys
from types import SimpleNamespace

import pytest
import torch

from tests.ut.model_loader.rfork.rfork_test_support import _stub


class _NPUTensorProxy:
    """Expose CPU tensor metadata while presenting an NPU device."""

    def __init__(self, tensor: torch.Tensor):
        self._tensor = tensor
        self.device = SimpleNamespace(type="npu")

    def __getattr__(self, name):
        return getattr(self._tensor, name)


def _metadata(record):
    message = record.getMessage()
    metadata_text = message.split(" metadata=", 1)[1].split(" errors=", 1)[0]
    return ast.literal_eval(metadata_text)


def _layout_record(caplog):
    records = [record for record in caplog.records if record.name == "rfork-tensor-safety-test"]
    assert len(records) == 1
    assert records[0].levelno == logging.INFO
    return records[0]


def test_cpu_offset_and_transpose_log_shape_stride_and_storage_sizes(tensor_runtime, caplog):
    base = torch.arange(24, dtype=torch.float32).reshape(4, 6)
    tensor = base[1:, 1:5].t()

    with caplog.at_level(logging.INFO, logger="rfork-tensor-safety-test"):
        tensor_runtime.tensor_layout.log_tensor_layout(
            "offset_transpose",
            tensor,
            stage="registration",
            session_id="receiver-session",
            peer_session_id="seed-session",
            processed_layout=False,
        )

    record = _layout_record(caplog)
    metadata = _metadata(record)
    assert metadata["shape"] == tuple(tensor.shape)
    assert metadata["stride"] == tuple(tensor.stride())
    assert metadata["logical_bytes"] == tensor.numel() * tensor.element_size()
    assert metadata["storage_bytes"] == base.untyped_storage().nbytes()
    assert metadata["storage_bytes"] > metadata["logical_bytes"]
    assert "stage=registration" in record.getMessage()
    assert "session=receiver-session" in record.getMessage()
    assert "peer_session=seed-session" in record.getMessage()


def test_npu_descriptor_size_is_logged_without_dtype_based_physical_size_inference(tensor_runtime, monkeypatch, caplog):
    tensor = _NPUTensorProxy(torch.arange(6, dtype=torch.float32))
    _stub(
        monkeypatch,
        "torch_npu",
        get_npu_format=lambda value: 29,
        get_storage_size=lambda value: 64,
    )

    with caplog.at_level(logging.INFO, logger="rfork-tensor-safety-test"):
        tensor_runtime.tensor_layout.log_tensor_layout(
            "packed_weight",
            tensor,
            stage="receiver_before_read",
            session_id="receiver-session",
            peer_session_id="seed-session",
            processed_layout=True,
        )

    metadata = _metadata(_layout_record(caplog))
    assert metadata["npu_format"] == 29
    assert metadata["npu_storage_numel"] == 64
    assert metadata["npu_storage_numel"] != tensor.numel() * tensor.element_size()


@pytest.mark.parametrize("module", [None, SimpleNamespace()])
def test_missing_npu_api_logs_unavailable_without_failing(tensor_runtime, monkeypatch, caplog, module):
    tensor = _NPUTensorProxy(torch.arange(6, dtype=torch.float32))
    monkeypatch.setitem(sys.modules, "torch_npu", module)

    with caplog.at_level(logging.INFO, logger="rfork-tensor-safety-test"):
        tensor_runtime.tensor_layout.log_tensor_layout(
            "missing_api",
            tensor,
            stage="registration",
            session_id="session",
            processed_layout=False,
        )

    metadata = _metadata(_layout_record(caplog))
    assert metadata["npu_format"] == "unavailable"
    assert metadata["npu_storage_numel"] == "unavailable"


def test_npu_api_errors_log_unavailable_without_failing(tensor_runtime, monkeypatch, caplog):
    def fail(_tensor):
        raise RuntimeError("descriptor unavailable")

    tensor = _NPUTensorProxy(torch.arange(6, dtype=torch.float32))
    _stub(monkeypatch, "torch_npu", get_npu_format=fail, get_storage_size=fail)

    with caplog.at_level(logging.INFO, logger="rfork-tensor-safety-test"):
        tensor_runtime.tensor_layout.log_tensor_layout(
            "raising_api",
            tensor,
            stage="registration",
            session_id="session",
            processed_layout=False,
        )

    metadata = _metadata(_layout_record(caplog))
    assert metadata["npu_format"] == "unavailable"
    assert metadata["npu_storage_numel"] == "unavailable"


def test_info_disabled_does_not_read_tensor_metadata(tensor_runtime, monkeypatch, caplog):
    class _ExplodingTensor:
        @property
        def device(self):
            raise AssertionError("metadata must not be read when INFO is disabled")

    monkeypatch.setattr(tensor_runtime.tensor_layout.logger, "isEnabledFor", lambda level: False)

    tensor_runtime.tensor_layout.log_tensor_layout(
        "disabled",
        _ExplodingTensor(),
        stage="registration",
        session_id="session",
        processed_layout=False,
    )

    assert not caplog.records


def test_backend_registration_layout_log_contains_transfer_session(tensor_runtime, monkeypatch, caplog):
    tensor = torch.arange(4, dtype=torch.float32)
    registrations = []

    class _MemoryRegistration:
        def __init__(self, *values):
            self.values = values

    class _Result:
        def is_error(self):
            return False

    backend = tensor_runtime.RForkTransferBackend.__new__(tensor_runtime.RForkTransferBackend)
    backend.transfer_engine = SimpleNamespace(
        batch_register_memory_ex=lambda values: registrations.extend(values) or _Result(),
    )
    backend.transfer_session_id = "receiver-session"
    backend._memory_registration_cls = _MemoryRegistration
    backend.registered_weight_blocks = []
    backend.registered_memory_addresses = []
    backend._registered_transferable_tensors = None
    backend._registered_transferable_storages = None
    backend._all_transferable_tensors_excluded = False

    monkeypatch.setattr(tensor_runtime.transfer_backend, "find_non_npu_state_tensors", lambda _model: [])
    monkeypatch.setattr(tensor_runtime.transfer_backend, "is_transferable_tensor", lambda _tensor: True)
    monkeypatch.setattr(
        tensor_runtime.transfer_backend,
        "collect_transferable_tensors",
        lambda *_args: [("weight", tensor)],
    )
    monkeypatch.setattr(
        tensor_runtime.transfer_backend.torch,
        "npu",
        SimpleNamespace(
            memory=SimpleNamespace(
                memory_snapshot=lambda: [
                    {
                        "blocks": [
                            {
                                "address": tensor.data_ptr(),
                                "size": tensor.numel() * tensor.element_size(),
                                "state": "active_allocated",
                            }
                        ]
                    }
                ]
            )
        ),
        raising=False,
    )

    with caplog.at_level(logging.INFO, logger="rfork-tensor-safety-test"):
        assert backend._register_memory_region_locked(object(), False)

    layout_records = [record for record in caplog.records if "RFork tensor layout:" in record.getMessage()]
    assert len(layout_records) == 1
    assert "stage=registration" in layout_records[0].getMessage()
    assert "session=receiver-session" in layout_records[0].getMessage()
    assert registrations


def test_backend_read_layout_log_contains_receiver_and_seed_sessions(tensor_runtime, caplog):
    tensor = torch.arange(4, dtype=torch.float32)

    class _Result:
        def is_error(self):
            return False

    backend = tensor_runtime.RForkTransferBackend.__new__(tensor_runtime.RForkTransferBackend)
    backend.transfer_engine = SimpleNamespace(
        batch_transfer_sync_read=lambda *args: _Result(),
    )
    backend.transfer_session_id = "receiver-session"
    backend._registered_transferable_tensors = [("weight", tensor)]
    backend.weight_shapes = {"weight": tuple(tensor.shape)}
    backend.weight_manifest = {
        "weight": (tensor.data_ptr(), tensor.numel(), tensor.element_size(), tuple(tensor.shape), "float32")
    }
    backend.excluded_weight_blocks = []
    seed_info = tensor_runtime.SeedTransferInfo(
        "seed-session",
        {
            "weight": [
                tensor.data_ptr(),
                tensor.numel(),
                tensor.element_size(),
                list(tensor.shape),
                "float32",
            ]
        },
    )

    with caplog.at_level(logging.INFO, logger="rfork-tensor-safety-test"):
        assert backend.read_weights_from_seed(object(), seed_info, False)

    layout_records = [record for record in caplog.records if "RFork tensor layout:" in record.getMessage()]
    assert len(layout_records) == 1
    assert "stage=receiver_before_read" in layout_records[0].getMessage()
    assert "session=receiver-session" in layout_records[0].getMessage()
    assert "peer_session=seed-session" in layout_records[0].getMessage()
