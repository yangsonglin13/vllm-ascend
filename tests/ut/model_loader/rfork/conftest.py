# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

import logging
from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from .rfork_test_support import _load_module, _stub
from .test_lease_release import runtime as runtime


@pytest.fixture
def tensor_runtime(monkeypatch):
    _stub(monkeypatch, "vllm", __path__=[])
    _stub(monkeypatch, "vllm.logger", logger=logging.getLogger("rfork-tensor-safety-test"))
    _stub(monkeypatch, "vllm.utils", __path__=[])
    _stub(
        monkeypatch,
        "vllm.utils.network_utils",
        get_ip=lambda: "127.0.0.1",
        get_open_port=lambda: 12345,
        join_host_port=lambda host, port: f"{host}:{port}",
    )
    for module_name in (
        "vllm_ascend",
        "vllm_ascend.model_loader",
        "vllm_ascend.model_loader.rfork",
    ):
        _stub(monkeypatch, module_name, __path__=[])

    @dataclass(frozen=True)
    class _SeedTransferInfo:
        session_id: str
        weights: dict
        shapes: dict | None = None

    _stub(monkeypatch, "vllm_ascend.model_loader.rfork.types", SeedTransferInfo=_SeedTransferInfo)
    _load_module(monkeypatch, "vllm_ascend.model_loader.rfork.manifest", "manifest.py")
    tensor_layout = _load_module(monkeypatch, "vllm_ascend.model_loader.rfork.tensor_layout", "tensor_layout.py")
    transfer_backend = _load_module(
        monkeypatch, "vllm_ascend.model_loader.rfork.transfer_backend", "transfer_backend.py"
    )
    monkeypatch.setattr(transfer_backend.RForkTransferBackend, "_initialize_transfer_engine", lambda self: None)
    return SimpleNamespace(
        tensor_layout=tensor_layout,
        transfer_backend=transfer_backend,
        RForkTransferBackend=transfer_backend.RForkTransferBackend,
        SeedTransferInfo=_SeedTransferInfo,
    )
