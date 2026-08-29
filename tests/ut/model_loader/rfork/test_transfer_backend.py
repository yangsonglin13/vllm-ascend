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
import weakref
from types import SimpleNamespace

import pytest
import torch

import vllm_ascend.model_loader.rfork.transfer_backend as transfer_backend
from vllm_ascend.model_loader.rfork.transfer_backend import (
    RForkTransferBackend,
    _collect_checkpoint_layout_tensors,
    _collect_processed_layout_tensors,
    _iter_transfer_chunks,
    _parse_weight_info,
    _reshape_tensor_to_seed_shape,
    get_remote_instance_transfer_engine_info,
)


def test_parse_weight_info_keeps_backward_compatibility():
    assert _parse_weight_info([1, 2, 4]) == (1, 2, 4, None)


def test_parse_weight_info_accepts_shape_metadata_from_json():
    assert _parse_weight_info([1, 6, 2, [2, 3]]) == (1, 6, 2, (2, 3))


def test_parse_weight_info_rejects_invalid_shape_metadata():
    assert _parse_weight_info([1, 6, 2, ["2", 3]]) is None
    assert _parse_weight_info([1, 6, 2, -1]) is None


@pytest.mark.parametrize(
    "weight_info",
    [
        [True, 6, 2],
        [1, True, 2],
        [1, 6, True],
        [0, 6, 2],
        [1, 0, 2],
        [1, 6, 0],
    ],
)
def test_parse_weight_info_rejects_bool_and_non_positive_values(weight_info):
    assert _parse_weight_info(weight_info) is None


def test_parse_weight_info_accepts_v2_named_shape_and_dtype():
    parsed = _parse_weight_info(
        {
            "ptr": 1,
            "numel": 6,
            "element_size": 2,
            "shape": [2, 3],
            "dtype": "torch.float16",
        }
    )
    assert parsed == (1, 6, 2, (2, 3), "float16")
    assert _parse_weight_info([1, 6, 2, [2, 3], "bfloat16"]) == (1, 6, 2, (2, 3), "bfloat16")


def test_reshape_tensor_to_seed_shape_updates_tensor_metadata_only():
    tensor = torch.arange(6).reshape(2, 3)
    original_ptr = tensor.data_ptr()

    assert _reshape_tensor_to_seed_shape("weight", tensor, (1, 2, 3))

    assert tuple(tensor.shape) == (1, 2, 3)
    assert tensor.data_ptr() == original_ptr


def test_reshape_tensor_to_seed_shape_rejects_numel_mismatch():
    tensor = torch.arange(6).reshape(2, 3)

    assert not _reshape_tensor_to_seed_shape("weight", tensor, (2, 2))
    assert tuple(tensor.shape) == (2, 3)


def test_iter_transfer_chunks_splits_single_large_tensor():
    chunk_limit = transfer_backend.MAX_TRANSFER_CHUNK_BYTES
    chunks = list(
        _iter_transfer_chunks(
            ["large"],
            [10_000],
            [20_000],
            [chunk_limit + 17],
        )
    )

    assert len(chunks) == 2
    assert [length for _, _, _, lengths in chunks for length in lengths] == [chunk_limit, 17]
    assert all(sum(lengths) <= chunk_limit for _, _, _, lengths in chunks)
    assert all(len(lengths) <= transfer_backend.MAX_TRANSFER_CHUNK_WEIGHTS for _, _, _, lengths in chunks)
    assert chunks[1][1] == [10_000 + chunk_limit]
    assert chunks[1][2] == [20_000 + chunk_limit]


def test_recv_from_source_refreshes_registered_shape_after_reshape(monkeypatch):
    tensor = torch.arange(6).reshape(2, 3)
    backend = RForkTransferBackend.__new__(RForkTransferBackend)
    backend.rfork_transfer_engine = SimpleNamespace(
        batch_transfer_sync_read=lambda *args: SimpleNamespace(is_error=lambda: False)
    )
    backend.rfork_transfer_engine_weights_info_dict = {
        "weight": (tensor.data_ptr(), tensor.numel(), tensor.element_size(), (2, 3), "int64")
    }
    backend.rfork_transfer_engine_weights_shape_dict = {"weight": (2, 3)}

    monkeypatch.setattr(
        transfer_backend,
        "_iter_transferable_tensors",
        lambda model, processed_layout: iter([("weight", tensor)]),
    )
    monkeypatch.setattr(
        transfer_backend,
        "get_remote_instance_transfer_engine_info",
        lambda *args: (
            "seed-session",
            {"weight": [1, tensor.numel(), tensor.element_size()]},
            {"weight": [1, 2, 3]},
        ),
    )

    assert backend.recv_from_source(object(), "127.0.0.1", 8000, "seed-key", True)
    assert tuple(tensor.shape) == (1, 2, 3)
    assert backend.rfork_transfer_engine_weights_shape_dict["weight"] == (1, 2, 3)
    assert backend.rfork_transfer_engine_weights_info_dict["weight"][3] == (1, 2, 3)
    assert backend.rfork_transfer_engine_weights_info_dict["weight"][4] == "int64"


def test_recv_from_source_retains_registered_transferable_tensor_owners(monkeypatch):
    tensor = torch.arange(6).reshape(2, 3)
    tensor_ref = weakref.ref(tensor)
    tensor_numel = tensor.numel()
    tensor_element_size = tensor.element_size()
    backend = RForkTransferBackend.__new__(RForkTransferBackend)
    backend.rfork_transfer_engine = SimpleNamespace(
        batch_transfer_sync_read=lambda *args: SimpleNamespace(is_error=lambda: False)
    )
    backend.rfork_transfer_engine_weights_shape_dict = {"weight": (2, 3)}
    registered_tensors = [("weight", tensor)]
    backend._registered_transferable_tensors = registered_tensors

    def fail_if_rescanned(model, processed_layout):
        raise AssertionError("recv_from_source should reuse the registered tensor cache")

    monkeypatch.setattr(transfer_backend, "_iter_transferable_tensors", fail_if_rescanned)
    monkeypatch.setattr(
        transfer_backend,
        "get_remote_instance_transfer_engine_info",
        lambda *args: (
            "seed-session",
            {"weight": [1, tensor_numel, tensor_element_size, [2, 3]]},
            None,
        ),
    )

    assert backend.recv_from_source(object(), "127.0.0.1", 8000, "seed-key", True)
    assert backend._registered_transferable_tensors is registered_tensors
    del registered_tensors
    del tensor
    gc.collect()
    assert tensor_ref() is not None


def test_recv_from_source_keeps_registered_tensors_when_seed_metadata_is_unavailable(monkeypatch):
    tensor = torch.arange(6).reshape(2, 3)
    backend = RForkTransferBackend.__new__(RForkTransferBackend)
    backend.rfork_transfer_engine = SimpleNamespace()
    registered_tensors = [("weight", tensor)]
    backend._registered_transferable_tensors = registered_tensors

    monkeypatch.setattr(
        transfer_backend,
        "get_remote_instance_transfer_engine_info",
        lambda *args: (None, None, None),
    )

    assert not backend.recv_from_source(object(), "127.0.0.1", 8000, "seed-key", True)
    assert backend._registered_transferable_tensors is registered_tensors


@pytest.mark.parametrize(
    "remote_info",
    [
        {"other": [1, 6, 2]},
        {"weight": [1, 6, 4]},
        {"weight": [1, 6, 2, [2, 3], "float32"]},
    ],
)
def test_recv_from_source_rejects_manifest_mismatch_before_native_transfer(monkeypatch, remote_info):
    tensor = torch.arange(6, dtype=torch.float16).reshape(2, 3)
    native_calls = []
    backend = RForkTransferBackend.__new__(RForkTransferBackend)
    backend.rfork_transfer_engine = SimpleNamespace(
        batch_transfer_sync_read=lambda *args: native_calls.append(args) or SimpleNamespace(is_error=lambda: False)
    )
    backend._registered_transferable_tensors = [("weight", tensor)]
    backend.rfork_transfer_engine_weights_shape_dict = {"weight": (2, 3)}
    monkeypatch.setattr(
        transfer_backend,
        "get_remote_instance_transfer_engine_info",
        lambda *args: ("seed-session", remote_info, None),
    )

    assert not backend.recv_from_source(object(), "127.0.0.1", 8000, "seed-key", True)
    assert native_calls == []


def test_recv_from_source_requires_shape_and_dtype_for_v2_key(monkeypatch):
    tensor = torch.arange(6).reshape(2, 3)
    backend = RForkTransferBackend.__new__(RForkTransferBackend)
    backend.rfork_transfer_engine = SimpleNamespace()
    backend.rfork_transfer_engine_weights_shape_dict = {"weight": (2, 3)}
    backend._registered_transferable_tensors = [("weight", tensor)]

    monkeypatch.setattr(
        transfer_backend,
        "get_remote_instance_transfer_engine_info",
        lambda *args: (
            "seed-session",
            {"weight": [1, tensor.numel(), tensor.element_size()]},
            None,
        ),
    )

    assert not backend.recv_from_source(object(), "127.0.0.1", 8000, "rfork-v2:digest", True)


def test_recv_from_source_validates_optional_manifest_metadata(monkeypatch):
    tensor = torch.arange(6, dtype=torch.float16).reshape(2, 3)
    native_calls = []
    backend = RForkTransferBackend.__new__(RForkTransferBackend)
    backend.rfork_transfer_engine = SimpleNamespace(
        batch_transfer_sync_read=lambda *args: native_calls.append(args) or SimpleNamespace(is_error=lambda: False)
    )
    backend._registered_transferable_tensors = [("weight", tensor)]
    backend.rfork_transfer_engine_weights_shape_dict = {"weight": (2, 3)}
    monkeypatch.setattr(
        transfer_backend,
        "get_remote_instance_transfer_engine_info",
        lambda *args: ("seed-session", {"weight": [1, 6, 2, [2, 3], "float16"]}, None),
    )

    metadata = {
        "tensor_count": 1,
        "total_bytes": tensor.numel() * tensor.element_size(),
        "weights": {
            "weight": {
                "numel": 6,
                "element_size": 2,
                "shape": [2, 3],
                "dtype": "float16",
            }
        },
    }
    assert backend.recv_from_source(object(), "127.0.0.1", 8000, "seed-key", True, metadata)
    assert len(native_calls) == 1


def test_unregister_memory_region_releases_registered_tensor_owners():
    tensor = torch.arange(6).reshape(2, 3)
    backend = RForkTransferBackend.__new__(RForkTransferBackend)
    backend.rfork_transfer_engine = SimpleNamespace(
        batch_unregister_memory=lambda *args: SimpleNamespace(is_error=lambda: False)
    )
    backend.rfork_transfer_engine_weights_info_dict = {"weight": (tensor.data_ptr(), tensor.numel(), 1)}
    backend.rfork_transfer_engine_weights_shape_dict = {"weight": tuple(tensor.shape)}
    backend.registered_weight_blocks = [(tensor.data_ptr(), tensor.numel() * tensor.element_size())]
    backend._registered_transferable_tensors = [("weight", tensor)]

    assert backend.unregister_memory_region()
    assert backend._registered_transferable_tensors is None
    assert backend.rfork_transfer_engine_weights_info_dict is None
    assert backend.rfork_transfer_engine_weights_shape_dict is None
    assert backend.registered_weight_blocks == []


def test_unregister_memory_region_failure_preserves_state_for_retry():
    tensor = torch.arange(6).reshape(2, 3)
    blocks = [(tensor.data_ptr(), tensor.numel() * tensor.element_size())]
    owners = [("weight", tensor)]
    attempts = []

    class _Ret:
        def __init__(self, error):
            self.error = error

        def is_error(self):
            return self.error

        def to_string(self):
            return "failure"

    def unregister(_addresses):
        attempts.append(_addresses)
        return _Ret(len(attempts) == 1)

    backend = RForkTransferBackend.__new__(RForkTransferBackend)
    backend.rfork_transfer_engine = SimpleNamespace(batch_unregister_memory=unregister)
    backend.rfork_transfer_engine_weights_info_dict = {"weight": (tensor.data_ptr(), 6, 1)}
    backend.rfork_transfer_engine_weights_shape_dict = {"weight": (2, 3)}
    backend.registered_weight_blocks = blocks
    backend._registered_transferable_tensors = owners

    assert not backend.unregister_memory_region()
    assert backend.registered_weight_blocks == blocks
    assert backend._registered_transferable_tensors is owners
    assert backend.rfork_transfer_engine_weights_info_dict is not None
    assert backend.rfork_transfer_engine_weights_shape_dict is not None

    assert backend.unregister_memory_region()
    assert attempts == [[blocks[0][0]], [blocks[0][0]]]
    assert backend.registered_weight_blocks == []
    assert backend._registered_transferable_tensors is None


def test_register_memory_region_rejects_second_registration_without_overwriting(monkeypatch):
    tensor = torch.arange(4)
    old_blocks = [(123, 64)]
    old_info = {"old": (123, 4, 4)}
    old_owners = [("old", tensor)]
    backend = RForkTransferBackend.__new__(RForkTransferBackend)
    backend.rfork_transfer_engine = SimpleNamespace()
    backend.registered_weight_blocks = old_blocks
    backend.rfork_transfer_engine_weights_info_dict = old_info
    backend.rfork_transfer_engine_weights_shape_dict = {"old": (4,)}
    backend._registered_transferable_tensors = old_owners

    monkeypatch.setattr(transfer_backend, "_iter_transferable_tensors", lambda *args: [("new", tensor)])
    assert not backend.register_memory_region(object(), False)
    assert backend.registered_weight_blocks == old_blocks
    assert backend.rfork_transfer_engine_weights_info_dict is old_info
    assert backend._registered_transferable_tensors is old_owners


def test_register_memory_region_rejects_partial_cpu_offloaded_state():
    backend = RForkTransferBackend.__new__(RForkTransferBackend)
    backend.rfork_transfer_engine = SimpleNamespace()
    backend.registered_weight_blocks = []
    backend.rfork_transfer_engine_weights_info_dict = None
    backend.rfork_transfer_engine_weights_shape_dict = None
    backend._registered_transferable_tensors = None

    assert not backend.register_memory_region(torch.nn.Linear(2, 2), False)


def test_register_memory_region_rejects_uncovered_tensor_without_state_change(monkeypatch):
    tensor = torch.arange(4)
    register_calls = []

    class _Ret:
        def is_error(self):
            return False

    backend = RForkTransferBackend.__new__(RForkTransferBackend)
    backend.rfork_transfer_engine = SimpleNamespace(
        batch_register_memory=lambda addresses, sizes: register_calls.append((addresses, sizes)) or _Ret()
    )
    backend.registered_weight_blocks = []
    backend.rfork_transfer_engine_weights_info_dict = None
    backend.rfork_transfer_engine_weights_shape_dict = None
    backend._registered_transferable_tensors = None

    monkeypatch.setattr(transfer_backend, "_is_transferable_tensor", lambda _tensor: True)
    monkeypatch.setattr(transfer_backend, "_iter_transferable_tensors", lambda *args: [("weight", tensor)])
    monkeypatch.setattr(
        transfer_backend.torch,
        "npu",
        SimpleNamespace(memory=SimpleNamespace(memory_snapshot=lambda: [])),
        raising=False,
    )

    assert not backend.register_memory_region(object(), False)
    assert register_calls == []
    assert backend.registered_weight_blocks == []
    assert backend._registered_transferable_tensors is None


def test_transferable_tensor_scan_depends_on_runtime_layout(monkeypatch):
    class _RuntimeImpl:
        def __init__(self):
            self.runtime_weight = torch.ones(2)

    class _Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(2))
            self.register_buffer("buffer", torch.ones(2))
            self.runtime_constant = torch.ones(2)
            self.impl = _RuntimeImpl()

    monkeypatch.setattr(transfer_backend, "_is_transferable_tensor", lambda tensor: True)
    model = _Model()

    processed_names = {name for name, _ in _collect_processed_layout_tensors(model)}
    checkpoint_names = {name for name, _ in _collect_checkpoint_layout_tensors(model)}

    assert processed_names == {"weight", "buffer", "runtime_constant", "impl.runtime_weight"}
    assert checkpoint_names == {"weight", "buffer", "impl.runtime_weight"}


def test_get_remote_instance_transfer_engine_info_non_200_returns_three_values(monkeypatch):
    monkeypatch.setattr(
        transfer_backend.requests,
        "get",
        lambda *args, **kwargs: SimpleNamespace(status_code=503),
    )

    assert get_remote_instance_transfer_engine_info("http://seed", "seed-key") == (None, None, None)


def test_remote_manifest_requests_use_timeout_and_auth_token(monkeypatch):
    calls = []

    class _Response:
        status_code = 200

        def __init__(self, payload):
            self.payload = payload

        def json(self):
            return self.payload

    responses = iter(
        [
            _Response({"rfork_transfer_engine_info": ["session", {"w": [1, 1, 2]}]}),
            _Response({"rfork_transfer_engine_shape_info": {"w": [1]}}),
        ]
    )

    def fake_get(*args, **kwargs):
        calls.append((args, kwargs))
        return next(responses)

    monkeypatch.setattr(transfer_backend.requests, "get", fake_get)
    assert get_remote_instance_transfer_engine_info("http://[::1]:8000", "seed-key", 2.5, "secret") == (
        "session",
        {"w": [1, 1, 2]},
        {"w": [1]},
    )
    assert len(calls) == 2
    assert all(call[1]["timeout"] == 2.5 for call in calls)
    assert all(call[1]["headers"] == {"X-RFORK-TOKEN": "secret"} for call in calls)


def test_remote_manifest_timeout_must_be_finite_and_positive(monkeypatch):
    monkeypatch.setattr(transfer_backend.requests, "get", lambda *args, **kwargs: None)
    with pytest.raises(ValueError):
        get_remote_instance_transfer_engine_info("http://seed", "key", 0)
    with pytest.raises(ValueError):
        get_remote_instance_transfer_engine_info("http://seed", "key", float("inf"))
