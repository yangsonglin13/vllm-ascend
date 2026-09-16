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
from typing import Any

import torch

import vllm_ascend.model_loader.rfork.transfer_backend as transfer_backend
from vllm_ascend.model_loader.rfork import tensor_layout
from vllm_ascend.model_loader.rfork.transfer_backend import (
    RForkTransferBackend,
    _split_tensors_by_excluded_blocks,
)
from vllm_ascend.model_loader.rfork.types import SeedTransferInfo


def _extend_and_return(items: list[Any], values: Any, result: Any) -> Any:
    items.extend(values)
    return result


def test_tensor_collection_deduplicates_exact_impl_alias_but_keeps_distinct_view(monkeypatch):
    weight = torch.nn.Parameter(torch.arange(4, dtype=torch.float32))
    model = torch.nn.Module()
    model.register_parameter("weight", weight)
    model.impl = SimpleNamespace(weight=weight, view=weight[:2])
    monkeypatch.setattr(tensor_layout, "is_transferable_tensor", lambda _tensor: True)

    collected = tensor_layout.collect_processed_layout_tensors(model)

    assert [(name, tensor.numel()) for name, tensor in collected] == [("weight", 4), ("impl.view", 2)]


def test_read_weights_from_seed_refreshes_registered_shape_after_reshape(monkeypatch):
    tensor = torch.arange(6).reshape(2, 3)
    backend = RForkTransferBackend.__new__(RForkTransferBackend)
    backend.transfer_engine = SimpleNamespace(
        batch_transfer_sync_read=lambda *args: SimpleNamespace(is_error=lambda: False)
    )
    backend.weight_manifest = {"weight": (tensor.data_ptr(), tensor.numel(), tensor.element_size(), (2, 3), "int64")}
    backend.weight_shapes = {"weight": (2, 3)}

    monkeypatch.setattr(
        transfer_backend,
        "collect_transferable_tensors",
        lambda model, processed_layout: [("weight", tensor)],
    )
    seed_info = SeedTransferInfo(
        "seed-session",
        {"weight": [1, tensor.numel(), tensor.element_size(), [1, 2, 3], "int64"]},
        {"weight": [1, 2, 3]},
    )

    assert backend.read_weights_from_seed(object(), seed_info, True)
    assert tuple(tensor.shape) == (1, 2, 3)
    assert backend.weight_shapes["weight"] == (1, 2, 3)
    assert backend.weight_manifest["weight"][3] == (1, 2, 3)
    assert backend.weight_manifest["weight"][4] == "int64"


def test_read_weights_from_seed_retains_registered_transferable_tensor_owners(monkeypatch):
    tensor = torch.arange(6).reshape(2, 3)
    tensor_ref = weakref.ref(tensor)
    tensor_numel = tensor.numel()
    tensor_element_size = tensor.element_size()
    backend = RForkTransferBackend.__new__(RForkTransferBackend)
    backend.transfer_engine = SimpleNamespace(
        batch_transfer_sync_read=lambda *args: SimpleNamespace(is_error=lambda: False)
    )
    backend.weight_shapes = {"weight": (2, 3)}
    registered_tensors = [("weight", tensor)]
    backend._registered_transferable_tensors = registered_tensors

    def fail_if_rescanned(model, processed_layout):
        raise AssertionError("read_weights_from_seed should reuse the registered tensor cache")

    monkeypatch.setattr(transfer_backend, "collect_transferable_tensors", fail_if_rescanned)
    seed_info = SeedTransferInfo(
        "seed-session",
        {"weight": [1, tensor_numel, tensor_element_size, [2, 3], "int64"]},
        None,
    )

    assert backend.read_weights_from_seed(object(), seed_info, True)
    assert backend._registered_transferable_tensors is registered_tensors
    del registered_tensors
    del tensor
    gc.collect()
    assert tensor_ref() is not None


def test_read_weights_from_seed_keeps_registered_tensors_when_seed_metadata_is_invalid(monkeypatch):
    tensor = torch.arange(6).reshape(2, 3)
    backend = RForkTransferBackend.__new__(RForkTransferBackend)
    backend.transfer_engine = SimpleNamespace()
    registered_tensors = [("weight", tensor)]
    backend._registered_transferable_tensors = registered_tensors

    seed_info = SeedTransferInfo("", {})

    assert not backend.read_weights_from_seed(object(), seed_info, True)
    assert backend._registered_transferable_tensors is registered_tensors


def test_unregister_memory_region_releases_registered_tensor_owners():
    tensor = torch.arange(6).reshape(2, 3)
    backend = RForkTransferBackend.__new__(RForkTransferBackend)
    backend.transfer_engine = SimpleNamespace(
        batch_unregister_memory=lambda *args: SimpleNamespace(is_error=lambda: False)
    )
    backend.weight_manifest = {"weight": (tensor.data_ptr(), tensor.numel(), 1)}
    backend.weight_shapes = {"weight": tuple(tensor.shape)}
    backend.registered_weight_blocks = [(tensor.data_ptr(), tensor.numel() * tensor.element_size())]
    backend.registered_memory_addresses = [tensor.data_ptr()]
    backend._registered_transferable_tensors = [("weight", tensor)]

    assert backend.unregister_memory_region()
    assert backend._registered_transferable_tensors is None
    assert backend.weight_manifest is None
    assert backend.weight_shapes is None
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
    backend.transfer_engine = SimpleNamespace(batch_unregister_memory=unregister)
    backend.weight_manifest = {"weight": (tensor.data_ptr(), 6, 1)}
    backend.weight_shapes = {"weight": (2, 3)}
    backend.registered_weight_blocks = blocks
    backend.registered_memory_addresses = [blocks[0][0]]
    backend._registered_transferable_tensors = owners

    assert not backend.unregister_memory_region()
    assert backend.registered_weight_blocks == blocks
    assert backend._registered_transferable_tensors is owners
    assert backend.weight_manifest is not None
    assert backend.weight_shapes is not None

    assert backend.unregister_memory_region()
    assert attempts == [[blocks[0][0]], [blocks[0][0]]]
    assert backend.registered_weight_blocks == []
    assert backend._registered_transferable_tensors is None


def test_register_memory_region_uses_logical_ranges_with_allocator_backing(monkeypatch):
    tensor = torch.arange(8, dtype=torch.uint8)
    logical = tensor[2:6]
    backing_start = tensor.data_ptr()
    backing_size = tensor.numel() * tensor.element_size()
    registrations: list[tuple[int, int, int, int]] = []

    class _MemoryRegistration:
        def __init__(self, logical_addr, logical_length, backing_addr, backing_length):
            self.values = (logical_addr, logical_length, backing_addr, backing_length)

    class _Ret:
        def is_error(self):
            return False

    backend = RForkTransferBackend.__new__(RForkTransferBackend)
    backend.transfer_engine = SimpleNamespace(
        batch_register_memory_ex=lambda items: _extend_and_return(
            registrations, (item.values for item in items), _Ret()
        )
    )
    backend._memory_registration_cls = _MemoryRegistration
    backend.registered_weight_blocks = []
    backend.registered_memory_addresses = []
    backend.weight_manifest = None
    backend.weight_shapes = None
    backend._registered_transferable_tensors = None

    monkeypatch.setattr(transfer_backend, "find_non_npu_state_tensors", lambda _model: [])
    monkeypatch.setattr(transfer_backend, "is_transferable_tensor", lambda _tensor: True)
    monkeypatch.setattr(transfer_backend, "collect_transferable_tensors", lambda *args: [("weight", logical)])
    monkeypatch.setattr(
        transfer_backend.torch,
        "npu",
        SimpleNamespace(
            memory=SimpleNamespace(
                memory_snapshot=lambda: [
                    {
                        "blocks": [
                            {
                                "address": backing_start,
                                "size": backing_size,
                                "state": "active_allocated",
                            }
                        ]
                    }
                ]
            )
        ),
        raising=False,
    )

    assert backend.register_memory_region(object(), False)
    assert registrations == [(logical.data_ptr(), logical.numel(), backing_start, backing_size)]
    assert backend.registered_memory_addresses == [logical.data_ptr()]


def test_register_memory_region_retries_stale_blocks_before_registering(monkeypatch):
    storage = torch.arange(10, dtype=torch.float32)
    stale_blocks = [(storage.data_ptr() + 4096, 256)]
    backend, registrations, unregistered_calls = _make_register_memory_region_backend(
        monkeypatch,
        [("weight", storage)],
        [
            {
                "address": storage.data_ptr(),
                "size": storage.numel() * storage.element_size(),
                "state": "active_allocated",
            }
        ],
        stale_blocks=stale_blocks,
    )

    assert backend.register_memory_region(object(), True)

    assert unregistered_calls == [[stale_blocks[0][0]]]
    assert registrations == [(storage.data_ptr(), storage.numel() * storage.element_size(), storage.data_ptr(), 40)]
    assert backend.registered_weight_blocks == [(storage.data_ptr(), 40)]


def test_register_memory_region_aborts_when_stale_unregister_fails(monkeypatch):
    storage = torch.arange(10, dtype=torch.float32)
    stale_blocks = [(storage.data_ptr() + 4096, 256)]
    backend, registrations, unregistered_calls = _make_register_memory_region_backend(
        monkeypatch,
        [("weight", storage)],
        [
            {
                "address": storage.data_ptr(),
                "size": storage.numel() * storage.element_size(),
                "state": "active_allocated",
            }
        ],
        stale_blocks=stale_blocks,
        unregister_error=True,
    )

    assert not backend.register_memory_region(object(), True)

    assert registrations == []
    assert unregistered_calls == [[stale_blocks[0][0]]]
    assert backend.registered_weight_blocks == stale_blocks


def test_split_tensors_by_excluded_blocks_separates_shared_storage():
    storage = torch.arange(10)
    shared_tensor = storage[:3]
    own_tensor = torch.arange(4)

    kept_tensors, excluded_names = _split_tensors_by_excluded_blocks(
        [("model.embed_tokens.weight", shared_tensor), ("layers.0.fc.weight", own_tensor)],
        [(storage.data_ptr(), shared_tensor.numel() * shared_tensor.element_size())],
    )

    assert [name for name, _ in kept_tensors] == ["layers.0.fc.weight"]
    assert excluded_names == ["model.embed_tokens.weight"]


def _make_register_memory_region_backend(
    monkeypatch,
    tensors,
    snapshot_blocks,
    stale_blocks=(),
    unregister_error=False,
):
    registrations: list[tuple[int, int, int, int]] = []
    unregistered_calls = []
    monkeypatch.setattr(
        transfer_backend, "is_transferable_tensor", lambda t: isinstance(t, torch.Tensor) and t.numel() > 0
    )

    class _MemoryRegistration:
        def __init__(self, *values):
            self.values = values

    class _Ret:
        def is_error(self):
            return False

    def batch_register_memory_ex(items):
        registrations.extend(item.values for item in items)
        return _Ret()

    def batch_unregister_memory(addresses):
        unregistered_calls.append(list(addresses))
        return SimpleNamespace(is_error=lambda: unregister_error, to_string=lambda: "mock unregister error")

    backend = RForkTransferBackend.__new__(RForkTransferBackend)
    backend.transfer_engine = SimpleNamespace(
        batch_register_memory_ex=batch_register_memory_ex,
        batch_unregister_memory=batch_unregister_memory,
    )
    backend._memory_registration_cls = _MemoryRegistration
    backend.registered_weight_blocks = list(stale_blocks)
    backend.registered_memory_addresses = [address for address, _ in stale_blocks]
    backend.weight_manifest = None
    backend.weight_shapes = None
    backend._registered_transferable_tensors = None
    monkeypatch.setattr(
        transfer_backend,
        "collect_transferable_tensors",
        lambda model, processed_layout: list(tensors),
    )
    snapshot = [{"blocks": snapshot_blocks}]
    monkeypatch.setattr(
        transfer_backend.torch,
        "npu",
        SimpleNamespace(memory=SimpleNamespace(memory_snapshot=lambda: snapshot)),
        raising=False,
    )
    return backend, registrations, unregistered_calls


def test_register_memory_region_skips_shared_weights(monkeypatch):
    storage = torch.arange(100, dtype=torch.float32)
    shared_weight = storage[:10]
    own_weight = storage[20:32].reshape(2, 6)
    backend, registrations, _ = _make_register_memory_region_backend(
        monkeypatch,
        [("model.embed_tokens.weight", shared_weight), ("layers.0.fc.weight", own_weight)],
        [
            {
                "address": storage.data_ptr(),
                "size": storage.numel() * storage.element_size(),
                "state": "active_allocated",
            }
        ],
    )

    excluded_blocks = [(storage.data_ptr(), 40)]
    assert backend.register_memory_region(object(), True, exclude_blocks=excluded_blocks)

    assert set(backend.weight_manifest) == {"layers.0.fc.weight"}
    assert [name for name, _ in backend._registered_transferable_tensors] == ["layers.0.fc.weight"]
    assert backend.excluded_weight_blocks == excluded_blocks
    assert registrations == [
        (
            own_weight.data_ptr(),
            own_weight.numel() * own_weight.element_size(),
            storage.data_ptr() + 40,
            360,
        )
    ]


def test_register_memory_region_skips_empty_batch_after_excluding_all_weights(monkeypatch):
    storage = torch.arange(10, dtype=torch.float32)
    backend, registrations, _ = _make_register_memory_region_backend(
        monkeypatch,
        [("model.embed_tokens.weight", storage)],
        [
            {
                "address": storage.data_ptr(),
                "size": storage.numel() * storage.element_size(),
                "state": "active_allocated",
            }
        ],
    )

    excluded_blocks = [(storage.data_ptr(), storage.numel() * storage.element_size())]
    assert backend.register_memory_region(object(), True, exclude_blocks=excluded_blocks)

    assert backend.weight_manifest == {}
    assert backend.weight_shapes == {}
    assert backend._registered_transferable_tensors == []
    assert backend.registered_weight_blocks == []
    assert registrations == []


def test_read_weights_from_seed_defensively_skips_pre_registered_weights(monkeypatch):
    storage = torch.arange(100, dtype=torch.float32)
    shared_weight = storage[:10]
    own_weight = storage[20:32]
    backend = RForkTransferBackend.__new__(RForkTransferBackend)
    backend.transfer_engine = SimpleNamespace(
        batch_transfer_sync_read=lambda *args: SimpleNamespace(is_error=lambda: False)
    )
    backend.weight_shapes = {}
    backend.excluded_weight_blocks = [(storage.data_ptr(), 40)]
    backend._registered_transferable_tensors = None

    monkeypatch.setattr(
        transfer_backend,
        "collect_transferable_tensors",
        lambda model, processed_layout: [
            ("model.embed_tokens.weight", shared_weight),
            ("layers.0.fc.weight", own_weight),
        ],
    )

    seed_info = SeedTransferInfo(
        "seed-session",
        {
            "layers.0.fc.weight": [
                7,
                own_weight.numel(),
                own_weight.element_size(),
                list(own_weight.shape),
                "float32",
            ]
        },
        None,
    )

    assert backend.read_weights_from_seed(object(), seed_info, True)
    assert backend.weight_shapes == {"layers.0.fc.weight": tuple(own_weight.shape)}


def test_read_weights_from_seed_fails_for_unknown_weight_outside_shared_blocks(monkeypatch):
    own_weight = torch.arange(12, dtype=torch.float32)
    backend = RForkTransferBackend.__new__(RForkTransferBackend)
    backend.transfer_engine = SimpleNamespace(
        batch_transfer_sync_read=lambda *args: SimpleNamespace(is_error=lambda: False)
    )
    backend.weight_shapes = {}
    backend.excluded_weight_blocks = [(4096, 128)]
    backend._registered_transferable_tensors = [("layers.0.fc.weight", own_weight)]

    seed_info = SeedTransferInfo("seed-session", {}, None)

    assert not backend.read_weights_from_seed(object(), seed_info, True)
