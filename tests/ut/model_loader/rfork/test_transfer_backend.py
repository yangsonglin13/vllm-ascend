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

from types import SimpleNamespace

import torch

import vllm_ascend.model_loader.rfork.transfer_backend as transfer_backend
from vllm_ascend.model_loader.rfork.transfer_backend import (
    RForkTransferBackend,
    _collect_checkpoint_layout_tensors,
    _collect_processed_layout_tensors,
    _parse_weight_info,
    _reshape_tensor_to_seed_shape,
    _split_tensors_by_excluded_blocks,
    _subtract_weight_blocks,
    get_remote_instance_transfer_engine_info,
)


def test_parse_weight_info_keeps_backward_compatibility():
    assert _parse_weight_info([1, 2, 4]) == (1, 2, 4, None)


def test_parse_weight_info_accepts_shape_metadata_from_json():
    assert _parse_weight_info([1, 6, 2, [2, 3]]) == (1, 6, 2, (2, 3))


def test_parse_weight_info_rejects_invalid_shape_metadata():
    assert _parse_weight_info([1, 6, 2, ["2", 3]]) is None
    assert _parse_weight_info([1, 6, 2, -1]) is None


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


def test_recv_from_source_refreshes_registered_shape_after_reshape(monkeypatch):
    tensor = torch.arange(6).reshape(2, 3)
    backend = RForkTransferBackend.__new__(RForkTransferBackend)
    backend.rfork_transfer_engine = SimpleNamespace(
        batch_transfer_sync_read=lambda *args: SimpleNamespace(is_error=lambda: False)
    )
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


def test_recv_from_source_reuses_registered_transferable_tensors(monkeypatch):
    tensor = torch.arange(6).reshape(2, 3)
    backend = RForkTransferBackend.__new__(RForkTransferBackend)
    backend.rfork_transfer_engine = SimpleNamespace(
        batch_transfer_sync_read=lambda *args: SimpleNamespace(is_error=lambda: False)
    )
    backend.rfork_transfer_engine_weights_shape_dict = {"weight": (2, 3)}
    backend._registered_transferable_tensors = [("weight", tensor)]

    def fail_if_rescanned(model, processed_layout):
        raise AssertionError("recv_from_source should reuse the registered tensor cache")

    monkeypatch.setattr(transfer_backend, "_iter_transferable_tensors", fail_if_rescanned)
    monkeypatch.setattr(
        transfer_backend,
        "get_remote_instance_transfer_engine_info",
        lambda *args: (
            "seed-session",
            {"weight": [1, tensor.numel(), tensor.element_size(), [2, 3]]},
            None,
        ),
    )

    assert backend.recv_from_source(object(), "127.0.0.1", 8000, "seed-key", True)
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


def test_subtract_weight_blocks_cuts_excluded_ranges():
    assert _subtract_weight_blocks([(0, 100)], []) == [(0, 100)]
    assert _subtract_weight_blocks([(0, 100)], [(0, 100)]) == []
    assert _subtract_weight_blocks([(0, 100)], [(10, 20)]) == [(0, 10), (30, 70)]
    # Exclusion straddling the left edge of the block.
    assert _subtract_weight_blocks([(0, 100)], [(-50, 60)]) == [(10, 90)]
    # Exclusion straddling the right edge of the block.
    assert _subtract_weight_blocks([(0, 100)], [(90, 50)]) == [(0, 90)]
    # Disjoint exclusion leaves the block untouched.
    assert _subtract_weight_blocks([(0, 100)], [(200, 10)]) == [(0, 100)]
    assert _subtract_weight_blocks(
        [(0, 100), (200, 100)],
        [(50, 10), (240, 10)],
    ) == [(0, 50), (60, 40), (200, 40), (250, 50)]


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


def test_split_tensors_by_excluded_blocks_noop_without_exclusion():
    tensor = torch.arange(6)
    kept_tensors, excluded_names = _split_tensors_by_excluded_blocks([("weight", tensor)], [])

    assert kept_tensors == [("weight", tensor)]
    assert excluded_names == []


def _make_register_memory_region_backend(monkeypatch, tensors, snapshot_blocks):
    registered_calls = []
    backend = RForkTransferBackend.__new__(RForkTransferBackend)
    backend.rfork_transfer_engine = SimpleNamespace(
        batch_register_memory=lambda addresses, sizes: registered_calls.append(
            (list(addresses), list(sizes))
        )
        or SimpleNamespace(is_error=lambda: False)
    )
    monkeypatch.setattr(
        transfer_backend,
        "_iter_transferable_tensors",
        lambda model, processed_layout: iter(tensors),
    )
    snapshot = [{"blocks": snapshot_blocks}]
    monkeypatch.setattr(
        torch,
        "npu",
        SimpleNamespace(memory=SimpleNamespace(memory_snapshot=lambda: snapshot)),
        raising=False,
    )
    return backend, registered_calls


def test_register_memory_region_skips_shared_weights(monkeypatch):
    storage = torch.arange(100, dtype=torch.float32)
    # The first 40 bytes simulate storage registered by the target model.
    shared_weight = storage[:10]
    own_weight = storage[20:32].reshape(2, 6)
    backend, registered_calls = _make_register_memory_region_backend(
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

    assert set(backend.rfork_transfer_engine_weights_info_dict) == {"layers.0.fc.weight"}
    assert [name for name, _ in backend._registered_transferable_tensors] == ["layers.0.fc.weight"]
    assert backend.excluded_weight_blocks == excluded_blocks
    # Only the residual range after the shared prefix is registered.
    assert registered_calls == [([storage.data_ptr() + 40], [360])]
    assert backend.registered_weight_blocks == [(storage.data_ptr() + 40, 360)]


def test_register_memory_region_registers_all_weights_without_exclusion(monkeypatch):
    storage = torch.arange(100, dtype=torch.float32)
    weight_a = storage[:10]
    weight_b = storage[20:32]
    backend, registered_calls = _make_register_memory_region_backend(
        monkeypatch,
        [("a.weight", weight_a), ("b.weight", weight_b)],
        [
            {
                "address": storage.data_ptr(),
                "size": storage.numel() * storage.element_size(),
                "state": "active_allocated",
            }
        ],
    )

    assert backend.register_memory_region(object(), True)

    assert set(backend.rfork_transfer_engine_weights_info_dict) == {"a.weight", "b.weight"}
    assert registered_calls == [([storage.data_ptr()], [400])]
    assert backend.excluded_weight_blocks == []


def test_register_memory_region_skips_empty_batch_after_excluding_all_weights(monkeypatch):
    storage = torch.arange(10, dtype=torch.float32)
    backend, registered_calls = _make_register_memory_region_backend(
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

    assert backend.rfork_transfer_engine_weights_info_dict == {}
    assert backend.rfork_transfer_engine_weights_shape_dict == {}
    assert backend._registered_transferable_tensors == []
    assert backend.registered_weight_blocks == []
    assert registered_calls == []


def test_recv_from_source_defensively_skips_pre_registered_weights(monkeypatch):
    storage = torch.arange(100, dtype=torch.float32)
    shared_weight = storage[:10]
    own_weight = storage[20:32]
    backend = RForkTransferBackend.__new__(RForkTransferBackend)
    backend.rfork_transfer_engine = SimpleNamespace(
        batch_transfer_sync_read=lambda *args: SimpleNamespace(is_error=lambda: False)
    )
    backend.rfork_transfer_engine_weights_shape_dict = {}
    backend.excluded_weight_blocks = [(storage.data_ptr(), 40)]
    backend._registered_transferable_tensors = None

    monkeypatch.setattr(
        transfer_backend,
        "_iter_transferable_tensors",
        lambda model, processed_layout: iter(
            [
                ("model.embed_tokens.weight", shared_weight),
                ("layers.0.fc.weight", own_weight),
            ]
        ),
    )

    monkeypatch.setattr(
        transfer_backend,
        "get_remote_instance_transfer_engine_info",
        lambda *args: (
            "seed-session",
            {"layers.0.fc.weight": [7, own_weight.numel(), own_weight.element_size()]},
            None,
        ),
    )

    assert backend.recv_from_source(object(), "127.0.0.1", 8000, "seed-key", True)
    assert backend.rfork_transfer_engine_weights_shape_dict == {
        "layers.0.fc.weight": tuple(own_weight.shape)
    }


def test_recv_from_source_fails_for_unknown_weight_outside_shared_blocks(monkeypatch):
    own_weight = torch.arange(12, dtype=torch.float32)
    backend = RForkTransferBackend.__new__(RForkTransferBackend)
    backend.rfork_transfer_engine = SimpleNamespace(
        batch_transfer_sync_read=lambda *args: SimpleNamespace(is_error=lambda: False)
    )
    backend.rfork_transfer_engine_weights_shape_dict = {}
    backend.excluded_weight_blocks = [(4096, 128)]
    backend._registered_transferable_tensors = [("layers.0.fc.weight", own_weight)]

    monkeypatch.setattr(
        transfer_backend,
        "get_remote_instance_transfer_engine_info",
        lambda *args: ("seed-session", {}, None),
    )

    assert not backend.recv_from_source(object(), "127.0.0.1", 8000, "seed-key", True)
