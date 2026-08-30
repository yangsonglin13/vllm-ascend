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
import sys
import threading
import weakref
from types import ModuleType, SimpleNamespace

import pytest
import torch

import vllm_ascend.model_loader.rfork.manifest as manifest_module
import vllm_ascend.model_loader.rfork.seed_client as seed_client_module
import vllm_ascend.model_loader.rfork.transfer_backend as transfer_backend
from vllm_ascend.model_loader.rfork.manifest import (
    collect_checkpoint_layout_tensors,
    collect_processed_layout_tensors,
    iter_transfer_chunks,
    parse_weight_info,
    reshape_tensor_to_seed_shape,
)
from vllm_ascend.model_loader.rfork.seed_client import fetch_seed_transfer_info
from vllm_ascend.model_loader.rfork.transfer_backend import (
    RForkTransferBackend,
    _split_tensors_by_excluded_blocks,
)
from vllm_ascend.model_loader.rfork.types import SeedTransferInfo


def test_parse_weight_info_keeps_backward_compatibility():
    assert parse_weight_info([1, 2, 4]) == (1, 2, 4, None)


def test_backend_initialization_rejects_transfer_engine_without_memory_registration(monkeypatch):
    yr_module = ModuleType("yr")
    datasystem_module = ModuleType("yr.datasystem")
    datasystem_module.TransferEngine = object
    datasystem_module.ErrorCode = SimpleNamespace(kNotReady=1)
    yr_module.datasystem = datasystem_module
    monkeypatch.setitem(sys.modules, "yr", yr_module)
    monkeypatch.setitem(sys.modules, "yr.datasystem", datasystem_module)

    with pytest.raises(ImportError, match="MemoryRegistration"):
        RForkTransferBackend()


def test_parse_weight_info_accepts_shape_metadata_from_json():
    assert parse_weight_info([1, 6, 2, [2, 3]]) == (1, 6, 2, (2, 3))


def test_parse_weight_info_rejects_invalid_shape_metadata():
    assert parse_weight_info([1, 6, 2, ["2", 3]]) is None
    assert parse_weight_info([1, 6, 2, -1]) is None


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
    assert parse_weight_info(weight_info) is None


def test_parse_weight_info_accepts_named_shape_and_dtype():
    parsed = parse_weight_info(
        {
            "ptr": 1,
            "numel": 6,
            "element_size": 2,
            "shape": [2, 3],
            "dtype": "torch.float16",
        }
    )
    assert parsed == (1, 6, 2, (2, 3), "float16")
    assert parse_weight_info([1, 6, 2, [2, 3], "bfloat16"]) == (1, 6, 2, (2, 3), "bfloat16")


def test_reshape_tensor_to_seed_shape_updates_tensor_metadata_only():
    tensor = torch.arange(6).reshape(2, 3)
    original_ptr = tensor.data_ptr()

    assert reshape_tensor_to_seed_shape("weight", tensor, (1, 2, 3))

    assert tuple(tensor.shape) == (1, 2, 3)
    assert tensor.data_ptr() == original_ptr


def test_reshape_tensor_to_seed_shape_rejects_numel_mismatch():
    tensor = torch.arange(6).reshape(2, 3)

    assert not reshape_tensor_to_seed_shape("weight", tensor, (2, 2))
    assert tuple(tensor.shape) == (2, 3)


def test_iter_transfer_chunks_splits_single_large_tensor():
    chunk_limit = transfer_backend.MAX_TRANSFER_CHUNK_BYTES
    chunks = list(
        iter_transfer_chunks(
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
        "collect_transferable_tensors",
        lambda model, processed_layout: [("weight", tensor)],
    )
    monkeypatch.setattr(
        transfer_backend,
        "fetch_seed_transfer_info",
        lambda *args: SeedTransferInfo(
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

    monkeypatch.setattr(transfer_backend, "collect_transferable_tensors", fail_if_rescanned)
    monkeypatch.setattr(
        transfer_backend,
        "fetch_seed_transfer_info",
        lambda *args: SeedTransferInfo(
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
        "fetch_seed_transfer_info",
        lambda *args: None,
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
        "fetch_seed_transfer_info",
        lambda *args: SeedTransferInfo("seed-session", remote_info, None),
    )

    assert not backend.recv_from_source(object(), "127.0.0.1", 8000, "seed-key", True)
    assert native_calls == []


def test_unregister_memory_region_releases_registered_tensor_owners():
    tensor = torch.arange(6).reshape(2, 3)
    backend = RForkTransferBackend.__new__(RForkTransferBackend)
    backend.rfork_transfer_engine = SimpleNamespace(
        batch_unregister_memory=lambda *args: SimpleNamespace(is_error=lambda: False)
    )
    backend.rfork_transfer_engine_weights_info_dict = {"weight": (tensor.data_ptr(), tensor.numel(), 1)}
    backend.rfork_transfer_engine_weights_shape_dict = {"weight": tuple(tensor.shape)}
    backend.registered_weight_blocks = [(tensor.data_ptr(), tensor.numel() * tensor.element_size())]
    backend.registered_memory_addresses = [tensor.data_ptr()]
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
    backend.registered_memory_addresses = [blocks[0][0]]
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


def test_register_memory_region_uses_logical_ranges_with_allocator_backing(monkeypatch):
    tensor = torch.arange(8, dtype=torch.uint8)
    logical = tensor[2:6]
    backing_start = tensor.data_ptr()
    backing_size = tensor.numel() * tensor.element_size()
    registrations = []

    class _MemoryRegistration:
        def __init__(self, logical_addr, logical_length, backing_addr, backing_length):
            self.values = (logical_addr, logical_length, backing_addr, backing_length)

    class _Ret:
        def is_error(self):
            return False

    backend = RForkTransferBackend.__new__(RForkTransferBackend)
    backend.rfork_transfer_engine = SimpleNamespace(
        batch_register_memory_ex=lambda items: registrations.extend(item.values for item in items) or _Ret()
    )
    backend._memory_registration_cls = _MemoryRegistration
    backend.registered_weight_blocks = []
    backend.registered_memory_addresses = []
    backend.rfork_transfer_engine_weights_info_dict = None
    backend.rfork_transfer_engine_weights_shape_dict = None
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


def test_extended_registration_merges_overlapping_logical_ranges(monkeypatch):
    tensor = torch.arange(8, dtype=torch.uint8)
    first = tensor[1:5]
    second = tensor[3:7]
    backing_start = tensor.data_ptr()
    registrations = []

    class _MemoryRegistration:
        def __init__(self, *values):
            self.values = values

    class _Ret:
        def is_error(self):
            return False

    backend = RForkTransferBackend.__new__(RForkTransferBackend)
    backend.rfork_transfer_engine = SimpleNamespace(
        batch_register_memory_ex=lambda items: registrations.extend(item.values for item in items) or _Ret()
    )
    backend._memory_registration_cls = _MemoryRegistration
    backend.registered_weight_blocks = []
    backend.registered_memory_addresses = []
    backend._registered_transferable_tensors = None
    monkeypatch.setattr(transfer_backend, "find_non_npu_state_tensors", lambda _model: [])
    monkeypatch.setattr(transfer_backend, "is_transferable_tensor", lambda _tensor: True)
    monkeypatch.setattr(
        transfer_backend,
        "collect_transferable_tensors",
        lambda *args: [("first", first), ("second", second)],
    )
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
                                "size": tensor.numel(),
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
    assert registrations == [(first.data_ptr(), 6, backing_start, tensor.numel())]


def test_register_memory_region_rejects_transfer_engine_without_extended_registration(monkeypatch):
    tensor = torch.arange(4, dtype=torch.uint8)
    calls = []

    class _Ret:
        def is_error(self):
            return False

    backend = RForkTransferBackend.__new__(RForkTransferBackend)
    backend.rfork_transfer_engine = SimpleNamespace(
        batch_register_memory=lambda addresses, sizes: calls.append((list(addresses), list(sizes))) or _Ret()
    )
    backend._memory_registration_cls = None
    backend.registered_weight_blocks = []
    backend.registered_memory_addresses = []
    backend._registered_transferable_tensors = None
    monkeypatch.setattr(transfer_backend, "find_non_npu_state_tensors", lambda _model: [])
    monkeypatch.setattr(transfer_backend, "is_transferable_tensor", lambda _tensor: True)
    monkeypatch.setattr(transfer_backend, "collect_transferable_tensors", lambda *args: [("weight", tensor)])
    monkeypatch.setattr(
        transfer_backend.torch,
        "npu",
        SimpleNamespace(
            memory=SimpleNamespace(
                memory_snapshot=lambda: [
                    {
                        "blocks": [
                            {
                                "address": tensor.data_ptr(),
                                "size": tensor.numel(),
                                "state": "active_allocated",
                            }
                        ]
                    }
                ]
            )
        ),
        raising=False,
    )

    assert not backend.register_memory_region(object(), False)
    assert calls == []
    assert backend.registered_weight_blocks == []
    assert backend.registered_memory_addresses == []


def test_unregister_uses_logical_registration_addresses():
    calls = []
    backend = RForkTransferBackend.__new__(RForkTransferBackend)
    backend.rfork_transfer_engine = SimpleNamespace(
        batch_unregister_memory=lambda addresses: calls.append(addresses) or SimpleNamespace(is_error=lambda: False)
    )
    backend.registered_weight_blocks = [(100, 1000)]
    backend.registered_memory_addresses = [120, 400]
    backend._registered_transferable_tensors = []

    assert backend.unregister_memory_region()
    assert calls == [[120, 400]]


def test_finalize_retries_not_ready_and_clears_state_only_after_success():
    attempts = []

    class _Ret:
        def __init__(self, error, code):
            self.error = error
            self.code = code

        def is_error(self):
            return self.error

        def get_code(self):
            return self.code

        def to_string(self):
            return self.code

    results = iter([_Ret(True, "not-ready"), _Ret(False, "ok")])
    backend = RForkTransferBackend.__new__(RForkTransferBackend)
    backend.rfork_transfer_engine = SimpleNamespace(finalize=lambda: attempts.append(True) or next(results))
    backend.rfork_transfer_engine_session_id = "session"
    backend.rfork_transfer_engine_weights_info_dict = {"weight": (1, 1, 1)}
    backend.rfork_transfer_engine_weights_shape_dict = {"weight": (1,)}
    backend.registered_weight_blocks = [(1, 1)]
    backend.registered_memory_addresses = [1]
    backend._registered_transferable_tensors = [("weight", torch.ones(1))]
    backend._not_ready_error_code = "not-ready"
    backend._is_initialized = True

    assert backend.finalize_transfer_engine()
    assert len(attempts) == 2
    assert backend._is_initialized is False
    assert backend.rfork_transfer_engine_session_id is None
    assert backend.registered_weight_blocks == []
    assert backend._registered_transferable_tensors is None


def test_finalize_failure_preserves_registered_tensor_owners():
    owners = [("weight", torch.ones(1))]

    class _Ret:
        def is_error(self):
            return True

        def get_code(self):
            return "fatal"

        def to_string(self):
            return "fatal"

    backend = RForkTransferBackend.__new__(RForkTransferBackend)
    backend.rfork_transfer_engine = SimpleNamespace(finalize=lambda: _Ret())
    backend.rfork_transfer_engine_session_id = "session"
    backend.registered_weight_blocks = [(1, 1)]
    backend.registered_memory_addresses = [1]
    backend._registered_transferable_tensors = owners
    backend._not_ready_error_code = "not-ready"
    backend._is_initialized = True

    assert not backend.finalize_transfer_engine()
    assert backend._is_initialized is True
    assert backend.registered_weight_blocks == [(1, 1)]
    assert backend._registered_transferable_tensors is owners


def test_finalize_is_idempotent_after_success():
    calls = []
    backend = RForkTransferBackend.__new__(RForkTransferBackend)
    backend.rfork_transfer_engine = SimpleNamespace(finalize=lambda: calls.append(True))
    backend._is_initialized = False

    assert backend.finalize_transfer_engine()
    assert calls == []


def test_transfer_lifecycle_lock_serializes_concurrent_calls():
    backend = RForkTransferBackend.__new__(RForkTransferBackend)
    backend._lifecycle_lock = threading.RLock()
    first_entered = threading.Event()
    release_first = threading.Event()
    second_done = threading.Event()
    calls = []

    def recv_locked(*args):
        calls.append(args)
        if len(calls) == 1:
            first_entered.set()
            assert release_first.wait(1)
        return True

    backend._recv_from_source_locked = recv_locked
    args = (object(), "127.0.0.1", 8000, "key", False)
    first = threading.Thread(target=backend.recv_from_source, args=args)
    second = threading.Thread(target=lambda: (backend.recv_from_source(*args), second_done.set()))

    first.start()
    assert first_entered.wait(1)
    second.start()
    assert not second_done.wait(0.05)
    assert len(calls) == 1
    release_first.set()
    first.join(1)
    second.join(1)

    assert not first.is_alive()
    assert not second.is_alive()
    assert len(calls) == 2


def test_register_memory_region_rejects_second_registration_without_overwriting(monkeypatch):
    tensor = torch.arange(4)
    old_blocks = [(123, 64)]
    old_info = {"old": (123, 4, 4)}
    old_owners = [("old", tensor)]
    backend = RForkTransferBackend.__new__(RForkTransferBackend)
    backend.rfork_transfer_engine = SimpleNamespace()
    backend.registered_weight_blocks = old_blocks
    backend.registered_memory_addresses = [123]
    backend.rfork_transfer_engine_weights_info_dict = old_info
    backend.rfork_transfer_engine_weights_shape_dict = {"old": (4,)}
    backend._registered_transferable_tensors = old_owners

    # A failed unregister keeps its state so the next attempt can retry it.
    monkeypatch.setattr(
        transfer_backend,
        "collect_transferable_tensors",
        lambda *args: pytest.fail("register_memory_region must not rescan before stale unregister succeeds"),
    )
    assert not backend.register_memory_region(object(), False)
    assert backend.registered_weight_blocks == old_blocks
    assert backend.rfork_transfer_engine_weights_info_dict is old_info
    assert backend._registered_transferable_tensors is old_owners


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

    monkeypatch.setattr(manifest_module, "is_transferable_tensor", lambda tensor: True)
    model = _Model()

    processed_names = {name for name, _ in collect_processed_layout_tensors(model)}
    checkpoint_names = {name for name, _ in collect_checkpoint_layout_tensors(model)}

    assert processed_names == {"weight", "buffer", "runtime_constant", "impl.runtime_weight"}
    assert checkpoint_names == {"weight", "buffer", "impl.runtime_weight"}


def test_fetch_seed_transfer_info_returns_none_for_non_200(monkeypatch):
    monkeypatch.setattr(
        seed_client_module.requests,
        "get",
        lambda *args, **kwargs: SimpleNamespace(status_code=503),
    )

    assert fetch_seed_transfer_info("http://seed", "seed-key", 10.0) is None


def test_remote_manifest_requests_use_timeout(monkeypatch):
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

    monkeypatch.setattr(seed_client_module.requests, "get", fake_get)
    assert fetch_seed_transfer_info("http://[::1]:8000", "seed-key", 2.5) == SeedTransferInfo(
        "session", {"w": [1, 1, 2]}, {"w": [1]}
    )
    assert len(calls) == 2
    assert all(call[1]["timeout"] == 2.5 for call in calls)


def test_backend_request_timeout_must_be_finite_and_positive():
    with pytest.raises(ValueError):
        transfer_backend._validate_request_timeout(0)
    with pytest.raises(ValueError):
        transfer_backend._validate_request_timeout(float("inf"))


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


def _make_register_memory_region_backend(
    monkeypatch,
    tensors,
    snapshot_blocks,
    stale_blocks=(),
    unregister_error=False,
):
    registrations = []
    unregistered_calls = []

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
    backend.rfork_transfer_engine = SimpleNamespace(
        batch_register_memory_ex=batch_register_memory_ex,
        batch_unregister_memory=batch_unregister_memory,
    )
    backend._memory_registration_cls = _MemoryRegistration
    backend.registered_weight_blocks = list(stale_blocks)
    backend.registered_memory_addresses = []
    backend.rfork_transfer_engine_weights_info_dict = None
    backend.rfork_transfer_engine_weights_shape_dict = None
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

    assert set(backend.rfork_transfer_engine_weights_info_dict) == {"layers.0.fc.weight"}
    assert [name for name, _ in backend._registered_transferable_tensors] == ["layers.0.fc.weight"]
    assert backend.excluded_weight_blocks == excluded_blocks
    assert registrations == [
        (
            own_weight.data_ptr(),
            own_weight.numel() * own_weight.element_size(),
            storage.data_ptr(),
            storage.numel() * storage.element_size(),
        )
    ]


def test_register_memory_region_registers_all_weights_without_exclusion(monkeypatch):
    storage = torch.arange(100, dtype=torch.float32)
    weight_a = storage[:10]
    weight_b = storage[20:32]
    backend, registrations, _ = _make_register_memory_region_backend(
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
    assert len(registrations) == 2
    assert backend.excluded_weight_blocks == []


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

    assert backend.rfork_transfer_engine_weights_info_dict == {}
    assert backend.rfork_transfer_engine_weights_shape_dict == {}
    assert backend._registered_transferable_tensors == []
    assert backend.registered_weight_blocks == []
    assert registrations == []


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
        "collect_transferable_tensors",
        lambda model, processed_layout: [
            ("model.embed_tokens.weight", shared_weight),
            ("layers.0.fc.weight", own_weight),
        ],
    )

    monkeypatch.setattr(
        transfer_backend,
        "fetch_seed_transfer_info",
        lambda *args: SeedTransferInfo(
            "seed-session",
            {"layers.0.fc.weight": [7, own_weight.numel(), own_weight.element_size()]},
            None,
        ),
    )

    assert backend.recv_from_source(object(), "127.0.0.1", 8000, "seed-key", True)
    assert backend.rfork_transfer_engine_weights_shape_dict == {"layers.0.fc.weight": tuple(own_weight.shape)}


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
        "fetch_seed_transfer_info",
        lambda *args: SeedTransferInfo("seed-session", {}, None),
    )

    assert not backend.recv_from_source(object(), "127.0.0.1", 8000, "seed-key", True)
