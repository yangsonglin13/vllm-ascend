# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

import ctypes
import threading
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from torch import nn

from tests.ut.model_loader.rfork.rfork_test_support import _stub


class _CallableImpl:
    def __init__(self, value):
        self.weights = value
        self.cycle = []
        self.cycle.append(self.cycle)

    def __call__(self, value):
        return value


def test_collector_keeps_logical_aliases_when_larger_view_is_seen_later(tensor_runtime, monkeypatch):
    storage = torch.arange(8, dtype=torch.float32)
    model = nn.Module()
    model.impl = _CallableImpl(storage[:2])
    model.impl.large = storage[:5]
    monkeypatch.setattr(tensor_runtime.tensor_layout, "is_transferable_tensor", lambda _tensor: True)

    collected = tensor_runtime.tensor_layout.collect_processed_layout_tensors(model)

    assert [(name, tensor.numel()) for name, tensor in collected] == [
        ("impl.weights", 2),
        ("impl.large", 5),
    ]
    assert collected[0][1].data_ptr() == collected[1][1].data_ptr()


def test_collector_deduplicates_exact_same_logical_tensor(tensor_runtime, monkeypatch):
    tensor = torch.arange(4, dtype=torch.float32)
    model = nn.Module()
    model.weight = nn.Parameter(tensor)
    monkeypatch.setattr(tensor_runtime.tensor_layout, "is_transferable_tensor", lambda _tensor: True)

    collected = tensor_runtime.tensor_layout.collect_processed_layout_tensors(model)

    assert [(name, value.data_ptr()) for name, value in collected] == [("weight", tensor.data_ptr())]


def test_collector_rejects_conflicting_same_name_view(tensor_runtime, monkeypatch):
    storage = torch.arange(8, dtype=torch.float32)
    small = storage[:2]
    large = storage[:5]
    collected: list[tuple[str, torch.Tensor]] = []
    seen_names: dict[str, int] = {}
    monkeypatch.setattr(tensor_runtime.tensor_layout, "is_transferable_tensor", lambda _tensor: True)

    with pytest.raises(ValueError, match="conflicting tensor entries"):
        tensor_runtime.tensor_layout._try_collect("weight", small, seen_names, collected)
        tensor_runtime.tensor_layout._try_collect("weight", large, seen_names, collected)


def test_collector_scans_custom_callable_impl_and_skips_function(tensor_runtime, monkeypatch):
    storage = torch.arange(4, dtype=torch.float32)
    model = nn.Module()
    model.impl = _CallableImpl(storage)

    def function_impl():
        return None

    function_impl.weights = storage
    model.impl.function = function_impl
    monkeypatch.setattr(tensor_runtime.tensor_layout, "is_transferable_tensor", lambda _tensor: True)

    collected = tensor_runtime.tensor_layout.collect_checkpoint_layout_tensors(model)
    names = [name for name, _ in collected]

    assert "impl.weights" in names
    assert all(not name.startswith("impl.function") for name in names)


def test_collector_accepts_dense_permutation_offset_and_singleton_views(tensor_runtime, monkeypatch):
    base = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    model = nn.Module()
    model.weight = nn.Parameter(base.t())
    model.register_buffer("offset", base.reshape(-1)[1:9])
    model.register_buffer("singleton", base[:1, :])
    monkeypatch.setattr(tensor_runtime.tensor_layout, "is_tensor_on_transfer_device", lambda _tensor: True)

    collected = tensor_runtime.tensor_layout.collect_processed_layout_tensors(model)

    assert [name for name, _ in collected] == ["weight", "offset", "singleton"]
    assert tensor_runtime.tensor_layout.is_non_overlapping_dense_tensor(base.t())
    assert tensor_runtime.tensor_layout.is_non_overlapping_dense_tensor(base.reshape(-1)[1:9])
    assert tensor_runtime.tensor_layout.is_non_overlapping_dense_tensor(base[:1, :])


@pytest.mark.parametrize("state_kind", ["parameter", "buffer"])
def test_collector_rejects_gapped_views_in_parameters_and_buffers(tensor_runtime, monkeypatch, state_kind):
    base = torch.arange(8, dtype=torch.float32)
    gapped = base[::2]
    model = nn.Module()
    if state_kind == "parameter":
        model.register_parameter("weight", nn.Parameter(gapped))
    else:
        model.register_buffer("buffer", gapped)
    monkeypatch.setattr(tensor_runtime.tensor_layout, "is_tensor_on_transfer_device", lambda _tensor: True)

    with pytest.raises(ValueError, match="gapped or overlapping storage"):
        tensor_runtime.tensor_layout.collect_processed_layout_tensors(model)


@pytest.mark.parametrize(
    "view_factory",
    [
        lambda base: base.expand(2, -1),
        lambda base: base.as_strided((2, 2), (1, 1)),
    ],
    ids=["broadcast", "overlap"],
)
def test_collector_rejects_overlapping_views(tensor_runtime, monkeypatch, view_factory):
    base = torch.arange(3, dtype=torch.float32)
    model = nn.Module()
    model.register_buffer("buffer", view_factory(base))
    monkeypatch.setattr(tensor_runtime.tensor_layout, "is_tensor_on_transfer_device", lambda _tensor: True)

    with pytest.raises(ValueError, match="gapped or overlapping storage"):
        tensor_runtime.tensor_layout.collect_processed_layout_tensors(model)


def test_registration_rejects_gapped_view_before_native_registration(tensor_runtime, monkeypatch):
    base = torch.arange(8, dtype=torch.float32)
    gapped = base[::2]
    registrations = []

    class _MemoryRegistration:
        def __init__(self, *values):
            self.values = values

    backend = tensor_runtime.RForkTransferBackend.__new__(tensor_runtime.RForkTransferBackend)
    backend.transfer_engine = SimpleNamespace(
        batch_register_memory_ex=lambda items: registrations.extend(item.values for item in items),
    )
    backend._memory_registration_cls = _MemoryRegistration
    backend.registered_weight_blocks = []
    backend.registered_memory_addresses = []
    backend._registered_transferable_tensors = None
    backend._registered_transferable_storages = None
    monkeypatch.setattr(tensor_runtime.transfer_backend, "find_non_npu_state_tensors", lambda _model: [])
    monkeypatch.setattr(tensor_runtime.transfer_backend, "is_transferable_tensor", lambda _tensor: True)
    monkeypatch.setattr(
        tensor_runtime.transfer_backend,
        "collect_transferable_tensors",
        lambda *_args: [("gapped", gapped)],
    )

    with pytest.raises(ValueError, match="gapped or overlapping storage"):
        backend.register_memory_region(object(), False)

    assert registrations == []


def test_read_rejects_gapped_cached_view_before_native_read(tensor_runtime):
    base = torch.arange(8, dtype=torch.float32)
    gapped = base[::2]
    reads = []
    backend = tensor_runtime.RForkTransferBackend.__new__(tensor_runtime.RForkTransferBackend)
    backend.transfer_engine = SimpleNamespace(batch_transfer_sync_read=lambda *args: reads.append(args))
    backend._registered_transferable_tensors = [("gapped", gapped)]
    backend.weight_shapes = {}
    seed_info = tensor_runtime.SeedTransferInfo(
        "seed-session",
        {"gapped": [1, gapped.numel(), gapped.element_size(), list(gapped.shape), "float32"]},
    )

    with pytest.raises(ValueError, match="gapped or overlapping storage"):
        backend.read_weights_from_seed(object(), seed_info, True)

    assert reads == []


def test_read_copies_dense_transpose_view_as_contiguous_bytes(tensor_runtime):
    source_storage = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    seed_view = source_storage.t()
    target_storage = torch.full_like(source_storage, -1)
    target_view = target_storage.t()
    read_lengths = []

    def read(_session_id, client_ptrs, seed_ptrs, lengths):
        read_lengths.extend(lengths)
        ctypes.memmove(client_ptrs[0], seed_ptrs[0], lengths[0])
        return SimpleNamespace(is_error=lambda: False)

    backend = tensor_runtime.RForkTransferBackend.__new__(tensor_runtime.RForkTransferBackend)
    backend.transfer_engine = SimpleNamespace(batch_transfer_sync_read=read)
    backend._registered_transferable_tensors = [("weight", target_view)]
    backend.weight_shapes = {"weight": tuple(target_view.shape)}
    backend.weight_manifest = {
        "weight": (
            target_view.data_ptr(),
            target_view.numel(),
            target_view.element_size(),
            tuple(target_view.shape),
            "float32",
        )
    }
    seed_info = tensor_runtime.SeedTransferInfo(
        "seed-session",
        {
            "weight": [
                seed_view.data_ptr(),
                seed_view.numel(),
                seed_view.element_size(),
                list(seed_view.shape),
                "float32",
            ]
        },
    )

    assert backend.read_weights_from_seed(object(), seed_info, True)

    assert read_lengths == [source_storage.numel() * source_storage.element_size()]
    assert torch.equal(target_storage, source_storage)


def test_registration_merges_small_then_large_alias_ranges(tensor_runtime, monkeypatch):
    storage = torch.arange(8, dtype=torch.float32)
    small = storage[:2]
    large = storage[:5]
    registrations = []

    class _MemoryRegistration:
        def __init__(self, *values):
            self.values = values

    class _Result:
        def is_error(self):
            return False

    backend = tensor_runtime.RForkTransferBackend.__new__(tensor_runtime.RForkTransferBackend)
    backend.transfer_engine = SimpleNamespace(
        batch_register_memory_ex=lambda items: registrations.extend(item.values for item in items) or _Result()
    )
    backend._memory_registration_cls = _MemoryRegistration
    backend.registered_weight_blocks = []
    backend.registered_memory_addresses = []
    backend._registered_transferable_tensors = None
    backend._registered_transferable_storages = None

    monkeypatch.setattr(tensor_runtime.transfer_backend, "find_non_npu_state_tensors", lambda _model: [])
    monkeypatch.setattr(tensor_runtime.transfer_backend, "is_transferable_tensor", lambda _tensor: True)
    monkeypatch.setattr(
        tensor_runtime.transfer_backend,
        "collect_transferable_tensors",
        lambda *_args: [("small", small), ("large", large)],
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
                                "address": storage.data_ptr(),
                                "size": storage.numel() * storage.element_size(),
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
    assert set(backend.weight_manifest) == {"small", "large"}
    assert backend.weight_manifest["small"][3] == (2,)
    assert backend.weight_manifest["large"][3] == (5,)
    assert registrations == [
        (
            small.data_ptr(),
            large.numel() * large.element_size(),
            storage.data_ptr(),
            storage.numel() * storage.element_size(),
        )
    ]


@pytest.mark.parametrize("successful_batches", [0, 1])
def test_registration_failure_tracks_failed_batch_and_storage_owners(tensor_runtime, monkeypatch, successful_batches):
    first = torch.arange(2, dtype=torch.float32)
    second = torch.arange(3, dtype=torch.float32)
    tensors = [("first", first), ("second", second)]
    registered_calls = []

    class _MemoryRegistration:
        def __init__(self, *values):
            self.values = values

    class _Result:
        def __init__(self, error, message):
            self.error = error
            self.message = message

        def is_error(self):
            return self.error

        def to_string(self):
            return self.message

    register_results = iter([_Result(False, "ok")] * successful_batches + [_Result(True, "partial failure")])

    def batch_register(items):
        registered_calls.append([item.values for item in items])
        return next(register_results)

    def batch_unregister(addresses):
        return _Result(True, "rollback unavailable")

    backend = tensor_runtime.RForkTransferBackend.__new__(tensor_runtime.RForkTransferBackend)
    backend.transfer_engine = SimpleNamespace(
        batch_register_memory_ex=batch_register,
        batch_unregister_memory=batch_unregister,
    )
    backend._memory_registration_cls = _MemoryRegistration
    backend.registered_weight_blocks = []
    backend.registered_memory_addresses = []
    backend._registered_transferable_tensors = None
    backend._registered_transferable_storages = None
    backend.weight_manifest = None
    backend.weight_shapes = None

    monkeypatch.setattr(tensor_runtime.transfer_backend, "find_non_npu_state_tensors", lambda _model: [])
    monkeypatch.setattr(tensor_runtime.transfer_backend, "is_transferable_tensor", lambda _tensor: True)
    monkeypatch.setattr(tensor_runtime.transfer_backend, "collect_transferable_tensors", lambda *_args: tensors)
    monkeypatch.setattr(tensor_runtime.transfer_backend, "MAX_MEMORY_REGISTRATION_BATCH_ITEMS", 1)
    monkeypatch.setattr(
        tensor_runtime.transfer_backend.torch,
        "npu",
        SimpleNamespace(
            memory=SimpleNamespace(
                memory_snapshot=lambda: [
                    {
                        "blocks": [
                            {
                                "address": first.data_ptr(),
                                "size": first.numel() * first.element_size(),
                                "state": "active_allocated",
                            },
                            {
                                "address": second.data_ptr(),
                                "size": second.numel() * second.element_size(),
                                "state": "active_allocated",
                            },
                        ]
                    }
                ]
            )
        ),
        raising=False,
    )

    assert not backend.register_memory_region(object(), False)
    assert len(registered_calls) == successful_batches + 1
    assert backend.registered_memory_addresses == [entry[0] for batch in registered_calls for entry in batch]
    assert backend._registered_transferable_tensors == tensors
    assert backend._registered_transferable_storages is not None
    assert all(owner is not None for owner in backend._registered_transferable_storages)
    first_storage_ptr = backend._registered_transferable_storages[0].data_ptr()
    first.data = torch.zeros_like(first)
    assert backend._registered_transferable_storages[0].data_ptr() == first_storage_ptr


def test_unregister_treats_explicit_not_found_as_idempotent(tensor_runtime, monkeypatch):
    not_found = object()
    calls = []

    class _Result:
        def __init__(self, error, code):
            self.error = error
            self.code = code

        def is_error(self):
            return self.error

        def get_code(self):
            return self.code

        def to_string(self):
            return "not found" if self.code is not None else "ok"

    def unregister(addresses):
        calls.append(list(addresses))
        if len(addresses) > 1:
            return _Result(True, not_found)
        if addresses == [10]:
            return _Result(True, not_found)
        return _Result(False, None)

    backend = tensor_runtime.RForkTransferBackend.__new__(tensor_runtime.RForkTransferBackend)
    backend.transfer_engine = SimpleNamespace(batch_unregister_memory=unregister)
    backend.registered_weight_blocks = [(10, 4), (20, 4)]
    backend.registered_memory_addresses = [10, 20]
    backend._registered_transferable_tensors = [("weight", torch.ones(1))]
    backend._not_found_error_code = not_found

    assert backend.unregister_memory_region()
    assert calls == [[10, 20], [10], [20]]
    assert backend.registered_memory_addresses == []
    assert backend._registered_transferable_tensors is None


def test_empty_transfer_succeeds_only_for_registered_all_shared_layout(tensor_runtime):
    backend = tensor_runtime.RForkTransferBackend.__new__(tensor_runtime.RForkTransferBackend)
    backend._registered_transferable_tensors = []
    backend._all_transferable_tensors_excluded = True
    backend.weight_manifest = {}
    backend.transfer_engine = SimpleNamespace()

    assert backend.read_weights_from_seed(object(), tensor_runtime.SeedTransferInfo("seed", {}, {}), True)
    assert not backend.read_weights_from_seed(
        object(), tensor_runtime.SeedTransferInfo("seed", {"weight": [1, 1, 1]}), True
    )
    backend._all_transferable_tensors_excluded = False
    assert not backend.read_weights_from_seed(object(), tensor_runtime.SeedTransferInfo("seed", {}, {}), True)


def test_unregister_does_not_infer_absence_from_error_text(tensor_runtime):
    backend = tensor_runtime.RForkTransferBackend.__new__(tensor_runtime.RForkTransferBackend)
    backend._not_found_error_code = object()
    backend.transfer_engine = SimpleNamespace(
        batch_unregister_memory=lambda addresses: SimpleNamespace(
            is_error=lambda: True, get_code=lambda: object(), to_string=lambda: "device context not found"
        )
    )
    backend.registered_weight_blocks = [(10, 4)]
    backend.registered_memory_addresses = [10]
    owners = [("weight", torch.ones(1))]
    backend._registered_transferable_tensors = owners
    assert not backend.unregister_memory_region()
    assert backend._registered_transferable_tensors is owners
    assert backend.registered_memory_addresses == [10]


def test_shared_reuse_requires_all_tensor_bytes_and_nonempty_state(tensor_runtime, monkeypatch):
    backend = tensor_runtime.RForkTransferBackend.__new__(tensor_runtime.RForkTransferBackend)
    weight = torch.arange(8, dtype=torch.float32)
    monkeypatch.setattr(tensor_runtime.transfer_backend, "find_non_npu_state_tensors", lambda model: [])
    monkeypatch.setattr(tensor_runtime.transfer_backend, "collect_transferable_tensors", lambda *args: [("w", weight)])
    address, size = weight.data_ptr(), weight.numel() * weight.element_size()
    assert backend.can_reuse_shared_weights(object(), False, [(address, size)])
    assert not backend.can_reuse_shared_weights(object(), False, [(address, size - 1)])
    monkeypatch.setattr(tensor_runtime.transfer_backend, "collect_transferable_tensors", lambda *args: [])
    assert not backend.can_reuse_shared_weights(object(), False, [(address, size)])


def test_shared_weights_and_empty_cleanup_do_not_create_engine(tensor_runtime, monkeypatch):
    create_engine = Mock(side_effect=AssertionError("idle backend must not initialize"))
    monkeypatch.setattr(tensor_runtime.RForkTransferBackend, "_initialize_transfer_engine", create_engine)
    backend = tensor_runtime.RForkTransferBackend()
    weight = torch.ones(8)
    monkeypatch.setattr(tensor_runtime.transfer_backend, "find_non_npu_state_tensors", lambda model: [])
    monkeypatch.setattr(tensor_runtime.transfer_backend, "collect_transferable_tensors", lambda *args: [("w", weight)])
    assert backend.can_reuse_shared_weights(
        object(), True, [(weight.data_ptr(), weight.numel() * weight.element_size())]
    )
    assert backend.unregister_memory_region()
    assert backend.finalize_transfer_engine()
    assert not backend.is_initialized()
    assert backend.transfer_engine is None
    create_engine.assert_not_called()


def test_idle_session_fallback_and_shutdown_do_not_initialize_engine(request, tensor_runtime, monkeypatch):
    session_runtime = request.getfixturevalue("runtime")
    create_engine = Mock(side_effect=AssertionError("idle session must not initialize"))
    monkeypatch.setattr(tensor_runtime.RForkTransferBackend, "_initialize_transfer_engine", create_engine)
    monkeypatch.setattr(session_runtime.session, "RForkTransferBackend", tensor_runtime.RForkTransferBackend)
    session = session_runtime.session.RForkSession(session_runtime.config, session_runtime.identity)
    assert session.prepare_for_fallback().memory_reset
    assert session.shutdown()
    create_engine.assert_not_called()


def test_first_registration_initializes_once_on_caller_and_reuses_engine(tensor_runtime, monkeypatch):
    ok = SimpleNamespace(is_error=lambda: False)
    engine = Mock()
    initialized_on = []
    engine.initialize.side_effect = lambda *args: initialized_on.append(threading.get_ident()) or ok
    engine.batch_register_memory_ex.return_value = ok
    engine.batch_unregister_memory.return_value = ok
    engine.finalize.return_value = ok
    factory = Mock(return_value=engine)
    _stub(monkeypatch, "yr", __path__=[])
    _stub(
        monkeypatch,
        "yr.datasystem",
        TransferEngine=factory,
        MemoryRegistration=lambda *args: args,
        ErrorCode=SimpleNamespace(kNotReady=1, kNotFound=2),
    )
    weight = torch.ones(8)
    monkeypatch.setattr(tensor_runtime.transfer_backend, "find_non_npu_state_tensors", lambda model: [])
    monkeypatch.setattr(tensor_runtime.transfer_backend, "collect_transferable_tensors", lambda *args: [("w", weight)])
    monkeypatch.setattr(tensor_runtime.transfer_backend, "is_transferable_tensor", lambda tensor: True)
    monkeypatch.setattr(
        torch,
        "npu",
        SimpleNamespace(
            current_device=lambda: 3,
            memory=SimpleNamespace(
                memory_snapshot=lambda: [
                    {
                        "blocks": [
                            {
                                "address": weight.data_ptr(),
                                "size": weight.numel() * weight.element_size(),
                                "state": "active_allocated",
                            }
                        ]
                    }
                ]
            ),
        ),
        raising=False,
    )
    backend = tensor_runtime.RForkTransferBackend()
    factory.assert_not_called()
    assert backend.register_memory_region(object(), True)
    assert backend.is_initialized()
    assert initialized_on == [threading.get_ident()]
    engine.initialize.assert_called_once_with("127.0.0.1:12345", "ascend", "npu:3")
    assert backend.unregister_memory_region()
    assert backend.register_memory_region(object(), True)
    factory.assert_called_once()
    assert backend.finalize_transfer_engine()
    assert backend.transfer_engine is None
    assert backend.unregister_memory_region()
    assert backend.finalize_transfer_engine()
    engine.finalize.assert_called_once()
    factory.assert_called_once()


def test_failed_lazy_initialization_leaves_empty_cleanup_safe(tensor_runtime, monkeypatch):
    _stub(monkeypatch, "yr", __path__=[])
    _stub(monkeypatch, "yr.datasystem")
    backend = tensor_runtime.RForkTransferBackend()
    with pytest.raises(ImportError, match="MemoryRegistration"):
        backend.register_memory_region(object(), True)
    assert not backend.is_initialized()
    assert backend.unregister_memory_region()
    assert backend.finalize_transfer_engine()
