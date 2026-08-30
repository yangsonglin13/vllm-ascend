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

import math
import threading
import time
from bisect import bisect_left
from collections.abc import Mapping
from typing import Any

import requests
import torch
from torch import nn
from vllm.logger import logger
from vllm.utils.network_utils import get_ip, get_open_port, join_host_port

MAX_TRANSFER_CHUNK_BYTES = 1024**3
MAX_TRANSFER_CHUNK_WEIGHTS = 512
DEFAULT_REQUEST_TIMEOUT_SEC = 10.0


def _validate_request_timeout(request_timeout_sec: Any) -> float:
    """Return a finite, positive timeout suitable for every RFork HTTP call."""
    if isinstance(request_timeout_sec, bool) or not isinstance(request_timeout_sec, (int, float)):
        raise ValueError("RFork request_timeout_sec must be a positive finite number")
    timeout = float(request_timeout_sec)
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("RFork request_timeout_sec must be a positive finite number")
    return timeout


def _auth_headers(auth_token: str | None) -> dict[str, str]:
    if isinstance(auth_token, str) and auth_token:
        return {"X-RFORK-TOKEN": auth_token}
    return {}


def _normalize_dtype_name(dtype: Any) -> str | None:
    """Normalize torch and JSON dtype spellings to a comparable name."""
    if dtype is None:
        return None
    if isinstance(dtype, torch.dtype):
        dtype_name = str(dtype)
    elif isinstance(dtype, str):
        dtype_name = dtype.strip()
    else:
        return None
    if dtype_name.startswith("torch."):
        dtype_name = dtype_name[6:]
    return dtype_name.lower() if dtype_name else None


def _is_positive_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _normalize_weight_shape(shape: Any) -> tuple[int, ...] | None:
    if shape is None:
        return None
    if not isinstance(shape, (list, tuple)):
        return None
    if not all(isinstance(dim, int) and not isinstance(dim, bool) and dim >= 0 for dim in shape):
        return None
    return tuple(shape)


def _parse_weight_info(weight_info: Any):
    """Parse a legacy or v2 RFork manifest entry.

    The original wire format is ``[ptr, numel, element_size]`` with an
    optional fourth shape item.  RFork v2 additionally permits a fifth dtype
    item, or a JSON object with named fields.  Legacy entries retain their
    four-item return shape so existing callers remain source compatible;
    v2 entries return ``(ptr, numel, element_size, shape, dtype)``.
    """
    is_v2 = False
    seed_shape = None
    seed_dtype = None

    if isinstance(weight_info, (list, tuple)):
        if len(weight_info) not in (3, 4, 5):
            return None
        seed_ptr, seed_len, seed_size = weight_info[:3]
        if len(weight_info) >= 4:
            shape_or_metadata = weight_info[3]
            if isinstance(shape_or_metadata, Mapping):
                is_v2 = True
                seed_shape = shape_or_metadata.get("shape")
                seed_dtype = shape_or_metadata.get("dtype")
            else:
                if shape_or_metadata is None:
                    return None
                seed_shape = shape_or_metadata
            if len(weight_info) == 5:
                is_v2 = True
                seed_dtype = weight_info[4]
    elif isinstance(weight_info, Mapping):
        is_v2 = True

        def _get_first(*keys: str):
            for key in keys:
                if key in weight_info:
                    return weight_info[key]
            return None

        seed_ptr = _get_first("ptr", "pointer", "seed_ptr", "address")
        seed_len = _get_first("numel", "count", "seed_len", "length")
        seed_size = _get_first("element_size", "itemsize", "seed_size", "size")
        seed_shape = weight_info.get("shape")
        seed_dtype = weight_info.get("dtype")
    else:
        return None

    if not all(_is_positive_int(value) for value in (seed_ptr, seed_len, seed_size)):
        return None

    if seed_shape is not None:
        seed_shape = _normalize_weight_shape(seed_shape)
        if seed_shape is None or _numel_from_shape(seed_shape) != seed_len:
            return None

    if seed_dtype is not None:
        seed_dtype = _normalize_dtype_name(seed_dtype)
        if seed_dtype is None:
            return None

    if is_v2:
        return seed_ptr, seed_len, seed_size, seed_shape, seed_dtype
    return seed_ptr, seed_len, seed_size, seed_shape


def _reshape_tensor_to_seed_shape(
    name: str,
    tensor: torch.Tensor,
    seed_shape: tuple[int, ...] | None,
    reshape_events: list[tuple[str, tuple[int, ...], tuple[int, ...]]] | None = None,
) -> bool:
    if seed_shape is None or tuple(tensor.shape) == seed_shape:
        return True

    if tensor.numel() != _numel_from_shape(seed_shape):
        logger.error(
            "Weight shape mismatch for %s, local shape %s cannot view as seed shape %s",
            name,
            tuple(tensor.shape),
            seed_shape,
        )
        return False

    local_shape = tuple(tensor.shape)
    try:
        tensor.data = tensor.data.view(seed_shape)
    except Exception as e:
        logger.error(
            "Failed to reshape RFork tensor %s from %s to seed shape %s: %s",
            name,
            local_shape,
            seed_shape,
            e,
        )
        return False

    if reshape_events is not None:
        reshape_events.append((name, local_shape, seed_shape))
    return True


def _update_registered_weight_shape(
    weight_shape_dict: dict[str, tuple[int, ...]] | None,
    name: str,
    tensor: torch.Tensor,
) -> None:
    if isinstance(weight_shape_dict, dict):
        weight_shape_dict[name] = tuple(tensor.shape)


def _update_registered_weight_info(
    weight_info_dict: dict[str, Any] | None,
    name: str,
    tensor: torch.Tensor,
) -> None:
    """Keep the v2 inline manifest synchronized after metadata-only reshape."""

    if isinstance(weight_info_dict, dict) and name in weight_info_dict:
        weight_info_dict[name] = (
            tensor.data_ptr(),
            tensor.numel(),
            tensor.element_size(),
            tuple(tensor.shape),
            _normalize_dtype_name(tensor.dtype),
        )


def _numel_from_shape(shape: tuple[int, ...]) -> int:
    numel = 1
    for dim in shape:
        numel *= dim
    return numel


def _is_transferable_tensor(tensor: torch.Tensor) -> bool:
    return not tensor.is_meta and tensor.numel() > 0 and _is_tensor_on_transfer_device(tensor)


def _is_tensor_on_transfer_device(tensor: torch.Tensor) -> bool:
    return tensor.device.type == "npu"


def _iter_tensors_in_value(prefix: str, value: Any, visited_object_ids: set[int], scan_objects: bool = False):
    if isinstance(value, torch.Tensor):
        yield prefix, value
        return

    if isinstance(value, (nn.Module, str, bytes)) or callable(value):
        return

    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            yield from _iter_tensors_in_value(f"{prefix}.{index}", item, visited_object_ids, scan_objects)
        return

    if isinstance(value, dict):
        for key, item in value.items():
            yield from _iter_tensors_in_value(f"{prefix}.{key}", item, visited_object_ids, scan_objects)
        return

    if not scan_objects or not hasattr(value, "__dict__"):
        return

    value_id = id(value)
    if value_id in visited_object_ids:
        return
    visited_object_ids.add(value_id)
    for attr_name, attr_value in vars(value).items():
        if attr_name.startswith("_"):
            continue
        yield from _iter_tensors_in_value(f"{prefix}.{attr_name}", attr_value, visited_object_ids, scan_objects)


def _try_collect_transferable_tensor(
    name: str,
    tensor: torch.Tensor,
    seen_data_ptrs: set[int],
    collected_tensors: list[tuple[str, torch.Tensor]],
) -> tuple[bool, bool]:
    if not _is_transferable_tensor(tensor):
        return False, False

    data_ptr = tensor.data_ptr()
    if data_ptr in seen_data_ptrs:
        return False, True

    seen_data_ptrs.add(data_ptr)
    collected_tensors.append((name, tensor))
    return True, False


def _collect_processed_layout_tensors(model: nn.Module) -> list[tuple[str, torch.Tensor]]:
    seen_data_ptrs: set[int] = set()
    collected_tensors: list[tuple[str, torch.Tensor]] = []

    for name, tensor in model.named_parameters():
        _try_collect_transferable_tensor(
            name,
            tensor,
            seen_data_ptrs,
            collected_tensors,
        )

    for name, tensor in model.named_buffers():
        _try_collect_transferable_tensor(
            name,
            tensor,
            seen_data_ptrs,
            collected_tensors,
        )

    # Post-load processing can replace checkpoint params with runtime tensors stored as direct attrs or inside `impl`.
    for module_prefix, module in model.named_modules():
        for attr_name, attr_value in vars(module).items():
            if attr_name.startswith("_") or isinstance(attr_value, nn.Module):
                continue

            scan_objects = attr_name == "impl"
            for tensor_name, tensor in _iter_tensors_in_value(attr_name, attr_value, set(), scan_objects):
                full_name = f"{module_prefix}.{tensor_name}" if module_prefix else tensor_name
                _try_collect_transferable_tensor(
                    full_name,
                    tensor,
                    seen_data_ptrs,
                    collected_tensors,
                )
    return collected_tensors


def _collect_checkpoint_layout_tensors(model: nn.Module) -> list[tuple[str, torch.Tensor]]:
    seen_data_ptrs: set[int] = set()
    collected_tensors: list[tuple[str, torch.Tensor]] = []

    for name, tensor in model.named_parameters():
        _try_collect_transferable_tensor(
            name,
            tensor,
            seen_data_ptrs,
            collected_tensors,
        )

    for name, tensor in model.named_buffers():
        _try_collect_transferable_tensor(
            name,
            tensor,
            seen_data_ptrs,
            collected_tensors,
        )

    # Before post-load processing, only impl-stored runtime tensors supplement params/buffers.
    for module_prefix, module in model.named_modules():
        impl = getattr(module, "impl", None)
        if impl is None or isinstance(impl, nn.Module):
            continue

        for tensor_name, tensor in _iter_tensors_in_value("impl", impl, set(), scan_objects=True):
            full_name = f"{module_prefix}.{tensor_name}" if module_prefix else tensor_name
            _try_collect_transferable_tensor(
                full_name,
                tensor,
                seen_data_ptrs,
                collected_tensors,
            )
    return collected_tensors


def _iter_transferable_tensors(model: nn.Module, processed_layout: bool):
    if processed_layout:
        yield from _collect_processed_layout_tensors(model)
    else:
        yield from _collect_checkpoint_layout_tensors(model)


def _find_non_npu_state_tensors(model: Any) -> list[str]:
    """Return materialized parameters/buffers RFork cannot safely transfer.

    Silently omitting CPU-offloaded model state could make a partial transfer
    look successful. Falling back is safer until RFork has an explicit mixed-
    device manifest protocol.
    """

    if not isinstance(model, nn.Module):
        return []
    non_npu_names: list[str] = []
    for iterator in (model.named_parameters(), model.named_buffers()):
        for name, tensor in iterator:
            if not tensor.is_meta and tensor.numel() > 0 and not _is_tensor_on_transfer_device(tensor):
                non_npu_names.append(name)
    return non_npu_names


def _block_contains_weight_ptr(address: int, size: int, sorted_weight_ptrs: list[int]) -> bool:
    index = bisect_left(sorted_weight_ptrs, address)
    return index < len(sorted_weight_ptrs) and sorted_weight_ptrs[index] < address + size


def _iter_transfer_chunks(
    weight_names: list[str],
    seed_ptr_list: list[int],
    client_ptr_list: list[int],
    client_len_list: list[int],
):
    """Yield native transfer batches bounded by bytes and pointer segments.

    ``batch_transfer_sync_read`` accepts one pointer/length per segment.  A
    single tensor can be larger than the native one-gigabyte limit, so split
    both source and destination pointers into bounded segments before packing
    batches.  Repeating the tensor name for each segment keeps diagnostics
    useful without changing the native API.
    """
    if not (len(weight_names) == len(seed_ptr_list) == len(client_ptr_list) == len(client_len_list)):
        raise ValueError("RFork transfer lists must have equal lengths")

    segment_names: list[str] = []
    segment_seed_ptrs: list[int] = []
    segment_client_ptrs: list[int] = []
    segment_lengths: list[int] = []
    for name, seed_ptr, client_ptr, length in zip(
        weight_names,
        seed_ptr_list,
        client_ptr_list,
        client_len_list,
        strict=True,
    ):
        if not _is_positive_int(length):
            raise ValueError("RFork transfer segment length must be a positive integer")
        offset = 0
        while offset < length:
            segment_length = min(MAX_TRANSFER_CHUNK_BYTES, length - offset)
            segment_names.append(name)
            segment_seed_ptrs.append(seed_ptr + offset)
            segment_client_ptrs.append(client_ptr + offset)
            segment_lengths.append(segment_length)
            offset += segment_length

    chunk_start = 0
    chunk_bytes = 0
    chunk_weights = 0

    for index, length in enumerate(segment_lengths):
        should_flush = chunk_weights > 0 and (
            chunk_bytes + length > MAX_TRANSFER_CHUNK_BYTES or chunk_weights >= MAX_TRANSFER_CHUNK_WEIGHTS
        )
        if should_flush:
            yield (
                segment_names[chunk_start:index],
                segment_seed_ptrs[chunk_start:index],
                segment_client_ptrs[chunk_start:index],
                segment_lengths[chunk_start:index],
            )
            chunk_start = index
            chunk_bytes = 0
            chunk_weights = 0

        chunk_bytes += length
        chunk_weights += 1

    if chunk_weights > 0:
        yield (
            segment_names[chunk_start:],
            segment_seed_ptrs[chunk_start:],
            segment_client_ptrs[chunk_start:],
            segment_lengths[chunk_start:],
        )


def _unpack_weight_info(parsed_weight_info: tuple[Any, ...]) -> tuple[int, int, int, Any, str | None]:
    """Normalize legacy four-item and v2 five-item parser results."""
    if len(parsed_weight_info) == 4:
        seed_ptr, seed_len, seed_size, seed_shape = parsed_weight_info
        return seed_ptr, seed_len, seed_size, seed_shape, None
    return parsed_weight_info


def _extract_manifest_entries(manifest_metadata: Any) -> Mapping[str, Any] | None:
    """Extract a per-tensor metadata mapping from common v2 envelopes."""
    if not isinstance(manifest_metadata, Mapping):
        return None
    for key in ("weights", "tensors", "manifest", "entries", "weight_info"):
        entries = manifest_metadata.get(key)
        if isinstance(entries, Mapping):
            return entries

    metadata_keys = {
        "tensor_count",
        "num_tensors",
        "count",
        "total_bytes",
        "byte_count",
        "bytes",
        "version",
        "protocol_version",
    }
    entries = {key: value for key, value in manifest_metadata.items() if key not in metadata_keys}
    return entries if entries and all(isinstance(key, str) for key in entries) else None


def _get_manifest_value(metadata: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in metadata:
            return metadata[key]
    return None


def _parse_manifest_scalar(value: Any, *, allow_zero: bool = False) -> int | None:
    if not isinstance(value, int) or isinstance(value, bool):
        return None
    if value < 0 or (value == 0 and not allow_zero):
        return None
    return value


class RForkTransferBackend:
    def __init__(self, request_timeout_sec: float = DEFAULT_REQUEST_TIMEOUT_SEC, auth_token: str = ""):
        self.rfork_transfer_engine: Any | None = None
        self.rfork_transfer_engine_session_id = None
        self.rfork_transfer_engine_weights_info_dict = None
        self.rfork_transfer_engine_weights_shape_dict = None
        self.registered_weight_blocks = []
        self.registered_memory_addresses = []
        self._registered_transferable_tensors: list[tuple[str, torch.Tensor]] | None = None
        self._memory_registration_cls: Any | None = None
        self._lifecycle_lock = threading.RLock()
        self.request_timeout_sec = _validate_request_timeout(request_timeout_sec)
        self.auth_token = auth_token if isinstance(auth_token, str) else ""
        self._is_initialized = False
        self.init_transfer_engine()

    def init_transfer_engine(self):
        try:
            from yr.datasystem import (  # type: ignore[import-not-found]
                ErrorCode,
                MemoryRegistration,
                TransferEngine,
            )
        except ImportError as e:
            err_msg = (
                "Failed to import the required TransferEngine, MemoryRegistration, and ErrorCode APIs from "
                "yr.datasystem. Install a YuanRong TransferEngine release that supports RFork extended memory "
                "registration and explicit finalization."
            )
            logger.error(err_msg)
            raise ImportError(err_msg) from e

        transfer_engine = TransferEngine()
        local_hostname = join_host_port(get_ip(), get_open_port())
        ret = transfer_engine.initialize(local_hostname, "ascend", f"npu:{torch.npu.current_device()}")
        if ret.is_error():
            err_msg = (
                f"TransferEngine initialization failed: "
                f"initialize({local_hostname}, 'ascend', "
                f"'npu:{int(torch.npu.current_device())}') -> {ret.to_string()}"
            )
            logger.error(err_msg)
            raise RuntimeError(err_msg)

        self.rfork_transfer_engine = transfer_engine
        self.rfork_transfer_engine_session_id = local_hostname
        self._memory_registration_cls = MemoryRegistration
        self._not_ready_error_code = ErrorCode.kNotReady
        self._is_initialized = True

    def is_initialized(self) -> bool:
        return self._is_initialized

    def _get_transfer_engine(self) -> Any:
        if self.rfork_transfer_engine is None:
            raise RuntimeError("TransferEngine is not initialized.")
        return self.rfork_transfer_engine

    def _get_lifecycle_lock(self) -> threading.RLock:
        lock = getattr(self, "_lifecycle_lock", None)
        if lock is None:
            lock = threading.RLock()
            self._lifecycle_lock = lock
        return lock

    def _clear_registration_state(self) -> None:
        self.rfork_transfer_engine_weights_info_dict = None
        self.rfork_transfer_engine_weights_shape_dict = None
        self.registered_weight_blocks = []
        self.registered_memory_addresses = []
        self._registered_transferable_tensors = None

    def register_memory_region(self, model, processed_layout: bool):
        with self._get_lifecycle_lock():
            return self._register_memory_region_locked(model, processed_layout)

    def _register_memory_region_locked(self, model, processed_layout: bool):
        # A failed unregister deliberately leaves this state intact so it can
        # be retried.  Never replace it with a new registration.
        if getattr(self, "registered_weight_blocks", None):
            logger.error("RFork memory is already registered; unregister it before registering again.")
            return False

        transfer_engine = self._get_transfer_engine()
        start_reg_mr_time = time.perf_counter()

        non_npu_state = _find_non_npu_state_tensors(model)
        if non_npu_state:
            logger.error(
                "RFork does not support mixed-device model state; non-NPU tensors include: %s",
                non_npu_state[:10],
            )
            return False

        weight_mr_dict = {}
        weight_shape_dict = {}
        transferable_tensors = list(_iter_transferable_tensors(model, processed_layout))
        if not transferable_tensors:
            logger.error("RFork refuses to register an empty transferable tensor manifest.")
            return False

        transferable_names = [name for name, _ in transferable_tensors]
        if any(not isinstance(name, str) or not name for name in transferable_names) or len(transferable_names) != len(
            set(transferable_names)
        ):
            logger.error("RFork refuses a manifest with duplicate or empty tensor names.")
            return False

        for name, weight in transferable_tensors:
            if not _is_transferable_tensor(weight):
                logger.error("RFork found an invalid transferable tensor entry: %r", name)
                return False
            weight_ptr = weight.data_ptr()
            weight_numel = weight.numel()
            weight_size = weight.element_size()
            if not all(_is_positive_int(value) for value in (weight_ptr, weight_numel, weight_size)):
                logger.error("RFork found an invalid tensor manifest entry for %s", name)
                return False
            weight_mr_dict[name] = (
                weight_ptr,
                weight_numel,
                weight_size,
                tuple(weight.shape),
                _normalize_dtype_name(weight.dtype),
            )
            weight_shape_dict[name] = tuple(weight.shape)

        try:
            memory_snapshot = torch.npu.memory.memory_snapshot()
        except Exception as e:
            logger.error("Failed to snapshot NPU memory for RFork registration: %s", e)
            return False

        # Include every active allocator block that overlaps a tensor.  This
        # handles tensors spanning adjacent allocator blocks, then merge only
        # contiguous ranges so the registered range covers every byte.
        tensor_ranges = [
            (weight.data_ptr(), weight.data_ptr() + weight.numel() * weight.element_size())
            for _, weight in transferable_tensors
        ]
        active_blocks: list[tuple[int, int]] = []
        for segment in memory_snapshot:
            for block in segment.get("blocks", []):
                address = block.get("address", -1)
                size = block.get("size", -1)
                state = block.get("state", "")
                if not _is_positive_int(address) or not _is_positive_int(size) or state != "active_allocated":
                    continue
                block_end = address + size
                if any(address < tensor_end and block_end > tensor_start for tensor_start, tensor_end in tensor_ranges):
                    active_blocks.append((address, size))

        merged_blocks: list[tuple[int, int]] = []
        for address, size in sorted(set(active_blocks)):
            if not merged_blocks or merged_blocks[-1][0] + merged_blocks[-1][1] < address:
                merged_blocks.append((address, size))
                continue
            merged_start, merged_size = merged_blocks[-1]
            merged_end = max(merged_start + merged_size, address + size)
            merged_blocks[-1] = (merged_start, merged_end - merged_start)

        # Every tensor must be fully covered by a registered range.  A pointer
        # merely falling inside an allocator block is insufficient.
        for name, weight in transferable_tensors:
            tensor_start = weight.data_ptr()
            tensor_end = tensor_start + weight.numel() * weight.element_size()
            if not any(start <= tensor_start and tensor_end <= start + size for start, size in merged_blocks):
                logger.error("RFork tensor %s is not fully covered by an active NPU allocator block", name)
                return False

        if not merged_blocks:
            logger.error("RFork found no allocator blocks for %d transferable tensors", len(transferable_tensors))
            return False

        logical_registrations: list[tuple[int, int, int, int]] = []
        for tensor_start, tensor_end in sorted(set(tensor_ranges)):
            backing_start, backing_size = next(
                (start, size) for start, size in merged_blocks if start <= tensor_start and tensor_end <= start + size
            )
            if (
                logical_registrations
                and logical_registrations[-1][2:] == (backing_start, backing_size)
                and tensor_start <= logical_registrations[-1][0] + logical_registrations[-1][1]
            ):
                logical_start, logical_size, _, _ = logical_registrations[-1]
                logical_registrations[-1] = (
                    logical_start,
                    max(logical_start + logical_size, tensor_end) - logical_start,
                    backing_start,
                    backing_size,
                )
            else:
                logical_registrations.append((tensor_start, tensor_end - tensor_start, backing_start, backing_size))

        memory_registration_cls = self._memory_registration_cls
        batch_register_memory_ex = getattr(transfer_engine, "batch_register_memory_ex", None)
        if memory_registration_cls is None or not callable(batch_register_memory_ex):
            logger.error(
                "RFork requires YuanRong TransferEngine with MemoryRegistration and batch_register_memory_ex support."
            )
            return False
        try:
            registrations = [memory_registration_cls(*registration) for registration in logical_registrations]
            ret = batch_register_memory_ex(registrations)
            registered_memory_addresses = [registration[0] for registration in logical_registrations]
        except Exception as e:
            logger.error(
                "TransferEngine memory registration raised for %d logical regions: %s",
                len(logical_registrations),
                e,
            )
            return False
        if ret.is_error():
            logger.error(
                "TransferEngine memory registration failed for %d logical regions, ret: %s",
                len(logical_registrations),
                ret.to_string(),
            )
            return False

        # Commit only after all validation and the native call succeed.  The
        # previous owner/tracking state is untouched on every failure path.
        self.rfork_transfer_engine_weights_info_dict = weight_mr_dict
        self.rfork_transfer_engine_weights_shape_dict = weight_shape_dict
        self.registered_weight_blocks = merged_blocks
        self.registered_memory_addresses = registered_memory_addresses
        self._registered_transferable_tensors = transferable_tensors
        logger.info(
            "register_memory_region time: %.4fs, weights: %d",
            time.perf_counter() - start_reg_mr_time,
            len(weight_mr_dict),
        )
        return True

    def unregister_memory_region(self) -> bool:
        with self._get_lifecycle_lock():
            return self._unregister_memory_region_locked()

    def _unregister_memory_region_locked(self) -> bool:
        transfer_engine = self._get_transfer_engine()
        start_unreg_mr_time = time.perf_counter()
        if not getattr(self, "registered_weight_blocks", None):
            self._clear_registration_state()
            logger.debug("unregister_memory_region skipped because no blocks are registered.")
            return True

        registered_blocks = self.registered_weight_blocks
        registered_memory_addresses = getattr(self, "registered_memory_addresses", None)
        if not registered_memory_addresses:
            registered_memory_addresses = [address for address, _ in registered_blocks]
        try:
            ret = transfer_engine.batch_unregister_memory(registered_memory_addresses)
        except Exception as e:
            # Keep all tracking and tensor owners for a subsequent retry.
            logger.error(
                "batch_unregister_memory raised for %d blocks: %s",
                len(registered_blocks),
                e,
            )
            return False
        if ret.is_error():
            logger.error(
                "batch_unregister_memory failed for %d blocks, ret: %s",
                len(registered_blocks),
                ret.to_string(),
            )
            return False
        self._clear_registration_state()
        logger.info(
            "unregister_memory_region time: %.4fs",
            time.perf_counter() - start_unreg_mr_time,
        )
        return True

    def finalize_transfer_engine(self, max_attempts: int = 2) -> bool:
        if not isinstance(max_attempts, int) or isinstance(max_attempts, bool) or max_attempts <= 0:
            raise ValueError("RFork TransferEngine finalize max_attempts must be a positive integer")
        with self._get_lifecycle_lock():
            if not getattr(self, "_is_initialized", False):
                return True
            transfer_engine = self._get_transfer_engine()
            for attempt in range(1, max_attempts + 1):
                try:
                    ret = transfer_engine.finalize()
                except Exception as e:
                    logger.error("TransferEngine finalize raised on attempt %d/%d: %s", attempt, max_attempts, e)
                    return False
                if not ret.is_error():
                    self._clear_registration_state()
                    self.rfork_transfer_engine_session_id = None
                    self._is_initialized = False
                    return True
                logger.warning(
                    "TransferEngine finalize failed on attempt %d/%d: %s",
                    attempt,
                    max_attempts,
                    ret.to_string(),
                )
                not_ready_code = getattr(self, "_not_ready_error_code", None)
                get_code = getattr(ret, "get_code", None)
                if not_ready_code is None or not callable(get_code) or get_code() != not_ready_code:
                    return False
            return False

    def recv_from_source(
        self,
        model,
        seed_instance_ip,
        seed_instance_service_port,
        local_seed_key,
        processed_layout: bool,
        manifest_metadata: Any | None = None,
        request_timeout_sec: float | None = None,
        auth_token: str | None = None,
    ):
        with self._get_lifecycle_lock():
            return self._recv_from_source_locked(
                model,
                seed_instance_ip,
                seed_instance_service_port,
                local_seed_key,
                processed_layout,
                manifest_metadata,
                request_timeout_sec,
                auth_token,
            )

    def _recv_from_source_locked(
        self,
        model,
        seed_instance_ip,
        seed_instance_service_port,
        local_seed_key,
        processed_layout: bool,
        manifest_metadata: Any | None = None,
        request_timeout_sec: float | None = None,
        auth_token: str | None = None,
    ):
        transfer_engine = self._get_transfer_engine()
        seed_host = str(seed_instance_ip).strip()
        seed_url: str | None = None
        if seed_host.startswith("http://") or seed_host.startswith("https://"):
            seed_url = seed_host.rstrip("/")
        else:
            # join_host_port brackets IPv6 literals while preserving normal
            # host:port formatting (and accepts callers that already bracketed
            # an IPv6 address).
            if seed_host.startswith("[") and "]" in seed_host:
                closing_bracket = seed_host.find("]")
                if closing_bracket == len(seed_host) - 1:
                    seed_host = seed_host[1:-1]
                elif seed_host[closing_bracket + 1 :] == f":{seed_instance_service_port}":
                    seed_url = f"http://{seed_host}"
                else:
                    seed_host = seed_host[1:closing_bracket]
            if seed_url is None:
                seed_url = f"http://{join_host_port(seed_host, seed_instance_service_port)}"
        if request_timeout_sec is None:
            request_timeout_sec = getattr(self, "request_timeout_sec", DEFAULT_REQUEST_TIMEOUT_SEC)
        request_timeout_sec = _validate_request_timeout(request_timeout_sec)
        if auth_token is None:
            auth_token = getattr(self, "auth_token", "")
        if request_timeout_sec == DEFAULT_REQUEST_TIMEOUT_SEC and not auth_token:
            # Keep the historical two-argument call shape for lightweight
            # test doubles; the helper still applies the finite default.
            seed_session_id, seed_weight_info, seed_weight_shapes = get_remote_instance_transfer_engine_info(
                seed_url,
                local_seed_key,
            )
        else:
            seed_session_id, seed_weight_info, seed_weight_shapes = get_remote_instance_transfer_engine_info(
                seed_url,
                local_seed_key,
                request_timeout_sec,
                auth_token,
            )
        if not isinstance(seed_session_id, str) or not seed_session_id or not isinstance(seed_weight_info, Mapping):
            logger.error("Cannot get transfer engine session or weight info.")
            return False

        transferable_tensors = getattr(self, "_registered_transferable_tensors", None)
        if transferable_tensors is None:
            transferable_tensors = list(_iter_transferable_tensors(model, processed_layout))
        # Keep the tensor owners alive for as long as their memory remains
        # registered. A receiver becomes a seed after transfer, and later model
        # finalization can replace tensors such as an MTP draft embedding.
        # unregister_memory_region() is the only safe place to release them.

        if not transferable_tensors:
            logger.error("RFork refuses to transfer an empty local tensor manifest.")
            return False

        local_names = [name for name, _ in transferable_tensors]
        local_name_set = set(local_names)
        if (
            len(local_names) != len(local_name_set)
            or not local_name_set
            or any(not isinstance(name, str) or not name for name in local_name_set)
        ):
            logger.error("RFork local tensor manifest has duplicate or empty names.")
            return False
        remote_names = list(seed_weight_info)
        remote_name_set = set(remote_names)
        if (
            len(remote_names) != len(remote_name_set)
            or not remote_name_set
            or any(not isinstance(name, str) or not name for name in remote_name_set)
            or local_name_set != remote_name_set
        ):
            logger.error(
                "RFork manifest names differ: local_only=%s, remote_only=%s",
                sorted(local_name_set - remote_name_set, key=str),
                sorted(remote_name_set - local_name_set, key=str),
            )
            return False

        if seed_weight_shapes is not None and not isinstance(seed_weight_shapes, Mapping):
            logger.error("RFork remote shape manifest is malformed.")
            return False
        if isinstance(seed_weight_shapes, Mapping) and seed_weight_shapes:
            shape_name_set = set(seed_weight_shapes)
            if shape_name_set != remote_name_set:
                logger.error("RFork remote shape manifest names differ from weight manifest.")
                return False

        requires_v2_manifest = str(local_seed_key).startswith("rfork-v2:")
        parsed_remote: dict[str, tuple[int, int, int, Any, str | None]] = {}
        remote_total_bytes = 0
        for name, weight_info in seed_weight_info.items():
            parsed_weight_info = _parse_weight_info(weight_info)
            if parsed_weight_info is None:
                logger.error("Invalid weight info for %s: %s", name, weight_info)
                return False
            seed_ptr, seed_len, seed_size, seed_shape, seed_dtype = _unpack_weight_info(parsed_weight_info)
            if seed_shape is None and isinstance(seed_weight_shapes, Mapping) and name in seed_weight_shapes:
                seed_shape = _normalize_weight_shape(seed_weight_shapes[name])
                if seed_shape is None:
                    logger.error("Invalid shape metadata for %s", name)
                    return False
            elif seed_shape is not None and isinstance(seed_weight_shapes, Mapping) and name in seed_weight_shapes:
                shape_from_endpoint = _normalize_weight_shape(seed_weight_shapes[name])
                if shape_from_endpoint is None or shape_from_endpoint != seed_shape:
                    logger.error("Conflicting shape metadata for %s", name)
                    return False
            if seed_shape is not None and _numel_from_shape(seed_shape) != seed_len:
                logger.error("Shape metadata does not match numel for %s", name)
                return False
            if requires_v2_manifest and (seed_shape is None or seed_dtype is None):
                logger.error("RFork v2 manifest entry for %s must include shape and dtype", name)
                return False
            parsed_remote[name] = (seed_ptr, seed_len, seed_size, seed_shape, seed_dtype)
            remote_total_bytes += seed_len * seed_size

        local_total_bytes = 0
        local_by_name: dict[str, torch.Tensor] = {}
        for name, tensor in transferable_tensors:
            # A cached registration has already established the transfer
            # device.  Keep this validation device-agnostic so callers can
            # exercise the protocol with mocked tensors in unit tests.
            if (
                not isinstance(name, str)
                or not name
                or not isinstance(tensor, torch.Tensor)
                or not _is_positive_int(tensor.data_ptr())
                or not _is_positive_int(tensor.numel())
                or not _is_positive_int(tensor.element_size())
            ):
                logger.error("Invalid local tensor manifest entry for %s", name)
                return False
            local_by_name[name] = tensor
            local_total_bytes += tensor.numel() * tensor.element_size()
            seed_ptr, seed_len, seed_size, _, seed_dtype = parsed_remote[name]
            del seed_ptr
            if seed_len != tensor.numel() or seed_size != tensor.element_size():
                logger.error(
                    "Weight info mismatch for %s, expected (%s, %s), got (%s, %s)",
                    name,
                    seed_len,
                    seed_size,
                    tensor.numel(),
                    tensor.element_size(),
                )
                return False
            if seed_dtype is not None and seed_dtype != _normalize_dtype_name(tensor.dtype):
                logger.error(
                    "Weight dtype mismatch for %s, expected %s, got %s",
                    name,
                    seed_dtype,
                    _normalize_dtype_name(tensor.dtype),
                )
                return False

        if len(local_by_name) != len(parsed_remote) or local_total_bytes != remote_total_bytes:
            logger.error(
                "RFork manifest count/bytes differ: local=(%d, %d), remote=(%d, %d)",
                len(local_by_name),
                local_total_bytes,
                len(parsed_remote),
                remote_total_bytes,
            )
            return False

        if manifest_metadata is not None:
            if not isinstance(manifest_metadata, Mapping):
                logger.error("RFork manifest metadata is malformed.")
                return False
            count_value = _get_manifest_value(manifest_metadata, "tensor_count", "num_tensors", "count")
            if count_value is not None and _parse_manifest_scalar(count_value) != len(local_by_name):
                logger.error("RFork manifest tensor count mismatch: %s", count_value)
                return False
            bytes_value = _get_manifest_value(manifest_metadata, "total_bytes", "byte_count", "bytes")
            if bytes_value is not None and _parse_manifest_scalar(bytes_value) != local_total_bytes:
                logger.error("RFork manifest byte count mismatch: %s", bytes_value)
                return False

            metadata_entries = _extract_manifest_entries(manifest_metadata)
            if metadata_entries is not None:
                if set(metadata_entries) != local_name_set:
                    logger.error("RFork optional manifest names differ from local manifest.")
                    return False
                for name, metadata in metadata_entries.items():
                    if not isinstance(metadata, Mapping):
                        logger.error("RFork optional manifest entry for %s is malformed", name)
                        return False
                    tensor = local_by_name[name]
                    remote_entry = parsed_remote[name]
                    expected_numel = _get_manifest_value(metadata, "numel", "count", "seed_len")
                    if expected_numel is not None and (
                        _parse_manifest_scalar(expected_numel) != tensor.numel()
                        or _parse_manifest_scalar(expected_numel) != remote_entry[1]
                    ):
                        logger.error("RFork optional manifest numel mismatch for %s", name)
                        return False
                    expected_size = _get_manifest_value(metadata, "element_size", "itemsize", "seed_size")
                    if expected_size is not None and (
                        _parse_manifest_scalar(expected_size) != tensor.element_size()
                        or _parse_manifest_scalar(expected_size) != remote_entry[2]
                    ):
                        logger.error("RFork optional manifest element size mismatch for %s", name)
                        return False
                    expected_shape = metadata.get("shape")
                    if expected_shape is not None:
                        expected_shape = _normalize_weight_shape(expected_shape)
                        if (
                            expected_shape is None
                            or _numel_from_shape(expected_shape) != tensor.numel()
                            or (remote_entry[3] is not None and expected_shape != remote_entry[3])
                        ):
                            logger.error("RFork optional manifest shape mismatch for %s", name)
                            return False
                    expected_dtype = metadata.get("dtype")
                    if expected_dtype is not None:
                        expected_dtype = _normalize_dtype_name(expected_dtype)
                        if (
                            expected_dtype is None
                            or expected_dtype != _normalize_dtype_name(tensor.dtype)
                            or (remote_entry[4] is not None and expected_dtype != remote_entry[4])
                        ):
                            logger.error("RFork optional manifest dtype mismatch for %s", name)
                            return False

        # All validation is complete before mutating tensor metadata.  A later
        # transfer failure therefore leaves the registered model unchanged.
        reshape_events: list[tuple[str, tuple[int, ...], tuple[int, ...]]] = []
        for name, tensor in transferable_tensors:
            seed_shape = parsed_remote[name][3]
            if not _reshape_tensor_to_seed_shape(name, tensor, seed_shape, reshape_events):
                return False
            _update_registered_weight_shape(
                getattr(self, "rfork_transfer_engine_weights_shape_dict", None),
                name,
                tensor,
            )
            _update_registered_weight_info(
                getattr(self, "rfork_transfer_engine_weights_info_dict", None),
                name,
                tensor,
            )

        seed_ptr_list = []
        client_ptr_list = []
        client_len_list = []
        weight_names = []
        for name, tensor in transferable_tensors:
            seed_ptr, _, _, _, _ = parsed_remote[name]
            seed_ptr_list.append(seed_ptr)
            client_ptr_list.append(tensor.data_ptr())
            client_len_list.append(tensor.numel() * tensor.element_size())
            weight_names.append(name)

        if reshape_events:
            sample_events = ", ".join(
                f"{name}: {local_shape}->{seed_shape}" for name, local_shape, seed_shape in reshape_events[:3]
            )
            if len(reshape_events) > 3:
                sample_events += ", ..."
            logger.debug(
                "RFork reshaped %d tensors to match seed shapes: %s",
                len(reshape_events),
                sample_events,
            )

        transfer_chunks = list(
            _iter_transfer_chunks(
                weight_names,
                seed_ptr_list,
                client_ptr_list,
                client_len_list,
            )
        )
        total_transfer_bytes = sum(client_len_list)

        transfer_start_time = time.perf_counter()
        logger.info(
            "transfer weights starts, weights: %d, chunks: %d, total bytes: %.2f GiB",
            len(client_len_list),
            len(transfer_chunks),
            total_transfer_bytes / (1024**3),
        )
        for index, (chunk_names, chunk_seed_ptrs, chunk_client_ptrs, chunk_lengths) in enumerate(transfer_chunks, 1):
            chunk_start_time = time.perf_counter()
            logger.debug(
                "transfer weights chunk %d/%d starts, weights: %d, bytes: %.2f GiB, first: %s, last: %s",
                index,
                len(transfer_chunks),
                len(chunk_lengths),
                sum(chunk_lengths) / (1024**3),
                chunk_names[0],
                chunk_names[-1],
            )
            ret = transfer_engine.batch_transfer_sync_read(
                seed_session_id,
                chunk_client_ptrs,
                chunk_seed_ptrs,
                chunk_lengths,
            )
            if ret.is_error():
                logger.error(
                    "Failed to transfer weights chunk %d/%d, first: %s, last: %s, ret=%s",
                    index,
                    len(transfer_chunks),
                    chunk_names[0],
                    chunk_names[-1],
                    ret.to_string(),
                )
                return False
            logger.debug(
                "transfer weights chunk %d/%d done, time: %.4fs",
                index,
                len(transfer_chunks),
                time.perf_counter() - chunk_start_time,
            )
        transfer_time = time.perf_counter() - transfer_start_time
        logger.info("transfer weights time: %.4fs", transfer_time)
        return True


def get_remote_instance_transfer_engine_info(
    seed_url: str,
    local_seed_key: str,
    request_timeout_sec: float = DEFAULT_REQUEST_TIMEOUT_SEC,
    auth_token: str = "",
):
    request_timeout_sec = _validate_request_timeout(request_timeout_sec)
    headers = _auth_headers(auth_token)
    try:
        response = requests.get(
            f"{seed_url}/get_rfork_transfer_engine_info",
            params={"seed_key": local_seed_key},
            headers=headers,
            timeout=request_timeout_sec,
        )
        if response.status_code != 200:
            logger.error(
                "GET %s/get_rfork_transfer_engine_info failed: %s",
                seed_url,
                response.status_code,
            )
            return None, None, None

        data = response.json()
        info = data.get("rfork_transfer_engine_info", None)
        if info is not None and isinstance(info, (list, tuple)) and len(info) in (2, 3):
            if request_timeout_sec == DEFAULT_REQUEST_TIMEOUT_SEC and not auth_token:
                shape_info = get_remote_instance_weight_shape_info(seed_url, local_seed_key)
            else:
                shape_info = get_remote_instance_weight_shape_info(
                    seed_url,
                    local_seed_key,
                    request_timeout_sec,
                    auth_token,
                )
            if len(info) == 3 and shape_info is None and isinstance(info[2], Mapping):
                shape_info = info[2]
            return info[0], info[1], shape_info
        if isinstance(info, Mapping):
            session_id = info.get("session_id", info.get("session"))
            weight_info = info.get("weights", info.get("weight_info", info.get("manifest")))
            shape_info = info.get("shape_info", info.get("shapes"))
            if session_id and isinstance(weight_info, Mapping):
                if shape_info is None:
                    if request_timeout_sec == DEFAULT_REQUEST_TIMEOUT_SEC and not auth_token:
                        shape_info = get_remote_instance_weight_shape_info(seed_url, local_seed_key)
                    else:
                        shape_info = get_remote_instance_weight_shape_info(
                            seed_url,
                            local_seed_key,
                            request_timeout_sec,
                            auth_token,
                        )
                return session_id, weight_info, shape_info

        logger.error(
            "Failed to get rfork_transfer_engine_info in response from %s.",
            seed_url,
        )
        return None, None, None
    except Exception as e:
        logger.error("Exception getting transfer engine info from %s: %s", seed_url, e)
        return None, None, None


def get_remote_instance_weight_shape_info(
    seed_url: str,
    local_seed_key: str,
    request_timeout_sec: float = DEFAULT_REQUEST_TIMEOUT_SEC,
    auth_token: str = "",
):
    request_timeout_sec = _validate_request_timeout(request_timeout_sec)
    headers = _auth_headers(auth_token)
    try:
        response = requests.get(
            f"{seed_url}/get_rfork_transfer_engine_shape_info",
            params={"seed_key": local_seed_key},
            headers=headers,
            timeout=request_timeout_sec,
        )
        if response.status_code != 200:
            logger.debug(
                "GET %s/get_rfork_transfer_engine_shape_info failed: %s",
                seed_url,
                response.status_code,
            )
            return None

        data = response.json()
        info = data.get("rfork_transfer_engine_shape_info", None)
        if info is None or isinstance(info, dict):
            return info

        logger.error(
            "Failed to get rfork_transfer_engine_shape_info in response from %s.",
            seed_url,
        )
        return None
    except Exception as e:
        logger.debug("Exception getting transfer engine shape info from %s: %s", seed_url, e)
        return None
