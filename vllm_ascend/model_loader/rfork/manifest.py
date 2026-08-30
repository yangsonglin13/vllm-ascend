# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

from collections.abc import Iterator, Mapping
from typing import Any

import torch
from torch import nn
from vllm.logger import logger

MAX_TRANSFER_CHUNK_BYTES = 1024**3
MAX_TRANSFER_CHUNK_WEIGHTS = 512


def normalize_dtype_name(dtype: Any) -> str | None:
    if dtype is None:
        return None
    if isinstance(dtype, torch.dtype):
        name = str(dtype)
    elif isinstance(dtype, str):
        name = dtype.strip()
    else:
        return None
    if name.startswith("torch."):
        name = name[6:]
    return name.lower() if name else None


def is_positive_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def normalize_weight_shape(shape: Any) -> tuple[int, ...] | None:
    if not isinstance(shape, (list, tuple)):
        return None
    if not all(isinstance(dim, int) and not isinstance(dim, bool) and dim >= 0 for dim in shape):
        return None
    return tuple(shape)


def numel_from_shape(shape: tuple[int, ...]) -> int:
    numel = 1
    for dim in shape:
        numel *= dim
    return numel


def parse_weight_info(weight_info: Any):
    has_full_metadata = False
    seed_shape = None
    seed_dtype = None
    if isinstance(weight_info, (list, tuple)):
        if len(weight_info) not in (3, 4, 5):
            return None
        seed_ptr, seed_len, seed_size = weight_info[:3]
        if len(weight_info) >= 4:
            shape_or_metadata = weight_info[3]
            if isinstance(shape_or_metadata, Mapping):
                has_full_metadata = True
                seed_shape = shape_or_metadata.get("shape")
                seed_dtype = shape_or_metadata.get("dtype")
            else:
                if shape_or_metadata is None:
                    return None
                seed_shape = shape_or_metadata
            if len(weight_info) == 5:
                has_full_metadata = True
                seed_dtype = weight_info[4]
    elif isinstance(weight_info, Mapping):
        has_full_metadata = True

        def first(*keys: str):
            return next((weight_info[key] for key in keys if key in weight_info), None)

        seed_ptr = first("ptr", "pointer", "seed_ptr", "address")
        seed_len = first("numel", "count", "seed_len", "length")
        seed_size = first("element_size", "itemsize", "seed_size", "size")
        seed_shape = weight_info.get("shape")
        seed_dtype = weight_info.get("dtype")
    else:
        return None

    if not all(is_positive_int(value) for value in (seed_ptr, seed_len, seed_size)):
        return None
    if seed_shape is not None:
        seed_shape = normalize_weight_shape(seed_shape)
        if seed_shape is None or numel_from_shape(seed_shape) != seed_len:
            return None
    if seed_dtype is not None:
        seed_dtype = normalize_dtype_name(seed_dtype)
        if seed_dtype is None:
            return None
    if has_full_metadata:
        return seed_ptr, seed_len, seed_size, seed_shape, seed_dtype
    return seed_ptr, seed_len, seed_size, seed_shape


def unpack_weight_info(parsed: tuple[Any, ...]) -> tuple[int, int, int, Any, str | None]:
    if len(parsed) == 4:
        seed_ptr, seed_len, seed_size, seed_shape = parsed
        return seed_ptr, seed_len, seed_size, seed_shape, None
    return parsed


def reshape_tensor_to_seed_shape(
    name: str,
    tensor: torch.Tensor,
    seed_shape: tuple[int, ...] | None,
    reshape_events: list[tuple[str, tuple[int, ...], tuple[int, ...]]] | None = None,
) -> bool:
    if seed_shape is None or tuple(tensor.shape) == seed_shape:
        return True
    if tensor.numel() != numel_from_shape(seed_shape):
        logger.error("Weight shape mismatch for %s: local=%s, seed=%s", name, tuple(tensor.shape), seed_shape)
        return False
    local_shape = tuple(tensor.shape)
    try:
        tensor.data = tensor.data.view(seed_shape)
    except Exception as exc:
        logger.error("Failed to reshape RFork tensor %s from %s to %s: %s", name, local_shape, seed_shape, exc)
        return False
    if reshape_events is not None:
        reshape_events.append((name, local_shape, seed_shape))
    return True


def update_registered_weight_shape(info: dict[str, Any] | None, name: str, tensor: torch.Tensor) -> None:
    if isinstance(info, dict) and name in info:
        info[name] = tuple(tensor.shape)


def update_registered_weight_info(info: dict[str, Any] | None, name: str, tensor: torch.Tensor) -> None:
    if not isinstance(info, dict) or name not in info:
        return
    current = info[name]
    if isinstance(current, tuple) and len(current) >= 5:
        info[name] = (*current[:3], tuple(tensor.shape), normalize_dtype_name(tensor.dtype))
    elif isinstance(current, tuple) and len(current) == 4:
        info[name] = (*current[:3], tuple(tensor.shape))


def is_tensor_on_transfer_device(tensor: torch.Tensor) -> bool:
    return tensor.device.type == "npu"


def is_transferable_tensor(tensor: torch.Tensor) -> bool:
    return not tensor.is_meta and tensor.numel() > 0 and is_tensor_on_transfer_device(tensor)


def _iter_tensors_in_value(
    prefix: str,
    value: Any,
    visited_object_ids: set[int],
    scan_objects: bool = False,
) -> Iterator[tuple[str, torch.Tensor]]:
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
        if not attr_name.startswith("_"):
            yield from _iter_tensors_in_value(
                f"{prefix}.{attr_name}",
                attr_value,
                visited_object_ids,
                scan_objects,
            )


def _try_collect(
    name: str,
    tensor: torch.Tensor,
    seen_data_ptrs: set[int],
    collected: list[tuple[str, torch.Tensor]],
) -> None:
    if not is_transferable_tensor(tensor):
        return
    data_ptr = tensor.data_ptr()
    if data_ptr not in seen_data_ptrs:
        seen_data_ptrs.add(data_ptr)
        collected.append((name, tensor))


def collect_processed_layout_tensors(model: nn.Module) -> list[tuple[str, torch.Tensor]]:
    seen: set[int] = set()
    collected: list[tuple[str, torch.Tensor]] = []
    for name, tensor in model.named_parameters():
        _try_collect(name, tensor, seen, collected)
    for name, tensor in model.named_buffers():
        _try_collect(name, tensor, seen, collected)
    for module_prefix, module in model.named_modules():
        for attr_name, attr_value in vars(module).items():
            if attr_name.startswith("_") or isinstance(attr_value, nn.Module):
                continue
            for tensor_name, tensor in _iter_tensors_in_value(
                attr_name,
                attr_value,
                set(),
                attr_name == "impl",
            ):
                full_name = f"{module_prefix}.{tensor_name}" if module_prefix else tensor_name
                _try_collect(full_name, tensor, seen, collected)
    return collected


def collect_checkpoint_layout_tensors(model: nn.Module) -> list[tuple[str, torch.Tensor]]:
    seen: set[int] = set()
    collected: list[tuple[str, torch.Tensor]] = []
    for name, tensor in model.named_parameters():
        _try_collect(name, tensor, seen, collected)
    for name, tensor in model.named_buffers():
        _try_collect(name, tensor, seen, collected)
    for module_prefix, module in model.named_modules():
        impl = getattr(module, "impl", None)
        if impl is None or isinstance(impl, nn.Module):
            continue
        for tensor_name, tensor in _iter_tensors_in_value("impl", impl, set(), scan_objects=True):
            full_name = f"{module_prefix}.{tensor_name}" if module_prefix else tensor_name
            _try_collect(full_name, tensor, seen, collected)
    return collected


def collect_transferable_tensors(model: nn.Module, processed_layout: bool) -> list[tuple[str, torch.Tensor]]:
    if processed_layout:
        return collect_processed_layout_tensors(model)
    return collect_checkpoint_layout_tensors(model)


def find_non_npu_state_tensors(model: Any) -> list[str]:
    if not isinstance(model, nn.Module):
        return []
    return [
        name
        for iterator in (model.named_parameters(), model.named_buffers())
        for name, tensor in iterator
        if not tensor.is_meta and tensor.numel() > 0 and not is_tensor_on_transfer_device(tensor)
    ]


def iter_transfer_chunks(
    weight_names: list[str],
    seed_ptrs: list[int],
    client_ptrs: list[int],
    lengths: list[int],
):
    if not (len(weight_names) == len(seed_ptrs) == len(client_ptrs) == len(lengths)):
        raise ValueError("RFork transfer lists must have equal lengths")
    segments: list[tuple[str, int, int, int]] = []
    for name, seed_ptr, client_ptr, length in zip(weight_names, seed_ptrs, client_ptrs, lengths, strict=True):
        if not is_positive_int(length):
            raise ValueError("RFork transfer segment length must be a positive integer")
        offset = 0
        while offset < length:
            segment_length = min(MAX_TRANSFER_CHUNK_BYTES, length - offset)
            segments.append((name, seed_ptr + offset, client_ptr + offset, segment_length))
            offset += segment_length

    start = 0
    chunk_bytes = 0
    for index, segment in enumerate(segments):
        should_flush = index > start and (
            chunk_bytes + segment[3] > MAX_TRANSFER_CHUNK_BYTES or index - start >= MAX_TRANSFER_CHUNK_WEIGHTS
        )
        if should_flush:
            chunk = segments[start:index]
            yield tuple(map(list, zip(*chunk, strict=True)))
            start = index
            chunk_bytes = 0
        chunk_bytes += segment[3]
    if start < len(segments):
        yield tuple(map(list, zip(*segments[start:], strict=True)))
