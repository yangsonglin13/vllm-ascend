# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

import math
import threading
import time
from typing import Any

import torch
from vllm.logger import logger
from vllm.utils.network_utils import get_ip, get_open_port, join_host_port

from vllm_ascend.model_loader.rfork.manifest import (
    MAX_TRANSFER_CHUNK_BYTES,
    MAX_TRANSFER_CHUNK_WEIGHTS,
    collect_transferable_tensors,
    find_non_npu_state_tensors,
    is_positive_int,
    is_transferable_tensor,
    iter_transfer_chunks,
    normalize_dtype_name,
    normalize_weight_shape,
    numel_from_shape,
    parse_weight_info,
    reshape_tensor_to_seed_shape,
    unpack_weight_info,
    update_registered_weight_info,
    update_registered_weight_shape,
)
from vllm_ascend.model_loader.rfork.seed_client import fetch_seed_transfer_info

DEFAULT_REQUEST_TIMEOUT_SEC = 10.0


def _validate_request_timeout(request_timeout_sec: Any) -> float:
    if isinstance(request_timeout_sec, bool) or not isinstance(request_timeout_sec, (int, float)):
        raise ValueError("RFork request_timeout_sec must be a positive finite number")
    timeout = float(request_timeout_sec)
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("RFork request_timeout_sec must be a positive finite number")
    return timeout


def _is_ptr_in_blocks(ptr: int, blocks: list[tuple[int, int]]) -> bool:
    return any(address <= ptr < address + size for address, size in blocks)


def _split_tensors_by_excluded_blocks(
    transferable_tensors: list[tuple[str, torch.Tensor]],
    excluded_blocks: list[tuple[int, int]],
) -> tuple[list[tuple[str, torch.Tensor]], list[str]]:
    if not excluded_blocks:
        return list(transferable_tensors), []

    kept_tensors: list[tuple[str, torch.Tensor]] = []
    excluded_names: list[str] = []
    for name, tensor in transferable_tensors:
        if _is_ptr_in_blocks(tensor.data_ptr(), excluded_blocks):
            excluded_names.append(name)
        else:
            kept_tensors.append((name, tensor))
    return kept_tensors, excluded_names


class RForkTransferBackend:
    """Own one YuanRong TransferEngine and its registered model memory."""

    def __init__(self, request_timeout_sec: float = DEFAULT_REQUEST_TIMEOUT_SEC) -> None:
        self.rfork_transfer_engine: Any | None = None
        self.rfork_transfer_engine_session_id: str | None = None
        self.rfork_transfer_engine_weights_info_dict: dict[str, Any] | None = None
        self.rfork_transfer_engine_weights_shape_dict: dict[str, tuple[int, ...]] | None = None
        self.registered_weight_blocks: list[tuple[int, int]] = []
        self.registered_memory_addresses: list[int] = []
        self.excluded_weight_blocks: list[tuple[int, int]] = []
        self._registered_transferable_tensors: list[tuple[str, torch.Tensor]] | None = None
        self._memory_registration_cls: Any | None = None
        self._not_ready_error_code: Any | None = None
        self._lifecycle_lock = threading.RLock()
        self.request_timeout_sec = _validate_request_timeout(request_timeout_sec)
        self._is_initialized = False
        self._initialize_transfer_engine()

    def _initialize_transfer_engine(self) -> None:
        try:
            from yr.datasystem import (  # type: ignore[import-not-found]
                ErrorCode,
                MemoryRegistration,
                TransferEngine,
            )
        except ImportError as exc:
            raise ImportError(
                "RFork requires the YuanRong TransferEngine, MemoryRegistration, and ErrorCode APIs."
            ) from exc

        engine = TransferEngine()
        endpoint = join_host_port(get_ip(), get_open_port())
        device_name = f"npu:{torch.npu.current_device()}"
        result = engine.initialize(endpoint, "ascend", device_name)
        if result.is_error():
            raise RuntimeError(
                f"YuanRong TransferEngine initialize({endpoint!r}, 'ascend', {device_name!r}) failed: "
                f"{result.to_string()}"
            )
        self.rfork_transfer_engine = engine
        self.rfork_transfer_engine_session_id = endpoint
        self._memory_registration_cls = MemoryRegistration
        self._not_ready_error_code = ErrorCode.kNotReady
        self._is_initialized = True

    def is_initialized(self) -> bool:
        return self._is_initialized

    def _engine(self) -> Any:
        if self.rfork_transfer_engine is None:
            raise RuntimeError("TransferEngine is not initialized.")
        return self.rfork_transfer_engine

    def _lock(self) -> threading.RLock:
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

    def register_memory_region(
        self,
        model,
        processed_layout: bool,
        exclude_blocks: list[tuple[int, int]] | None = None,
    ) -> bool:
        with self._lock():
            if self.registered_weight_blocks:
                # A failed unregister deliberately leaves this state intact. Retry
                # the unregister now; never overwrite a live registration.
                stale_block_count = len(self.registered_weight_blocks)
                if not self.unregister_memory_region():
                    return False
                logger.info(
                    "Retried unregister for %d blocks left registered by a failed reset.",
                    stale_block_count,
                )

            # Exclude target-owned ranges because HCCL rejects overlaps.
            excluded_blocks = list(exclude_blocks) if exclude_blocks else []
            self.excluded_weight_blocks = excluded_blocks

            tensors, excluded_names = _split_tensors_by_excluded_blocks(
                collect_transferable_tensors(model, processed_layout),
                excluded_blocks,
            )
            if excluded_names:
                logger.info(
                    "Skipping %d weights shared with the target model (already registered), e.g. %s",
                    len(excluded_names),
                    ", ".join(excluded_names[:3]),
                )

            non_npu_state = find_non_npu_state_tensors(model)
            if non_npu_state:
                logger.error("RFork does not support mixed-device model state: %s", non_npu_state[:10])
                return False

            # An empty manifest is legitimate when every tensor was deliberately
            # excluded because the target model already registered its storage.
            if not tensors and not excluded_names:
                logger.error("RFork refuses an empty tensor manifest.")
                return False
            names = [name for name, _ in tensors]
            if len(names) != len(set(names)) or any(not name for name in names):
                logger.error("RFork tensor manifest contains duplicate or empty names.")
                return False

            weights: dict[str, Any] = {}
            shapes: dict[str, tuple[int, ...]] = {}
            tensor_ranges: list[tuple[int, int]] = []
            for name, tensor in tensors:
                if not is_transferable_tensor(tensor):
                    logger.error("RFork found an invalid transferable tensor: %s", name)
                    return False
                pointer = tensor.data_ptr()
                numel = tensor.numel()
                element_size = tensor.element_size()
                if not all(is_positive_int(value) for value in (pointer, numel, element_size)):
                    logger.error("RFork found invalid tensor metadata for %s", name)
                    return False
                shape = tuple(tensor.shape)
                weights[name] = (pointer, numel, element_size, shape, normalize_dtype_name(tensor.dtype))
                shapes[name] = shape
                tensor_ranges.append((pointer, pointer + numel * element_size))

            try:
                memory_snapshot = torch.npu.memory.memory_snapshot()
            except Exception as exc:
                logger.error("RFork failed to snapshot NPU memory: %s", exc)
                return False

            active_blocks: list[tuple[int, int]] = []
            for segment in memory_snapshot:
                for block in segment.get("blocks", []):
                    address = block.get("address", -1)
                    size = block.get("size", -1)
                    if (
                        is_positive_int(address)
                        and is_positive_int(size)
                        and block.get("state") == "active_allocated"
                        and any(address < end and address + size > start for start, end in tensor_ranges)
                    ):
                        active_blocks.append((address, size))

            merged_blocks: list[tuple[int, int]] = []
            for address, size in sorted(set(active_blocks)):
                if not merged_blocks or merged_blocks[-1][0] + merged_blocks[-1][1] < address:
                    merged_blocks.append((address, size))
                else:
                    start, current_size = merged_blocks[-1]
                    merged_blocks[-1] = (start, max(start + current_size, address + size) - start)

            for name, tensor in tensors:
                start = tensor.data_ptr()
                end = start + tensor.numel() * tensor.element_size()
                if not any(block <= start and end <= block + size for block, size in merged_blocks):
                    logger.error("RFork tensor %s is not covered by an active allocator block.", name)
                    return False
            if not merged_blocks and tensors:
                logger.error("RFork found no allocator blocks for registered tensors.")
                return False

            logical_specs: list[tuple[int, int, int, int]] = []
            for tensor_start, tensor_end in sorted(set(tensor_ranges)):
                backing_start, backing_size = next(
                    (start, size)
                    for start, size in merged_blocks
                    if start <= tensor_start and tensor_end <= start + size
                )
                if (
                    logical_specs
                    and logical_specs[-1][2:] == (backing_start, backing_size)
                    and tensor_start <= logical_specs[-1][0] + logical_specs[-1][1]
                ):
                    logical_start, logical_size, _, _ = logical_specs[-1]
                    logical_specs[-1] = (
                        logical_start,
                        max(logical_start + logical_size, tensor_end) - logical_start,
                        backing_start,
                        backing_size,
                    )
                else:
                    logical_specs.append((tensor_start, tensor_end - tensor_start, backing_start, backing_size))

            engine = self._engine()
            register_many = getattr(engine, "batch_register_memory_ex", None)
            if self._memory_registration_cls is None or not callable(register_many):
                logger.error("YuanRong TransferEngine lacks RFork extended memory registration.")
                return False
            try:
                result = register_many([self._memory_registration_cls(*spec) for spec in logical_specs])
            except Exception as exc:
                logger.error("YuanRong memory registration raised: %s", exc)
                return False
            if result.is_error():
                logger.error("YuanRong memory registration failed: %s", result.to_string())
                return False

            self.rfork_transfer_engine_weights_info_dict = weights
            self.rfork_transfer_engine_weights_shape_dict = shapes
            self.registered_weight_blocks = merged_blocks
            self.registered_memory_addresses = [spec[0] for spec in logical_specs]
            self._registered_transferable_tensors = tensors
            return True

    def unregister_memory_region(self) -> bool:
        with self._lock():
            if not self.registered_weight_blocks:
                self._clear_registration_state()
                return True
            if not self.registered_memory_addresses:
                logger.error("RFork registration state has no logical addresses.")
                return False
            try:
                result = self._engine().batch_unregister_memory(self.registered_memory_addresses)
            except Exception as exc:
                logger.error("YuanRong memory unregistration raised: %s", exc)
                return False
            if result.is_error():
                logger.error("YuanRong memory unregistration failed: %s", result.to_string())
                return False
            self._clear_registration_state()
            return True

    def finalize_transfer_engine(self, max_attempts: int = 2) -> bool:
        if not isinstance(max_attempts, int) or isinstance(max_attempts, bool) or max_attempts <= 0:
            raise ValueError("RFork TransferEngine finalize max_attempts must be positive")
        with self._lock():
            if not getattr(self, "_is_initialized", False):
                return True
            for attempt in range(1, max_attempts + 1):
                try:
                    result = self._engine().finalize()
                except Exception as exc:
                    logger.error("YuanRong Finalize raised: %s", exc)
                    return False
                if not result.is_error():
                    self._clear_registration_state()
                    self.rfork_transfer_engine_session_id = None
                    self._is_initialized = False
                    return True
                logger.warning(
                    "YuanRong Finalize attempt %d/%d failed: %s",
                    attempt,
                    max_attempts,
                    result.to_string(),
                )
                get_code = getattr(result, "get_code", None)
                if not callable(get_code) or get_code() != self._not_ready_error_code:
                    return False
            return False

    @staticmethod
    def _seed_url(host: object, port: int) -> str:
        seed_host = str(host).strip()
        if seed_host.startswith(("http://", "https://")):
            return seed_host.rstrip("/")
        if seed_host.startswith("[") and "]" in seed_host:
            closing = seed_host.find("]")
            if closing == len(seed_host) - 1:
                seed_host = seed_host[1:-1]
            elif seed_host[closing + 1 :] == f":{port}":
                return f"http://{seed_host}"
            else:
                seed_host = seed_host[1:closing]
        return f"http://{join_host_port(seed_host, port)}"

    def recv_from_source(
        self,
        model,
        seed_instance_ip,
        seed_instance_service_port: int,
        local_seed_key: str,
        processed_layout: bool,
    ) -> bool:
        with self._lock():
            return self._recv_from_source_locked(
                model,
                seed_instance_ip,
                seed_instance_service_port,
                local_seed_key,
                processed_layout,
            )

    def _recv_from_source_locked(
        self,
        model,
        seed_instance_ip,
        seed_instance_service_port: int,
        local_seed_key: str,
        processed_layout: bool,
    ) -> bool:
        seed_info = fetch_seed_transfer_info(
            self._seed_url(seed_instance_ip, seed_instance_service_port),
            local_seed_key,
            getattr(self, "request_timeout_sec", DEFAULT_REQUEST_TIMEOUT_SEC),
        )
        if seed_info is None:
            return False

        tensors = getattr(self, "_registered_transferable_tensors", None)
        if tensors is None:
            tensors = collect_transferable_tensors(model, processed_layout)
        if not tensors:
            logger.error("RFork refuses to transfer an empty local manifest.")
            return False
        local_names = [name for name, _ in tensors]
        if len(local_names) != len(set(local_names)) or any(not name for name in local_names):
            logger.error("RFork local manifest contains duplicate or empty names.")
            return False
        local_name_set = set(local_names)
        remote_name_set = set(seed_info.weights)

        # Draft workers deliberately skip tensors whose memory is already
        # registered by the target model. The two manifests may therefore
        # differ by exactly those shared names.
        excluded_blocks = getattr(self, "excluded_weight_blocks", None) or []
        local_only = local_name_set - remote_name_set
        remote_only = remote_name_set - local_name_set
        skipped_shared_names: set[str] = set()
        if local_only:
            for name, tensor in tensors:
                if name in local_only and _is_ptr_in_blocks(tensor.data_ptr(), excluded_blocks):
                    logger.debug(
                        "Skip RFork weight %s shared with the target model: not present in the seed manifest.",
                        name,
                    )
                    skipped_shared_names.add(name)
            if local_only - skipped_shared_names:
                logger.error(
                    "RFork manifest names differ: local_only=%s, remote_only=%s",
                    sorted(local_name_set - remote_name_set),
                    sorted(remote_name_set - local_name_set),
                )
                return False
        if remote_only:
            # A seed built without draft exclusion advertises shared names the
            # local filtered manifest does not contain.
            full_model_tensors = dict(collect_transferable_tensors(model, processed_layout))
            for name in remote_only:
                tensor = full_model_tensors.get(name)
                if tensor is not None and _is_ptr_in_blocks(tensor.data_ptr(), excluded_blocks):
                    skipped_shared_names.add(name)
            if remote_only - skipped_shared_names:
                logger.error(
                    "RFork manifest names differ: local_only=%s, remote_only=%s",
                    sorted(local_name_set - remote_name_set),
                    sorted(remote_name_set - local_name_set),
                )
                return False

        if seed_info.shapes is not None and set(seed_info.shapes) != remote_name_set:
            logger.error("RFork shape manifest names differ from weight manifest.")
            return False

        parsed_remote: dict[str, tuple[int, int, int, tuple[int, ...] | None, str | None]] = {}
        for name, raw_info in seed_info.weights.items():
            if name in skipped_shared_names:
                continue
            parsed = parse_weight_info(raw_info)
            if parsed is None:
                logger.error("RFork invalid remote weight metadata for %s", name)
                return False
            pointer, numel, element_size, shape, dtype = unpack_weight_info(parsed)
            if seed_info.shapes is not None:
                manifest_shape = normalize_weight_shape(seed_info.shapes[name])
                if manifest_shape is None:
                    logger.error("RFork invalid remote shape metadata for %s", name)
                    return False
                if shape is None:
                    shape = manifest_shape
                elif shape != manifest_shape:
                    logger.error("RFork conflicting remote shape metadata for %s", name)
                    return False
            if shape is not None and numel_from_shape(shape) != numel:
                logger.error("RFork shape metadata does not match numel for %s", name)
                return False
            parsed_remote[name] = (pointer, numel, element_size, shape, dtype)

        local_bytes = 0
        remote_bytes = sum(numel * element_size for _, numel, element_size, _, _ in parsed_remote.values())
        for name, tensor in tensors:
            if name in skipped_shared_names:
                continue
            _, remote_numel, remote_size, _, remote_dtype = parsed_remote[name]
            if remote_numel != tensor.numel() or remote_size != tensor.element_size():
                logger.error("RFork weight size mismatch for %s", name)
                return False
            if remote_dtype is not None and remote_dtype != normalize_dtype_name(tensor.dtype):
                logger.error("RFork weight dtype mismatch for %s", name)
                return False
            local_bytes += tensor.numel() * tensor.element_size()
        if local_bytes != remote_bytes:
            logger.error("RFork manifest byte totals differ: local=%d, remote=%d", local_bytes, remote_bytes)
            return False

        reshape_events: list[tuple[str, tuple[int, ...], tuple[int, ...]]] = []
        for name, tensor in tensors:
            if name in skipped_shared_names:
                continue
            if not reshape_tensor_to_seed_shape(name, tensor, parsed_remote[name][3], reshape_events):
                return False
            update_registered_weight_shape(
                getattr(self, "rfork_transfer_engine_weights_shape_dict", None), name, tensor
            )
            update_registered_weight_info(getattr(self, "rfork_transfer_engine_weights_info_dict", None), name, tensor)

        names: list[str] = []
        seed_ptrs: list[int] = []
        client_ptrs: list[int] = []
        lengths: list[int] = []
        for name, tensor in tensors:
            if name in skipped_shared_names:
                continue
            names.append(name)
            seed_ptrs.append(parsed_remote[name][0])
            client_ptrs.append(tensor.data_ptr())
            lengths.append(tensor.numel() * tensor.element_size())

        chunks = list(iter_transfer_chunks(names, seed_ptrs, client_ptrs, lengths))
        start = time.perf_counter()
        logger.info(
            "RFork transfer starts: tensors=%d, chunks=%d, bytes=%.2f GiB",
            len(names),
            len(chunks),
            sum(lengths) / (1024**3),
        )
        for index, (chunk_names, chunk_seed_ptrs, chunk_client_ptrs, chunk_lengths) in enumerate(chunks, 1):
            result = self._engine().batch_transfer_sync_read(
                seed_info.session_id,
                chunk_client_ptrs,
                chunk_seed_ptrs,
                chunk_lengths,
            )
            if result.is_error():
                logger.error(
                    "RFork transfer chunk %d/%d failed (%s..%s): %s",
                    index,
                    len(chunks),
                    chunk_names[0],
                    chunk_names[-1],
                    result.to_string(),
                )
                return False
        logger.info("RFork transfer completed in %.4fs", time.perf_counter() - start)
        return True


__all__ = [
    "DEFAULT_REQUEST_TIMEOUT_SEC",
    "MAX_TRANSFER_CHUNK_BYTES",
    "MAX_TRANSFER_CHUNK_WEIGHTS",
    "RForkTransferBackend",
]
