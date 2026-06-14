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

import time
from bisect import bisect_left
from typing import Any

import requests
import torch
from torch import nn
from vllm.logger import logger
from vllm.utils.network_utils import get_ip, get_open_port, join_host_port

from vllm_ascend import envs

MAX_TRANSFER_CHUNK_BYTES = 1024**3
MAX_TRANSFER_CHUNK_WEIGHTS = 512


def _normalize_weight_shape(shape: Any) -> tuple[int, ...] | None:
    if shape is None:
        return None
    if not isinstance(shape, (list, tuple)):
        return None
    if not all(isinstance(dim, int) and dim >= 0 for dim in shape):
        return None
    return tuple(shape)


def _parse_weight_info(weight_info: Any):
    if not isinstance(weight_info, (list, tuple)) or len(weight_info) not in (3, 4):
        return None

    seed_ptr, seed_len, seed_size = weight_info[:3]
    if not all(isinstance(value, int) for value in (seed_ptr, seed_len, seed_size)):
        return None

    seed_shape = None
    if len(weight_info) == 4:
        seed_shape = _normalize_weight_shape(weight_info[3])
        if seed_shape is None:
            return None

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


def _iter_tensors_in_public_attrs(prefix: str, value: Any):
    if not hasattr(value, "__dict__"):
        return

    for attr_name, attr_value in vars(value).items():
        if attr_name.startswith("_"):
            continue
        yield from _iter_tensors_in_value(f"{prefix}.{attr_name}", attr_value, set())


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


def _collect_transferable_tensors(model: nn.Module) -> list[tuple[str, torch.Tensor]]:
    scan_start_tic = time.time()
    seen_data_ptrs: set[int] = set()
    collected_tensors: list[tuple[str, torch.Tensor]] = []

    parameter_count = 0
    parameter_kept_count = 0
    parameter_duplicate_count = 0
    parameter_scan_tic = time.time()
    for name, tensor in model.named_parameters():
        parameter_count += 1
        is_kept, is_duplicate = _try_collect_transferable_tensor(
            name,
            tensor,
            seen_data_ptrs,
            collected_tensors,
        )
        parameter_kept_count += int(is_kept)
        parameter_duplicate_count += int(is_duplicate)
    parameter_scan_time = time.time() - parameter_scan_tic

    buffer_count = 0
    buffer_kept_count = 0
    buffer_duplicate_count = 0
    buffer_scan_tic = time.time()
    for name, tensor in model.named_buffers():
        buffer_count += 1
        is_kept, is_duplicate = _try_collect_transferable_tensor(
            name,
            tensor,
            seen_data_ptrs,
            collected_tensors,
        )
        buffer_kept_count += int(is_kept)
        buffer_duplicate_count += int(is_duplicate)
    buffer_scan_time = time.time() - buffer_scan_tic

    module_count = 0
    module_attr_count = 0
    module_attr_tensor_count = 0
    module_attr_kept_count = 0
    module_attr_duplicate_count = 0
    module_attr_scan_time = 0.0
    impl_attr_count = 0
    impl_shallow_tensor_count = 0
    impl_kept_count = 0
    impl_duplicate_count = 0
    impl_shallow_scan_time = 0.0
    impl_recursive_scan_enabled = envs.VLLM_ASCEND_RFORK_SCAN_IMPL_RECURSIVE
    module_scan_tic = time.time()
    # Some Ascend post-load paths replace checkpoint parameters with runtime
    # tensors stored as plain module attributes, e.g. MLA/SFA W_UV and W_UK_T.
    for module_prefix, module in model.named_modules():
        module_count += 1
        for attr_name, attr_value in vars(module).items():
            if attr_name.startswith("_") or isinstance(attr_value, nn.Module):
                continue

            attr_scan_tic = time.time()
            if attr_name == "impl":
                if impl_recursive_scan_enabled:
                    attr_tensors = list(_iter_tensors_in_value(attr_name, attr_value, set(), scan_objects=True))
                else:
                    attr_tensors = list(_iter_tensors_in_public_attrs(attr_name, attr_value))
            else:
                attr_tensors = list(_iter_tensors_in_value(attr_name, attr_value, set()))
            attr_scan_time = time.time() - attr_scan_tic
            if attr_name == "impl":
                impl_attr_count += 1
                impl_shallow_tensor_count += len(attr_tensors)
                impl_shallow_scan_time += attr_scan_time
            else:
                module_attr_count += 1
                module_attr_tensor_count += len(attr_tensors)
                module_attr_scan_time += attr_scan_time

            for tensor_name, tensor in attr_tensors:
                full_name = f"{module_prefix}.{tensor_name}" if module_prefix else tensor_name
                is_kept, is_duplicate = _try_collect_transferable_tensor(
                    full_name,
                    tensor,
                    seen_data_ptrs,
                    collected_tensors,
                )
                if attr_name == "impl":
                    impl_kept_count += int(is_kept)
                    impl_duplicate_count += int(is_duplicate)
                else:
                    module_attr_kept_count += int(is_kept)
                    module_attr_duplicate_count += int(is_duplicate)
    module_scan_time = time.time() - module_scan_tic

    logger.info(
        "iter_transferable_tensors details: total=%.4fs, "
        "named_parameters=%.4fs/%d/%d/%d, named_buffers=%.4fs/%d/%d/%d, "
        "module_loop=%.4fs, module_attrs=%.4fs/%d/%d/%d/%d, "
        "impl_attrs=%.4fs/%d/%d/%d/%d, impl_recursive=%s, "
        "modules=%d, tensors=%d, unique_ptrs=%d",
        time.time() - scan_start_tic,
        parameter_scan_time,
        parameter_count,
        parameter_kept_count,
        parameter_duplicate_count,
        buffer_scan_time,
        buffer_count,
        buffer_kept_count,
        buffer_duplicate_count,
        module_scan_time,
        module_attr_scan_time,
        module_attr_count,
        module_attr_tensor_count,
        module_attr_kept_count,
        module_attr_duplicate_count,
        impl_shallow_scan_time,
        impl_attr_count,
        impl_shallow_tensor_count,
        impl_kept_count,
        impl_duplicate_count,
        impl_recursive_scan_enabled,
        module_count,
        len(collected_tensors),
        len(seen_data_ptrs),
    )
    return collected_tensors


def _iter_transferable_tensors(model: nn.Module):
    yield from _collect_transferable_tensors(model)


def _block_contains_weight_ptr(address: int, size: int, sorted_weight_ptrs: list[int]) -> bool:
    index = bisect_left(sorted_weight_ptrs, address)
    return index < len(sorted_weight_ptrs) and sorted_weight_ptrs[index] < address + size


def _iter_transfer_chunks(
    weight_names: list[str],
    seed_ptr_list: list[int],
    client_ptr_list: list[int],
    client_len_list: list[int],
):
    chunk_start = 0
    chunk_bytes = 0
    chunk_weights = 0

    for index, length in enumerate(client_len_list):
        should_flush = chunk_weights > 0 and (
            chunk_bytes + length > MAX_TRANSFER_CHUNK_BYTES or chunk_weights >= MAX_TRANSFER_CHUNK_WEIGHTS
        )
        if should_flush:
            yield (
                weight_names[chunk_start:index],
                seed_ptr_list[chunk_start:index],
                client_ptr_list[chunk_start:index],
                client_len_list[chunk_start:index],
            )
            chunk_start = index
            chunk_bytes = 0
            chunk_weights = 0

        chunk_bytes += length
        chunk_weights += 1

    if chunk_weights > 0:
        yield (
            weight_names[chunk_start:],
            seed_ptr_list[chunk_start:],
            client_ptr_list[chunk_start:],
            client_len_list[chunk_start:],
        )


class RForkTransferBackend:
    def __init__(self):
        self.rfork_transfer_engine: Any | None = None
        self.rfork_transfer_engine_session_id = None
        self.rfork_transfer_engine_weights_info_dict = None
        self.rfork_transfer_engine_weights_shape_dict = None
        self.registered_weight_blocks = []
        self._is_initialized = False
        self.init_transfer_engine()

    def init_transfer_engine(self):
        try:
            from yr.datasystem import TransferEngine  # type: ignore[import-not-found]
        except ImportError as e:
            err_msg = (
                "Failed to import TransferEngine from yr.datasystem. "
                "Please install @yuanrong-datasystem/transfer_engine."
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
        self._is_initialized = True

    def is_initialized(self) -> bool:
        return self._is_initialized

    def _get_transfer_engine(self) -> Any:
        if self.rfork_transfer_engine is None:
            raise RuntimeError("TransferEngine is not initialized.")
        return self.rfork_transfer_engine

    def register_memory_region(self, model):
        transfer_engine = self._get_transfer_engine()
        start_reg_mr_tic = time.time()

        weight_mr_dict = {}
        weight_shape_dict = {}
        weight_addr_set = set()
        weight_bytes = 0
        collect_weights_tic = time.time()
        for name, weight in _iter_transferable_tensors(model):
            weight_mr_dict[name] = (
                weight.data_ptr(),
                weight.numel(),
                weight.element_size(),
            )
            weight_shape_dict[name] = tuple(weight.shape)
            weight_addr_set.add(weight.data_ptr())
            weight_bytes += weight.numel() * weight.element_size()
        collect_weights_time = time.time() - collect_weights_tic

        sort_ptrs_tic = time.time()
        sorted_weight_ptrs = sorted(weight_addr_set)
        sort_ptrs_time = time.time() - sort_ptrs_tic

        memory_snapshot_tic = time.time()
        memory_snapshot = torch.npu.memory.memory_snapshot()
        memory_snapshot_time = time.time() - memory_snapshot_tic

        scan_snapshot_tic = time.time()
        weight_blocks_for_reg_mr = []
        for segment in memory_snapshot:
            current_weight_block = None
            for block in segment.get("blocks", []):
                address = block.get("address", -1)
                size = block.get("size", -1)
                state = block.get("state", "")
                if address < 0 or size < 0 or state == "":
                    continue
                if state == "active_allocated" and _block_contains_weight_ptr(address, size, sorted_weight_ptrs):
                    if current_weight_block is None:
                        current_weight_block = (address, size)
                    elif current_weight_block[0] + current_weight_block[1] == address:
                        current_weight_block = (
                            current_weight_block[0],
                            current_weight_block[1] + size,
                        )
                    else:
                        weight_blocks_for_reg_mr.append(current_weight_block)
                        current_weight_block = (address, size)
            if current_weight_block is not None:
                weight_blocks_for_reg_mr.append(current_weight_block)
        scan_snapshot_time = time.time() - scan_snapshot_tic

        addresses, sizes = zip(*weight_blocks_for_reg_mr) if weight_blocks_for_reg_mr else ((), ())
        batch_register_tic = time.time()
        ret = transfer_engine.batch_register_memory(addresses, sizes)
        batch_register_time = time.time() - batch_register_tic
        if ret.is_error():
            logger.error(
                "batch_register_memory failed for %d blocks, ret: %s",
                len(weight_blocks_for_reg_mr),
                ret.to_string(),
            )
            return False

        self.rfork_transfer_engine_weights_info_dict = weight_mr_dict
        self.rfork_transfer_engine_weights_shape_dict = weight_shape_dict
        self.registered_weight_blocks = weight_blocks_for_reg_mr

        registered_bytes = sum(sizes)
        logger.info(
            "register_memory_region details: collect_weights=%.4fs, "
            "sort_ptrs=%.4fs, memory_snapshot=%.4fs, scan_snapshot=%.4fs, "
            "batch_register=%.4fs, weights=%d, unique_ptrs=%d, "
            "snapshot_segments=%d, registered_blocks=%d, weight_bytes=%.2f GiB, "
            "registered_bytes=%.2f GiB",
            collect_weights_time,
            sort_ptrs_time,
            memory_snapshot_time,
            scan_snapshot_time,
            batch_register_time,
            len(weight_mr_dict),
            len(weight_addr_set),
            len(memory_snapshot),
            len(weight_blocks_for_reg_mr),
            weight_bytes / (1024**3),
            registered_bytes / (1024**3),
        )
        logger.info(
            "register_memory_region time: %.4fs, weights: %d",
            time.time() - start_reg_mr_tic,
            len(weight_mr_dict),
        )
        return True

    def unregister_memory_region(self) -> bool:
        transfer_engine = self._get_transfer_engine()
        start_unreg_mr_tic = time.time()
        if not self.registered_weight_blocks:
            self.rfork_transfer_engine_weights_info_dict = None
            self.rfork_transfer_engine_weights_shape_dict = None
            logger.debug("unregister_memory_region skipped because no blocks are registered.")
            return True

        ret = transfer_engine.batch_unregister_memory([address for address, _ in self.registered_weight_blocks])
        if ret.is_error():
            logger.error(
                "batch_unregister_memory failed for %d blocks, ret: %s",
                len(self.registered_weight_blocks),
                ret.to_string(),
            )
            return False
        self.rfork_transfer_engine_weights_info_dict = None
        self.rfork_transfer_engine_weights_shape_dict = None
        self.registered_weight_blocks = []
        logger.info(
            "unregister_memory_region time: %.4fs",
            time.time() - start_unreg_mr_tic,
        )
        return True

    def recv_from_source(
        self,
        model,
        seed_instance_ip,
        seed_instance_service_port,
        local_seed_key,
    ):
        transfer_engine = self._get_transfer_engine()
        recv_start_tic = time.time()
        seed_url = f"http://{seed_instance_ip}:{seed_instance_service_port}"
        get_remote_info_tic = time.time()
        seed_session_id, seed_weight_info, seed_weight_shapes = get_remote_instance_transfer_engine_info(
            seed_url,
            local_seed_key,
        )
        get_remote_info_time = time.time() - get_remote_info_tic
        if seed_session_id is None or seed_weight_info is None:
            logger.error("Cannot get transfer engine session or weight info.")
            return False

        prepare_metadata_tic = time.time()
        seed_ptr_list = []
        client_ptr_list = []
        client_len_list = []
        weight_names = []
        reshape_events: list[tuple[str, tuple[int, ...], tuple[int, ...]]] = []
        for name, tensor in _iter_transferable_tensors(model):
            weight_info = seed_weight_info.get(name, None)
            if weight_info is None:
                logger.error("Cannot find weight info for %s.", name)
                return False

            parsed_weight_info = _parse_weight_info(weight_info)
            if parsed_weight_info is None:
                logger.error("Invalid weight info for %s: %s", name, weight_info)
                return False

            seed_ptr, seed_len, seed_size, seed_shape = parsed_weight_info
            if seed_shape is None and isinstance(seed_weight_shapes, dict):
                seed_shape = _normalize_weight_shape(seed_weight_shapes.get(name))
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

            if not _reshape_tensor_to_seed_shape(name, tensor, seed_shape, reshape_events):
                return False
            _update_registered_weight_shape(
                self.rfork_transfer_engine_weights_shape_dict,
                name,
                tensor,
            )

            seed_ptr_list.append(seed_ptr)
            client_ptr_list.append(tensor.data_ptr())
            client_len_list.append(tensor.numel() * tensor.element_size())
            weight_names.append(name)
        prepare_metadata_time = time.time() - prepare_metadata_tic

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

        build_chunks_tic = time.time()
        transfer_chunks = list(
            _iter_transfer_chunks(
                weight_names,
                seed_ptr_list,
                client_ptr_list,
                client_len_list,
            )
        )
        build_chunks_time = time.time() - build_chunks_tic
        total_transfer_bytes = sum(client_len_list)
        logger.info(
            "recv_from_source prepare details: get_remote_info=%.4fs, "
            "prepare_metadata=%.4fs, build_chunks=%.4fs, weights=%d, "
            "seed_weights=%d, shape_info=%s, reshaped=%d, total bytes=%.2f GiB",
            get_remote_info_time,
            prepare_metadata_time,
            build_chunks_time,
            len(client_len_list),
            len(seed_weight_info),
            isinstance(seed_weight_shapes, dict),
            len(reshape_events),
            total_transfer_bytes / (1024**3),
        )

        start_transfer_tic = time.time()
        logger.info(
            "transfer weights starts, weights: %d, chunks: %d, total bytes: %.2f GiB",
            len(client_len_list),
            len(transfer_chunks),
            total_transfer_bytes / (1024**3),
        )
        for index, (chunk_names, chunk_seed_ptrs, chunk_client_ptrs, chunk_lengths) in enumerate(transfer_chunks, 1):
            chunk_start_tic = time.time()
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
                time.time() - chunk_start_tic,
            )

        transfer_time = time.time() - start_transfer_tic
        logger.info("transfer weights time: %.4fs", transfer_time)
        logger.info(
            "recv_from_source total time: %.4fs, pre_transfer_prepare=%.4fs, transfer=%.4fs",
            time.time() - recv_start_tic,
            start_transfer_tic - recv_start_tic,
            transfer_time,
        )
        return True


def get_remote_instance_transfer_engine_info(seed_url: str, local_seed_key: str):
    try:
        get_info_tic = time.time()
        response = requests.get(
            f"{seed_url}/get_rfork_transfer_engine_info",
            params={"seed_key": local_seed_key},
        )
        get_info_time = time.time() - get_info_tic
        if response.status_code != 200:
            logger.error(
                "GET %s/get_rfork_transfer_engine_info failed: %s, time: %.4fs",
                seed_url,
                response.status_code,
                get_info_time,
            )
            return None, None, None

        parse_info_tic = time.time()
        data = response.json()
        info = data.get("rfork_transfer_engine_info", None)
        parse_info_time = time.time() - parse_info_tic
        if info is not None and isinstance(info, list) and len(info) == 2:
            get_shape_tic = time.time()
            shape_info = get_remote_instance_weight_shape_info(seed_url, local_seed_key)
            get_shape_time = time.time() - get_shape_tic
            logger.info(
                "get_remote_instance_transfer_engine_info details: "
                "get_info=%.4fs, parse_info=%.4fs, get_shape=%.4fs, "
                "seed_weights=%d, shape_info=%s",
                get_info_time,
                parse_info_time,
                get_shape_time,
                len(info[1]) if isinstance(info[1], dict) else -1,
                isinstance(shape_info, dict),
            )
            return info[0], info[1], shape_info

        logger.error(
            "Failed to get rfork_transfer_engine_info in response from %s.",
            seed_url,
        )
        return None, None, None
    except Exception as e:
        logger.error("Exception getting transfer engine info from %s: %s", seed_url, e)
        return None, None, None


def get_remote_instance_weight_shape_info(seed_url: str, local_seed_key: str):
    try:
        get_shape_tic = time.time()
        response = requests.get(
            f"{seed_url}/get_rfork_transfer_engine_shape_info",
            params={"seed_key": local_seed_key},
        )
        get_shape_time = time.time() - get_shape_tic
        if response.status_code != 200:
            logger.debug(
                "GET %s/get_rfork_transfer_engine_shape_info failed: %s, time: %.4fs",
                seed_url,
                response.status_code,
                get_shape_time,
            )
            return None

        parse_shape_tic = time.time()
        data = response.json()
        info = data.get("rfork_transfer_engine_shape_info", None)
        parse_shape_time = time.time() - parse_shape_tic
        if info is None or isinstance(info, dict):
            logger.info(
                "get_remote_instance_weight_shape_info details: get_shape=%.4fs, parse_shape=%.4fs, shapes=%d",
                get_shape_time,
                parse_shape_time,
                len(info) if isinstance(info, dict) else 0,
            )
            return info

        logger.error(
            "Failed to get rfork_transfer_engine_shape_info in response from %s.",
            seed_url,
        )
        return None
    except Exception as e:
        logger.debug("Exception getting transfer engine shape info from %s: %s", seed_url, e)
        return None
