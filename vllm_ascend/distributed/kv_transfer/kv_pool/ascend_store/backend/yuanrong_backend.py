import os
import time
from dataclasses import dataclass
from typing import Any

import torch
from vllm.config import ParallelConfig
from vllm.logger import logger
from vllm.utils.network_utils import split_host_port

from vllm_ascend import envs
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend.backend import Backend

_MULTI_BUFFER_API_MODES = {"auto", "on", "off"}


def _sum_transfer_bytes(sizes: list[list[int]]) -> int:
    return sum(sum(size_group) for size_group in sizes)


def _resolve_multi_buffer_apis(client: Any, mode: str):
    if mode == "off":
        return None, None

    get_api = getattr(client, "mget_h2d_from_multi_buffers", None)
    put_api = getattr(client, "mset_d2h_from_multi_buffers", None)
    if get_api is not None and put_api is not None:
        return get_api, put_api

    if mode == "on":
        missing = []
        if get_api is None:
            missing.append("mget_h2d_from_multi_buffers")
        if put_api is None:
            missing.append("mset_d2h_from_multi_buffers")
        raise RuntimeError(
            "VLLM_ASCEND_YUANRONG_MULTI_BUFFER_API=on requires an "
            f"openyuanrong-datasystem SDK with: {', '.join(missing)}"
        )

    return None, None


@dataclass
class YuanrongConfig:
    worker_addr: str
    enable_exclusive_connection: bool
    enable_remote_h2d: bool
    multi_buffer_api: str

    @staticmethod
    def load_from_env() -> "YuanrongConfig":
        worker_addr = os.getenv("DS_WORKER_ADDR")
        if not worker_addr:
            raise ValueError("Environment variable DS_WORKER_ADDR is required, expected format '<host>:<port>'.")
        multi_buffer_api = envs.VLLM_ASCEND_YUANRONG_MULTI_BUFFER_API
        if multi_buffer_api not in _MULTI_BUFFER_API_MODES:
            raise ValueError(
                "VLLM_ASCEND_YUANRONG_MULTI_BUFFER_API must be one of "
                f"{sorted(_MULTI_BUFFER_API_MODES)}, got '{multi_buffer_api}'."
            )

        return YuanrongConfig(
            worker_addr=worker_addr,
            enable_exclusive_connection=bool(int(os.getenv("DS_ENABLE_EXCLUSIVE_CONNECTION", "0"))),
            enable_remote_h2d=bool(int(os.getenv("DS_ENABLE_REMOTE_H2D", "0"))),
            multi_buffer_api=multi_buffer_api,
        )


class YuanrongHelper:
    def __init__(self, blob_cls, blob_list_cls):
        self._blob_cls = blob_cls
        self._blob_list_cls = blob_list_cls
        self._device_id: int | None = None

    @property
    def device_id(self) -> int:
        if self._device_id is None:
            raise RuntimeError("Yuanrong backend device id is not initialized.")
        return self._device_id

    def make_blob_lists(self, addrs_list: list[list[int]], sizes_list: list[list[int]]) -> list[Any]:
        total = len(addrs_list)
        if total != len(sizes_list):
            raise ValueError("Address list and size list length mismatch.")

        device_id = self.device_id

        blob_lists: list[Any] = []
        for addrs, sizes in zip(addrs_list, sizes_list):
            if len(addrs) != len(sizes):
                raise ValueError("Address list and size list length mismatch.")
            blobs = [
                self._blob_cls(addr, size)  # type: ignore[misc]
                for addr, size in zip(addrs, sizes)
            ]
            blob_lists.append(
                self._blob_list_cls(device_id, blobs)  # type: ignore[misc]
            )
        return blob_lists


class YuanrongBackend(Backend):
    def __init__(self, parallel_config: ParallelConfig):
        try:
            from yr.datasystem.hetero_client import Blob, DeviceBlobList, HeteroClient  # type: ignore[import-not-found]
            from yr.datasystem.kv_client import SetParam  # type: ignore[import-not-found]
            from yr.datasystem.object_client import WriteMode  # type: ignore[import-not-found]
        except ImportError as exc:
            raise ImportError("Please install openyuanrong-datasystem to use the yuanrong backend.") from exc

        self.rank = parallel_config.rank
        self._helper = YuanrongHelper(Blob, DeviceBlobList)
        self._ds_set_param = SetParam()
        self._ds_set_param.write_mode = WriteMode.NONE_L2_CACHE_EVICT

        self.config = YuanrongConfig.load_from_env()
        try:
            host, port = split_host_port(self.config.worker_addr)
        except Exception as exc:
            raise ValueError(f"Invalid DS_WORKER_ADDR '{self.config.worker_addr}', expected '<host>:<port>'.") from exc
        self._hetero_client = HeteroClient(
            host,
            int(port),
            enable_exclusive_connection=self.config.enable_exclusive_connection,
            enable_remote_h2d=self.config.enable_remote_h2d,
        )
        self._hetero_client.init()
        self._multi_buffer_get, self._multi_buffer_put = _resolve_multi_buffer_apis(
            self._hetero_client, self.config.multi_buffer_api
        )
        selected_api = "multi_buffer" if self._multi_buffer_get is not None else "legacy_blob"
        logger.info(
            "Yuanrong descriptor API configured=%s, selected=%s",
            self.config.multi_buffer_api,
            selected_api,
        )

    def _ensure_device_ready(self):
        if self._helper._device_id is None:
            self.set_device()

    def set_device(self):
        device = torch.device(f"npu:{self.rank}")
        torch.npu.set_device(device)
        self._helper._device_id = int(torch.npu.current_device())

    def register_buffer(self, ptrs: list[int], lengths: list[int]):
        # Yuanrong APIs consume device pointers directly when building blob
        # lists. No explicit pre-registration is required.
        self._ensure_device_ready()

    def exists(self, keys: list[str]) -> list[int]:
        if len(keys) == 0:
            return []
        try:
            exists = self._hetero_client.exist(keys)  # type: ignore[union-attr]
            return [1 if value else 0 for value in exists]
        except Exception as exc:
            logger.error("Failed to check keys %s: %s", keys, exc)
            return [0] * len(keys)

    def get(self, keys: list[str], addrs: list[list[int]], sizes: list[list[int]]):
        if len(keys) == 0:
            return
        try:
            self._ensure_device_ready()
            failed_keys: list[str]
            start_time = time.perf_counter()
            try:
                if self._multi_buffer_get is not None:
                    failed_keys = self._multi_buffer_get(keys, self._helper.device_id, addrs, sizes, 0)
                else:
                    blob_lists = self._helper.make_blob_lists(addrs, sizes)
                    failed_keys = self._hetero_client.mget_h2d(  # type: ignore[union-attr]
                        keys, blob_lists, 0
                    )
            finally:
                elapsed_ms = (time.perf_counter() - start_time) * 1000
                logger.info("Yuanrong load_kvc took %.3f ms, bytes=%d", elapsed_ms, _sum_transfer_bytes(sizes))
            for key in failed_keys:
                logger.error("Failed to get key %s", key)
        except Exception as exc:
            logger.error("Failed to get keys %s: %s", keys, exc)

    def put(self, keys: list[str], addrs: list[list[int]], sizes: list[list[int]]):
        if len(keys) == 0:
            return
        try:
            self._ensure_device_ready()
            start_time = time.perf_counter()
            try:
                if self._multi_buffer_put is not None:
                    self._multi_buffer_put(keys, self._helper.device_id, addrs, sizes, self._ds_set_param)
                else:
                    blob_lists = self._helper.make_blob_lists(addrs, sizes)
                    self._hetero_client.mset_d2h(  # type: ignore[union-attr]
                        keys, blob_lists, self._ds_set_param
                    )
            finally:
                elapsed_ms = (time.perf_counter() - start_time) * 1000
                logger.info("Yuanrong store_kvc took %.3f ms, bytes=%d", elapsed_ms, _sum_transfer_bytes(sizes))
        except Exception as exc:
            logger.error("Failed to put keys %s: %s", keys, exc)
