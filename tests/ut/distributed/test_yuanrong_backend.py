# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from unittest.mock import MagicMock

import pytest

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend.yuanrong_backend import (
    YuanrongBackend,
    YuanrongConfig,
    YuanrongHelper,
    _resolve_multi_buffer_apis,
)


def _make_backend():
    backend = YuanrongBackend.__new__(YuanrongBackend)
    backend._ensure_device_ready = MagicMock()
    backend._helper = MagicMock()
    backend._helper.device_id = 3
    backend._helper.normalize_keys.side_effect = lambda keys: [f"normalized-{key}" for key in keys]
    backend._helper.make_blob_lists.return_value = ["blob-list"]
    backend._hetero_client = MagicMock()
    backend._ds_set_param = object()
    return backend


def test_multi_buffer_get_and_put_use_new_sdk_api():
    backend = _make_backend()
    backend._multi_buffer_get = MagicMock(return_value=[])
    backend._multi_buffer_put = MagicMock()
    keys = ["key0"]
    addrs = [[100, 200]]
    sizes = [[10, 20]]

    backend.get(keys, addrs, sizes)
    backend.put(keys, addrs, sizes)

    normalized_keys = ["normalized-key0"]
    backend._multi_buffer_get.assert_called_once_with(normalized_keys, 3, addrs, sizes, 0)
    backend._multi_buffer_put.assert_called_once_with(normalized_keys, 3, addrs, sizes, backend._ds_set_param)
    backend._helper.make_blob_lists.assert_not_called()
    backend._hetero_client.mget_h2d.assert_not_called()
    backend._hetero_client.mset_d2h.assert_not_called()


def test_old_sdk_falls_back_to_blob_wrappers():
    backend = _make_backend()
    backend._multi_buffer_get = None
    backend._multi_buffer_put = None
    backend._hetero_client.mget_h2d.return_value = []
    keys = ["key0"]
    addrs = [[100]]
    sizes = [[10]]

    backend.get(keys, addrs, sizes)
    backend.put(keys, addrs, sizes)

    normalized_keys = ["normalized-key0"]
    assert backend._helper.make_blob_lists.call_count == 2
    backend._hetero_client.mget_h2d.assert_called_once_with(normalized_keys, ["blob-list"], 0)
    backend._hetero_client.mset_d2h.assert_called_once_with(normalized_keys, ["blob-list"], backend._ds_set_param)


def test_device_id_requires_set_device():
    helper = YuanrongHelper(MagicMock(), MagicMock())

    with pytest.raises(RuntimeError, match="device id is not initialized"):
        _ = helper.device_id


@pytest.mark.parametrize("mode", ["auto", "on"])
def test_multi_buffer_api_mode_selects_new_sdk(mode):
    client = MagicMock()

    get_api, put_api = _resolve_multi_buffer_apis(client, mode)

    assert get_api is client.mget_h2d_from_multi_buffers
    assert put_api is client.mset_d2h_from_multi_buffers


@pytest.mark.parametrize("mode", ["auto", "off"])
def test_multi_buffer_api_mode_selects_legacy_sdk(mode):
    client = object()

    assert _resolve_multi_buffer_apis(client, mode) == (None, None)


def test_multi_buffer_api_on_rejects_old_sdk():
    with pytest.raises(RuntimeError, match="mget_h2d_from_multi_buffers.*mset_d2h_from_multi_buffers"):
        _resolve_multi_buffer_apis(object(), "on")


def test_yuanrong_config_rejects_invalid_multi_buffer_api(monkeypatch):
    monkeypatch.setenv("DS_WORKER_ADDR", "127.0.0.1:31501")
    monkeypatch.setenv("VLLM_ASCEND_YUANRONG_MULTI_BUFFER_API", "invalid")

    with pytest.raises(ValueError, match="must be one of"):
        YuanrongConfig.load_from_env()
