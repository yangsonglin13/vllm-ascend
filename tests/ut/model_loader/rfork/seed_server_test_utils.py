# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

import importlib.util
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest


@pytest.fixture
def live_session(runtime, monkeypatch):
    """Real session, FastAPI/uvicorn listener and HTTP client; native NPU and planner injected."""
    path = Path(__file__).resolve().parents[4] / "vllm_ascend/model_loader/rfork/seed_server.py"
    name = "rfork_live_seed_server_test"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    monkeypatch.setattr(runtime.session, "start_rfork_server", module.start_rfork_server)
    monkeypatch.setattr(runtime.session, "RForkSeedServerStartupError", module.RForkSeedServerStartupError)
    cfg = replace(
        runtime.config,
        seed_bind_host="127.0.0.1",
        seed_advertise_host="127.0.0.1",
        seed_timeout_sec=3.0,
        request_timeout_sec=1.0,
    )
    session = runtime.session.RForkSession(cfg, runtime.identity)
    session.planner = Mock(seed_key="live-key")
    session.planner.report_seed_once.return_value = True
    session.planner.remove_seed.return_value = True
    session.transfer_backend.transfer_session_id = "native-session"
    session.transfer_backend.weight_manifest = {"weight": [1234, 4, 4, [4], "float32"]}
    session.transfer_backend.weight_shapes = {"weight": (4,)}
    handles = []
    original = module.start_rfork_server

    def start(*args, **kwargs):
        handle = original(*args, **kwargs)
        handles.append(handle)
        return handle

    monkeypatch.setattr(runtime.session, "start_rfork_server", start)
    client = sys.modules["vllm_ascend.model_loader.rfork.seed_client"]
    yield SimpleNamespace(session=session, handles=handles, client=client, runtime=runtime)
    session.planner.remove_seed.side_effect = None
    session.planner.remove_seed.return_value = True
    session.shutdown()
    for handle in handles:
        assert handle.stop(), "live seed server leaked a thread"
