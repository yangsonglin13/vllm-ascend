# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

import ast
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from .rfork_test_support import _load_module
from .test_late_lease_isolated import loader_runtime as loader_runtime


@pytest.fixture
def safety(monkeypatch):
    return _load_module(monkeypatch, "rfork_mutation_safety_test", "safety.py")


@pytest.mark.parametrize("draft", [False, True])
@pytest.mark.parametrize("mode", ["sleep", "target_sleep", "weight_transfer"])
def test_mutable_model_bypasses_seed_and_registration(request, monkeypatch, mode, draft):
    r = request.getfixturevalue("loader_runtime")
    r.config.runner_type = "draft" if draft else "generate"
    if mode == "sleep":
        r.config.enable_sleep_mode = True
    elif mode == "target_sleep":
        r.vc.model_config = SimpleNamespace(enable_sleep_mode=True)
    else:
        r.vc.weight_transfer_config = SimpleNamespace()
    ensure = Mock()
    initialize = Mock()
    monkeypatch.setattr(r.loader, "_ensure_rfork_session", ensure)
    monkeypatch.setattr(r.module, "initialize_model", initialize)
    fallback = Mock(return_value=r.fallback)
    monkeypatch.setattr(sys.modules["vllm.model_executor.model_loader"], "get_model", fallback)
    assert r.loader.load_model(r.vc, r.config) is r.fallback
    ensure.assert_not_called()
    initialize.assert_not_called()
    assert not r.session.mock_calls
    assert fallback.call_args.kwargs["load_config"].load_format == "auto"
    assert r.loader.load_config.load_format == "rfork"


@pytest.mark.parametrize("config_slot", ["target", "draft"])
@pytest.mark.parametrize("session_slot", ["rfork_session", "rfork_draft_session"])
def test_mutation_rejects_even_cleanup_retained_sessions(safety, config_slot, session_slot):
    config = SimpleNamespace(load_config=SimpleNamespace(), speculative_config=SimpleNamespace())
    load_config = config.load_config
    if config_slot == "draft":
        load_config = SimpleNamespace()
        config.speculative_config.draft_load_config = load_config
    setattr(load_config, session_slot, SimpleNamespace(state="CLEANUP_REQUIRED"))
    with pytest.raises(RuntimeError, match="Restart with --load-format auto"):
        safety.ensure_no_rfork_session(config, "reload_weights")


def test_mutation_allowed_when_rfork_was_bypassed(safety):
    config = SimpleNamespace(load_config=SimpleNamespace(load_format="rfork"))
    safety.ensure_no_rfork_session(config, "sleep")
    safety.ensure_no_rfork_session(SimpleNamespace(), "reload_weights")


@pytest.fixture
def worker_methods(safety):
    # Execute the actual worker method bodies with runtime imports excluded;
    # assertions below check that guards run before NPU/engine side effects.
    path = Path(__file__).resolve().parents[4] / "vllm_ascend/worker/worker.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    worker = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "NPUWorker")
    names = {"sleep", "start_weight_update", "update_weights", "finish_weight_update", "reload_weights"}
    methods = [node for node in worker.body if isinstance(node, ast.FunctionDef) and node.name in names]
    namespace = {"ensure_no_rfork_session": safety.ensure_no_rfork_session, "torch": torch}
    exec(compile(ast.Module(body=methods, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


@pytest.mark.parametrize(
    "operation", ["sleep", "start_weight_update", "update_weights", "finish_weight_update", "reload_weights"]
)
def test_worker_rejects_mutation_before_any_side_effect(worker_methods, operation):
    worker = SimpleNamespace(vllm_config=SimpleNamespace(load_config=SimpleNamespace(rfork_session=object())))
    args = ({},) if operation == "update_weights" else ()
    with pytest.raises(RuntimeError, match=f"{operation} is unavailable"):
        worker_methods[operation](worker, *args)


def test_worker_reload_without_session_preserves_arguments(worker_methods):
    reload = Mock()
    worker = SimpleNamespace(vllm_config=SimpleNamespace(), model_runner=SimpleNamespace(reload_weights=reload))
    worker_methods["reload_weights"](worker, "weights", revision="version")
    reload.assert_called_once_with("weights", revision="version")


@pytest.mark.parametrize("operation", ["start_weight_update", "update_weights", "finish_weight_update"])
def test_worker_weight_updates_without_session_reach_engine(worker_methods, monkeypatch, operation):
    engine = Mock()
    worker = SimpleNamespace(
        vllm_config=SimpleNamespace(weight_transfer_config=object()),
        weight_transfer_engine=engine,
        _weight_update_active=operation != "start_weight_update",
        _check_weight_transfer_engine=lambda: None,
        _check_nz_disabled=lambda: None,
        _is_checkpoint_format=False,
        model_runner=SimpleNamespace(model=Mock()),
        device="cpu",
    )
    monkeypatch.setattr(torch, "npu", SimpleNamespace(synchronize=Mock()), raising=False)
    args = (
        ({"weights": "chunk"},)
        if operation == "update_weights"
        else (False,)
        if operation == "start_weight_update"
        else ()
    )
    worker_methods[operation](worker, *args)
    if operation == "update_weights":
        engine.parse_update_info.assert_called_once_with(*args)
        engine.receive_weights.assert_called_once()
        torch.npu.synchronize.assert_called_once()
    assert worker._weight_update_active is (operation != "finish_weight_update")
