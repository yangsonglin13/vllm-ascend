# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest

from .rfork_test_support import _load_module


@pytest.fixture
def safety(monkeypatch):
    return _load_module(monkeypatch, "rfork_mutation_safety_test", "safety.py")


@pytest.mark.parametrize("config_slot", ["target", "draft"])
def test_mutation_rejects_any_retained_rfork_session(safety, config_slot):
    config = SimpleNamespace(load_config=SimpleNamespace(), speculative_config=SimpleNamespace())
    load_config = config.load_config
    if config_slot == "draft":
        load_config = SimpleNamespace()
        config.speculative_config.draft_load_config = load_config
    load_config.rfork_session = SimpleNamespace(state="CLEANUP_REQUIRED")

    with pytest.raises(RuntimeError, match="Restart with --load-format auto"):
        safety.ensure_no_rfork_session(config, "reload_weights")


def test_mutation_is_allowed_when_rfork_was_bypassed(safety):
    safety.ensure_no_rfork_session(SimpleNamespace(load_config=SimpleNamespace(load_format="rfork")), "sleep")


@pytest.fixture
def worker_methods(safety):
    path = Path(__file__).resolve().parents[4] / "vllm_ascend/worker/worker.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    worker = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "NPUWorker")
    names = {"sleep", "start_weight_update", "update_weights", "finish_weight_update", "reload_weights"}
    methods = [node for node in worker.body if isinstance(node, ast.FunctionDef) and node.name in names]
    namespace = {"ensure_no_rfork_session": safety.ensure_no_rfork_session}
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
