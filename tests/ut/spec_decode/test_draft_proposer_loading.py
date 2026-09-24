# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from vllm_ascend.spec_decode.draft_proposer import AscendDraftModelProposer


@pytest.mark.parametrize("separate_load_config", [False, True])
def test_draft_model_uses_explicit_load_config(monkeypatch, separate_load_config):
    proposer = AscendDraftModelProposer.__new__(AscendDraftModelProposer)
    target_load_config = object()
    draft_load_config = object() if separate_load_config else None
    draft_vllm_config = SimpleNamespace(load_config=target_load_config)
    proposer.speculative_config = SimpleNamespace(draft_load_config=draft_load_config)
    monkeypatch.setattr(proposer, "_create_draft_vllm_config", lambda: draft_vllm_config)

    loaded_model = object()
    get_model = Mock(return_value=loaded_model)
    monkeypatch.setattr("vllm_ascend.spec_decode.draft_proposer.get_model", get_model)
    monkeypatch.setattr("vllm.compilation.backends.set_model_tag", lambda tag: nullcontext())

    assert proposer._get_model() is loaded_model
    get_model.assert_called_once_with(
        vllm_config=draft_vllm_config,
        prefix="draft_model",
        load_config=draft_load_config,
    )
