# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Huawei Technologies Co., Ltd.

import logging
from types import SimpleNamespace

import pytest

from .rfork_test_support import _load_module, _stub


@pytest.fixture
def load_state(monkeypatch):
    _stub(monkeypatch, "vllm.logger", logger=logging.getLogger("rfork-load-state-test"))
    _load_module(monkeypatch, "vllm_ascend.model_loader.rfork.types", "types.py")
    return _load_module(monkeypatch, "vllm_ascend.model_loader.rfork.load_state", "load_state.py")


def test_capture_and_restore_load_derived_attributes(load_state):
    state = {
        "has_own_lm_head": False,
        "has_own_embed_tokens": True,
        "enable_confidence_head": False,
        "fc_rotation_applied": True,
    }
    assert load_state.capture_load_derived_state(SimpleNamespace(**state)) == state

    restored = SimpleNamespace(has_own_lm_head=True)
    load_state.restore_load_derived_state(restored, state)
    assert all(getattr(restored, name) is value for name, value in state.items())


def test_restore_validates_all_attributes_before_applying_any(load_state):
    model = SimpleNamespace(has_own_lm_head=True)
    with pytest.raises(ValueError, match="must be a boolean"):
        load_state.restore_load_derived_state(model, {"has_own_lm_head": False, "has_own_embed_tokens": 1})
    assert model.has_own_lm_head is True
    assert not hasattr(model, "has_own_embed_tokens")


def test_restore_runs_model_hook_and_propagates_failures(load_state):
    calls = []
    load_state.restore_load_derived_state(
        SimpleNamespace(_restore_load_derived_state=lambda: calls.append("hook")),
        None,
    )
    assert calls == ["hook"]

    def broken_hook():
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        load_state.restore_load_derived_state(SimpleNamespace(_restore_load_derived_state=broken_hook), None)


def test_binds_seed_shared_embedding_head_and_buffer(load_state):
    target_embed = SimpleNamespace(weight=object())
    target_head = SimpleNamespace(weight=object())
    target_buffer = object()
    draft_head = SimpleNamespace(weight=object())
    draft = SimpleNamespace(
        model=SimpleNamespace(
            embed_tokens=SimpleNamespace(weight=object()),
            topk_indices_buffer=object(),
            layers={"0": SimpleNamespace(shared_head=SimpleNamespace(head=draft_head))},
        ),
        lm_head=draft_head,
    )
    target = SimpleNamespace(
        model=SimpleNamespace(embed_tokens=target_embed, topk_indices_buffer=target_buffer),
        lm_head=target_head,
    )

    load_state.force_bind_seed_shared_modules(
        draft,
        target,
        (
            "model.embed_tokens.weight",
            "model.layers.0.shared_head.head.weight",
            "model.topk_indices_buffer",
        ),
    )

    assert draft.model.embed_tokens is target_embed
    assert draft.model.layers["0"].shared_head.head is target_head
    assert draft.lm_head is target_head
    assert draft.model.topk_indices_buffer is target_buffer


@pytest.mark.parametrize(
    ("name", "message"),
    [("future.weight", "no known module mapping"), ("lm_head.weight", "no accessible lm_head")],
)
def test_rejects_unresolvable_shared_names(load_state, name, message):
    with pytest.raises(RuntimeError, match=message):
        load_state.force_bind_seed_shared_modules(SimpleNamespace(), SimpleNamespace(), (name,))


def _owner(session):
    return SimpleNamespace(
        speculative_config=SimpleNamespace(draft_load_config=SimpleNamespace(rfork_draft_session=session))
    )


def test_finish_binds_before_promoting_deferred_seed(load_state):
    events = []
    target_embed = SimpleNamespace(weight=object())
    draft = SimpleNamespace(model=SimpleNamespace(embed_tokens=SimpleNamespace(weight=object())))
    target = SimpleNamespace(model=SimpleNamespace(embed_tokens=target_embed))
    session = SimpleNamespace(
        get_seed_shared_names=lambda: ("model.embed_tokens.weight",),
        has_deferred_seed_start=lambda: events.append("pending") or True,
        complete_deferred_seed_start=lambda: events.append("complete"),
    )

    load_state.finish_rfork_deferred_seed_start(_owner(session), draft, target)

    assert draft.model.embed_tokens is target_embed
    assert events == ["pending", "complete"]


def test_binding_failure_blocks_deferred_seed_promotion(load_state):
    promoted = []
    session = SimpleNamespace(
        get_seed_shared_names=lambda: ("bogus.weight",),
        has_deferred_seed_start=lambda: True,
        complete_deferred_seed_start=lambda: promoted.append(True),
    )

    with pytest.raises(RuntimeError, match="no known module mapping"):
        load_state.finish_rfork_deferred_seed_start(_owner(session), SimpleNamespace(), SimpleNamespace())
    assert promoted == []
