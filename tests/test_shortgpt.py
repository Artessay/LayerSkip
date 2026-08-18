"""Tests for the core ShortGPT block scorer and selector."""

from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from evaluation.pruning.shortgpt import (
    block_influence,
    score_shortgpt_blocks,
    select_shortgpt_layers,
    validate_shortgpt_trace,
)


def test_block_influence_masks_padding_tokens():
    input_hidden = torch.tensor(
        [[[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]]]
    )
    output_hidden = torch.tensor(
        [[[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]]]
    )

    distance_sum, count = block_influence(
        input_hidden,
        output_hidden,
        attention_mask=torch.tensor([[1, 1, 0]]),
    )

    assert distance_sum == pytest.approx(1.0)
    assert count == 2

    unmasked_sum, unmasked_count = block_influence(input_hidden, output_hidden)
    assert unmasked_sum == pytest.approx(3.0)
    assert unmasked_count == 3


@pytest.mark.parametrize(
    "input_hidden,output_hidden,mask,error",
    [
        (torch.zeros(1, 2), torch.zeros(1, 3), None, ValueError),
        (torch.zeros(2), torch.zeros(2), None, ValueError),
        (torch.zeros(1, 2, 3), torch.zeros(1, 2, 3), torch.ones(1, 3), ValueError),
    ],
)
def test_block_influence_validates_shapes(input_hidden, output_hidden, mask, error):
    with pytest.raises(error):
        block_influence(input_hidden, output_hidden, mask)


class _RotateBlock(torch.nn.Module):
    def forward(self, hidden_states, **kwargs):
        del kwargs
        rotation = hidden_states.new_tensor([[0.0, -1.0], [1.0, 0.0]])
        return hidden_states @ rotation.T


class _PaddingOnlyFlipBlock(torch.nn.Module):
    def forward(self, *, hidden_states, **kwargs):
        del kwargs
        should_flip = hidden_states[..., :1] < 0
        return (torch.where(should_flip, -hidden_states, hidden_states), None)


class _NegatingFinalNorm(torch.nn.Module):
    def forward(self, hidden_states):
        return -hidden_states


class _ToyCausalLM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList(
            [_RotateBlock(), _PaddingOnlyFlipBlock()]
        )
        self.final_norm = _NegatingFinalNorm()

    def forward(
        self,
        input_ids,
        attention_mask=None,
        use_cache=False,
        output_hidden_states=False,
    ):
        del attention_mask, use_cache, output_hidden_states
        hidden = F.one_hot(input_ids, num_classes=2).float()
        hidden = self.layers[0](hidden)
        hidden = self.layers[1](hidden_states=hidden)[0]
        return SimpleNamespace(last_hidden_state=self.final_norm(hidden))


class _ToyWrapper:
    def __init__(self):
        self.model = _ToyCausalLM()
        self.num_layers = len(self.model.layers)
        self.device = "cpu"

    def _resolve_transformer_layers(self):
        return self.model.layers


def test_score_shortgpt_blocks_uses_raw_last_block_output_and_padding_mask():
    wrapper = _ToyWrapper()
    wrapper.model.train()
    batches = [
        {
            # id 0 is valid.  id 1 is padding and is flipped only by layer 1.
            "input_ids": torch.tensor([[0, 1]]),
            "attention_mask": torch.tensor([[1, 0]]),
        }
    ]

    scores = score_shortgpt_blocks(wrapper, batches)

    # Layer 0 rotates the valid vector by 90 degrees.  Layer 1 leaves that
    # vector unchanged; its padded vector and the model's final norm must not
    # affect the score.
    assert scores == pytest.approx([1.0, 0.0])
    assert wrapper.model.training is True
    for layer in wrapper.model.layers:
        assert len(layer._forward_pre_hooks) == 0
        assert len(layer._forward_hooks) == 0


def test_score_shortgpt_blocks_rejects_empty_or_all_padding_calibration():
    wrapper = _ToyWrapper()
    with pytest.raises(ValueError, match="at least one batch"):
        score_shortgpt_blocks(wrapper, [])

    with pytest.raises(ValueError, match="no valid calibration tokens"):
        score_shortgpt_blocks(
            wrapper,
            [
                {
                    "input_ids": torch.tensor([[0, 1]]),
                    "attention_mask": torch.tensor([[0, 0]]),
                }
            ],
        )


def test_select_shortgpt_layers_is_stable_and_zero_based():
    scores = [0.2, 0.1, 0.1, 0.3]

    assert select_shortgpt_layers(scores, num_remove=2) == [1, 2]
    assert select_shortgpt_layers(scores, prune_ratio=0.5) == [1, 2]
    assert select_shortgpt_layers(scores, prune_ratio=0.26) == [1]
    assert select_shortgpt_layers(scores, num_remove=0) == []
    assert select_shortgpt_layers(scores, prune_ratio=1.0) == [1, 2, 0, 3]


@pytest.mark.parametrize(
    "kwargs,error",
    [
        ({}, ValueError),
        ({"prune_ratio": 0.5, "num_remove": 1}, ValueError),
        ({"prune_ratio": -0.1}, ValueError),
        ({"prune_ratio": 1.1}, ValueError),
        ({"prune_ratio": float("nan")}, ValueError),
        ({"prune_ratio": True}, TypeError),
        ({"num_remove": -1}, ValueError),
        ({"num_remove": 5}, ValueError),
        ({"num_remove": 1.0}, TypeError),
    ],
)
def test_select_shortgpt_layers_validates_budget(kwargs, error):
    with pytest.raises(error):
        select_shortgpt_layers([0.1, 0.2, 0.3, 0.4], **kwargs)


@pytest.mark.parametrize(
    "scores,error",
    [
        ([], ValueError),
        ([0.1, float("inf")], ValueError),
        ([0.1, float("nan")], ValueError),
        ([0.1, "bad"], TypeError),
        (torch.zeros(2, 2), ValueError),
    ],
)
def test_select_shortgpt_layers_validates_scores(scores, error):
    with pytest.raises(error):
        select_shortgpt_layers(scores, num_remove=1)


def _completed_trace():
    return {
        "method": "shortgpt",
        "num_layers": 4,
        "num_remove": 2,
        "scores": [0.4, 0.1, 0.3, 0.1],
        # Tied layers 1 and 3 retain original layer order.
        "removal_order": [1, 3, 2, 0],
        # Application order is sorted even though the budget prefix is [1, 3].
        "selected_layers": [1, 3],
        "corpus_stats": {"segments": 8},
        "complete": True,
    }


def test_validate_shortgpt_trace_accepts_complete_trace_without_mutating_it():
    trace = _completed_trace()
    original = deepcopy(trace)

    assert (
        validate_shortgpt_trace(trace, num_layers=4, num_remove=2)
        is None
    )
    assert trace == original


def test_validate_shortgpt_trace_accepts_explicit_zero_budget_skip():
    trace = {
        "method": "shortgpt",
        "num_layers": 4,
        "num_remove": 0,
        "scores": [],
        "removal_order": [],
        "selected_layers": [],
        "complete": True,
        "search_skipped": "zero_budget",
    }

    validate_shortgpt_trace(trace, num_layers=4, num_remove=0)


@pytest.mark.parametrize(
    "field,value,error,match",
    [
        ("method", "sleb", ValueError, "method"),
        ("num_layers", 5, ValueError, "num_layers"),
        ("num_remove", 1, ValueError, "num_remove"),
        ("scores", [0.4, 0.1, 0.3], ValueError, "scores"),
        ("scores", [0.4, 0.1, float("nan"), 0.1], ValueError, "finite"),
        ("scores", [0.4, 0.1, float("inf"), 0.1], ValueError, "finite"),
        ("removal_order", [1, 3, 2], ValueError, "complete stable"),
        ("removal_order", [3, 1, 2, 0], ValueError, "stable"),
        ("removal_order", [1, 3, 2, True], TypeError, "integer"),
        ("selected_layers", [3, 1], ValueError, "sorted pruning-budget"),
        ("selected_layers", [1, 2], ValueError, "budget prefix"),
        ("selected_layers", [1, True], TypeError, "integer"),
        ("complete", False, ValueError, "must be complete"),
        ("complete", 1, TypeError, "boolean"),
        ("search_skipped", "zero_budget", ValueError, "positive-budget"),
    ],
)
def test_validate_shortgpt_trace_rejects_tampered_core_fields(
    field,
    value,
    error,
    match,
):
    trace = _completed_trace()
    trace[field] = value

    with pytest.raises(error, match=match):
        validate_shortgpt_trace(trace, num_layers=4, num_remove=2)


def test_validate_shortgpt_trace_rejects_missing_required_field():
    trace = _completed_trace()
    del trace["scores"]

    with pytest.raises(ValueError, match="missing required fields.*scores"):
        validate_shortgpt_trace(trace, num_layers=4, num_remove=2)


@pytest.mark.parametrize(
    "field,value,error,match",
    [
        ("search_skipped", None, ValueError, "zero-budget"),
        ("scores", [0.1], ValueError, "must be empty"),
        ("removal_order", [0], ValueError, "must be empty"),
        ("selected_layers", [0], ValueError, "must be empty"),
    ],
)
def test_validate_shortgpt_trace_rejects_tampered_zero_budget_trace(
    field,
    value,
    error,
    match,
):
    trace = {
        "method": "shortgpt",
        "num_layers": 4,
        "num_remove": 0,
        "scores": [],
        "removal_order": [],
        "selected_layers": [],
        "complete": True,
        "search_skipped": "zero_budget",
    }
    trace[field] = value

    with pytest.raises(error, match=match):
        validate_shortgpt_trace(trace, num_layers=4, num_remove=0)
