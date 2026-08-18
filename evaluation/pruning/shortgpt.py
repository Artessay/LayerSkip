"""Core scoring and layer-selection utilities for the ShortGPT baseline.

ShortGPT ranks complete transformer blocks by the cosine distance between each
block's input and output hidden states.  The helpers in this module deliberately
capture block boundaries with hooks: Hugging Face's final ``hidden_states``
entry may include the model's final normalization and therefore is not a clean
output for the last transformer block.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from numbers import Integral, Real
from typing import Any, List, Optional, Tuple

import torch
import torch.nn.functional as F


def block_influence(
    input_hidden: torch.Tensor,
    output_hidden: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
) -> Tuple[float, int]:
    """Return the summed ShortGPT cosine distance and valid-token count.

    The final dimension is treated as the hidden dimension; every preceding
    position is one token.  Distances are computed directly, rather than by
    constructing the quadratic token-by-token Gram matrix used by the original
    reference implementation.

    Args:
        input_hidden: Hidden states entering a transformer block.
        output_hidden: Hidden states returned by the same transformer block.
        attention_mask: Optional mask matching ``input_hidden.shape[:-1]``.
            Non-zero entries identify valid tokens.

    Returns:
        ``(sum_distance, valid_token_count)`` as a Python float and integer.
    """
    if not isinstance(input_hidden, torch.Tensor) or not isinstance(
        output_hidden, torch.Tensor
    ):
        raise TypeError("input_hidden and output_hidden must be torch tensors")
    if input_hidden.shape != output_hidden.shape:
        raise ValueError(
            "input_hidden and output_hidden must have identical shapes, got "
            f"{tuple(input_hidden.shape)} and {tuple(output_hidden.shape)}"
        )
    if input_hidden.ndim < 2:
        raise ValueError(
            "hidden states must include token and hidden dimensions, got "
            f"shape {tuple(input_hidden.shape)}"
        )
    if input_hidden.shape[-1] == 0:
        raise ValueError("the hidden dimension must be non-empty")

    distances = 1.0 - F.cosine_similarity(
        input_hidden.detach().float(),
        output_hidden.detach().float(),
        dim=-1,
        eps=1e-8,
    )

    if attention_mask is None:
        valid_count = distances.numel()
        if valid_count == 0:
            return 0.0, 0
        return float(distances.sum(dtype=torch.float64).item()), valid_count

    if not isinstance(attention_mask, torch.Tensor):
        raise TypeError("attention_mask must be a torch tensor or None")
    expected_mask_shape = input_hidden.shape[:-1]
    if tuple(attention_mask.shape) != tuple(expected_mask_shape):
        raise ValueError(
            "attention_mask must match the hidden-state token dimensions, got "
            f"{tuple(attention_mask.shape)} and expected {tuple(expected_mask_shape)}"
        )

    valid_mask = attention_mask.to(device=distances.device).ne(0)
    valid_count = int(valid_mask.sum().item())
    if valid_count == 0:
        return 0.0, 0
    return (
        float(distances.masked_select(valid_mask).sum(dtype=torch.float64).item()),
        valid_count,
    )


def _block_input(args: Tuple[Any, ...], kwargs: Mapping[str, Any]) -> torch.Tensor:
    hidden = args[0] if args else kwargs.get("hidden_states")
    if not isinstance(hidden, torch.Tensor):
        raise TypeError("a transformer block's first input must be hidden states")
    return hidden


def _block_output(output: Any) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, (tuple, list)):
        if output and isinstance(output[0], torch.Tensor):
            return output[0]
        raise TypeError("a transformer block tuple must start with hidden states")
    if isinstance(output, Mapping):
        for key in ("last_hidden_state", "hidden_states"):
            hidden = output.get(key)
            if isinstance(hidden, torch.Tensor):
                return hidden
    hidden = getattr(output, "last_hidden_state", None)
    if isinstance(hidden, torch.Tensor):
        return hidden
    raise TypeError("could not extract hidden states from transformer block output")


def score_shortgpt_blocks(
    model_wrapper: Any,
    token_batches: Iterable[Mapping[str, torch.Tensor]],
) -> List[float]:
    """Score every transformer block with streaming input/output hooks.

    ``model_wrapper`` is expected to expose ``model``, ``num_layers``,
    ``device``, and ``_resolve_transformer_layers()``.  Each token batch must
    contain ``input_ids`` and ``attention_mask``.  Hooks consume a block input
    as soon as its output becomes available, so complete hidden-state tuples are
    never retained.

    Returns:
        Average Block Influence for each block, in original 0-based order.
    """
    required_attributes = (
        "model",
        "num_layers",
        "device",
        "_resolve_transformer_layers",
    )
    missing = [name for name in required_attributes if not hasattr(model_wrapper, name)]
    if missing:
        raise TypeError(f"model_wrapper is missing required attributes: {missing}")

    num_layers = model_wrapper.num_layers
    if isinstance(num_layers, bool) or not isinstance(num_layers, Integral):
        raise TypeError("model_wrapper.num_layers must be an integer")
    num_layers = int(num_layers)
    if num_layers <= 0:
        raise ValueError("model_wrapper.num_layers must be positive")

    layers = list(model_wrapper._resolve_transformer_layers())
    if len(layers) != num_layers:
        raise ValueError(
            "resolved transformer-layer count does not match num_layers: "
            f"{len(layers)} != {num_layers}"
        )
    if any(not isinstance(layer, torch.nn.Module) for layer in layers):
        raise TypeError("all resolved transformer layers must be torch modules")

    model = model_wrapper.model
    if not isinstance(model, torch.nn.Module):
        raise TypeError("model_wrapper.model must be a torch module")

    score_sums = [0.0] * num_layers
    token_counts = [0] * num_layers
    pending_inputs: List[Optional[torch.Tensor]] = [None] * num_layers
    active_attention_mask: Optional[torch.Tensor] = None
    handles = []

    def make_pre_hook(layer_idx: int):
        def pre_hook(module, args, kwargs):
            del module
            if pending_inputs[layer_idx] is not None:
                raise RuntimeError(
                    f"transformer block {layer_idx} was entered again before returning"
                )
            pending_inputs[layer_idx] = _block_input(args, kwargs).detach()

        return pre_hook

    def make_forward_hook(layer_idx: int):
        def forward_hook(module, args, kwargs, output):
            del module, args, kwargs
            input_hidden = pending_inputs[layer_idx]
            pending_inputs[layer_idx] = None
            if input_hidden is None:
                raise RuntimeError(
                    f"missing captured input for transformer block {layer_idx}"
                )
            output_hidden = _block_output(output)
            distance_sum, valid_count = block_influence(
                input_hidden,
                output_hidden,
                active_attention_mask,
            )
            score_sums[layer_idx] += distance_sum
            token_counts[layer_idx] += valid_count

        return forward_hook

    for layer_idx, layer in enumerate(layers):
        handles.append(
            layer.register_forward_pre_hook(
                make_pre_hook(layer_idx),
                with_kwargs=True,
            )
        )
        handles.append(
            layer.register_forward_hook(
                make_forward_hook(layer_idx),
                with_kwargs=True,
            )
        )

    was_training = model.training
    model.eval()
    saw_batch = False
    try:
        with torch.inference_mode():
            for batch_idx, batch in enumerate(token_batches):
                saw_batch = True
                if not isinstance(batch, Mapping):
                    raise TypeError(f"token batch {batch_idx} must be a mapping")
                if "input_ids" not in batch or "attention_mask" not in batch:
                    raise ValueError(
                        f"token batch {batch_idx} must contain input_ids and attention_mask"
                    )
                if not isinstance(batch["input_ids"], torch.Tensor) or not isinstance(
                    batch["attention_mask"], torch.Tensor
                ):
                    raise TypeError("input_ids and attention_mask must be torch tensors")

                model_inputs = {
                    key: value.to(model_wrapper.device)
                    if isinstance(value, torch.Tensor)
                    else value
                    for key, value in batch.items()
                }
                active_attention_mask = model_inputs["attention_mask"]
                model_inputs["use_cache"] = False
                model_inputs["output_hidden_states"] = False
                try:
                    model(**model_inputs)
                finally:
                    active_attention_mask = None

                unfinished = [
                    idx for idx, hidden in enumerate(pending_inputs) if hidden is not None
                ]
                if unfinished:
                    raise RuntimeError(
                        "transformer blocks did not finish their forward hooks: "
                        f"{unfinished}"
                    )
    finally:
        active_attention_mask = None
        pending_inputs[:] = [None] * num_layers
        for handle in handles:
            handle.remove()
        if was_training:
            model.train()

    if not saw_batch:
        raise ValueError("token_batches must contain at least one batch")
    unscored = [idx for idx, count in enumerate(token_counts) if count == 0]
    if unscored:
        raise ValueError(f"no valid calibration tokens scored for layers {unscored}")

    return [
        score_sums[layer_idx] / token_counts[layer_idx]
        for layer_idx in range(num_layers)
    ]


def _normalise_scores(scores: Sequence[Real]) -> List[float]:
    if isinstance(scores, torch.Tensor):
        if scores.ndim != 1:
            raise ValueError("scores tensor must be one-dimensional")
        raw_scores: Sequence[Any] = scores.detach().cpu().tolist()
    else:
        if isinstance(scores, (str, bytes)):
            raise TypeError("scores must be a sequence of real numbers")
        try:
            raw_scores = list(scores)
        except TypeError as exc:
            raise TypeError("scores must be a sequence of real numbers") from exc

    if not raw_scores:
        raise ValueError("scores must contain at least one layer")

    normalised = []
    for layer_idx, value in enumerate(raw_scores):
        if isinstance(value, bool) or not isinstance(value, Real):
            raise TypeError(f"score {layer_idx} must be a real number")
        score = float(value)
        if not math.isfinite(score):
            raise ValueError(f"score {layer_idx} must be finite")
        normalised.append(score)
    return normalised


def select_shortgpt_layers(
    scores: Sequence[Real],
    prune_ratio: Optional[float] = None,
    num_remove: Optional[int] = None,
) -> List[int]:
    """Select 0-based block indices with the lowest ShortGPT scores.

    Exactly one pruning budget must be supplied.  Ratio budgets use
    ``floor(num_layers * prune_ratio)`` so the requested fraction is never
    exceeded.  Ties preserve original layer order.
    """
    normalised_scores = _normalise_scores(scores)
    num_layers = len(normalised_scores)

    if (prune_ratio is None) == (num_remove is None):
        raise ValueError("provide exactly one of prune_ratio and num_remove")

    if prune_ratio is not None:
        if isinstance(prune_ratio, bool) or not isinstance(prune_ratio, Real):
            raise TypeError("prune_ratio must be a real number")
        prune_ratio = float(prune_ratio)
        if not math.isfinite(prune_ratio):
            raise ValueError("prune_ratio must be finite")
        if not 0.0 <= prune_ratio <= 1.0:
            raise ValueError("prune_ratio must be within [0, 1]")
        remove_count = math.floor(num_layers * prune_ratio)
    else:
        if isinstance(num_remove, bool) or not isinstance(num_remove, Integral):
            raise TypeError("num_remove must be an integer")
        remove_count = int(num_remove)
        if not 0 <= remove_count <= num_layers:
            raise ValueError(f"num_remove must be within [0, {num_layers}]")

    ranked_indices = sorted(
        range(num_layers),
        key=lambda layer_idx: (normalised_scores[layer_idx], layer_idx),
    )
    return ranked_indices[:remove_count]


def _trace_integer(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    return int(value)


def _trace_layer_list(
    value: Any,
    *,
    name: str,
    num_layers: int,
) -> List[int]:
    if not isinstance(value, list):
        raise TypeError(f"{name} must be a list")

    layers = []
    for index, layer in enumerate(value):
        layer = _trace_integer(f"{name}[{index}]", layer)
        if layer < 0 or layer >= num_layers:
            raise ValueError(
                f"{name}[{index}]={layer} is outside [0, {num_layers})"
            )
        layers.append(layer)
    if len(set(layers)) != len(layers):
        raise ValueError(f"{name} must not contain duplicate layer IDs")
    return layers


def validate_shortgpt_trace(
    trace: Mapping[str, Any],
    *,
    num_layers: int,
    num_remove: int,
) -> None:
    """Validate a completed ShortGPT trace against its requested budget.

    Normal searches must contain a finite score for every original block and a
    complete ``removal_order`` equal to the stable score ordering (ties retain
    the lower original layer ID).  ``selected_layers`` is the sorted set formed
    by the first ``num_remove`` entries of that ordering.

    A zero-budget run is the sole exception to full scoring: it must explicitly
    record ``search_skipped='zero_budget'`` and use empty score and layer lists.
    The function does not mutate ``trace`` and permits unrelated metadata such
    as corpus statistics.
    """

    if not isinstance(trace, Mapping):
        raise TypeError("ShortGPT trace must be a mapping")

    num_layers = _trace_integer("num_layers", num_layers)
    num_remove = _trace_integer("num_remove", num_remove)
    if num_layers <= 0:
        raise ValueError("num_layers must be positive")
    if num_remove < 0 or num_remove > num_layers:
        raise ValueError(f"num_remove must be within [0, {num_layers}]")

    required = {
        "method",
        "num_layers",
        "num_remove",
        "scores",
        "removal_order",
        "selected_layers",
        "complete",
    }
    missing = sorted(required.difference(trace))
    if missing:
        raise ValueError(f"ShortGPT trace is missing required fields: {missing}")

    if trace["method"] != "shortgpt":
        raise ValueError(
            f"trace.method must be 'shortgpt', got {trace['method']!r}"
        )

    recorded_num_layers = _trace_integer(
        "trace.num_layers",
        trace["num_layers"],
    )
    if recorded_num_layers != num_layers:
        raise ValueError(
            f"trace.num_layers={recorded_num_layers} does not match requested "
            f"{num_layers}"
        )
    recorded_num_remove = _trace_integer(
        "trace.num_remove",
        trace["num_remove"],
    )
    if recorded_num_remove != num_remove:
        raise ValueError(
            f"trace.num_remove={recorded_num_remove} does not match requested "
            f"{num_remove}"
        )

    complete = trace["complete"]
    if not isinstance(complete, bool):
        raise TypeError("trace.complete must be a boolean")
    if not complete:
        raise ValueError("ShortGPT trace must be complete before it can be reused")

    if num_remove == 0:
        if trace.get("search_skipped") != "zero_budget":
            raise ValueError(
                "a zero-budget ShortGPT trace must set "
                "search_skipped='zero_budget'"
            )
        for field in ("scores", "removal_order", "selected_layers"):
            value = trace[field]
            if not isinstance(value, list):
                raise TypeError(f"trace.{field} must be a list")
            if value:
                raise ValueError(
                    f"trace.{field} must be empty when the zero-budget search "
                    "is skipped"
                )
        return

    if "search_skipped" in trace:
        raise ValueError(
            "a positive-budget ShortGPT trace cannot mark its search as skipped"
        )
    if not isinstance(trace["scores"], list):
        raise TypeError("trace.scores must be a list")
    scores = _normalise_scores(trace["scores"])
    if len(scores) != num_layers:
        raise ValueError(
            f"trace.scores must contain {num_layers} entries, got {len(scores)}"
        )

    expected_order = sorted(
        range(num_layers),
        key=lambda layer_idx: (scores[layer_idx], layer_idx),
    )
    removal_order = _trace_layer_list(
        trace["removal_order"],
        name="trace.removal_order",
        num_layers=num_layers,
    )
    if removal_order != expected_order:
        raise ValueError(
            "trace.removal_order must be the complete stable ascending score "
            f"ordering {expected_order}, got {removal_order}"
        )

    selected_layers = _trace_layer_list(
        trace["selected_layers"],
        name="trace.selected_layers",
        num_layers=num_layers,
    )
    expected_selected = sorted(expected_order[:num_remove])
    if selected_layers != expected_selected:
        raise ValueError(
            "trace.selected_layers must be the sorted pruning-budget prefix "
            f"{expected_selected}, got {selected_layers}"
        )


__all__ = [
    "block_influence",
    "score_shortgpt_blocks",
    "select_shortgpt_layers",
    "validate_shortgpt_trace",
]
