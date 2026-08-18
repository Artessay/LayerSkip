"""Pure greedy search for TALE task-aware layer elimination.

This is an independent, model-agnostic implementation of the search procedure
described in ``TELL-TALE: Task Efficient LLMs with Task Aware Layer
Elimination`` (Naim et al., Findings of ACL 2026):

* https://aclanthology.org/2026.findings-acl.1136/
* Official repository: https://github.com/omyokun/tale
* Reference search implementation (fixed revision):
  https://github.com/omyokun/tale/blob/d10cec53295ab4ce544e553509935ffcf0ac3e0d/src/evaluate.py

Only the algorithmic search semantics are adapted: evaluate every remaining
layer, greedily keep the highest-scoring removal, and compare it against the
fixed dense-baseline floor ``baseline - threshold``.  No official model,
dataset, evaluation, or CLI code is reused, and no source text is copied.
In particular, the tolerance is *not* subtracted from the current greedy
configuration.

The module is model- and task-agnostic.  Callers provide ``evaluate_fn``, which
maps a sorted tuple of original, zero-based layer IDs to a scalar task score.
The returned trace is JSON-serialisable and may be supplied back through
``trace`` to resume a partially evaluated round.
"""

from __future__ import annotations

import copy
import math
from numbers import Real
from typing import Any, Callable, Dict, List, Optional, Tuple


EvaluateFn = Callable[[Tuple[int, ...]], float]
CheckpointFn = Callable[[Dict[str, Any]], None]

_SCHEMA_VERSION = 1


def _score(value: Any, label: str) -> float:
    """Return ``value`` as a finite float or raise a useful error."""

    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{label} must be a real number, got {value!r}")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite, got {value!r}")
    return result


def _optional_count(name: str, value: Optional[int], num_layers: int) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer or None")
    if value < 0 or value > num_layers:
        raise ValueError(f"{name} must be within [0, {num_layers}], got {value}")
    return value


def _config(removed_layers: List[int], score: float) -> Dict[str, Any]:
    return {
        "removed_layers": sorted(removed_layers),
        "score": float(score),
    }


def _accepted_configs(state: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Reconstruct the greedy trajectory, including the dense baseline."""

    configs = [_config([], state["baseline"])]
    removed: List[int] = []
    for round_state in state["rounds"]:
        if not round_state.get("accepted", False):
            continue
        removed.append(round_state["selected_layer"])
        configs.append(_config(removed, round_state["selected_score"]))
    return configs


def _refresh_derived(state: Dict[str, Any]) -> None:
    """Refresh BEST/BSBA/final/budget views from accepted rounds."""

    if state.get("baseline") is None:
        return

    configs = _accepted_configs(state)
    state["best"] = copy.deepcopy(
        max(configs, key=lambda item: (item["score"], len(item["removed_layers"])))
    )

    baseline = state["baseline"]
    bsba_configs = [item for item in configs if item["score"] >= baseline]
    state["bsba"] = copy.deepcopy(
        max(bsba_configs, key=lambda item: len(item["removed_layers"]))
    )

    threshold_configs = [
        item for item in configs if item["score"] >= state["threshold_score"]
    ]
    state["threshold_final"] = copy.deepcopy(
        max(threshold_configs, key=lambda item: len(item["removed_layers"]))
    )

    target_remove = state["search_config"]["target_remove"]
    if target_remove is None:
        state["budget"] = None
        state["budget_status"] = "not_requested"
        return

    matching = [
        item for item in configs if len(item["removed_layers"]) == target_remove
    ]
    if matching:
        state["budget"] = copy.deepcopy(matching[0])
        state["budget_status"] = "reached"
        return

    state["budget"] = None
    if not state.get("completed", False):
        state["budget_status"] = "pending"
    elif state.get("stop_reason") == "threshold":
        state["budget_status"] = "unreached_threshold"
    elif state.get("stop_reason") == "max_remove":
        state["budget_status"] = "unreached_max_remove"
    elif state.get("stop_reason") == "all_layers_removed":
        state["budget_status"] = "unreached_all_layers_removed"
    else:
        state["budget_status"] = "unreached"


def _checkpoint(state: Dict[str, Any], checkpoint_fn: Optional[CheckpointFn]) -> None:
    if checkpoint_fn is None:
        return
    checkpoint_fn(copy.deepcopy(state))


def _new_state(
    *,
    num_layers: int,
    threshold: float,
    max_remove: Optional[int],
    target_remove: Optional[int],
    stop_at_threshold: bool,
) -> Dict[str, Any]:
    return {
        "schema_version": _SCHEMA_VERSION,
        "num_layers": num_layers,
        "threshold": threshold,
        "baseline": None,
        "baseline_score": None,
        "threshold_score": None,
        "rounds": [],
        "removal_order": [],
        "best": None,
        "bsba": None,
        "threshold_final": None,
        "budget": None,
        "budget_status": "not_requested" if target_remove is None else "pending",
        "completed": False,
        "stop_reason": None,
        "search_config": {
            "num_layers": num_layers,
            "threshold": threshold,
            "max_remove": max_remove,
            "target_remove": target_remove,
            "stop_at_threshold": stop_at_threshold,
        },
    }


def _search_limit(
    *,
    num_layers: int,
    max_remove: Optional[int],
    target_remove: Optional[int],
) -> Tuple[int, str]:
    """Return the hard accepted-depth cap and its stop reason."""

    if max_remove is not None:
        return max_remove, "max_remove"
    if target_remove is not None:
        return target_remove, "target_remove"
    return num_layers, "all_layers_removed"


def _stop_reason_at_depth(
    *,
    depth: int,
    num_layers: int,
    depth_limit: int,
    limit_reason: str,
) -> Optional[str]:
    """Return the canonical terminal reason at ``depth``, if any."""

    # This ordering mirrors the search loop.  In particular, removing every
    # layer is reported as such even when max_remove == num_layers.
    if depth >= num_layers:
        return "all_layers_removed"
    if depth >= depth_limit:
        return limit_reason
    return None


def _validate_resumed_trace(
    state: Dict[str, Any], expected_search_config: Dict[str, Any]
) -> None:
    """Validate that a resumed trace is a reachable search checkpoint."""

    schema_version = state.get("schema_version")
    if (
        isinstance(schema_version, bool)
        or not isinstance(schema_version, int)
        or schema_version != _SCHEMA_VERSION
    ):
        raise ValueError("Unsupported or missing TALE trace schema_version")
    saved_search_config = state.get("search_config")
    if not isinstance(saved_search_config, dict):
        raise ValueError("trace search_config must be a dictionary")
    if set(saved_search_config) != set(expected_search_config) or any(
        type(saved_search_config[name]) is not type(expected_value)
        or saved_search_config[name] != expected_value
        for name, expected_value in expected_search_config.items()
    ):
        raise ValueError("trace search_config does not match the requested search")

    saved_num_layers = state.get("num_layers")
    if (
        isinstance(saved_num_layers, bool)
        or not isinstance(saved_num_layers, int)
        or saved_num_layers != expected_search_config["num_layers"]
    ):
        raise ValueError("trace num_layers does not match search_config")
    saved_top_threshold = _score(state.get("threshold"), "trace threshold")
    if saved_top_threshold != expected_search_config["threshold"]:
        raise ValueError("trace threshold does not match search_config")
    state["threshold"] = saved_top_threshold

    baseline = _score(state.get("baseline"), "trace baseline")
    if state.get("baseline_score") is not None:
        alias = _score(state["baseline_score"], "trace baseline_score")
        if alias != baseline:
            raise ValueError("trace baseline and baseline_score disagree")
    state["baseline"] = baseline
    state["baseline_score"] = baseline

    expected_threshold = baseline - expected_search_config["threshold"]
    saved_threshold = _score(state.get("threshold_score"), "trace threshold_score")
    if saved_threshold != expected_threshold:
        raise ValueError("trace threshold_score is inconsistent with dense baseline")

    rounds = state.get("rounds")
    removal_order = state.get("removal_order")
    if not isinstance(rounds, list) or not isinstance(removal_order, list):
        raise ValueError("trace rounds and removal_order must be lists")

    num_layers = expected_search_config["num_layers"]
    depth_limit, limit_reason = _search_limit(
        num_layers=num_layers,
        max_remove=expected_search_config["max_remove"],
        target_remove=expected_search_config["target_remove"],
    )
    reconstructed_order: List[int] = []
    saw_incomplete = False
    saw_rejected = False
    for expected_index, round_state in enumerate(rounds, start=1):
        if not isinstance(round_state, dict):
            raise ValueError("trace rounds must contain dictionaries")
        if (
            isinstance(round_state.get("round_index"), bool)
            or not isinstance(round_state.get("round_index"), int)
            or round_state["round_index"] != expected_index
        ):
            raise ValueError("trace round_index sequence is invalid")

        base_removed_layers = round_state.get("base_removed_layers")
        if (
            not isinstance(base_removed_layers, list)
            or any(
                isinstance(layer, bool) or not isinstance(layer, int)
                for layer in base_removed_layers
            )
            or base_removed_layers != sorted(reconstructed_order)
        ):
            raise ValueError("trace round base does not match accepted removal order")
        if (
            _stop_reason_at_depth(
                depth=len(reconstructed_order),
                num_layers=num_layers,
                depth_limit=depth_limit,
                limit_reason=limit_reason,
            )
            is not None
        ):
            raise ValueError("trace contains a round after reaching the depth limit")

        candidates = round_state.get("candidates")
        if not isinstance(candidates, list):
            raise ValueError("trace round candidates must be a list")
        if any(not isinstance(candidate, dict) for candidate in candidates):
            raise ValueError("trace candidates must be dictionaries")
        candidate_layers = [candidate.get("layer") for candidate in candidates]

        base_set = set(reconstructed_order)
        remaining_layers = [
            layer for layer in range(num_layers) if layer not in base_set
        ]
        for candidate in candidates:
            layer = candidate.get("layer")
            if isinstance(layer, bool) or not isinstance(layer, int):
                raise ValueError("trace candidate layer IDs must be integers")
            if layer < 0 or layer >= num_layers or layer in base_set:
                raise ValueError("trace contains an invalid candidate layer ID")
            expected_removed = sorted(base_set | {layer})
            candidate_removed = candidate.get("removed_layers")
            if (
                not isinstance(candidate_removed, list)
                or any(
                    isinstance(removed, bool) or not isinstance(removed, int)
                    for removed in candidate_removed
                )
                or candidate_removed != expected_removed
            ):
                raise ValueError("trace candidate removal set is inconsistent")
            candidate["score"] = _score(candidate.get("score"), "trace candidate score")
        if candidate_layers != remaining_layers[: len(candidate_layers)]:
            raise ValueError(
                "trace candidate layers must be a prefix of remaining layer order"
            )

        if "selected_layer" not in round_state:
            if expected_index != len(rounds):
                raise ValueError("only the final trace round may be incomplete")
            if any(
                field in round_state
                for field in ("selected_score", "meets_threshold", "accepted")
            ):
                raise ValueError("an incomplete trace round cannot contain a decision")
            saw_incomplete = True
            continue

        remaining_count = len(remaining_layers)
        if len(candidates) != remaining_count:
            raise ValueError("a completed trace round must contain every candidate")
        selected = max(
            candidates,
            key=lambda item: (item["score"], -item["layer"]),
        )
        saved_selected_layer = round_state.get("selected_layer")
        if isinstance(saved_selected_layer, bool) or not isinstance(
            saved_selected_layer, int
        ):
            raise ValueError("trace selected_layer must be an integer")
        saved_selected_score = _score(
            round_state.get("selected_score"), "trace selected_score"
        )
        if (
            saved_selected_layer != selected["layer"]
            or saved_selected_score != selected["score"]
        ):
            raise ValueError("trace selected candidate is not the stable argmax")
        round_state["selected_score"] = saved_selected_score

        meets_threshold = round_state.get("meets_threshold")
        if not isinstance(meets_threshold, bool):
            raise ValueError(
                "completed trace rounds require a boolean meets_threshold flag"
            )
        expected_meets_threshold = selected["score"] >= saved_threshold
        if meets_threshold is not expected_meets_threshold:
            raise ValueError("trace meets_threshold disagrees with the selected score")

        accepted = round_state.get("accepted")
        if not isinstance(accepted, bool):
            raise ValueError("completed trace rounds require a boolean accepted flag")
        expected_accepted = (
            expected_meets_threshold
            or not expected_search_config["stop_at_threshold"]
        )
        if accepted is not expected_accepted:
            raise ValueError(
                "trace accepted flag disagrees with the threshold stopping policy"
            )
        if accepted:
            reconstructed_order.append(selected["layer"])
        else:
            if expected_index != len(rounds):
                raise ValueError("a rejected trace round must be final")
            saw_rejected = True

    if any(
        isinstance(layer, bool) or not isinstance(layer, int)
        for layer in removal_order
    ):
        raise ValueError("trace removal_order must contain integer layer IDs")
    if removal_order != reconstructed_order:
        raise ValueError("trace removal_order disagrees with accepted rounds")

    completed = state.get("completed")
    if not isinstance(completed, bool):
        raise ValueError("trace completed must be a bool")
    stop_reason = state.get("stop_reason")

    if saw_incomplete and completed:
        raise ValueError("a completed trace cannot contain an incomplete round")

    terminal_reason = _stop_reason_at_depth(
        depth=len(reconstructed_order),
        num_layers=num_layers,
        depth_limit=depth_limit,
        limit_reason=limit_reason,
    )
    if saw_rejected:
        if not completed:
            raise ValueError("a rejected trace round must complete the search")
        if stop_reason != "threshold":
            raise ValueError("a rejected trace round requires stop_reason='threshold'")
    elif terminal_reason is not None:
        if not completed:
            raise ValueError("a trace at the depth limit must be completed")
        if stop_reason != terminal_reason:
            raise ValueError(
                f"trace stop_reason must be {terminal_reason!r} at this depth"
            )
    else:
        if completed:
            raise ValueError("trace is marked completed before a stopping condition")
        if stop_reason is not None:
            raise ValueError("an unfinished trace must have stop_reason=None")


def run_tale_search(
    *,
    num_layers: int,
    evaluate_fn: EvaluateFn,
    baseline_score: Optional[float] = None,
    threshold: float = 0.08,
    max_remove: Optional[int] = None,
    target_remove: Optional[int] = None,
    stop_at_threshold: bool = True,
    trace: Optional[Dict[str, Any]] = None,
    checkpoint_fn: Optional[CheckpointFn] = None,
) -> Dict[str, Any]:
    """Run or resume TALE's greedy task-aware layer search.

    Args:
        num_layers: Number of dense-model transformer layers.
        evaluate_fn: Called with a sorted tuple of original zero-based removed
            layer IDs and returns a scalar score for that configuration.
        baseline_score: Optional precomputed dense score.  When omitted,
            ``evaluate_fn(())`` is used exactly once (unless restored).
        threshold: Absolute score tolerance below the *dense* baseline.
        max_remove: Optional hard cap on accepted removals.  This pure search
            API mirrors the authors' loop and can represent removal of all
            layers; the Hugging Face integration caps it at ``num_layers - 1``
            because a structurally evaluated model must retain one block.
        target_remove: Optional exact removal depth exposed as ``budget``.
            If ``max_remove`` is omitted, this also caps the search depth.
        stop_at_threshold: When true, a round whose best candidate is below the
            fixed floor is recorded but rejected.  Consequently, a deeper
            ``target_remove`` is returned as unavailable.  Set this false to
            accept below-floor candidates and search through to the target (or
            ``max_remove``), allowing later recovery to be observed.
        trace: A prior trace/checkpoint produced with identical search options.
        checkpoint_fn: Receives a deep-copied, JSON-serialisable trace after
            baseline evaluation, after every candidate, and after decisions.

    Returns:
        The complete search trace.  ``best`` is the highest-scoring accepted
        configuration (ties prefer greater depth), ``bsba`` is the deepest
        accepted configuration at or above the dense baseline, and
        ``threshold_final`` is the deepest accepted configuration at or above
        the fixed threshold floor.  ``budget`` is ``None`` with an explicit
        ``budget_status`` when the requested target was not reached.
    """

    if isinstance(num_layers, bool) or not isinstance(num_layers, int):
        raise TypeError("num_layers must be an integer")
    if num_layers <= 0:
        raise ValueError("num_layers must be positive")
    if not callable(evaluate_fn):
        raise TypeError("evaluate_fn must be callable")
    threshold = _score(threshold, "threshold")
    if threshold < 0:
        raise ValueError("threshold must be non-negative")
    max_remove = _optional_count("max_remove", max_remove, num_layers)
    target_remove = _optional_count("target_remove", target_remove, num_layers)
    if not isinstance(stop_at_threshold, bool):
        raise TypeError("stop_at_threshold must be a bool")
    if checkpoint_fn is not None and not callable(checkpoint_fn):
        raise TypeError("checkpoint_fn must be callable or None")

    search_config = {
        "num_layers": num_layers,
        "threshold": threshold,
        "max_remove": max_remove,
        "target_remove": target_remove,
        "stop_at_threshold": stop_at_threshold,
    }
    depth_limit, limit_reason = _search_limit(
        num_layers=num_layers,
        max_remove=max_remove,
        target_remove=target_remove,
    )

    if trace is None:
        state = _new_state(
            num_layers=num_layers,
            threshold=threshold,
            max_remove=max_remove,
            target_remove=target_remove,
            stop_at_threshold=stop_at_threshold,
        )
        dense_score = (
            _score(baseline_score, "baseline_score")
            if baseline_score is not None
            else _score(evaluate_fn(()), "evaluate_fn baseline score")
        )
        state["baseline"] = dense_score
        state["baseline_score"] = dense_score
        state["threshold_score"] = dense_score - threshold
        initial_stop_reason = _stop_reason_at_depth(
            depth=0,
            num_layers=num_layers,
            depth_limit=depth_limit,
            limit_reason=limit_reason,
        )
        if initial_stop_reason is not None:
            state["completed"] = True
            state["stop_reason"] = initial_stop_reason
        _refresh_derived(state)
        _checkpoint(state, checkpoint_fn)
    else:
        if not isinstance(trace, dict):
            raise TypeError("trace must be a dictionary or None")
        state = copy.deepcopy(trace)
        _validate_resumed_trace(state, search_config)
        if baseline_score is not None:
            supplied_baseline = _score(baseline_score, "baseline_score")
            if supplied_baseline != state["baseline"]:
                raise ValueError("baseline_score disagrees with restored trace")
        _refresh_derived(state)
        if state.get("completed", False):
            return state

    if state["completed"]:
        return state

    while True:
        current_depth = len(state["removal_order"])
        if current_depth >= num_layers:
            state["completed"] = True
            state["stop_reason"] = "all_layers_removed"
            break
        if current_depth >= depth_limit:
            state["completed"] = True
            state["stop_reason"] = limit_reason
            break

        base_removed = sorted(state["removal_order"])
        if state["rounds"] and "selected_layer" not in state["rounds"][-1]:
            round_state = state["rounds"][-1]
        else:
            round_state = {
                "round_index": len(state["rounds"]) + 1,
                "base_removed_layers": base_removed,
                "candidates": [],
            }
            state["rounds"].append(round_state)

        evaluated_layers = {
            candidate["layer"] for candidate in round_state["candidates"]
        }
        remaining_layers = [
            layer
            for layer in range(num_layers)
            if layer not in set(base_removed)
        ]

        for layer in remaining_layers:
            if layer in evaluated_layers:
                continue
            removed_layers = sorted(set(base_removed) | {layer})
            candidate_score = _score(
                evaluate_fn(tuple(removed_layers)),
                f"evaluate_fn score for removed layers {removed_layers}",
            )
            round_state["candidates"].append(
                {
                    "layer": layer,
                    "removed_layers": removed_layers,
                    "score": candidate_score,
                }
            )
            _refresh_derived(state)
            _checkpoint(state, checkpoint_fn)

        selected = max(
            round_state["candidates"],
            key=lambda item: (item["score"], -item["layer"]),
        )
        meets_threshold = selected["score"] >= state["threshold_score"]
        accepted = meets_threshold or not stop_at_threshold
        round_state.update(
            {
                "selected_layer": selected["layer"],
                "selected_score": selected["score"],
                "meets_threshold": meets_threshold,
                "accepted": accepted,
            }
        )

        if accepted:
            state["removal_order"].append(selected["layer"])
            accepted_stop_reason = _stop_reason_at_depth(
                depth=len(state["removal_order"]),
                num_layers=num_layers,
                depth_limit=depth_limit,
                limit_reason=limit_reason,
            )
            if accepted_stop_reason is not None:
                state["completed"] = True
                state["stop_reason"] = accepted_stop_reason
            _refresh_derived(state)
            _checkpoint(state, checkpoint_fn)
            if state["completed"]:
                return state
            continue

        state["completed"] = True
        state["stop_reason"] = "threshold"
        _refresh_derived(state)
        _checkpoint(state, checkpoint_fn)
        return state

    _refresh_derived(state)
    _checkpoint(state, checkpoint_fn)
    return state


__all__ = ["run_tale_search"]
