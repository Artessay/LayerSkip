"""Pure greedy search used by SLEB block pruning.

SLEB scores a *candidate model*, not an isolated transformer block.  After
each selection, every surviving block is scored again together with the blocks
that were selected in earlier rounds.  This module deliberately contains no
model or dataset code: callers provide ``score_fn``, which receives the exact
set of original, zero-based block IDs that should be removed.

The returned trace is JSON-serializable.  ``rounds`` contains only completed
rounds, while ``in_progress_round`` permits candidate-level checkpointing and
resume.  ``removal_order`` preserves greedy selection order and
``selected_layers`` is the same set sorted for structural application.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Callable, Mapping
from numbers import Integral, Real
from typing import Any, Optional


ScoreFn = Callable[[tuple[int, ...]], float]
CheckpointFn = Callable[[dict[str, Any]], None]


def _integer(name: str, value: Any, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer, got {type(value).__name__}")
    normalized = int(value)
    if normalized < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {normalized}")
    return normalized


def _finite_score(value: Any, *, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{context} must be a real number, got {type(value).__name__}")
    score = float(value)
    if not math.isfinite(score):
        raise ValueError(f"{context} must be finite, got {score}")
    return score


def _eligible_layers(
    alive: list[int], early_barrier: int, latter_barrier: int
) -> list[int]:
    stop = len(alive) - latter_barrier if latter_barrier else len(alive)
    return alive[early_barrier:stop]


def _new_trace(
    *, num_layers: int, num_remove: int, early_barrier: int, latter_barrier: int
) -> dict[str, Any]:
    return {
        "method": "sleb",
        "num_layers": num_layers,
        "num_remove": num_remove,
        "early_barrier": early_barrier,
        "latter_barrier": latter_barrier,
        "rounds": [],
        "in_progress_round": None,
        "removal_order": [],
        "selected_layers": [],
        "complete": num_remove == 0,
    }


def _layer_list(value: Any, *, context: str) -> list[int]:
    if not isinstance(value, list):
        raise TypeError(f"{context} must be a list")
    result = []
    for index, layer in enumerate(value):
        result.append(_integer(f"{context}[{index}]", layer))
    return result


def _validate_candidate(
    candidate: Any,
    *,
    expected_layer: int,
    previously_removed: list[int],
    context: str,
) -> dict[str, Any]:
    if not isinstance(candidate, Mapping):
        raise TypeError(f"{context} must be a mapping")

    layer = _integer(f"{context}.layer", candidate.get("layer"))
    if layer != expected_layer:
        raise ValueError(
            f"{context}.layer must be {expected_layer}, got {layer}; "
            "candidate traces must follow alive-layer order"
        )

    removed_layers = _layer_list(
        candidate.get("removed_layers"), context=f"{context}.removed_layers"
    )
    expected_removed = sorted([*previously_removed, layer])
    if removed_layers != expected_removed:
        raise ValueError(
            f"{context}.removed_layers must be {expected_removed}, "
            f"got {removed_layers}"
        )

    nll = _finite_score(candidate.get("nll"), context=f"{context}.nll")
    return {"layer": layer, "removed_layers": removed_layers, "nll": nll}


def _resume_trace(
    trace: Mapping[str, Any],
    *,
    num_layers: int,
    num_remove: int,
    early_barrier: int,
    latter_barrier: int,
) -> tuple[dict[str, Any], list[int]]:
    """Validate a checkpoint and return a canonical copy plus current alive IDs."""

    required = {
        "method",
        "num_layers",
        "num_remove",
        "early_barrier",
        "latter_barrier",
        "rounds",
        "removal_order",
        "selected_layers",
        "complete",
    }
    missing = sorted(required.difference(trace))
    if missing:
        raise ValueError(f"SLEB trace is missing required fields: {missing}")
    if trace["method"] != "sleb":
        raise ValueError(f"trace.method must be 'sleb', got {trace['method']!r}")

    expected_scalars = {
        "num_layers": num_layers,
        "num_remove": num_remove,
        "early_barrier": early_barrier,
        "latter_barrier": latter_barrier,
    }
    for field, expected in expected_scalars.items():
        actual = _integer(f"trace.{field}", trace[field])
        if actual != expected:
            raise ValueError(
                f"trace.{field}={actual} does not match requested {expected}"
            )

    rounds = trace["rounds"]
    if not isinstance(rounds, list):
        raise TypeError("trace.rounds must be a list")
    if len(rounds) > num_remove:
        raise ValueError("trace has more completed rounds than num_remove")

    alive = list(range(num_layers))
    removal_order: list[int] = []
    canonical_rounds: list[dict[str, Any]] = []

    for round_index, round_record in enumerate(rounds):
        context = f"trace.rounds[{round_index}]"
        if not isinstance(round_record, Mapping):
            raise TypeError(f"{context} must be a mapping")
        recorded_index = _integer(f"{context}.round", round_record.get("round"))
        if recorded_index != round_index:
            raise ValueError(
                f"{context}.round must be {round_index}, got {recorded_index}"
            )

        candidates = round_record.get("candidates")
        if not isinstance(candidates, list):
            raise TypeError(f"{context}.candidates must be a list")
        expected_candidates = _eligible_layers(alive, early_barrier, latter_barrier)
        if len(candidates) != len(expected_candidates):
            raise ValueError(
                f"{context} must contain all {len(expected_candidates)} candidates, "
                f"got {len(candidates)}"
            )

        canonical_candidates = [
            _validate_candidate(
                candidate,
                expected_layer=expected_layer,
                previously_removed=removal_order,
                context=f"{context}.candidates[{candidate_index}]",
            )
            for candidate_index, (candidate, expected_layer) in enumerate(
                zip(candidates, expected_candidates)
            )
        ]
        # Python's min is stable, so equal NLLs retain alive-layer order.
        best = min(canonical_candidates, key=lambda item: item["nll"])
        selected_layer = _integer(
            f"{context}.selected_layer", round_record.get("selected_layer")
        )
        if selected_layer != best["layer"]:
            raise ValueError(
                f"{context}.selected_layer must be stable argmin {best['layer']}, "
                f"got {selected_layer}"
            )
        selected_nll = _finite_score(
            round_record.get("selected_nll"), context=f"{context}.selected_nll"
        )
        if selected_nll != best["nll"]:
            raise ValueError(
                f"{context}.selected_nll must be {best['nll']}, got {selected_nll}"
            )

        canonical_rounds.append(
            {
                "round": round_index,
                "candidates": canonical_candidates,
                "selected_layer": selected_layer,
                "selected_nll": selected_nll,
            }
        )
        removal_order.append(selected_layer)
        alive.remove(selected_layer)

    recorded_order = _layer_list(
        trace["removal_order"], context="trace.removal_order"
    )
    if recorded_order != removal_order:
        raise ValueError(
            f"trace.removal_order must be {removal_order}, got {recorded_order}"
        )
    selected_layers = _layer_list(
        trace["selected_layers"], context="trace.selected_layers"
    )
    if selected_layers != sorted(removal_order):
        raise ValueError(
            "trace.selected_layers must be the sorted removal_order "
            f"{sorted(removal_order)}, got {selected_layers}"
        )

    complete = trace["complete"]
    if not isinstance(complete, bool):
        raise TypeError("trace.complete must be a boolean")

    state = _new_trace(
        num_layers=num_layers,
        num_remove=num_remove,
        early_barrier=early_barrier,
        latter_barrier=latter_barrier,
    )
    state["rounds"] = canonical_rounds
    state["removal_order"] = removal_order
    state["selected_layers"] = sorted(removal_order)
    state["complete"] = complete

    in_progress = trace.get("in_progress_round")
    if in_progress is not None:
        if complete:
            raise ValueError("a complete trace cannot contain an in-progress round")
        if len(rounds) >= num_remove:
            raise ValueError("trace cannot start another round after num_remove rounds")
        if not isinstance(in_progress, Mapping):
            raise TypeError("trace.in_progress_round must be a mapping or None")

        round_index = _integer(
            "trace.in_progress_round.round", in_progress.get("round")
        )
        if round_index != len(rounds):
            raise ValueError(
                "trace.in_progress_round.round must equal the number of "
                f"completed rounds ({len(rounds)})"
            )
        candidates = in_progress.get("candidates")
        if not isinstance(candidates, list):
            raise TypeError("trace.in_progress_round.candidates must be a list")
        expected_candidates = _eligible_layers(alive, early_barrier, latter_barrier)
        if len(candidates) > len(expected_candidates):
            raise ValueError("in-progress round has too many candidate scores")

        canonical_candidates = [
            _validate_candidate(
                candidate,
                expected_layer=expected_candidates[candidate_index],
                previously_removed=removal_order,
                context=f"trace.in_progress_round.candidates[{candidate_index}]",
            )
            for candidate_index, candidate in enumerate(candidates)
        ]
        state["in_progress_round"] = {
            "round": round_index,
            "candidates": canonical_candidates,
        }

    should_be_complete = len(rounds) == num_remove and in_progress is None
    if complete != should_be_complete:
        raise ValueError(
            f"trace.complete must be {should_be_complete} for its recorded rounds"
        )

    return state, alive


def _checkpoint(state: dict[str, Any], checkpoint_fn: Optional[CheckpointFn]) -> None:
    if checkpoint_fn is not None:
        checkpoint_fn(copy.deepcopy(state))


def run_sleb_search(
    *,
    num_layers: int,
    num_remove: int,
    score_fn: ScoreFn,
    early_barrier: int = 0,
    latter_barrier: int = 0,
    trace: Optional[Mapping[str, Any]] = None,
    checkpoint_fn: Optional[CheckpointFn] = None,
) -> dict[str, Any]:
    """Run or resume the iterative SLEB block-elimination search.

    Args:
        num_layers: Number of transformer blocks in the original model.
        num_remove: Number of blocks to select for removal.
        score_fn: Called once per candidate with a sorted tuple of original,
            zero-based removed block IDs. It must return a finite NLL-like score;
            the lowest score is selected.
        early_barrier: Number of earliest alive blocks excluded from search.
            The paper-faithful default is zero; the official code uses one.
        latter_barrier: Number of latest alive blocks excluded from search.
            The paper-faithful default is zero; the official code uses one.
        trace: Optional trace produced by an interrupted invocation. Completed
            rounds and a valid candidate prefix are reused without rescoring.
        checkpoint_fn: Optional callback receiving an independent trace snapshot
            after every candidate and completed round.

    Returns:
        A JSON-serializable trace containing scores, selection order, and status.
    """

    num_layers = _integer("num_layers", num_layers, minimum=1)
    num_remove = _integer("num_remove", num_remove)
    early_barrier = _integer("early_barrier", early_barrier)
    latter_barrier = _integer("latter_barrier", latter_barrier)

    if early_barrier + latter_barrier > num_layers:
        raise ValueError(
            "early_barrier + latter_barrier cannot exceed num_layers "
            f"({early_barrier} + {latter_barrier} > {num_layers})"
        )
    max_remove = num_layers - early_barrier - latter_barrier
    if num_remove > max_remove:
        raise ValueError(
            f"num_remove={num_remove} exceeds the {max_remove} searchable layers "
            "left by the barriers"
        )
    if not callable(score_fn):
        raise TypeError("score_fn must be callable")
    if checkpoint_fn is not None and not callable(checkpoint_fn):
        raise TypeError("checkpoint_fn must be callable or None")

    if trace is None:
        state = _new_trace(
            num_layers=num_layers,
            num_remove=num_remove,
            early_barrier=early_barrier,
            latter_barrier=latter_barrier,
        )
        alive = list(range(num_layers))
        if state["complete"]:
            _checkpoint(state, checkpoint_fn)
            return state
    else:
        if not isinstance(trace, Mapping):
            raise TypeError("trace must be a mapping or None")
        state, alive = _resume_trace(
            trace,
            num_layers=num_layers,
            num_remove=num_remove,
            early_barrier=early_barrier,
            latter_barrier=latter_barrier,
        )
        if state["complete"]:
            return state

    while len(state["rounds"]) < num_remove:
        round_index = len(state["rounds"])
        eligible = _eligible_layers(alive, early_barrier, latter_barrier)
        if not eligible:
            raise RuntimeError("no eligible SLEB candidates remain")

        in_progress = state["in_progress_round"]
        if in_progress is None:
            in_progress = {"round": round_index, "candidates": []}
            state["in_progress_round"] = in_progress

        candidates = in_progress["candidates"]
        for layer in eligible[len(candidates) :]:
            removed_layers = tuple(sorted([*state["removal_order"], layer]))
            nll = _finite_score(
                score_fn(removed_layers),
                context=f"score_fn result for removed layers {removed_layers}",
            )
            candidates.append(
                {
                    "layer": layer,
                    "removed_layers": list(removed_layers),
                    "nll": nll,
                }
            )
            _checkpoint(state, checkpoint_fn)

        # Candidate order follows alive IDs; min therefore resolves ties stably.
        best = min(candidates, key=lambda item: item["nll"])
        round_record = {
            "round": round_index,
            "candidates": copy.deepcopy(candidates),
            "selected_layer": best["layer"],
            "selected_nll": best["nll"],
        }
        state["rounds"].append(round_record)
        state["removal_order"].append(best["layer"])
        state["selected_layers"] = sorted(state["removal_order"])
        state["in_progress_round"] = None
        alive.remove(best["layer"])
        state["complete"] = len(state["rounds"]) == num_remove
        _checkpoint(state, checkpoint_fn)

    return state


__all__ = ["run_sleb_search"]
