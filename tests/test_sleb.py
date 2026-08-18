"""Tests for the model-independent SLEB greedy search."""

import copy

import pytest

from evaluation.pruning.sleb import run_sleb_search


class _StopSearch(Exception):
    pass


def test_iterative_rescoring_changes_the_second_choice():
    calls = []

    def score(removed):
        calls.append(removed)
        if len(removed) == 1:
            return {0: 1.0, 1: 2.0, 2: 3.0, 3: 4.0}[removed[0]]
        # After layer 0 is selected, interaction makes layer 2 preferable to 1.
        return {(0, 1): 10.0, (0, 2): 0.5, (0, 3): 2.0}[removed]

    result = run_sleb_search(num_layers=4, num_remove=2, score_fn=score)

    assert result["method"] == "sleb"
    assert result["removal_order"] == [0, 2]
    assert result["selected_layers"] == [0, 2]
    assert result["complete"] is True
    assert calls == [
        (0,),
        (1,),
        (2,),
        (3,),
        (0, 1),
        (0, 2),
        (0, 3),
    ]
    assert result["rounds"][1]["candidates"][1] == {
        "layer": 2,
        "removed_layers": [0, 2],
        "nll": 0.5,
    }


def test_equal_scores_use_stable_original_layer_order():
    result = run_sleb_search(
        num_layers=3,
        num_remove=2,
        score_fn=lambda removed: 1.0,
    )

    assert result["removal_order"] == [0, 1]


def test_barriers_protect_early_and_latter_layers():
    calls = []

    def score(removed):
        calls.append(removed)
        return sum(removed)

    result = run_sleb_search(
        num_layers=6,
        num_remove=3,
        score_fn=score,
        early_barrier=1,
        latter_barrier=2,
    )

    assert result["removal_order"] == [1, 2, 3]
    assert all(0 not in removed for removed in calls)
    assert all(4 not in removed and 5 not in removed for removed in calls)
    assert len(calls) == 3 + 2 + 1


@pytest.mark.parametrize(
    ("num_layers", "num_remove", "early_barrier", "latter_barrier"),
    [
        (5, 3, 0, 0),
        (8, 4, 1, 1),
        (6, 2, 2, 1),
    ],
)
def test_candidate_evaluation_count_matches_greedy_formula(
    num_layers, num_remove, early_barrier, latter_barrier
):
    calls = []

    result = run_sleb_search(
        num_layers=num_layers,
        num_remove=num_remove,
        early_barrier=early_barrier,
        latter_barrier=latter_barrier,
        score_fn=lambda removed: calls.append(removed) or float(sum(removed)),
    )

    searchable = num_layers - early_barrier - latter_barrier
    expected = sum(searchable - round_index for round_index in range(num_remove))
    assert len(calls) == expected
    assert sum(len(round_["candidates"]) for round_ in result["rounds"]) == expected


def test_resume_from_completed_round_does_not_rescore_it():
    calls = []
    checkpoints = []

    def score(removed):
        calls.append(removed)
        return float(sum(removed))

    def stop_after_first_round(snapshot):
        checkpoints.append(snapshot)
        if len(snapshot["rounds"]) == 1 and snapshot["in_progress_round"] is None:
            raise _StopSearch

    with pytest.raises(_StopSearch):
        run_sleb_search(
            num_layers=4,
            num_remove=2,
            score_fn=score,
            checkpoint_fn=stop_after_first_round,
        )

    partial = checkpoints[-1]
    assert partial["removal_order"] == [0]
    assert partial["complete"] is False
    assert len(calls) == 4

    result = run_sleb_search(
        num_layers=4,
        num_remove=2,
        score_fn=score,
        trace=partial,
    )

    assert result["removal_order"] == [0, 1]
    assert len(calls) == 4 + 3


def test_resume_reuses_candidate_level_checkpoint_prefix():
    calls = []
    checkpoints = []

    def score(removed):
        calls.append(removed)
        return float(sum(removed))

    def stop_after_two_candidates(snapshot):
        checkpoints.append(copy.deepcopy(snapshot))
        in_progress = snapshot["in_progress_round"]
        if (
            not snapshot["rounds"]
            and in_progress is not None
            and len(in_progress["candidates"]) == 2
        ):
            raise _StopSearch

    with pytest.raises(_StopSearch):
        run_sleb_search(
            num_layers=4,
            num_remove=2,
            score_fn=score,
            checkpoint_fn=stop_after_two_candidates,
        )

    partial = checkpoints[-1]
    assert calls == [(0,), (1,)]

    result = run_sleb_search(
        num_layers=4,
        num_remove=2,
        score_fn=score,
        trace=partial,
    )

    assert result["removal_order"] == [0, 1]
    assert calls == [(0,), (1,), (2,), (3,), (0, 1), (0, 2), (0, 3)]


def test_zero_removals_returns_complete_trace_without_scoring():
    calls = []
    result = run_sleb_search(
        num_layers=4,
        num_remove=0,
        score_fn=lambda removed: calls.append(removed) or 0.0,
    )

    assert calls == []
    assert result["rounds"] == []
    assert result["removal_order"] == []
    assert result["complete"] is True


@pytest.mark.parametrize(
    "kwargs, error",
    [
        ({"num_layers": 0, "num_remove": 0}, ValueError),
        ({"num_layers": 4, "num_remove": -1}, ValueError),
        ({"num_layers": 4, "num_remove": 3, "early_barrier": 1, "latter_barrier": 1}, ValueError),
        ({"num_layers": 4, "num_remove": 0, "early_barrier": 3, "latter_barrier": 2}, ValueError),
        ({"num_layers": 4, "num_remove": 1, "early_barrier": -1}, ValueError),
        ({"num_layers": True, "num_remove": 0}, TypeError),
    ],
)
def test_invalid_search_dimensions_are_rejected(kwargs, error):
    with pytest.raises(error):
        run_sleb_search(score_fn=lambda removed: 0.0, **kwargs)


@pytest.mark.parametrize("bad_score", [float("nan"), float("inf"), "1.0", True])
def test_score_must_be_a_finite_real_number(bad_score):
    with pytest.raises((TypeError, ValueError)):
        run_sleb_search(
            num_layers=2,
            num_remove=1,
            score_fn=lambda removed: bad_score,
        )
