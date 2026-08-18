"""Tests for TALE's greedy search and resumable trace state machine."""

import copy

import pytest

from evaluation.pruning.tale import run_tale_search


class _StopSearch(Exception):
    pass


def _healthy_score(removed):
    """Prefer low layer IDs while keeping every candidate above the floor."""

    return 1.0 - 0.01 * len(removed) - 0.001 * sum(removed)


def _healthy_trace(**search_options):
    options = {
        "num_layers": 3,
        "baseline_score": 1.0,
        "threshold": 0.1,
        "max_remove": 1,
    }
    options.update(search_options)
    return run_tale_search(evaluate_fn=_healthy_score, **options)


def _threshold_trace():
    return run_tale_search(
        num_layers=3,
        evaluate_fn=lambda removed: 0.5,
        baseline_score=1.0,
        threshold=0.1,
        max_remove=2,
    )


def _resume(trace, evaluate_fn=None):
    if evaluate_fn is None:

        def evaluate_fn(removed):
            raise AssertionError(f"completed trace unexpectedly scored {removed}")

    return run_tale_search(
        evaluate_fn=evaluate_fn,
        trace=trace,
        **trace["search_config"],
    )


def _baseline_checkpoint(**search_options):
    checkpoints = []

    def stop_at_baseline(snapshot):
        if not snapshot["rounds"]:
            checkpoints.append(snapshot)
            raise _StopSearch

    with pytest.raises(_StopSearch):
        run_tale_search(
            num_layers=3,
            evaluate_fn=_healthy_score,
            baseline_score=1.0,
            threshold=0.1,
            checkpoint_fn=stop_at_baseline,
            **search_options,
        )

    return checkpoints[-1]


def _checkpoint_after_first_accept(**search_options):
    checkpoints = []

    def stop_after_first_accept(snapshot):
        if len(snapshot["removal_order"]) == 1:
            checkpoints.append(snapshot)
            raise _StopSearch

    with pytest.raises(_StopSearch):
        run_tale_search(
            num_layers=3,
            evaluate_fn=_healthy_score,
            baseline_score=1.0,
            threshold=0.1,
            checkpoint_fn=stop_after_first_accept,
            **search_options,
        )

    return checkpoints[-1]


def test_baseline_only_trace_cannot_claim_completion_before_a_limit():
    trace = _baseline_checkpoint(max_remove=2)
    trace["completed"] = True
    trace["stop_reason"] = "max_remove"

    with pytest.raises(ValueError, match="completed before a stopping condition"):
        _resume(trace)


def test_completed_flag_must_be_boolean():
    trace = _healthy_trace()
    trace["completed"] = 1

    with pytest.raises(ValueError, match="completed must be a bool"):
        _resume(trace)


@pytest.mark.parametrize("stop_reason", [None, "threshold", "target_remove"])
def test_completed_depth_requires_its_canonical_stop_reason(stop_reason):
    trace = _healthy_trace(max_remove=1)
    trace["stop_reason"] = stop_reason

    with pytest.raises(ValueError, match="stop_reason must be 'max_remove'"):
        _resume(trace)


def test_trace_at_depth_limit_cannot_remain_unfinished():
    trace = _healthy_trace(max_remove=1)
    trace["completed"] = False
    trace["stop_reason"] = None

    with pytest.raises(ValueError, match="at the depth limit must be completed"):
        _resume(trace)


@pytest.mark.parametrize(
    ("search_options", "claimed_reason"),
    [
        ({"max_remove": 2}, "max_remove"),
        ({"target_remove": 2}, "target_remove"),
    ],
)
def test_trace_cannot_claim_a_depth_reason_before_reaching_it(
    search_options, claimed_reason
):
    trace = _checkpoint_after_first_accept(**search_options)
    trace["completed"] = True
    trace["stop_reason"] = claimed_reason

    with pytest.raises(ValueError, match="completed before a stopping condition"):
        _resume(trace)


def test_meets_threshold_must_match_the_selected_score():
    trace = _healthy_trace()
    trace["rounds"][0]["meets_threshold"] = False

    with pytest.raises(ValueError, match="meets_threshold disagrees"):
        _resume(trace)


@pytest.mark.parametrize("below_floor", [False, True])
def test_accepted_must_match_threshold_policy(below_floor):
    if below_floor:
        trace = run_tale_search(
            num_layers=3,
            evaluate_fn=lambda removed: 0.5,
            baseline_score=1.0,
            threshold=0.1,
            max_remove=1,
            stop_at_threshold=False,
        )
    else:
        trace = _healthy_trace()
    trace["rounds"][0]["accepted"] = False

    with pytest.raises(ValueError, match="accepted flag disagrees"):
        _resume(trace)


def test_rejected_round_must_complete_the_search():
    trace = _threshold_trace()
    trace["completed"] = False
    trace["stop_reason"] = None

    with pytest.raises(ValueError, match="rejected trace round must complete"):
        _resume(trace)


def test_rejected_round_requires_threshold_stop_reason():
    trace = _threshold_trace()
    trace["stop_reason"] = "max_remove"

    with pytest.raises(ValueError, match="requires stop_reason='threshold'"):
        _resume(trace)


def test_completed_trace_cannot_contain_an_incomplete_round():
    checkpoints = []

    def stop_after_one_candidate(snapshot):
        if snapshot["rounds"] and len(snapshot["rounds"][-1]["candidates"]) == 1:
            checkpoints.append(snapshot)
            raise _StopSearch

    with pytest.raises(_StopSearch):
        run_tale_search(
            num_layers=3,
            evaluate_fn=_healthy_score,
            baseline_score=1.0,
            threshold=0.1,
            max_remove=2,
            checkpoint_fn=stop_after_one_candidate,
        )

    trace = checkpoints[-1]
    trace["completed"] = True
    trace["stop_reason"] = "max_remove"
    with pytest.raises(ValueError, match="cannot contain an incomplete round"):
        _resume(trace)


def test_incomplete_candidate_scores_must_be_a_remaining_layer_prefix():
    checkpoints = []

    def stop_after_one_candidate(snapshot):
        if snapshot["rounds"] and len(snapshot["rounds"][-1]["candidates"]) == 1:
            checkpoints.append(snapshot)
            raise _StopSearch

    with pytest.raises(_StopSearch):
        run_tale_search(
            num_layers=3,
            evaluate_fn=_healthy_score,
            baseline_score=1.0,
            threshold=0.1,
            max_remove=2,
            checkpoint_fn=stop_after_one_candidate,
        )

    trace = checkpoints[-1]
    trace["rounds"][0]["candidates"][0].update(
        {"layer": 1, "removed_layers": [1]}
    )
    with pytest.raises(ValueError, match="prefix of remaining layer order"):
        _resume(trace, evaluate_fn=_healthy_score)


@pytest.mark.parametrize(
    ("field", "bad_value", "message"),
    [
        ("num_layers", 99, "num_layers does not match"),
        ("threshold", 0.2, "threshold does not match"),
    ],
)
def test_top_level_search_metadata_must_match_search_config(
    field, bad_value, message
):
    trace = _healthy_trace()
    trace[field] = bad_value

    with pytest.raises(ValueError, match=message):
        _resume(trace)


def test_candidate_removed_layers_rejects_bool_as_an_integer_alias():
    trace = _healthy_trace()
    trace["rounds"][0]["candidates"][0]["removed_layers"] = [False]

    with pytest.raises(ValueError, match="removal set is inconsistent"):
        _resume(trace)


@pytest.mark.parametrize("bad_score", [True, "0.99"])
def test_trace_scores_must_be_json_numbers(bad_score):
    trace = _healthy_trace()
    trace["rounds"][0]["candidates"][0]["score"] = bad_score

    with pytest.raises(TypeError, match="must be a real number"):
        _resume(trace)


def test_schema_version_rejects_bool_as_an_integer_alias():
    trace = _healthy_trace()
    trace["schema_version"] = True

    with pytest.raises(ValueError, match="schema_version"):
        _resume(trace)


def test_candidate_level_resume_does_not_repeat_scoring():
    calls = []
    checkpoints = []

    def score(removed):
        calls.append(removed)
        return _healthy_score(removed)

    def stop_after_two_candidates(snapshot):
        if (
            snapshot["rounds"]
            and "selected_layer" not in snapshot["rounds"][-1]
            and len(snapshot["rounds"][-1]["candidates"]) == 2
        ):
            checkpoints.append(snapshot)
            raise _StopSearch

    with pytest.raises(_StopSearch):
        run_tale_search(
            num_layers=3,
            evaluate_fn=score,
            baseline_score=1.0,
            threshold=0.1,
            max_remove=1,
            checkpoint_fn=stop_after_two_candidates,
        )

    assert calls == [(0,), (1,)]
    result = _resume(checkpoints[-1], evaluate_fn=score)

    assert calls == [(0,), (1,), (2,)]
    assert result["removal_order"] == [0]
    assert result["completed"] is True
    assert result["stop_reason"] == "max_remove"


def test_completed_threshold_trace_resumes_without_scoring():
    trace = _threshold_trace()

    assert _resume(copy.deepcopy(trace)) == trace


@pytest.mark.parametrize(
    ("search_options", "expected_reason"),
    [
        ({"num_layers": 3, "max_remove": 1}, "max_remove"),
        ({"num_layers": 3, "target_remove": 1}, "target_remove"),
        ({"num_layers": 2}, "all_layers_removed"),
        ({"num_layers": 2, "max_remove": 2}, "all_layers_removed"),
    ],
)
def test_completed_depth_traces_resume_without_scoring(
    search_options, expected_reason
):
    trace = run_tale_search(
        evaluate_fn=_healthy_score,
        baseline_score=1.0,
        threshold=0.1,
        **search_options,
    )

    assert trace["completed"] is True
    assert trace["stop_reason"] == expected_reason
    assert _resume(copy.deepcopy(trace)) == trace


@pytest.mark.parametrize(
    ("search_options", "expected_reason"),
    [
        ({"max_remove": 0}, "max_remove"),
        ({"target_remove": 0}, "target_remove"),
    ],
)
def test_zero_depth_limit_writes_a_resumable_completed_baseline(
    search_options, expected_reason
):
    calls = []
    checkpoints = []
    trace = run_tale_search(
        num_layers=3,
        evaluate_fn=lambda removed: calls.append(removed) or _healthy_score(removed),
        baseline_score=1.0,
        threshold=0.1,
        checkpoint_fn=checkpoints.append,
        **search_options,
    )

    assert calls == []
    assert len(checkpoints) == 1
    assert checkpoints[0]["completed"] is True
    assert trace["stop_reason"] == expected_reason
    assert _resume(copy.deepcopy(trace)) == trace
