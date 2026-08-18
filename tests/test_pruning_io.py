"""Tests for pruning-search checkpoint identity and persistence."""

from __future__ import annotations

import json

import pytest

from evaluation.pruning.io import (
    SEARCH_SCHEMA_VERSION,
    load_search_envelope,
    make_search_envelope,
    save_search_envelope,
    search_trace_path,
)


def test_trace_path_is_stable_and_scoped_by_task_and_config(tmp_path):
    kwargs = {
        "results_dir": tmp_path,
        "model_name": "org/model",
        "method": "tale",
        "search_config": {"threshold": 0.08},
    }

    first = search_trace_path(**kwargs, task_name="mmlu")
    assert first == search_trace_path(**kwargs, task_name="mmlu")
    assert first != search_trace_path(**kwargs, task_name="hellaswag")
    assert first != search_trace_path(
        **{**kwargs, "search_config": {"threshold": 0.09}},
        task_name="mmlu",
    )
    assert first.parts[-5:-1] == ("model", "pruning", "tale", "mmlu")


def test_save_and_load_matching_envelope_atomically(tmp_path):
    path = tmp_path / "nested" / "trace.json"
    envelope = make_search_envelope(
        model_name="org/model",
        method="sleb",
        search_config={"num_remove": 2},
        algorithm_trace={"complete": False, "rounds": []},
        provenance={"torch": "test"},
    )

    save_search_envelope(envelope, path)
    loaded = load_search_envelope(
        path,
        model_name="org/model",
        method="sleb",
        search_config={"num_remove": 2},
    )

    assert loaded == envelope
    assert loaded["schema_version"] == SEARCH_SCHEMA_VERSION
    assert not list(path.parent.glob("*.tmp"))
    assert json.loads(path.read_text()) == envelope


def test_load_rejects_checkpoint_from_different_search_config(tmp_path):
    path = tmp_path / "trace.json"
    envelope = make_search_envelope(
        model_name="org/model",
        method="shortgpt",
        search_config={"num_remove": 1},
        algorithm_trace={"complete": True},
        provenance={},
    )
    save_search_envelope(envelope, path)

    with pytest.raises(ValueError, match="mismatched search_config"):
        load_search_envelope(
            path,
            model_name="org/model",
            method="shortgpt",
            search_config={"num_remove": 2},
        )


def test_load_missing_trace_returns_none(tmp_path):
    assert (
        load_search_envelope(
            tmp_path / "missing.json",
            model_name="org/model",
            method="sleb",
            search_config={},
        )
        is None
    )
