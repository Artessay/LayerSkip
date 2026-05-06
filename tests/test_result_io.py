"""Tests for result JSON persistence helpers."""

import json

import pytest

from evaluation.utils.result_io import save_json


def test_save_json_creates_parent_dirs(tmp_path):
    result = {
        "model": "test",
        "strategy": "none",
        "strategy_config": {},
        "results": {"mmlu": {"accuracy": 0.7}},
        "elapsed_seconds": 1.0,
    }
    path = tmp_path / "nested" / "result.json"

    save_json(result, path)

    with open(path) as f:
        loaded = json.load(f)

    assert loaded["model"] == "test"
    assert loaded["results"]["mmlu"]["accuracy"] == pytest.approx(0.7)