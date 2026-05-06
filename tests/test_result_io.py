"""Tests for result JSON persistence helpers."""

import json
from types import SimpleNamespace

import pytest

from evaluation.utils.result_io import (
    build_task_evaluation_config,
    model_basename,
    save_json,
    task_result_path,
    task_samples_path,
)


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


def test_model_basename_handles_model_ids_and_paths():
    assert model_basename("meta-llama/Meta-Llama-3-8B-Instruct") == (
        "Meta-Llama-3-8B-Instruct"
    )
    assert model_basename("/data/meta-llama/Meta-Llama-3-8B-Instruct") == (
        "Meta-Llama-3-8B-Instruct"
    )
    assert model_basename(r"C:\models\Meta-Llama-3-8B-Instruct") == (
        "Meta-Llama-3-8B-Instruct"
    )


def test_task_paths_use_model_basename(tmp_path):
    task = SimpleNamespace(num_fewshot=0, max_samples=10, seed=42)
    config = build_task_evaluation_config(
        model_name="/data/meta-llama/Meta-Llama-3-8B-Instruct",
        strategy_name="none",
        strategy_kwargs={},
        strategy_config={},
        task_name="gsm8k",
        task=task,
        task_kwargs={"gsm8k": {"max_samples": 10}},
        batch_size=1,
        device="cpu",
        requested_device="cpu",
        dtype="auto",
        trust_remote_code=False,
    )

    result_path = task_result_path(tmp_path, config)
    samples_path = task_samples_path(tmp_path, config)

    assert config["model"] == "Meta-Llama-3-8B-Instruct"
    assert result_path.relative_to(tmp_path).parts[0] == "Meta-Llama-3-8B-Instruct"
    assert result_path.suffix == ".json"
    assert samples_path.suffix == ".jsonl"
    assert samples_path.with_suffix(".json") == result_path