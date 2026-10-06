"""Tests for the eval.py command-line interface."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from eval import (
    _as_local_calibration_dataset_path,
    _apply_local_dataset_paths,
    _as_local_dataset_path,
    _as_local_model_path,
    _build_strategy_kwargs,
    _parse_manualskip_layers,
    _restore_dataset_paths,
    build_parser,
    main,
)
from evaluation.tasks.mmlu import MMLUTask


def test_output_defaults_to_results_directory():
    parser = build_parser()
    args = parser.parse_args([])

    assert args.output == "results"


def test_local_defaults_to_false():
    parser = build_parser()
    args = parser.parse_args([])

    assert args.local is False


def test_local_argument_sets_flag():
    parser = build_parser()
    args = parser.parse_args(["--local"])

    assert args.local is True


def test_results_dir_argument_removed():
    parser = build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(["--results_dir", "custom-results"])


def test_manualskip_layers_parse_space_separated_values():
    assert _parse_manualskip_layers(["2", "4", "8"]) == [2, 4, 8]


def test_manualskip_layers_parse_comma_and_bracket_values():
    assert _parse_manualskip_layers(["[2,4]", "8"]) == [2, 4, 8]


def test_build_strategy_kwargs_for_manualskip():
    parser = build_parser()
    args = parser.parse_args(
        ["--strategy", "manualskip", "--manualskip_layers", "2", "4", "8"]
    )

    assert _build_strategy_kwargs(args, "manualskip") == {"skip_layers": [2, 4, 8]}


def test_build_strategy_kwargs_for_calibratedskip():
    parser = build_parser()
    args = parser.parse_args(
        [
            "--strategy",
            "calibratedskip",
            "--calibratedskip_metrics",
            "activation_ratio",
            "gradient_value",
            "gradient_trace",
            "shapley_value",
            "--calibration_max_samples",
            "16",
        ]
    )

    assert _build_strategy_kwargs(args, "calibratedskip") == {
        "calibration_metrics": [
            "activation_ratio",
            "gradient_value",
            "gradient_trace",
            "shapley_value",
        ],
        "calibration_max_samples": 16,
    }


@pytest.mark.parametrize("strategy", ["shortgpt", "sleb", "tale"])
def test_parser_accepts_pruning_methods(strategy):
    parser = build_parser()

    args = parser.parse_args(["--strategy", strategy])

    assert args.strategy == [strategy]


def test_build_strategy_kwargs_for_shortgpt():
    parser = build_parser()
    args = parser.parse_args(
        [
            "--strategy",
            "shortgpt",
            "--shortgpt_prune_ratio",
            "0.3",
            "--shortgpt_num_remove",
            "6",
            "--shortgpt_dataset",
            "custom/pg19",
            "--shortgpt_split",
            "test",
            "--shortgpt_max_samples",
            "24",
            "--shortgpt_sequence_length",
            "512",
            "--shortgpt_search_batch_size",
            "2",
            "--seed",
            "7",
        ]
    )

    assert _build_strategy_kwargs(args, "shortgpt") == {
        "prune_ratio": 0.3,
        "num_remove": 6,
        "dataset_path": "custom/pg19",
        "split": "test",
        "max_samples": 24,
        "sequence_length": 512,
        "search_batch_size": 2,
        "seed": 7,
    }


def test_build_strategy_kwargs_for_sleb_uses_paper_barrier_defaults():
    parser = build_parser()
    args = parser.parse_args(["--strategy", "sleb", "--seed", "99"])

    assert _build_strategy_kwargs(args, "sleb") == {
        "prune_ratio": 0.2,
        "num_remove": None,
        "dataset_path": "wikitext",
        "dataset_name": "wikitext-2-raw-v1",
        "split": "train",
        "max_samples": 128,
        "sequence_length": 2048,
        "search_batch_size": 1,
        "early_barrier": 0,
        "latter_barrier": 0,
        "seed": 0,
    }


def test_build_strategy_kwargs_for_sleb_accepts_method_seed_override():
    parser = build_parser()
    args = parser.parse_args(["--strategy", "sleb", "--sleb_seed", "7"])

    assert _build_strategy_kwargs(args, "sleb")["seed"] == 7


def test_build_strategy_kwargs_for_tale_continue_below_threshold():
    parser = build_parser()
    args = parser.parse_args(
        [
            "--strategy",
            "tale",
            "--tale_threshold",
            "0.05",
            "--tale_search_max_samples",
            "32",
            "--tale_max_remove",
            "10",
            "--tale_target_remove",
            "8",
            "--tale_variant",
            "budget",
            "--tale_continue_below_threshold",
            "--seed",
            "17",
        ]
    )

    assert _build_strategy_kwargs(args, "tale") == {
        "threshold": 0.05,
        "search_max_samples": 32,
        "max_remove": 10,
        "target_remove": 8,
        "variant": "budget",
        "stop_at_threshold": False,
        "seed": 17,
    }


def test_build_strategy_kwargs_for_tale_uses_full_search_split_by_default():
    parser = build_parser()
    args = parser.parse_args(["--strategy", "tale"])

    assert _build_strategy_kwargs(args, "tale")["search_max_samples"] is None


def test_as_local_model_path_prefixes_hub_id():
    assert _as_local_model_path("meta-llama/Llama-3.2-1B-Instruct") == (
        "/data/meta-llama/Llama-3.2-1B-Instruct"
    )


def test_as_local_model_path_keeps_absolute_path():
    assert _as_local_model_path("/data/meta-llama/Llama-3.2-1B-Instruct") == (
        "/data/meta-llama/Llama-3.2-1B-Instruct"
    )


def test_as_local_dataset_path_prefixes_hub_id(monkeypatch):
    monkeypatch.setattr("eval.DATASET_LOCAL_ROOT", Path("/hf/datasets/source"))

    assert _as_local_dataset_path("cais/mmlu") == "/hf/datasets/source/cais/mmlu"


def test_as_local_dataset_path_keeps_absolute_path():
    assert _as_local_dataset_path("/data/cais/mmlu") == "/data/cais/mmlu"


def test_local_calibration_dataset_paths_use_hf_home(monkeypatch):
    monkeypatch.setattr("eval.DATASET_LOCAL_ROOT", Path("/hf/datasets/source"))

    assert (
        _as_local_calibration_dataset_path("emozilla/pg19")
        == "/hf/datasets/source/emozilla/pg19"
    )
    assert (
        _as_local_calibration_dataset_path("wikitext")
        == "/hf/datasets/source/Salesforce/wikitext"
    )


def test_apply_local_dataset_paths_restores_original_paths(monkeypatch):
    monkeypatch.setattr("eval.DATASET_LOCAL_ROOT", Path("/hf/datasets/source"))
    original_path = MMLUTask.DATASET_PATH
    originals = _apply_local_dataset_paths(["mmlu"])
    try:
        assert MMLUTask.DATASET_PATH == "/hf/datasets/source/cais/mmlu"
    finally:
        _restore_dataset_paths(originals)

    assert MMLUTask.DATASET_PATH == original_path


@patch("eval.Evaluator")
def test_main_uses_output_as_results_directory(mock_evaluator):
    mock_instance = MagicMock()
    mock_instance.run.return_value = {
        "model": "mock-model",
        "strategy": "none",
        "strategy_config": {},
        "results": {"mmlu": {"accuracy": 0.5}},
        "result_files": {"mmlu": "custom-results/mock.json"},
        "elapsed_seconds": 1.0,
    }
    mock_evaluator.return_value = mock_instance

    main([
        "--model",
        "mock-model",
        "--strategy",
        "none",
        "--tasks",
        "mmlu",
        "--output",
        "custom-results",
    ])

    assert mock_evaluator.call_args.kwargs["results_dir"] == "custom-results"


@patch("eval.Evaluator")
def test_main_local_uses_data_model_path(mock_evaluator):
    mock_instance = MagicMock()
    mock_instance.run.return_value = {
        "model": "/data/meta-llama/Llama-3.2-1B-Instruct",
        "strategy": "none",
        "strategy_config": {},
        "results": {"mmlu": {"accuracy": 0.5}},
        "elapsed_seconds": 1.0,
    }
    mock_evaluator.return_value = mock_instance

    main([
        "--model",
        "meta-llama/Llama-3.2-1B-Instruct",
        "--strategy",
        "none",
        "--tasks",
        "mmlu",
        "--local",
    ])

    assert (
        mock_evaluator.call_args.kwargs["model_name"]
        == "/data/meta-llama/Llama-3.2-1B-Instruct"
    )
    assert MMLUTask.DATASET_PATH == "cais/mmlu"


@patch("eval.Evaluator")
def test_main_local_maps_pruning_search_datasets(mock_evaluator, monkeypatch):
    monkeypatch.setattr("eval.DATASET_LOCAL_ROOT", Path("/hf/datasets/source"))
    mock_instance = MagicMock()
    mock_instance.run.return_value = {
        "model": "/data/mock-model",
        "strategy": "shortgpt",
        "strategy_config": {},
        "results": {"mmlu": {"accuracy": 0.5}},
        "elapsed_seconds": 1.0,
    }
    mock_evaluator.return_value = mock_instance

    main([
        "--model",
        "mock-model",
        "--strategy",
        "shortgpt",
        "sleb",
        "--tasks",
        "mmlu",
        "--local",
    ])

    calls = mock_evaluator.call_args_list
    assert calls[0].kwargs["strategy_kwargs"]["dataset_path"] == (
        "/hf/datasets/source/emozilla/pg19"
    )
    assert calls[1].kwargs["strategy_kwargs"]["dataset_path"] == (
        "/hf/datasets/source/Salesforce/wikitext"
    )
    assert MMLUTask.DATASET_PATH == "cais/mmlu"


@patch("eval.Evaluator")
def test_main_passes_manualskip_layers(mock_evaluator):
    mock_instance = MagicMock()
    mock_instance.run.return_value = {
        "model": "mock-model",
        "strategy": "manualskip",
        "strategy_config": {"skip_layers": [2, 4]},
        "results": {"mmlu": {"accuracy": 0.5}},
        "elapsed_seconds": 1.0,
    }
    mock_evaluator.return_value = mock_instance

    main([
        "--model",
        "mock-model",
        "--strategy",
        "manualskip",
        "--manualskip_layers",
        "2",
        "4",
        "--tasks",
        "mmlu",
    ])

    assert mock_evaluator.call_args.kwargs["strategy_kwargs"] == {"skip_layers": [2, 4]}
