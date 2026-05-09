"""Tests for the eval.py command-line interface."""

from unittest.mock import MagicMock, patch

import pytest

from eval import (
    _apply_local_dataset_paths,
    _as_local_path,
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


def test_as_local_path_prefixes_hub_id():
    assert _as_local_path("cais/mmlu") == "/data/cais/mmlu"


def test_as_local_path_keeps_absolute_path():
    assert _as_local_path("/data/cais/mmlu") == "/data/cais/mmlu"


def test_apply_local_dataset_paths_restores_original_paths():
    original_path = MMLUTask.DATASET_PATH
    originals = _apply_local_dataset_paths(["mmlu"])
    try:
        assert MMLUTask.DATASET_PATH == "/data/cais/mmlu"
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
