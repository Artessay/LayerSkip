"""Tests for the eval.py command-line interface."""

from unittest.mock import MagicMock, patch

import pytest

from eval import build_parser, main


def test_output_defaults_to_results_directory():
    parser = build_parser()
    args = parser.parse_args([])

    assert args.output == "results"


def test_results_dir_argument_removed():
    parser = build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(["--results_dir", "custom-results"])


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