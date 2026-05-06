"""Tests for the Evaluator orchestrator."""

import json
from unittest.mock import MagicMock, patch

import pytest

from evaluation.evaluator import Evaluator


@pytest.fixture(autouse=True)
def mock_cuda_auto_detection():
    with patch("torch.cuda.is_available", return_value=False):
        yield


class TestEvaluatorCompare:

    def _make_run_result(self, strategy_name, mmlu_acc, hellaswag_acc):
        return {
            "model": "test-model",
            "strategy": strategy_name,
            "strategy_config": {},
            "results": {
                "mmlu": {"accuracy": mmlu_acc},
                "hellaswag": {"accuracy": hellaswag_acc},
            },
            "elapsed_seconds": 1.0,
        }

    def test_compare_results_structure(self):
        runs = [
            self._make_run_result("none", 0.70, 0.75),
            self._make_run_result("layerskip", 0.65, 0.72),
            self._make_run_result("caml", 0.68, 0.74),
        ]
        cmp = Evaluator.compare_results(runs)
        assert "mmlu" in cmp
        assert "hellaswag" in cmp
        assert "none" in cmp["mmlu"]["accuracy"]
        assert "layerskip" in cmp["mmlu"]["accuracy"]
        assert "caml" in cmp["mmlu"]["accuracy"]

    def test_compare_results_values(self):
        runs = [
            self._make_run_result("none", 0.70, 0.75),
            self._make_run_result("gateskip", 0.66, 0.71),
        ]
        cmp = Evaluator.compare_results(runs)
        assert cmp["mmlu"]["accuracy"]["none"] == pytest.approx(0.70)
        assert cmp["mmlu"]["accuracy"]["gateskip"] == pytest.approx(0.66)
        assert cmp["hellaswag"]["accuracy"]["none"] == pytest.approx(0.75)

    def test_compare_results_empty(self):
        cmp = Evaluator.compare_results([])
        assert cmp == {}


class TestEvaluatorPrint:

    def test_print_results_does_not_raise(self, capsys):
        result = {
            "model": "meta-llama/Llama-3.2-1B-Instruct",
            "strategy": "layerskip",
            "strategy_config": {"exit_ratio": 0.75},
            "results": {"mmlu": {"accuracy": 0.65}},
            "elapsed_seconds": 10.5,
        }
        Evaluator.print_results(result)
        captured = capsys.readouterr()
        assert "layerskip" in captured.out
        assert "mmlu" in captured.out
        assert "65.00%" in captured.out

    def test_print_comparison_does_not_raise(self, capsys):
        comparison = {
            "mmlu": {
                "accuracy": {
                    "none": 0.70,
                    "layerskip": 0.65,
                }
            }
        }
        Evaluator.print_comparison(comparison)
        captured = capsys.readouterr()
        assert "mmlu" in captured.out
        assert "none" in captured.out
        assert "layerskip" in captured.out


class TestEvaluatorInit:

    def test_device_auto_resolves(self):
        ev = Evaluator(model_name="mock", device="auto")
        assert ev.device in ("cuda", "cpu")

    def test_default_tasks_empty(self):
        ev = Evaluator(model_name="mock")
        assert ev.task_names == []

    def test_strategy_kwargs_stored(self):
        ev = Evaluator(
            model_name="mock",
            strategy_name="layerskip",
            strategy_kwargs={"exit_ratio": 0.5},
        )
        assert ev.strategy_name == "layerskip"
        assert ev.strategy_kwargs["exit_ratio"] == 0.5


class TestEvaluatorBuildModel:
    """Test that _build_model passes the right strategy to HFModel."""

    @patch("evaluation.evaluator.HFModel")
    def test_build_model_with_none_strategy(self, MockHFModel):
        ev = Evaluator(model_name="mock-model", strategy_name="none")
        ev._build_model()
        call_kwargs = MockHFModel.call_args[1]
        assert call_kwargs["strategy"] is None

    @patch("evaluation.evaluator.HFModel")
    def test_build_model_with_layerskip(self, MockHFModel):
        ev = Evaluator(
            model_name="mock-model",
            strategy_name="layerskip",
            strategy_kwargs={"exit_ratio": 0.6},
        )
        ev._build_model()
        call_kwargs = MockHFModel.call_args[1]
        from evaluation.strategies.layerskip import LayerSkipStrategy
        assert isinstance(call_kwargs["strategy"], LayerSkipStrategy)
        assert call_kwargs["strategy"].exit_ratio == 0.6

    @patch("evaluation.evaluator.HFModel")
    def test_build_model_with_caml(self, MockHFModel):
        ev = Evaluator(
            model_name="mock-model",
            strategy_name="caml",
            strategy_kwargs={"confidence_threshold": 0.85},
        )
        ev._build_model()
        call_kwargs = MockHFModel.call_args[1]
        from evaluation.strategies.caml import CAMLStrategy
        assert isinstance(call_kwargs["strategy"], CAMLStrategy)
        assert call_kwargs["strategy"].confidence_threshold == 0.85

    @patch("evaluation.evaluator.HFModel")
    def test_build_model_with_gateskip(self, MockHFModel):
        ev = Evaluator(
            model_name="mock-model",
            strategy_name="gateskip",
            strategy_kwargs={"skip_budget": 0.2},
        )
        ev._build_model()
        call_kwargs = MockHFModel.call_args[1]
        from evaluation.strategies.gateskip import GateSkipStrategy
        assert isinstance(call_kwargs["strategy"], GateSkipStrategy)
        assert call_kwargs["strategy"].skip_budget == 0.2


class TestEvaluatorRun:
    """Test the full run() method with mocked model and tasks."""

    @patch("evaluation.evaluator.HFModel")
    @patch("evaluation.evaluator.get_task")
    def test_run_returns_correct_structure(self, mock_get_task, MockHFModel, tmp_path):
        # Mock task
        mock_task = MagicMock()
        mock_task.evaluate.return_value = {"accuracy": 0.72}
        mock_task.num_fewshot = 5
        mock_task.max_samples = None
        mock_task.seed = 42
        mock_get_task.return_value = mock_task

        # Mock model
        mock_model_instance = MagicMock()
        mock_model_instance.strategy = None
        MockHFModel.return_value = mock_model_instance

        ev = Evaluator(
            model_name="mock-model",
            strategy_name="none",
            tasks=["mmlu"],
            results_dir=tmp_path,
        )
        result = ev.run()

        assert result["model"] == "mock-model"
        assert result["strategy"] == "none"
        assert "results" in result
        assert "mmlu" in result["results"]
        assert result["results"]["mmlu"]["accuracy"] == pytest.approx(0.72)
        assert "mmlu" in result["result_files"]
        assert "elapsed_seconds" in result

    @patch("evaluation.evaluator.HFModel")
    @patch("evaluation.evaluator.get_task")
    def test_run_multiple_tasks(self, mock_get_task, MockHFModel, tmp_path):
        def side_effect(name, **kwargs):
            task = MagicMock()
            task.evaluate.return_value = {"accuracy": 0.5 + hash(name) % 10 / 100}
            task.num_fewshot = kwargs.get("num_fewshot", 0)
            task.max_samples = kwargs.get("max_samples")
            task.seed = kwargs.get("seed", 42)
            return task

        mock_get_task.side_effect = side_effect
        mock_model_instance = MagicMock()
        mock_model_instance.strategy = None
        MockHFModel.return_value = mock_model_instance

        ev = Evaluator(
            model_name="mock-model",
            strategy_name="none",
            tasks=["mmlu", "hellaswag", "winogrande"],
            results_dir=tmp_path,
        )
        result = ev.run()

        assert set(result["results"].keys()) == {"mmlu", "hellaswag", "winogrande"}

    @patch("evaluation.evaluator.HFModel")
    @patch("evaluation.evaluator.get_task")
    def test_run_saves_each_task_result_json(self, mock_get_task, MockHFModel, tmp_path):
        def side_effect(name, **kwargs):
            task = MagicMock()
            task.evaluate.return_value = {"accuracy": 0.75 if name == "mmlu" else 0.8}
            task.num_fewshot = kwargs.get("num_fewshot", 0)
            task.max_samples = kwargs.get("max_samples")
            task.seed = kwargs.get("seed", 42)
            return task

        mock_get_task.side_effect = side_effect
        mock_model_instance = MagicMock()
        mock_model_instance.strategy = MagicMock()
        mock_model_instance.strategy.config = {"exit_ratio": 0.6}
        MockHFModel.return_value = mock_model_instance

        ev = Evaluator(
            model_name="org/mock-model",
            strategy_name="layerskip",
            strategy_kwargs={"exit_ratio": 0.6},
            tasks=["mmlu", "hellaswag"],
            task_kwargs={
                "mmlu": {"num_fewshot": 5, "max_samples": 10, "seed": 7},
                "hellaswag": {"num_fewshot": 0, "max_samples": 10, "seed": 7},
            },
            batch_size=2,
            device="cpu",
            dtype="float16",
            max_length=512,
            trust_remote_code=True,
            results_dir=tmp_path,
        )
        result = ev.run()

        assert set(result["result_files"].keys()) == {"mmlu", "hellaswag"}
        assert result["result_files"]["mmlu"] != result["result_files"]["hellaswag"]

        for task_name, result_file in result["result_files"].items():
            with open(result_file) as f:
                saved = json.load(f)

            assert saved["model"] == "org/mock-model"
            assert saved["strategy"] == "layerskip"
            assert saved["task"] == task_name
            assert saved["results"] == result["results"][task_name]
            assert saved["evaluation_config"]["strategy"]["kwargs"] == {"exit_ratio": 0.6}
            assert saved["evaluation_config"]["runtime"]["batch_size"] == 2
            assert saved["evaluation_config"]["runtime"]["max_length"] == 512
            assert saved["evaluation_config"]["task"]["resolved_kwargs"]["max_samples"] == 10
