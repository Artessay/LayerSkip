"""Tests for evaluation task implementations (using mock data)."""

import pytest
from unittest.mock import MagicMock, patch
from typing import Any, Dict, List

from evaluation.tasks import get_task, TASK_REGISTRY
from evaluation.tasks.base_task import BaseTask
from evaluation.tasks.mmlu import MMLUTask
from evaluation.tasks.hellaswag import HellaSwagTask
from evaluation.tasks.winogrande import WinoGrandeTask
from evaluation.tasks.gsm8k import GSM8KTask, _extract_answer, _extract_generated_answer
from evaluation.tasks.humaneval import HumanEvalTask, _sanitize_code, _execute_code


# ------------------------------------------------------------------ #
# Task registry                                                        #
# ------------------------------------------------------------------ #

def test_task_registry_keys():
    for name in ("mmlu", "hellaswag", "winogrande", "gsm8k", "humaneval"):
        assert name in TASK_REGISTRY


def test_get_task_unknown():
    with pytest.raises(ValueError, match="Unknown task"):
        get_task("nonexistent_task")


def test_get_task_returns_correct_type():
    assert isinstance(get_task("mmlu"), MMLUTask)
    assert isinstance(get_task("hellaswag"), HellaSwagTask)
    assert isinstance(get_task("winogrande"), WinoGrandeTask)
    assert isinstance(get_task("gsm8k"), GSM8KTask)
    assert isinstance(get_task("humaneval"), HumanEvalTask)


def test_get_task_kwargs_override():
    task = get_task("mmlu", num_fewshot=0, max_samples=50)
    assert task.num_fewshot == 0
    assert task.max_samples == 50


@pytest.mark.parametrize(
    "task_factory,method_name",
    [
        pytest.param(
            lambda: MMLUTask(subjects=["abstract_algebra"]),
            "_load_dataset",
            id="mmlu-test",
        ),
        pytest.param(
            lambda: MMLUTask(subjects=["abstract_algebra"]),
            "_load_fewshot_dataset",
            id="mmlu-dev",
        ),
        pytest.param(lambda: HellaSwagTask(), "_load_dataset", id="hellaswag"),
        pytest.param(lambda: WinoGrandeTask(), "_load_dataset", id="winogrande"),
        pytest.param(lambda: GSM8KTask(), "_load_dataset", id="gsm8k-test"),
        pytest.param(lambda: GSM8KTask(), "_load_fewshot_dataset", id="gsm8k-train"),
        pytest.param(lambda: HumanEvalTask(), "_load_dataset", id="humaneval"),
    ],
)
def test_dataset_loaders_do_not_pass_trust_remote_code(
    monkeypatch, task_factory, method_name
):
    import datasets

    load_calls = []

    def fake_load_dataset(*args, **kwargs):
        load_calls.append((args, kwargs))
        return []

    monkeypatch.setattr(datasets, "load_dataset", fake_load_dataset)

    getattr(task_factory(), method_name)()

    assert load_calls
    for call_args, call_kwargs in load_calls:
        assert call_args
        assert "trust_remote_code" not in call_kwargs


# ------------------------------------------------------------------ #
# MMLU task                                                            #
# ------------------------------------------------------------------ #

class TestMMLUTask:

    def _make_doc(self):
        return {
            "question": "What is 2+2?",
            "choices": ["3", "4", "5", "6"],
            "answer": 1,  # "B" = index 1 = "4"
        }

    def test_doc_to_text(self):
        task = MMLUTask()
        doc = self._make_doc()
        text = task.doc_to_text(doc)
        assert "2+2" in text
        assert "A." in text
        assert "B." in text
        assert "Answer:" in text

    def test_doc_to_target(self):
        task = MMLUTask()
        doc = self._make_doc()
        assert task.doc_to_target(doc).strip() == "B"

    def test_construct_requests_returns_four(self):
        task = MMLUTask()
        doc = self._make_doc()
        ctx = task.doc_to_text(doc)
        requests = task.construct_requests(doc, ctx)
        assert len(requests) == 4
        for req in requests:
            assert isinstance(req, tuple)
            assert len(req) == 2

    def test_process_results_correct(self):
        task = MMLUTask()
        doc = self._make_doc()
        # Mock: answer is B (index 1), highest LL is at index 1
        results = [(-2.0, False), (-0.5, True), (-3.0, False), (-4.0, False)]
        out = task.process_results(doc, results)
        assert out["accuracy"] == 1

    def test_process_results_wrong(self):
        task = MMLUTask()
        doc = self._make_doc()
        # Highest LL at index 0 but correct is index 1
        results = [(-0.5, True), (-2.0, False), (-3.0, False), (-4.0, False)]
        out = task.process_results(doc, results)
        assert out["accuracy"] == 0

    def test_aggregation_mean(self):
        task = MMLUTask()
        agg = task.aggregation()
        assert "accuracy" in agg
        assert agg["accuracy"]([1, 0, 1, 1]) == pytest.approx(0.75)

    def test_higher_is_better(self):
        task = MMLUTask()
        assert task.higher_is_better()["accuracy"] is True

    def test_name(self):
        assert MMLUTask().name == "mmlu"


# ------------------------------------------------------------------ #
# HellaSwag task                                                       #
# ------------------------------------------------------------------ #

class TestHellaSwagTask:

    def _make_doc(self):
        return {
            "ctx": "A person is walking to the store.",
            "endings": [
                "They trip and fall.",
                "They pick up a basketball.",
                "They buy groceries.",
                "They fly into space.",
            ],
            "label": "2",  # correct ending index
        }

    def test_doc_to_text(self):
        task = HellaSwagTask()
        doc = self._make_doc()
        text = task.doc_to_text(doc)
        assert "store" in text

    def test_construct_requests_returns_four(self):
        task = HellaSwagTask()
        doc = self._make_doc()
        ctx = task.doc_to_text(doc)
        requests = task.construct_requests(doc, ctx)
        assert len(requests) == 4

    def test_process_results_correct(self):
        task = HellaSwagTask()
        doc = self._make_doc()
        # Highest normalised LL for index 2
        results = [(-4.0, False), (-3.0, False), (-1.0, True), (-5.0, False)]
        out = task.process_results(doc, results)
        assert out["accuracy"] == 1

    def test_name(self):
        assert HellaSwagTask().name == "hellaswag"


# ------------------------------------------------------------------ #
# WinoGrande task                                                      #
# ------------------------------------------------------------------ #

class TestWinoGrandeTask:

    def _make_doc(self):
        return {
            "sentence": "Sarah was nicer than Emma because _ was more patient.",
            "option1": "Sarah",
            "option2": "Emma",
            "answer": "1",  # 1-indexed: option1 (Sarah)
        }

    def test_construct_requests_returns_two(self):
        task = WinoGrandeTask()
        doc = self._make_doc()
        ctx = task.doc_to_text(doc)
        requests = task.construct_requests(doc, ctx)
        assert len(requests) == 2

    def test_continuations_include_option(self):
        task = WinoGrandeTask()
        doc = self._make_doc()
        ctx = task.doc_to_text(doc)
        requests = task.construct_requests(doc, ctx)
        assert "Sarah" in requests[0][1]
        assert "Emma" in requests[1][1]

    def test_process_results_correct(self):
        task = WinoGrandeTask()
        doc = self._make_doc()
        results = [(-1.0, True), (-2.0, False)]  # option1 wins
        out = task.process_results(doc, results)
        assert out["accuracy"] == 1

    def test_process_results_wrong(self):
        task = WinoGrandeTask()
        doc = self._make_doc()
        results = [(-2.0, False), (-1.0, True)]  # option2 wins, but correct is option1
        out = task.process_results(doc, results)
        assert out["accuracy"] == 0

    def test_name(self):
        assert WinoGrandeTask().name == "winogrande"


# ------------------------------------------------------------------ #
# GSM8K task                                                           #
# ------------------------------------------------------------------ #

class TestGSM8KTask:

    @pytest.mark.parametrize("text,expected", [
        ("Some work\n#### 42", "42"),
        ("Work here\n#### 1,234", "1234"),
        ("No hash pattern 99", "99"),
        ("Nothing here", None),
    ])
    def test_extract_answer(self, text, expected):
        assert _extract_answer(text) == expected

    @pytest.mark.parametrize("text,expected", [
        ("The answer is 42.", "42"),
        ("= 100", "100"),
        ("#### 7", "7"),
        ("blah blah 55", "55"),
    ])
    def test_extract_generated_answer(self, text, expected):
        assert _extract_generated_answer(text) == expected

    def _make_doc(self):
        return {
            "question": "If apples cost $2 each and you buy 3, how much do you spend?",
            "answer": "3 apples × $2 = $6\n#### 6",
        }

    def test_doc_to_text(self):
        task = GSM8KTask()
        doc = self._make_doc()
        text = task.doc_to_text(doc)
        assert "Question:" in text
        assert "Answer:" in text

    def test_construct_requests_single(self):
        task = GSM8KTask()
        doc = self._make_doc()
        ctx = task.doc_to_text(doc)
        requests = task.construct_requests(doc, ctx)
        assert len(requests) == 1
        prompt, gen_kwargs = requests[0]
        assert isinstance(gen_kwargs, dict)
        assert "max_new_tokens" in gen_kwargs

    def test_process_results_correct(self):
        task = GSM8KTask()
        doc = self._make_doc()
        results = ["3 × 2 = 6\n#### 6"]
        out = task.process_results(doc, results)
        assert out["exact_match"] == 1

    def test_process_results_wrong(self):
        task = GSM8KTask()
        doc = self._make_doc()
        results = ["#### 7"]
        out = task.process_results(doc, results)
        assert out["exact_match"] == 0

    def test_name(self):
        assert GSM8KTask().name == "gsm8k"


# ------------------------------------------------------------------ #
# HumanEval task                                                       #
# ------------------------------------------------------------------ #

class TestHumanEvalTask:

    def test_sanitize_code_removes_fences(self):
        code = "```python\ndef foo():\n    return 1\n```"
        assert "```" not in _sanitize_code(code)
        assert "def foo" in _sanitize_code(code)

    def test_execute_code_simple_pass(self):
        code = "x = 1 + 1\nassert x == 2"
        assert _execute_code(code, timeout=5) is True

    def test_execute_code_fail(self):
        code = "assert 1 == 2"
        assert _execute_code(code, timeout=5) is False

    def test_execute_code_syntax_error(self):
        code = "def foo(: pass"
        assert _execute_code(code, timeout=5) is False

    def _make_doc(self):
        return {
            "task_id": "HumanEval/0",
            "prompt": "def add(a, b):\n    \"\"\"Add two numbers.\"\"\"\n",
            "canonical_solution": "    return a + b\n",
            "test": "def check(candidate):\n    assert candidate(1, 2) == 3\n",
            "entry_point": "add",
        }

    def test_doc_to_text(self):
        task = HumanEvalTask()
        doc = self._make_doc()
        text = task.doc_to_text(doc)
        assert "def add" in text

    def test_construct_requests_single(self):
        task = HumanEvalTask(num_samples_per_task=1)
        doc = self._make_doc()
        ctx = task.doc_to_text(doc)
        requests = task.construct_requests(doc, ctx)
        assert len(requests) == 1

    def test_construct_requests_multiple(self):
        task = HumanEvalTask(num_samples_per_task=3)
        doc = self._make_doc()
        ctx = task.doc_to_text(doc)
        requests = task.construct_requests(doc, ctx)
        assert len(requests) == 3

    def test_process_results_pass(self):
        task = HumanEvalTask()
        doc = self._make_doc()
        # Correct implementation
        results = ["    return a + b\n"]
        out = task.process_results(doc, results)
        assert out["pass@1"] == 1

    def test_process_results_fail(self):
        task = HumanEvalTask()
        doc = self._make_doc()
        results = ["    return a - b\n"]
        out = task.process_results(doc, results)
        assert out["pass@1"] == 0

    def test_name(self):
        assert HumanEvalTask().name == "humaneval"


# ------------------------------------------------------------------ #
# BaseTask evaluate() with mock model                                  #
# ------------------------------------------------------------------ #

class _MockTask(BaseTask):
    """Minimal concrete task for testing the evaluate() plumbing."""

    VERSION = 0
    DATASET_PATH = "mock"

    def _load_dataset(self):
        return [
            {"ctx": "Q1", "choices": ["A", "B", "C", "D"], "answer": 0},
            {"ctx": "Q2", "choices": ["A", "B", "C", "D"], "answer": 2},
        ]

    def doc_to_text(self, doc):
        return doc["ctx"]

    def doc_to_target(self, doc):
        return doc["choices"][doc["answer"]]

    def construct_requests(self, doc, ctx):
        return [(ctx, c) for c in doc["choices"]]

    def process_results(self, doc, results):
        lls = [r[0] for r in results]
        predicted = max(range(len(lls)), key=lls.__getitem__)
        return {"accuracy": int(predicted == doc["answer"])}

    def aggregation(self):
        from evaluation.utils.metrics import mean
        return {"accuracy": mean}

    def higher_is_better(self):
        return {"accuracy": True}


class TestBaseTaskEvaluate:

    def test_evaluate_with_mock_model(self):
        task = _MockTask(num_fewshot=0)
        mock_model = MagicMock()
        # Return log-likelihoods so correct answer always wins
        # Doc 0: answer=0, Doc 1: answer=2
        mock_model.loglikelihood.return_value = [
            (-0.1, True), (-2.0, False), (-3.0, False), (-4.0, False),  # Doc 0 → correct
            (-2.0, False), (-3.0, False), (-0.1, True), (-4.0, False),  # Doc 1 → correct
        ]
        results = task.evaluate(mock_model)
        assert results["accuracy"] == pytest.approx(1.0)

    def test_evaluate_max_samples(self):
        task = _MockTask(num_fewshot=0, max_samples=1)
        mock_model = MagicMock()
        mock_model.loglikelihood.return_value = [
            (-0.1, True), (-2.0, False), (-3.0, False), (-4.0, False),
        ]
        results = task.evaluate(mock_model)
        assert "accuracy" in results

    def test_infer_request_type_loglikelihood(self):
        requests = [("context", "continuation")]
        assert BaseTask._infer_request_type(requests) == "loglikelihood"

    def test_infer_request_type_generation(self):
        requests = [("prompt", {"max_new_tokens": 100})]
        assert BaseTask._infer_request_type(requests) == "generation"
