"""Integration tests for pruning search orchestration."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from evaluation.pruning import runner as runner_module
from evaluation.pruning.runner import PruningSearchRunner, causal_token_nll


class _DummyWrapper:
    def __init__(self, num_layers=4):
        self.num_layers = num_layers
        self.tokenizer = object()
        self.device = "cpu"
        self.model = SimpleNamespace(config=SimpleNamespace(_commit_hash="revision"))
        self.strategy = object()
        self.strategy_history = []

    def set_strategy(self, strategy):
        self.strategy_history.append(strategy)
        self.strategy = strategy


def _install_mutable_shortgpt_corpus(monkeypatch):
    state = {"token": 1, "iterations": 0}

    class _MutableShortGPTCorpus:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.stats = {}

        def __iter__(self):
            state["iterations"] += 1
            self.stats = {"segments": 3}
            yield {
                "input_ids": torch.tensor([[state["token"], 2, 3]]),
                "attention_mask": torch.tensor([[1, 1, 1]]),
            }

    monkeypatch.setattr(
        runner_module,
        "ShortGPTTokenBatches",
        _MutableShortGPTCorpus,
    )
    return state


def test_shortgpt_runner_ranks_once_persists_and_restores_strategy(
    monkeypatch, tmp_path
):
    wrapper = _DummyWrapper()
    original_strategy = wrapper.strategy
    calls = []
    corpus_state = _install_mutable_shortgpt_corpus(monkeypatch)

    def fake_score(model_wrapper, batches):
        calls.append((model_wrapper, batches))
        assert model_wrapper.strategy is None
        assert len(list(batches)) == 1
        return [0.4, 0.1, 0.3, 0.2]

    monkeypatch.setattr(runner_module, "score_shortgpt_blocks", fake_score)
    search = PruningSearchRunner(
        model_wrapper=wrapper,
        model_name="org/model",
        method="shortgpt",
        config={"num_remove": 2},
        results_dir=tmp_path,
    )

    result = search.run_global()

    assert result["selected_layers"] == [1, 3]
    assert result["selected_layers_1based"] == [2, 4]
    assert result["trace"]["removal_order"] == [1, 3, 2, 0]
    assert result["trace"]["corpus_stats"]["segments"] == 3
    assert len(result["trace"]["corpus_stats"]["sha256"]) == 64
    assert wrapper.strategy is original_strategy
    assert len(calls) == 1
    # A first search fingerprints the same streaming pass consumed by scoring.
    assert corpus_state["iterations"] == 1

    monkeypatch.setattr(
        runner_module,
        "score_shortgpt_blocks",
        lambda *args: (_ for _ in ()).throw(AssertionError("rescored")),
    )
    reused = search.run_global()
    assert reused["selected_layers"] == [1, 3]
    # Cache validation streams CPU tokens once but does not launch scoring.
    assert corpus_state["iterations"] == 2


def test_shortgpt_rescores_when_exact_token_batches_change(monkeypatch, tmp_path):
    wrapper = _DummyWrapper()
    corpus_state = _install_mutable_shortgpt_corpus(monkeypatch)
    score_calls = []

    def fake_score(model_wrapper, batches):
        del model_wrapper
        score_calls.append(corpus_state["token"])
        assert len(list(batches)) == 1
        return [0.4, 0.1, 0.3, 0.2]

    monkeypatch.setattr(runner_module, "score_shortgpt_blocks", fake_score)
    search = PruningSearchRunner(
        model_wrapper=wrapper,
        model_name="org/model",
        method="shortgpt",
        config={"num_remove": 2},
        results_dir=tmp_path,
    )

    first = search.run_global()
    first_sha256 = first["trace"]["corpus_stats"]["sha256"]
    assert search.run_global()["trace_path"] == first["trace_path"]
    assert score_calls == [1]

    corpus_state["token"] = 9
    changed = search.run_global()

    assert changed["trace_path"] == first["trace_path"]
    assert changed["trace"]["corpus_stats"]["sha256"] != first_sha256
    assert score_calls == [1, 9]
    # First score, valid-cache check, stale-cache check, then replacement score.
    assert corpus_state["iterations"] == 4


def test_shortgpt_zero_budget_skips_corpus_scoring(monkeypatch, tmp_path):
    wrapper = _DummyWrapper()
    monkeypatch.setattr(
        runner_module,
        "score_shortgpt_blocks",
        lambda *args: (_ for _ in ()).throw(AssertionError("scored")),
    )
    search = PruningSearchRunner(
        model_wrapper=wrapper,
        model_name="org/model",
        method="shortgpt",
        config={"num_remove": 0},
        results_dir=tmp_path,
    )

    result = search.run_global()

    assert result["selected_layers"] == []
    assert result["trace"]["search_skipped"] == "zero_budget"


def test_local_model_execution_identity_hashes_only_inference_artifacts(tmp_path):
    model_dir = tmp_path / "local-model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text('{"layers": 4}', encoding="utf-8")
    (model_dir / "tokenizer.json").write_text('{"vocab": 1}', encoding="utf-8")
    (model_dir / "model.safetensors").write_bytes(b"weights-v1")
    (model_dir / "README.md").write_text("notes-v1", encoding="utf-8")
    (model_dir / "experiment_config.json").write_text(
        "experiment-v1",
        encoding="utf-8",
    )
    (model_dir / "optimizer.pt").write_bytes(b"optimizer-v1")
    (model_dir / ".cache").mkdir()
    (model_dir / ".cache" / "model.safetensors").write_bytes(b"cached-v1")
    (model_dir / ".git").mkdir()
    (model_dir / ".git" / "config.json").write_text("git-v1", encoding="utf-8")

    def identity():
        search = PruningSearchRunner(
            model_wrapper=_DummyWrapper(),
            model_name=str(model_dir),
            method="shortgpt",
            config={"num_remove": 0},
            results_dir=tmp_path / "results",
        )
        return search.execution_identity, search.provenance

    first, first_provenance = identity()
    assert first["model_source"] == "local"
    assert first["model_revision"] is None
    assert len(first["local_artifacts_sha256"]) == 64
    assert first_provenance["local_artifact_count"] == 3

    (model_dir / "README.md").write_text("notes-v2", encoding="utf-8")
    (model_dir / "experiment_config.json").write_text(
        "experiment-v2",
        encoding="utf-8",
    )
    (model_dir / "optimizer.pt").write_bytes(b"optimizer-v2")
    (model_dir / ".cache" / "model.safetensors").write_bytes(b"cached-v2")
    (model_dir / ".git" / "config.json").write_text("git-v2", encoding="utf-8")
    unrelated_changed, _ = identity()
    assert unrelated_changed == first

    (model_dir / "tokenizer.json").write_text('{"vocab": 2}', encoding="utf-8")
    tokenizer_changed, _ = identity()
    assert tokenizer_changed["local_artifacts_sha256"] != first[
        "local_artifacts_sha256"
    ]

    (model_dir / "model.safetensors").write_bytes(b"weights-v2")
    model_changed, _ = identity()
    assert model_changed["local_artifacts_sha256"] != tokenizer_changed[
        "local_artifacts_sha256"
    ]


def test_remote_model_execution_identity_uses_loaded_revision(tmp_path):
    search = PruningSearchRunner(
        model_wrapper=_DummyWrapper(),
        model_name="org/model",
        method="shortgpt",
        config={"num_remove": 0},
        results_dir=tmp_path,
    )

    assert search.execution_identity["model_source"] == "remote"
    assert search.execution_identity["model_revision"] == "revision"
    assert search.execution_identity["local_artifacts_sha256"] is None


def test_token_fingerprint_rejects_non_cpu_tensors():
    batch = {
        "input_ids": torch.empty((1, 2), dtype=torch.long, device="meta"),
        "attention_mask": torch.empty((1, 2), dtype=torch.long, device="meta"),
    }

    with pytest.raises(ValueError, match="must remain on CPU"):
        runner_module._token_batches_fingerprint([batch])


def test_sleb_runner_uses_seed_zero_and_reuses_valid_completed_trace(
    monkeypatch, tmp_path
):
    wrapper = _DummyWrapper(num_layers=3)
    corpus_calls = []

    def fake_corpus(tokenizer, **kwargs):
        corpus_calls.append(kwargs)
        return [
            {
                "input_ids": torch.tensor([[1, 2]]),
                "attention_mask": torch.tensor([[1, 1]]),
            }
        ], {"segments": 1}

    monkeypatch.setattr(runner_module, "build_sleb_token_batches", fake_corpus)
    monkeypatch.setattr(
        runner_module,
        "causal_token_nll",
        lambda wrapper, batches, removed: {0: 3.0, 1: 1.0, 2: 2.0}[
            removed[-1]
        ],
    )
    search = PruningSearchRunner(
        model_wrapper=wrapper,
        model_name="org/model",
        method="sleb",
        config={"num_remove": 1},
        results_dir=tmp_path,
    )

    result = search.run_global()

    assert result["selected_layers"] == [1]
    assert corpus_calls[0]["seed"] == 0
    assert result["trace"]["corpus_stats"]["segments"] == 1
    assert len(result["trace"]["corpus_stats"]["sha256"]) == 64

    assert search.run_global()["selected_layers"] == [1]
    # Reconstructing the small corpus is intentional: its exact token hash is
    # part of checkpoint identity, while completed candidate scores are reused.
    assert len(corpus_calls) == 2


def test_sleb_zero_budget_skips_corpus_loading(monkeypatch, tmp_path):
    wrapper = _DummyWrapper(num_layers=3)
    monkeypatch.setattr(
        runner_module,
        "build_sleb_token_batches",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("loaded")),
    )
    search = PruningSearchRunner(
        model_wrapper=wrapper,
        model_name="org/model",
        method="sleb",
        config={"num_remove": 0},
        results_dir=tmp_path,
    )

    result = search.run_global()

    assert result["selected_layers"] == []
    assert result["trace"]["corpus_stats"] == {
        "search_skipped": "zero_budget"
    }


class _Task:
    VERSION = 2
    DATASET_PATH = "task/data"
    calibration_split_name = "validation"
    seed = 11
    num_fewshot = 0
    max_samples = None
    primary_metric = "accuracy"

    def __init__(self, wrapper):
        self.wrapper = wrapper
        self.docs = [{"id": 1}, {"id": 2}]
        self.evaluation_calls = []

    def calibration_docs(self, max_samples=None, seed=None):
        assert max_samples == 2
        assert seed == 11
        return list(self.docs)

    def aggregation(self):
        return {"accuracy": lambda values: sum(values) / len(values)}

    def higher_is_better(self):
        return {"accuracy": True}

    def evaluate_docs(self, model, docs):
        assert model is self.wrapper
        assert docs == self.docs
        removed = (
            ()
            if model.strategy is None
            else tuple(layer - 1 for layer in model.strategy.skip_layers)
        )
        self.evaluation_calls.append(removed)
        scores = {
            (): 0.90,
            (0,): 0.88,
            (1,): 0.91,
            (2,): 0.80,
            (3,): 0.70,
        }
        return {"accuracy": scores[removed]}


def test_tale_runner_searches_labeled_docs_and_restores_strategy(tmp_path):
    wrapper = _DummyWrapper()
    original_strategy = wrapper.strategy
    task = _Task(wrapper)
    search = PruningSearchRunner(
        model_wrapper=wrapper,
        model_name="org/model",
        method="tale",
        config={"search_max_samples": 2, "max_remove": 1},
        results_dir=tmp_path,
    )

    result = search.run_task("mmlu", task)

    assert result["selected_layers"] == [1]
    assert result["variant"] == "threshold_final"
    assert task.evaluation_calls == [(), (0,), (1,), (2,), (3,)]
    assert wrapper.strategy is original_strategy

    task.evaluation_calls.clear()
    assert search.run_task("mmlu", task)["selected_layers"] == [1]
    assert task.evaluation_calls == []


def test_tale_target_remove_caps_search_without_explicit_max(tmp_path):
    wrapper = _DummyWrapper()
    task = _Task(wrapper)
    search = PruningSearchRunner(
        model_wrapper=wrapper,
        model_name="org/model",
        method="tale",
        config={
            "search_max_samples": 2,
            "target_remove": 1,
            "variant": "budget",
        },
        results_dir=tmp_path,
    )

    result = search.run_task("mmlu", task)

    assert result["selected_layers"] == [1]
    assert result["trace"]["stop_reason"] == "target_remove"
    assert task.evaluation_calls == [(), (0,), (1,), (2,), (3,)]


def test_tale_trace_identity_changes_with_prompt_configuration(tmp_path):
    wrapper = _DummyWrapper()
    task = _Task(wrapper)
    first_search = PruningSearchRunner(
        model_wrapper=wrapper,
        model_name="org/model",
        method="tale",
        config={"search_max_samples": 2, "max_remove": 1},
        results_dir=tmp_path,
    )
    first = first_search.run_task("mmlu", task)
    task.evaluation_calls.clear()

    task.num_fewshot = 1
    second = first_search.run_task("mmlu", task)

    assert second["trace_path"] != first["trace_path"]
    assert task.evaluation_calls


@pytest.mark.parametrize(
    "config,match",
    [
        ({"max_remove": 4}, "retain at least one"),
        ({"target_remove": 4}, "retain at least one"),
        ({"variant": "budget"}, "requires target_remove"),
    ],
)
def test_tale_runner_rejects_unsafe_or_incomplete_budget(config, match, tmp_path):
    wrapper = _DummyWrapper()
    task = _Task(wrapper)
    search = PruningSearchRunner(
        model_wrapper=wrapper,
        model_name="org/model",
        method="tale",
        config={"search_max_samples": 2, **config},
        results_dir=tmp_path,
    )

    with pytest.raises(ValueError, match=match):
        search.run_task("mmlu", task)


class _NLLWrapper:
    device = "cpu"

    def __init__(self):
        self.strategy = object()
        self.seen_skip_layers = None
        self.kwargs = None

    def set_strategy(self, strategy):
        self.strategy = strategy

    def _forward_model(self, **kwargs):
        self.kwargs = kwargs
        self.seen_skip_layers = self.strategy.skip_layers
        batch, length = kwargs["input_ids"].shape
        return SimpleNamespace(logits=torch.zeros(batch, length, 5))


def test_causal_token_nll_masks_padding_and_restores_strategy():
    wrapper = _NLLWrapper()
    original_strategy = wrapper.strategy
    batch = {
        "input_ids": torch.tensor([[1, 2, 3, 0]]),
        "attention_mask": torch.tensor([[1, 1, 1, 0]]),
    }

    value = causal_token_nll(wrapper, [batch], removed_layers=(0, 2))

    assert value == pytest.approx(torch.log(torch.tensor(5.0)).item())
    assert wrapper.seen_skip_layers == (1, 3)
    assert wrapper.kwargs["use_cache"] is False
    assert wrapper.strategy is original_strategy


def test_causal_token_nll_keeps_native_logits_dtype(monkeypatch):
    wrapper = _NLLWrapper()

    def bf16_forward(**kwargs):
        wrapper.kwargs = kwargs
        wrapper.seen_skip_layers = wrapper.strategy.skip_layers
        batch, length = kwargs["input_ids"].shape
        return SimpleNamespace(
            logits=torch.zeros(batch, length, 5, dtype=torch.bfloat16)
        )

    wrapper._forward_model = bf16_forward
    seen_dtypes = []
    original_cross_entropy = runner_module.F.cross_entropy

    def capture_dtype(logits, labels, **kwargs):
        seen_dtypes.append(logits.dtype)
        return original_cross_entropy(logits, labels, **kwargs)

    monkeypatch.setattr(runner_module.F, "cross_entropy", capture_dtype)
    batch = {
        "input_ids": torch.tensor([[1, 2, 3]]),
        "attention_mask": torch.tensor([[1, 1, 1]]),
    }

    causal_token_nll(wrapper, [batch], removed_layers=(0,))

    assert seen_dtypes == [torch.bfloat16]
