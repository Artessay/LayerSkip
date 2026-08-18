"""End-to-end search orchestration for ShortGPT, SLEB, and TALE."""

from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import platform
from typing import Any, Dict, Iterable, Iterator, Mapping, Optional, Sequence

import torch
import torch.nn.functional as F

from evaluation.pruning.corpora import (
    ShortGPTTokenBatches,
    build_sleb_token_batches,
)
from evaluation.pruning.io import (
    load_search_envelope,
    make_search_envelope,
    save_search_envelope,
    search_trace_path,
)
from evaluation.pruning.shortgpt import (
    score_shortgpt_blocks,
    select_shortgpt_layers,
    validate_shortgpt_trace,
)
from evaluation.pruning.sleb import run_sleb_search
from evaluation.pruning.tale import run_tale_search
from evaluation.strategies.manualskip import ManualSkipStrategy


logger = logging.getLogger(__name__)

PRUNING_METHODS = ("shortgpt", "sleb", "tale")
TALE_VARIANTS = ("threshold_final", "best", "bsba", "budget")
_NLL_TOKEN_CHUNK_SIZE = 128
_ARTIFACT_HASH_CHUNK_SIZE = 1024 * 1024
_LOCAL_ARTIFACT_EXCLUDED_DIRS = frozenset(
    {
        ".cache",
        ".git",
        ".hg",
        ".locks",
        ".svn",
        "__pycache__",
    }
)
_LOCAL_ARTIFACT_IGNORED_FILES = frozenset(
    {
        "optimizer.pt",
        "rng_state.pth",
        "scaler.pt",
        "scheduler.pt",
        "training_args.bin",
        "trainer_state.json",
    }
)
_LOCAL_ARTIFACT_FILENAMES = frozenset(
    {
        "added_tokens.json",
        "adapter_config.json",
        "awq_config.json",
        "chat_template.json",
        "chat_template.jinja",
        "config.json",
        "feature_extractor_config.json",
        "generation_config.json",
        "gptq_config.json",
        "merges.txt",
        "preprocessor_config.json",
        "processor_config.json",
        "quantization_config.json",
        "quantize_config.json",
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer.model",
        "tokenizer_config.json",
        "vocab.json",
        "vocab.txt",
    }
)
_LOCAL_ARTIFACT_BINARY_SUFFIXES = (
    ".bin",
    ".bpe",
    ".ckpt",
    ".gguf",
    ".h5",
    ".model",
    ".msgpack",
    ".onnx",
    ".pt",
    ".pth",
    ".safetensors",
    ".tiktoken",
)
_LOCAL_ARTIFACT_CODE_PREFIXES = (
    "configuration_",
    "modeling_",
    "processing_",
    "tokenization_",
)


def _library_version(module_name: str) -> Optional[str]:
    try:
        module = __import__(module_name)
    except ImportError:
        return None
    value = getattr(module, "__version__", None)
    return str(value) if value is not None else None


def _model_revision(model_wrapper: Any) -> Optional[str]:
    config = getattr(getattr(model_wrapper, "model", None), "config", None)
    for name in ("_commit_hash", "revision", "model_revision"):
        value = getattr(config, name, None) if config is not None else None
        if value:
            return str(value)
    return None


def _is_local_artifact_file(relative_path: Path) -> bool:
    """Return whether a local checkpoint file affects inference/tokenization."""

    name = relative_path.name.lower()
    if name in _LOCAL_ARTIFACT_IGNORED_FILES:
        return False
    if name in _LOCAL_ARTIFACT_FILENAMES:
        return True
    if name.endswith(_LOCAL_ARTIFACT_BINARY_SUFFIXES):
        return True
    if name.endswith(".index.json"):
        return True
    if name.endswith(".jinja") and "chat_templates" in relative_path.parts:
        return True
    return name.endswith(".py") and name.startswith(_LOCAL_ARTIFACT_CODE_PREFIXES)


def _local_artifacts_fingerprint(
    model_name: str,
) -> tuple[Optional[str], Optional[int]]:
    """Hash recognized inference artifacts when ``model_name`` is local.

    Weight shards are read from disk in bounded chunks.  Repository metadata,
    caches, optimizer state, documentation, and other unrelated files are not
    part of the identity.
    """

    candidate = Path(model_name).expanduser()
    if not candidate.is_dir():
        return None, None
    root = candidate.resolve()
    artifacts: list[tuple[str, Path]] = []
    for directory, directory_names, file_names in os.walk(root, followlinks=False):
        directory_names[:] = sorted(
            name
            for name in directory_names
            if name not in _LOCAL_ARTIFACT_EXCLUDED_DIRS
        )
        directory_path = Path(directory)
        for file_name in sorted(file_names):
            path = directory_path / file_name
            relative_path = path.relative_to(root)
            if _is_local_artifact_file(relative_path) and path.is_file():
                artifacts.append((relative_path.as_posix(), path))

    if not artifacts:
        raise ValueError(
            f"Local model directory {candidate} contains no recognized "
            "model or tokenizer artifacts"
        )

    digest = hashlib.sha256(b"layerskip-local-artifacts-v1\0")
    for relative_name, path in sorted(artifacts):
        encoded_name = relative_name.encode("utf-8")
        digest.update(len(encoded_name).to_bytes(8, "big"))
        digest.update(encoded_name)
        file_digest = hashlib.sha256()
        file_size = 0
        with path.open("rb") as handle:
            while True:
                chunk = handle.read(_ARTIFACT_HASH_CHUNK_SIZE)
                if not chunk:
                    break
                file_digest.update(chunk)
                file_size += len(chunk)
        digest.update(file_size.to_bytes(8, "big"))
        digest.update(file_digest.digest())
    return digest.hexdigest(), len(artifacts)


def _provenance(model_wrapper: Any, model_name: str) -> Dict[str, Any]:
    model = getattr(model_wrapper, "model", None)
    tokenizer = getattr(model_wrapper, "tokenizer", None)
    local_artifacts_sha256, local_artifact_count = _local_artifacts_fingerprint(
        model_name
    )
    model_source = "local" if local_artifacts_sha256 is not None else "remote"
    model_dtype = getattr(model, "dtype", None)
    if model_dtype is None and isinstance(model, torch.nn.Module):
        try:
            model_dtype = next(model.parameters()).dtype
        except StopIteration:
            pass
    return {
        "model_revision": _model_revision(model_wrapper),
        "model_source": model_source,
        "local_artifacts_sha256": local_artifacts_sha256,
        "local_artifact_count": local_artifact_count,
        "model_class": type(model).__name__ if model is not None else None,
        "model_dtype": str(model_dtype) if model_dtype is not None else None,
        "tokenizer_class": (
            type(tokenizer).__name__ if tokenizer is not None else None
        ),
        "tokenizer_name_or_path": (
            str(getattr(tokenizer, "name_or_path"))
            if getattr(tokenizer, "name_or_path", None)
            else None
        ),
        "apply_chat_template": bool(
            getattr(model_wrapper, "_use_chat_template", False)
        ),
        "model_batch_size": getattr(model_wrapper, "batch_size", None),
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "transformers": _library_version("transformers"),
        "datasets": _library_version("datasets"),
    }


def _execution_identity(provenance: Mapping[str, Any]) -> Dict[str, Any]:
    """Return search-affecting runtime fields used in checkpoint identity."""

    fields = (
        "model_source",
        "local_artifacts_sha256",
        "model_class",
        "model_dtype",
        "tokenizer_class",
        "tokenizer_name_or_path",
        "apply_chat_template",
        "model_batch_size",
        "torch",
        "transformers",
        "datasets",
    )
    identity = {field: provenance.get(field) for field in fields}
    identity["model_revision"] = (
        provenance.get("model_revision")
        if provenance.get("model_source") == "remote"
        else None
    )
    return identity


def _nonnegative_int(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < 0:
        raise ValueError(f"{name} must be non-negative")
    return value


def _resolve_num_remove(
    *,
    num_layers: int,
    num_remove: Optional[int],
    prune_ratio: float,
) -> int:
    if num_remove is not None:
        if isinstance(num_remove, bool) or not isinstance(num_remove, int):
            raise TypeError("num_remove must be an integer or None")
        resolved = num_remove
    else:
        if isinstance(prune_ratio, bool):
            raise TypeError("prune_ratio must be a real number")
        try:
            ratio = float(prune_ratio)
        except (TypeError, ValueError) as exc:
            raise TypeError("prune_ratio must be a real number") from exc
        if not math.isfinite(ratio) or ratio < 0.0 or ratio >= 1.0:
            raise ValueError("prune_ratio must be within [0, 1)")
        resolved = math.floor(num_layers * ratio)
    if resolved < 0 or resolved >= num_layers:
        raise ValueError(
            f"num_remove must retain at least one of {num_layers} layers, got {resolved}"
        )
    return resolved


@contextmanager
def temporarily_bypass_layers(
    model_wrapper: Any,
    removed_layers: Iterable[int],
) -> Iterator[None]:
    """Apply original zero-based layer IDs with the existing bypass engine."""

    removed = tuple(sorted(removed_layers))
    previous_strategy = getattr(model_wrapper, "strategy", None)
    strategy = (
        ManualSkipStrategy(skip_layers=[layer + 1 for layer in removed])
        if removed
        else None
    )
    model_wrapper.set_strategy(strategy)
    try:
        yield
    finally:
        model_wrapper.set_strategy(previous_strategy)


def causal_token_nll(
    model_wrapper: Any,
    token_batches: Sequence[Mapping[str, torch.Tensor]],
    removed_layers: Iterable[int],
) -> float:
    """Return mean next-token NLL for a fixed candidate-pruned model."""

    loss_sum = 0.0
    token_count = 0
    with temporarily_bypass_layers(model_wrapper, removed_layers):
        with torch.inference_mode():
            for batch in token_batches:
                input_ids = batch["input_ids"].to(model_wrapper.device)
                attention_mask = batch["attention_mask"].to(model_wrapper.device)
                if input_ids.shape[1] < 2:
                    continue
                outputs = model_wrapper._forward_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=False,
                    use_cache=False,
                )
                logits = outputs.logits
                if logits.ndim != 3 or logits.shape[:2] != input_ids.shape:
                    raise ValueError(
                        "SLEB model logits must have shape [batch, sequence, vocab]"
                    )

                # Keep logits in the model's native dtype, matching the
                # official SLEB loss.  Chunking the sequence bounds the
                # contiguous CE workspace; converting a full 2048 x 128k
                # vocabulary tensor to FP32 can otherwise add ~1 GiB per
                # sample on current Llama models.
                shift_length = input_ids.shape[1] - 1
                for start in range(0, shift_length, _NLL_TOKEN_CHUNK_SIZE):
                    end = min(start + _NLL_TOKEN_CHUNK_SIZE, shift_length)
                    chunk_logits = logits[:, start:end, :]
                    chunk_labels = input_ids[:, start + 1 : end + 1].clone()
                    valid = attention_mask[:, start + 1 : end + 1].ne(0)
                    chunk_labels.masked_fill_(~valid, -100)
                    valid_count = int(valid.sum().item())
                    if valid_count == 0:
                        continue
                    loss = F.cross_entropy(
                        chunk_logits.reshape(-1, chunk_logits.shape[-1]),
                        chunk_labels.reshape(-1),
                        ignore_index=-100,
                        reduction="sum",
                    )
                    loss_sum += float(loss.detach().float().cpu())
                    token_count += valid_count
    if token_count == 0:
        raise ValueError("No valid next-token labels were available for SLEB scoring")
    return loss_sum / token_count


def _docs_fingerprint(docs: Sequence[Mapping[str, Any]]) -> str:
    digest = hashlib.sha256()
    for doc in docs:
        encoded = json.dumps(
            doc,
            sort_keys=True,
            ensure_ascii=False,
            default=str,
            separators=(",", ":"),
        ).encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def _update_token_batch_fingerprint(
    digest: Any,
    *,
    batch_index: int,
    batch: Mapping[str, torch.Tensor],
    label: str,
) -> None:
    """Hash one CPU token batch without copying any tensor from an accelerator."""

    if not isinstance(batch, Mapping):
        raise TypeError(f"{label} batch {batch_index} must be a mapping")
    digest.update(batch_index.to_bytes(8, "big"))
    for key in ("input_ids", "attention_mask"):
        value = batch.get(key)
        if not isinstance(value, torch.Tensor):
            raise TypeError(
                f"{label} batch {batch_index} field {key!r} must be a tensor"
            )
        if value.device.type != "cpu":
            raise ValueError(
                f"{label} batch {batch_index} field {key!r} must remain on CPU "
                "while its cache fingerprint is computed"
            )
        tensor = value.detach().contiguous()
        metadata = json.dumps(
            {
                "key": key,
                "dtype": str(tensor.dtype),
                "shape": list(tensor.shape),
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        digest.update(len(metadata).to_bytes(8, "big"))
        digest.update(metadata)
        digest.update(tensor.numpy().tobytes())


class _FingerprintingTokenBatches:
    """Re-iterable streaming wrapper that records the last complete pass."""

    def __init__(self, source: Iterable[Mapping[str, torch.Tensor]], label: str):
        self.source = source
        self.label = label
        self.sha256: Optional[str] = None
        self.batch_count: Optional[int] = None

    def __iter__(self) -> Iterator[Mapping[str, torch.Tensor]]:
        digest = hashlib.sha256()
        count = 0
        self.sha256 = None
        self.batch_count = None
        for batch_index, batch in enumerate(self.source):
            _update_token_batch_fingerprint(
                digest,
                batch_index=batch_index,
                batch=batch,
                label=self.label,
            )
            count += 1
            yield batch
        self.sha256 = digest.hexdigest()
        self.batch_count = count

    def fingerprint(self) -> str:
        for _ in self:
            pass
        if self.sha256 is None:
            raise RuntimeError(f"Could not fingerprint the complete {self.label} corpus")
        if self.batch_count == 0:
            raise ValueError(f"The {self.label} corpus contains no token batches")
        return self.sha256


def _token_batches_fingerprint(
    token_batches: Sequence[Mapping[str, torch.Tensor]],
) -> str:
    """Hash the exact CPU token tensors compared by all SLEB candidates."""

    digest = hashlib.sha256()
    for batch_index, batch in enumerate(token_batches):
        _update_token_batch_fingerprint(
            digest,
            batch_index=batch_index,
            batch=batch,
            label="SLEB",
        )
    return digest.hexdigest()


def _task_inputs_fingerprint(
    task: Any,
    docs: Sequence[Mapping[str, Any]],
) -> str:
    fingerprint = getattr(task, "evaluation_inputs_fingerprint", None)
    if callable(fingerprint):
        return str(fingerprint(list(docs)))

    # Lightweight task doubles and third-party task adapters may not inherit
    # BaseTask.  Their stable public settings still prevent obvious cache
    # collisions; built-in tasks use the exact request fingerprint above.
    payload = {
        "docs_sha256": _docs_fingerprint(docs),
        "num_fewshot": getattr(task, "num_fewshot", None),
        "seed": getattr(task, "seed", None),
        "max_samples": getattr(task, "max_samples", None),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()


def _primary_metric(task: Any) -> str:
    explicit = getattr(task, "primary_metric", None)
    if isinstance(explicit, str) and explicit:
        return explicit
    higher = task.higher_is_better()
    for metric, is_higher in higher.items():
        if is_higher:
            return metric
    raise ValueError(f"Task {task.name!r} exposes no higher-is-better metric for TALE")


def _standard_result(
    *,
    method: str,
    num_layers: int,
    selected_layers: Sequence[int],
    trace_path: Path,
    trace: Mapping[str, Any],
    variant: Optional[str] = None,
) -> Dict[str, Any]:
    raw_selected = list(selected_layers)
    if any(isinstance(layer, bool) or not isinstance(layer, int) for layer in raw_selected):
        raise ValueError(f"{method} selected layer IDs must be integers")
    selected = sorted(raw_selected)
    if len(selected) != len(set(selected)):
        raise ValueError(f"{method} selected duplicate layer IDs")
    if any(layer < 0 or layer >= num_layers for layer in selected):
        raise ValueError(
            f"{method} selected layer IDs outside [0, {num_layers - 1}]"
        )
    if len(selected) >= num_layers:
        raise ValueError(f"{method} must retain at least one transformer layer")
    result = {
        "method": method,
        "num_layers": num_layers,
        "num_removed": len(selected),
        "selected_layers": selected,
        "selected_layers_1based": [layer + 1 for layer in selected],
        "layer_id_space": "original_model_0_based",
        "trace_path": str(trace_path),
        "trace": dict(trace),
    }
    if variant is not None:
        result["variant"] = variant
    return result


class PruningSearchRunner:
    """Run paper-aligned layer selection while keeping traces resumable."""

    def __init__(
        self,
        *,
        model_wrapper: Any,
        model_name: str,
        method: str,
        config: Optional[Mapping[str, Any]] = None,
        results_dir: str | Path = "results",
    ) -> None:
        if method not in PRUNING_METHODS:
            raise ValueError(f"Unknown pruning method {method!r}")
        self.model_wrapper = model_wrapper
        self.model_name = model_name
        self.method = method
        self.config = dict(config or {})
        self.results_dir = Path(results_dir)
        if (
            isinstance(model_wrapper.num_layers, bool)
            or not isinstance(model_wrapper.num_layers, int)
            or model_wrapper.num_layers <= 0
        ):
            raise ValueError("model_wrapper.num_layers must be a positive integer")
        self.num_layers = model_wrapper.num_layers
        self.provenance = _provenance(model_wrapper, model_name)
        self.execution_identity = _execution_identity(self.provenance)

    def run_global(self) -> Dict[str, Any]:
        if self.method == "shortgpt":
            return self._run_shortgpt()
        if self.method == "sleb":
            return self._run_sleb()
        raise ValueError("TALE is task-specific; call run_task(task_name, task)")

    def run_task(self, task_name: str, task: Any) -> Dict[str, Any]:
        if self.method != "tale":
            raise ValueError(f"{self.method} is model-global; call run_global()")
        return self._run_tale(task_name, task)

    def _trace_io(
        self,
        *,
        search_config: Mapping[str, Any],
        task_name: Optional[str] = None,
    ) -> tuple[Path, Optional[Dict[str, Any]]]:
        path = search_trace_path(
            self.results_dir,
            model_name=self.model_name,
            method=self.method,
            search_config=search_config,
            task_name=task_name,
        )
        envelope = load_search_envelope(
            path,
            model_name=self.model_name,
            method=self.method,
            search_config=search_config,
            task_name=task_name,
        )
        return path, envelope

    def _save(
        self,
        *,
        path: Path,
        search_config: Mapping[str, Any],
        algorithm_trace: Mapping[str, Any],
        task_name: Optional[str] = None,
    ) -> None:
        envelope = make_search_envelope(
            model_name=self.model_name,
            method=self.method,
            search_config=search_config,
            algorithm_trace=algorithm_trace,
            provenance=self.provenance,
            task_name=task_name,
        )
        save_search_envelope(envelope, path)

    def _run_shortgpt(self) -> Dict[str, Any]:
        num_remove = _resolve_num_remove(
            num_layers=self.num_layers,
            num_remove=self.config.get("num_remove"),
            prune_ratio=self.config.get("prune_ratio", 0.25),
        )
        search_config = {
            "num_layers": self.num_layers,
            "num_remove": num_remove,
            "dataset_path": self.config.get("dataset_path", "emozilla/pg19"),
            "dataset_name": self.config.get("dataset_name"),
            "split": self.config.get("split", "validation"),
            "text_column": self.config.get("text_column", "text"),
            "max_samples": self.config.get("max_samples"),
            "sequence_length": int(self.config.get("sequence_length", 256)),
            "search_batch_size": int(self.config.get("search_batch_size", 1)),
            "seed": int(self.config.get("seed", 42)),
            "execution": self.execution_identity,
        }

        def build_batches() -> ShortGPTTokenBatches:
            return ShortGPTTokenBatches(
                tokenizer=self.model_wrapper.tokenizer,
                dataset_path=search_config["dataset_path"],
                dataset_name=search_config["dataset_name"],
                split=search_config["split"],
                text_column=search_config["text_column"],
                max_samples=search_config["max_samples"],
                sequence_length=search_config["sequence_length"],
                batch_size=search_config["search_batch_size"],
                seed=search_config["seed"],
            )

        path, envelope = self._trace_io(search_config=search_config)
        if envelope is not None:
            trace = envelope["algorithm_trace"]
            if trace.get("complete") is True:
                validate_shortgpt_trace(
                    trace,
                    num_layers=self.num_layers,
                    num_remove=num_remove,
                )
                if num_remove == 0:
                    selected = trace["selected_layers"]
                    logger.info("Reusing completed ShortGPT search from %s", path)
                    return _standard_result(
                        method=self.method,
                        num_layers=self.num_layers,
                        selected_layers=selected,
                        trace_path=path,
                        trace=trace,
                    )

                # A dataset or tokenizer may have changed in place without
                # changing its configured path.  Reconstruct the exact CPU
                # token stream before accepting a completed ranking.
                validation_batches = _FingerprintingTokenBatches(
                    build_batches(),
                    "ShortGPT",
                )
                actual_corpus_sha256 = validation_batches.fingerprint()
                corpus_stats = trace.get("corpus_stats")
                cached_corpus_sha256 = (
                    corpus_stats.get("sha256")
                    if isinstance(corpus_stats, Mapping)
                    else None
                )
                if cached_corpus_sha256 == actual_corpus_sha256:
                    selected = trace["selected_layers"]
                    logger.info("Reusing completed ShortGPT search from %s", path)
                    return _standard_result(
                        method=self.method,
                        num_layers=self.num_layers,
                        selected_layers=selected,
                        trace_path=path,
                        trace=trace,
                    )
                logger.info(
                    "Recomputing stale ShortGPT search at %s: calibration "
                    "token fingerprint changed (%r -> %s)",
                    path,
                    cached_corpus_sha256,
                    actual_corpus_sha256,
                )

        if num_remove == 0:
            trace = {
                "method": "shortgpt",
                "num_layers": self.num_layers,
                "num_remove": 0,
                "scores": [],
                "removal_order": [],
                "selected_layers": [],
                "corpus_stats": {},
                "search_skipped": "zero_budget",
                "complete": True,
            }
            self._save(path=path, search_config=search_config, algorithm_trace=trace)
            return _standard_result(
                method=self.method,
                num_layers=self.num_layers,
                selected_layers=[],
                trace_path=path,
                trace=trace,
            )

        batches = build_batches()
        fingerprinted_batches = _FingerprintingTokenBatches(batches, "ShortGPT")
        # ShortGPT must score the dense model.  Preserve any caller-owned
        # strategy so using the runner outside Evaluator has no lasting side
        # effects.  The CPU token fingerprint is updated during this same
        # streaming pass, so a first search neither repeats nor materializes
        # the PG19 corpus.
        with temporarily_bypass_layers(self.model_wrapper, ()):
            scores = score_shortgpt_blocks(self.model_wrapper, fingerprinted_batches)
        corpus_sha256 = fingerprinted_batches.sha256
        if corpus_sha256 is None:
            raise RuntimeError(
                "ShortGPT scoring must consume the complete calibration corpus"
            )
        removal_order = select_shortgpt_layers(scores, num_remove=self.num_layers)
        selected = sorted(removal_order[:num_remove])
        corpus_stats = dict(batches.stats)
        corpus_stats["sha256"] = corpus_sha256
        trace = {
            "method": "shortgpt",
            "num_layers": self.num_layers,
            "num_remove": num_remove,
            "scores": [float(score) for score in scores],
            "removal_order": removal_order,
            "selected_layers": selected,
            "corpus_stats": corpus_stats,
            "complete": True,
        }
        self._save(path=path, search_config=search_config, algorithm_trace=trace)
        return _standard_result(
            method=self.method,
            num_layers=self.num_layers,
            selected_layers=selected,
            trace_path=path,
            trace=trace,
        )

    def _run_sleb(self) -> Dict[str, Any]:
        num_remove = _resolve_num_remove(
            num_layers=self.num_layers,
            num_remove=self.config.get("num_remove"),
            prune_ratio=self.config.get("prune_ratio", 0.2),
        )
        early_barrier = _nonnegative_int(
            "SLEB early_barrier", self.config.get("early_barrier", 0)
        )
        latter_barrier = _nonnegative_int(
            "SLEB latter_barrier", self.config.get("latter_barrier", 0)
        )
        searchable_layers = self.num_layers - early_barrier - latter_barrier
        if searchable_layers < 0:
            raise ValueError(
                "SLEB early_barrier + latter_barrier cannot exceed model depth"
            )
        if num_remove > searchable_layers:
            raise ValueError(
                f"SLEB num_remove={num_remove} exceeds {searchable_layers} "
                "layers left by the barriers"
            )
        search_config = {
            "num_layers": self.num_layers,
            "num_remove": num_remove,
            "early_barrier": early_barrier,
            "latter_barrier": latter_barrier,
            "dataset_path": self.config.get("dataset_path", "wikitext"),
            "dataset_name": self.config.get("dataset_name", "wikitext-2-raw-v1"),
            "split": self.config.get("split", "train"),
            "text_column": self.config.get("text_column", "text"),
            "max_samples": int(self.config.get("max_samples", 128)),
            "sequence_length": int(self.config.get("sequence_length", 2048)),
            "search_batch_size": int(self.config.get("search_batch_size", 1)),
            "seed": int(self.config.get("seed", 0)),
            "execution": self.execution_identity,
        }
        if num_remove == 0:
            batches: Sequence[Mapping[str, torch.Tensor]] = []
            corpus_stats: Dict[str, Any] = {"search_skipped": "zero_budget"}
            search_config["corpus_sha256"] = None
        else:
            batches, corpus_stats = build_sleb_token_batches(
                self.model_wrapper.tokenizer,
                dataset_path=search_config["dataset_path"],
                dataset_name=search_config["dataset_name"],
                split=search_config["split"],
                text_column=search_config["text_column"],
                max_samples=search_config["max_samples"],
                sequence_length=search_config["sequence_length"],
                batch_size=search_config["search_batch_size"],
                seed=search_config["seed"],
            )
            corpus_sha256 = _token_batches_fingerprint(batches)
            corpus_stats = {**corpus_stats, "sha256": corpus_sha256}
            search_config["corpus_sha256"] = corpus_sha256

        # The exact token IDs are part of the path/config.  A local dataset or
        # tokenizer change therefore starts a clean search instead of mixing
        # old candidate NLLs with a newly constructed corpus.
        path, envelope = self._trace_io(search_config=search_config)
        restored_trace = envelope["algorithm_trace"] if envelope is not None else None

        def score_fn(removed: tuple[int, ...]) -> float:
            return causal_token_nll(self.model_wrapper, batches, removed)

        def checkpoint_fn(snapshot: Dict[str, Any]) -> None:
            snapshot["corpus_stats"] = corpus_stats
            self._save(
                path=path,
                search_config=search_config,
                algorithm_trace=snapshot,
            )

        trace = run_sleb_search(
            num_layers=self.num_layers,
            num_remove=num_remove,
            score_fn=score_fn,
            early_barrier=early_barrier,
            latter_barrier=latter_barrier,
            trace=restored_trace,
            checkpoint_fn=checkpoint_fn,
        )
        if restored_trace is not None and restored_trace.get("complete") is True:
            logger.info("Reusing completed SLEB search from %s", path)
        trace["corpus_stats"] = corpus_stats
        self._save(path=path, search_config=search_config, algorithm_trace=trace)
        return _standard_result(
            method=self.method,
            num_layers=self.num_layers,
            selected_layers=trace["selected_layers"],
            trace_path=path,
            trace=trace,
        )

    def _run_tale(self, task_name: str, task: Any) -> Dict[str, Any]:
        if task_name == "humaneval" or task.calibration_split_name == "unavailable":
            raise ValueError(
                "TALE requires a labeled search split disjoint from test data; "
                "HumanEval provides no legal search split"
            )
        # The authors' single-GPU implementation defaults to the full task
        # split.  Cost-limited runs must opt into a cap explicitly so an
        # omitted setting never silently changes the search protocol.
        search_max_samples = self.config.get("search_max_samples")
        if search_max_samples is not None:
            search_max_samples = int(search_max_samples)
            if search_max_samples <= 0:
                raise ValueError("TALE search_max_samples must be positive or None")
        seed = int(self.config.get("seed", getattr(task, "seed", 42)))
        docs = task.calibration_docs(max_samples=search_max_samples, seed=seed)
        if not docs:
            raise ValueError(f"TALE search split for {task_name!r} is empty")
        metric = str(self.config.get("metric") or _primary_metric(task))
        if metric not in task.aggregation():
            raise ValueError(
                f"TALE metric {metric!r} is not provided by task {task_name!r}"
            )
        metric_directions = task.higher_is_better()
        if metric_directions.get(metric) is not True:
            raise ValueError(
                f"TALE metric {metric!r} for task {task_name!r} must be "
                "higher-is-better"
            )

        configured_max_remove = self.config.get("max_remove")
        target_remove = self.config.get("target_remove")
        if target_remove is not None:
            target_remove = _nonnegative_int("TALE target_remove", target_remove)
            if target_remove >= self.num_layers:
                raise ValueError(
                    "TALE target_remove must retain at least one layer and be non-negative"
                )
        if configured_max_remove is None:
            # An exact target is also the natural hard search cap.  This keeps
            # a one-layer budget from accidentally launching an N-layer TALE
            # search merely because no separate max was specified.
            max_remove = (
                None if target_remove is not None else self.num_layers - 1
            )
        else:
            max_remove = _nonnegative_int(
                "TALE max_remove", configured_max_remove
            )
            if max_remove >= self.num_layers:
                raise ValueError(
                    "TALE max_remove must retain at least one layer and be non-negative"
                )
        if (
            target_remove is not None
            and max_remove is not None
            and target_remove > max_remove
        ):
            raise ValueError("TALE target_remove cannot exceed max_remove")
        variant = str(self.config.get("variant", "threshold_final"))
        if variant not in TALE_VARIANTS:
            raise ValueError(f"Unknown TALE variant {variant!r}; choose {TALE_VARIANTS}")
        if variant == "budget" and target_remove is None:
            raise ValueError("TALE variant 'budget' requires target_remove")
        threshold = float(self.config.get("threshold", 0.08))
        stop_at_threshold = bool(self.config.get("stop_at_threshold", True))
        docs_hash = _docs_fingerprint(docs)
        inputs_hash = _task_inputs_fingerprint(task, docs)
        search_config = {
            "num_layers": self.num_layers,
            "task": task_name,
            "task_version": getattr(type(task), "VERSION", 0),
            "dataset_path": getattr(task, "DATASET_PATH", None),
            "search_split": task.calibration_split_name,
            "search_max_samples": search_max_samples,
            "search_num_samples": len(docs),
            "search_docs_sha256": docs_hash,
            "search_inputs_sha256": inputs_hash,
            "task_num_fewshot": getattr(task, "num_fewshot", None),
            "task_seed": getattr(task, "seed", None),
            "seed": seed,
            "metric": metric,
            "threshold": threshold,
            "max_remove": max_remove,
            "target_remove": target_remove,
            "stop_at_threshold": stop_at_threshold,
            "execution": self.execution_identity,
        }
        path, envelope = self._trace_io(
            search_config=search_config,
            task_name=task_name,
        )
        restored_trace = envelope["algorithm_trace"] if envelope is not None else None

        def evaluate_fn(removed: tuple[int, ...]) -> float:
            with temporarily_bypass_layers(self.model_wrapper, removed):
                metrics = task.evaluate_docs(self.model_wrapper, docs)
            if metric not in metrics:
                raise ValueError(
                    f"Task {task_name!r} did not return TALE metric {metric!r}"
                )
            return float(metrics[metric])

        def checkpoint_fn(snapshot: Dict[str, Any]) -> None:
            snapshot["task_search"] = {
                "task": task_name,
                "split": task.calibration_split_name,
                "metric": metric,
                "num_samples": len(docs),
                "docs_sha256": docs_hash,
                "inputs_sha256": inputs_hash,
            }
            self._save(
                path=path,
                search_config=search_config,
                algorithm_trace=snapshot,
                task_name=task_name,
            )

        was_complete = bool(
            restored_trace is not None and restored_trace.get("completed") is True
        )
        trace = run_tale_search(
            num_layers=self.num_layers,
            evaluate_fn=evaluate_fn,
            threshold=threshold,
            max_remove=max_remove,
            target_remove=target_remove,
            stop_at_threshold=stop_at_threshold,
            trace=restored_trace,
            checkpoint_fn=None if was_complete else checkpoint_fn,
        )
        if was_complete:
            logger.info("Reusing completed TALE search for %s from %s", task_name, path)
        else:
            checkpoint_fn(trace)

        chosen = trace.get(variant)
        if chosen is None:
            status = trace.get("budget_status") if variant == "budget" else "unavailable"
            raise ValueError(f"TALE variant {variant!r} is unavailable ({status})")
        return _standard_result(
            method=self.method,
            num_layers=self.num_layers,
            selected_layers=chosen["removed_layers"],
            trace_path=path,
            trace=trace,
            variant=variant,
        )


__all__ = [
    "PRUNING_METHODS",
    "TALE_VARIANTS",
    "PruningSearchRunner",
    "causal_token_nll",
    "temporarily_bypass_layers",
]
