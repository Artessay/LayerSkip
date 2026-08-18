"""Calibration-corpus loaders used by pruning baselines.

The loaders intentionally keep benchmark search data separate from task test
data.  ShortGPT follows its reference implementation and reads PG19
validation books in non-overlapping chunks; SLEB follows its reference
implementation and builds a reusable token stream from shuffled WikiText-2
training documents.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import random
from typing import Any, Dict, Iterator, List, Optional, Sequence

import torch


def _load_text_split(dataset_path: str, dataset_name: Optional[str], split: str):
    from datasets import load_dataset

    if dataset_name:
        return load_dataset(dataset_path, dataset_name, split=split)
    return load_dataset(dataset_path, split=split)


def _token_ids(tokenizer: Any, text: str) -> List[int]:
    encoded = tokenizer(text, add_special_tokens=True, return_attention_mask=False)
    if hasattr(encoded, "input_ids"):
        ids = encoded.input_ids
    else:
        ids = encoded["input_ids"]
    if isinstance(ids, torch.Tensor):
        ids = ids.detach().cpu().tolist()
    if ids and isinstance(ids[0], list):
        if len(ids) != 1:
            raise ValueError("Expected one tokenized text at a time")
        ids = ids[0]
    return [int(token_id) for token_id in ids]


def _pad_id(tokenizer: Any) -> int:
    value = getattr(tokenizer, "pad_token_id", None)
    if value is None:
        value = getattr(tokenizer, "eos_token_id", None)
    return int(value) if value is not None else 0


def _collate_segments(
    segments: Sequence[Sequence[int]],
    *,
    pad_token_id: int,
) -> Dict[str, torch.Tensor]:
    if not segments:
        raise ValueError("Cannot collate an empty segment list")
    max_length = max(len(segment) for segment in segments)
    input_rows: List[List[int]] = []
    mask_rows: List[List[int]] = []
    for segment in segments:
        pad_length = max_length - len(segment)
        input_rows.append([*segment, *([pad_token_id] * pad_length)])
        mask_rows.append([*([1] * len(segment)), *([0] * pad_length)])
    return {
        "input_ids": torch.tensor(input_rows, dtype=torch.long),
        "attention_mask": torch.tensor(mask_rows, dtype=torch.long),
    }


@dataclass
class ShortGPTTokenBatches:
    """Re-iterable description of the one-pass PG19 ShortGPT corpus."""

    tokenizer: Any
    dataset_path: str = "emozilla/pg19"
    dataset_name: Optional[str] = None
    split: str = "validation"
    text_column: str = "text"
    max_samples: Optional[int] = None
    sequence_length: int = 256
    batch_size: int = 1
    seed: int = 42
    stats: Dict[str, int] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        if self.max_samples is not None and self.max_samples <= 0:
            raise ValueError("max_samples must be positive or None")
        if self.sequence_length <= 0:
            raise ValueError("sequence_length must be positive")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        dataset = _load_text_split(
            self.dataset_path,
            self.dataset_name,
            self.split,
        )
        indices = list(range(len(dataset)))
        random.Random(self.seed).shuffle(indices)
        if self.max_samples is not None:
            indices = indices[: self.max_samples]

        self.stats = {
            "available_documents": len(dataset),
            "selected_documents": len(indices),
            "nonempty_documents": 0,
            "tokens": 0,
            "segments": 0,
            "batches": 0,
        }
        pending: List[List[int]] = []
        pad_token_id = _pad_id(self.tokenizer)

        for index in indices:
            row = dataset[index]
            text = row.get(self.text_column) if hasattr(row, "get") else None
            if not isinstance(text, str):
                raise ValueError(
                    f"Dataset row {index} has no string column {self.text_column!r}"
                )
            ids = _token_ids(self.tokenizer, text)
            if not ids:
                continue
            self.stats["nonempty_documents"] += 1
            self.stats["tokens"] += len(ids)
            for start in range(0, len(ids), self.sequence_length):
                segment = ids[start : start + self.sequence_length]
                if not segment:
                    continue
                pending.append(segment)
                self.stats["segments"] += 1
                if len(pending) == self.batch_size:
                    self.stats["batches"] += 1
                    yield _collate_segments(pending, pad_token_id=pad_token_id)
                    pending = []

        if pending:
            self.stats["batches"] += 1
            yield _collate_segments(pending, pad_token_id=pad_token_id)


def build_sleb_token_batches(
    tokenizer: Any,
    *,
    dataset_path: str = "wikitext",
    dataset_name: Optional[str] = "wikitext-2-raw-v1",
    split: str = "train",
    text_column: str = "text",
    max_samples: int = 128,
    sequence_length: int = 2048,
    batch_size: int = 1,
    seed: int = 0,
) -> tuple[List[Dict[str, torch.Tensor]], Dict[str, int]]:
    """Build the fixed WikiText token batches scored by every SLEB candidate.

    This mirrors the official code's shuffled-document concatenation while
    using mean token NLL instead of its constant-scaled summed loss.  Those two
    objectives induce the same candidate ordering for a fixed corpus.
    """

    if max_samples <= 0:
        raise ValueError("max_samples must be positive")
    if sequence_length <= 1:
        raise ValueError("sequence_length must be at least 2")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    dataset = _load_text_split(dataset_path, dataset_name, split)
    available_documents = len(dataset)

    # The official SLEB loader uses Hugging Face Dataset.shuffle(seed=seed)
    # before taking the first ``nsamples`` documents.  Reusing that method is
    # important: it is backed by NumPy's generator and does not produce the
    # same permutation as ``random.Random(seed).shuffle``.  The small fallback
    # keeps lightweight sequence-like test/local adapters usable.
    shuffle = getattr(dataset, "shuffle", None)
    if callable(shuffle):
        dataset = shuffle(seed=seed)
        indices = list(range(min(max_samples, len(dataset))))
    else:
        indices = list(range(len(dataset)))
        random.Random(seed).shuffle(indices)
        indices = indices[: min(max_samples, len(indices))]
    texts: List[str] = []
    for index in indices:
        row = dataset[index]
        text = row.get(text_column) if hasattr(row, "get") else None
        if not isinstance(text, str):
            raise ValueError(
                f"Dataset row {index} has no string column {text_column!r}"
            )
        texts.append(text)

    ids = _token_ids(tokenizer, "\n\n".join(texts))
    full_segments = len(ids) // sequence_length
    if full_segments == 0:
        raise ValueError(
            "SLEB search corpus is shorter than one sequence; lower "
            "sequence_length or increase max_samples"
        )

    segments = [
        ids[start : start + sequence_length]
        for start in range(0, full_segments * sequence_length, sequence_length)
    ]
    pad_token_id = _pad_id(tokenizer)
    batches = [
        _collate_segments(
            segments[start : start + batch_size],
            pad_token_id=pad_token_id,
        )
        for start in range(0, len(segments), batch_size)
    ]
    return batches, {
        "available_documents": available_documents,
        "selected_documents": len(indices),
        "tokenized_tokens": len(ids),
        "used_tokens": full_segments * sequence_length,
        "discarded_tail_tokens": len(ids) - full_segments * sequence_length,
        "segments": full_segments,
        "batches": len(batches),
    }


__all__ = ["ShortGPTTokenBatches", "build_sleb_token_batches"]
