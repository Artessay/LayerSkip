"""Tests for paper-aligned pruning calibration corpora."""

from __future__ import annotations

import torch

from evaluation.pruning import corpora


class _LengthTokenizer:
    pad_token_id = 99
    eos_token_id = 98

    def __init__(self) -> None:
        self.texts = []

    def __call__(self, text, **kwargs):
        del kwargs
        self.texts.append(text)
        return {"input_ids": list(range(1, len(text) + 1))}


class _ShuffleDataset:
    def __init__(self, rows, calls=None):
        self.rows = list(rows)
        self.calls = calls if calls is not None else []

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.rows[index]

    def shuffle(self, seed):
        self.calls.append(seed)
        return _ShuffleDataset(reversed(self.rows), self.calls)


def test_shortgpt_chunks_documents_and_masks_partial_tail(monkeypatch):
    monkeypatch.setattr(
        corpora,
        "_load_text_split",
        lambda *args: [{"text": "abcde"}],
    )
    tokenizer = _LengthTokenizer()
    batches = corpora.ShortGPTTokenBatches(
        tokenizer=tokenizer,
        sequence_length=3,
        batch_size=2,
    )

    materialized = list(batches)

    assert len(materialized) == 1
    assert torch.equal(
        materialized[0]["input_ids"],
        torch.tensor([[1, 2, 3], [4, 5, 99]]),
    )
    assert torch.equal(
        materialized[0]["attention_mask"],
        torch.tensor([[1, 1, 1], [1, 1, 0]]),
    )
    assert batches.stats == {
        "available_documents": 1,
        "selected_documents": 1,
        "nonempty_documents": 1,
        "tokens": 5,
        "segments": 2,
        "batches": 1,
    }


def test_sleb_uses_huggingface_shuffle_before_document_prefix(monkeypatch):
    dataset = _ShuffleDataset(
        [{"text": "A"}, {"text": "B"}, {"text": "C"}]
    )
    monkeypatch.setattr(corpora, "_load_text_split", lambda *args: dataset)
    tokenizer = _LengthTokenizer()

    batches, stats = corpora.build_sleb_token_batches(
        tokenizer,
        max_samples=2,
        sequence_length=2,
        batch_size=2,
        seed=7,
    )

    assert dataset.calls == [7]
    assert tokenizer.texts == ["C\n\nB"]
    assert len(batches) == 1
    assert torch.equal(
        batches[0]["input_ids"],
        torch.tensor([[1, 2], [3, 4]]),
    )
    assert stats == {
        "available_documents": 3,
        "selected_documents": 2,
        "tokenized_tokens": 4,
        "used_tokens": 4,
        "discarded_tail_tokens": 0,
        "segments": 2,
        "batches": 1,
    }


def test_sleb_rejects_corpus_shorter_than_one_sequence(monkeypatch):
    monkeypatch.setattr(
        corpora,
        "_load_text_split",
        lambda *args: [{"text": "x"}],
    )

    try:
        corpora.build_sleb_token_batches(
            _LengthTokenizer(),
            max_samples=1,
            sequence_length=2,
        )
    except ValueError as exc:
        assert "shorter than one sequence" in str(exc)
    else:
        raise AssertionError("expected a short-corpus error")
