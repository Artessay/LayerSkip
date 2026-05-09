"""Tests for the HuggingFace model wrapper."""

import sys
from types import SimpleNamespace

from evaluation.models.hf_model import HFModel


class _MockTokenizer:
    pad_token = None
    eos_token = "<eos>"
    pad_token_id = 0
    eos_token_id = 1
    padding_side = "right"
    chat_template = None

    def __init__(self):
        self.encode_calls = []

    def encode(self, text, add_special_tokens=True):
        self.encode_calls.append((text, add_special_tokens))
        return [101 if add_special_tokens else 201, len(text)]


class _MockChatTokenizer(_MockTokenizer):
    chat_template = "llama-3-template"

    def __init__(self):
        super().__init__()
        self.chat_template_calls = []

    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=True):
        self.chat_template_calls.append(
            {
                "messages": messages,
                "tokenize": tokenize,
                "add_generation_prompt": add_generation_prompt,
            }
        )
        return [128000, 128006, 42, 128007]


class _MockChatBatchEncodingTokenizer(_MockChatTokenizer):

    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=True):
        self.chat_template_calls.append(
            {
                "messages": messages,
                "tokenize": tokenize,
                "add_generation_prompt": add_generation_prompt,
            }
        )
        return {
            "input_ids": [128000, 128006, 42, 128007],
            "attention_mask": [1, 1, 1, 1],
        }


class _MockModel:
    def __init__(self):
        self.config = SimpleNamespace(num_hidden_layers=32)
        self.generation_config = SimpleNamespace(max_length=4096)

    def eval(self):
        return None


def test_init_clears_model_generation_max_length(monkeypatch):
    mock_model = _MockModel()

    fake_transformers = SimpleNamespace(
        AutoTokenizer=SimpleNamespace(
            from_pretrained=lambda *args, **kwargs: _MockTokenizer(),
        ),
        AutoModelForCausalLM=SimpleNamespace(
            from_pretrained=lambda *args, **kwargs: mock_model,
        ),
    )
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    HFModel("mock-model", device="cpu")

    assert mock_model.generation_config.max_length is None


def test_tokenizer_chat_template_used_for_prompt_encoding(monkeypatch):
    mock_tokenizer = _MockChatTokenizer()

    fake_transformers = SimpleNamespace(
        AutoTokenizer=SimpleNamespace(
            from_pretrained=lambda *args, **kwargs: mock_tokenizer,
        ),
        AutoModelForCausalLM=SimpleNamespace(
            from_pretrained=lambda *args, **kwargs: _MockModel(),
        ),
    )
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    model = HFModel("mock-model", device="cpu")

    assert model._encode_prompt_ids("Question: 1 + 1?\nAnswer:") == [
        128000,
        128006,
        42,
        128007,
    ]
    assert mock_tokenizer.chat_template_calls == [
        {
            "messages": [
                {"role": "user", "content": "Question: 1 + 1?\nAnswer:"}
            ],
            "tokenize": True,
            "add_generation_prompt": True,
        }
    ]
    assert mock_tokenizer.encode_calls == []


def test_tokenizer_chat_template_batch_encoding_is_flattened(monkeypatch):
    mock_tokenizer = _MockChatBatchEncodingTokenizer()

    fake_transformers = SimpleNamespace(
        AutoTokenizer=SimpleNamespace(
            from_pretrained=lambda *args, **kwargs: mock_tokenizer,
        ),
        AutoModelForCausalLM=SimpleNamespace(
            from_pretrained=lambda *args, **kwargs: _MockModel(),
        ),
    )
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    model = HFModel("mock-model", device="cpu")

    assert model._encode_prompt_ids("Question: 1 + 1?\nAnswer:") == [
        128000,
        128006,
        42,
        128007,
    ]
    assert model._prepare_prompt_inputs("Question: 1 + 1?\nAnswer:")[
        "input_ids"
    ].tolist() == [[128000, 128006, 42, 128007]]
    assert mock_tokenizer.encode_calls == []


def test_tokenizer_without_chat_template_keeps_plain_prompt_encoding(monkeypatch):
    mock_tokenizer = _MockTokenizer()

    fake_transformers = SimpleNamespace(
        AutoTokenizer=SimpleNamespace(
            from_pretrained=lambda *args, **kwargs: mock_tokenizer,
        ),
        AutoModelForCausalLM=SimpleNamespace(
            from_pretrained=lambda *args, **kwargs: _MockModel(),
        ),
    )
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    model = HFModel("meta-llama/Meta-Llama-3-8B-Instruct", device="cpu")

    assert model._encode_prompt_ids("plain prompt") == [101, 12]
    assert mock_tokenizer.encode_calls == [("plain prompt", True)]
