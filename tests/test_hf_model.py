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
