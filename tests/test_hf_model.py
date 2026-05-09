"""Tests for the HuggingFace model wrapper."""

import sys
from types import SimpleNamespace

from evaluation.models.hf_model import HFModel
from evaluation.strategies.manualskip import ManualSkipStrategy


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

    def decode(
        self,
        token_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    ):
        return getattr(self, "decoded_text", "")


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


class _MockTokenizerWithTerminators(_MockTokenizer):
    unk_token_id = None

    def convert_tokens_to_ids(self, token):
        return {
            "<|eot_id|>": 128009,
            "<|end_of_text|>": 128001,
        }.get(token)


class _MockModel:
    def __init__(self):
        self.config = SimpleNamespace(num_hidden_layers=32)
        self.generation_config = SimpleNamespace(max_length=4096)

    def eval(self):
        return None


class _MockGenerativeModel(_MockModel):

    def generate(self, **kwargs):
        import torch

        input_ids = kwargs["input_ids"]
        generated = torch.tensor([[301]], dtype=input_ids.dtype, device=input_ids.device)
        return torch.cat([input_ids, generated], dim=1)


class _AddOneLayer:
    def __init__(self):
        self.calls = 0

    def forward(self, hidden_states, **kwargs):
        self.calls += 1
        return (hidden_states + 1,)

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class _MockLayerBypassModel(_MockModel):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(num_hidden_layers=3)
        self.layers = [_AddOneLayer(), _AddOneLayer(), _AddOneLayer()]
        self.model = SimpleNamespace(layers=self.layers, norm=None)
        self.lm_head = lambda hidden: hidden

    def __call__(
        self,
        input_ids,
        attention_mask=None,
        output_hidden_states=False,
        use_cache=False,
        **kwargs,
    ):
        hidden = input_ids.float().unsqueeze(-1)
        states = [hidden.clone()] if output_hidden_states else None
        for layer in self.layers:
            hidden = layer(
                hidden,
                attention_mask=attention_mask,
                output_attentions=False,
                use_cache=use_cache,
            )[0]
            if output_hidden_states:
                states.append(hidden.clone())
        return SimpleNamespace(hidden_states=tuple(states), logits=hidden)


class _ToyCalibrationLayer:
    def __init__(self, scale):
        import torch

        self.weight = torch.nn.Parameter(torch.tensor([scale], dtype=torch.float32))

    def parameters(self):
        return [self.weight]

    def forward(self, hidden_states, **kwargs):
        return (hidden_states * self.weight,)

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class _MockCalibrationModel(_MockModel):
    def __init__(self):
        import torch

        super().__init__()
        self.config = SimpleNamespace(num_hidden_layers=2)
        self.layers = [_ToyCalibrationLayer(1.5), _ToyCalibrationLayer(0.5)]
        self.model = SimpleNamespace(layers=self.layers, norm=torch.nn.Identity())
        self.lm_head = torch.nn.Linear(1, 256, bias=False)

    def zero_grad(self, set_to_none=False):
        for parameter in self.parameters():
            parameter.grad = None

    def parameters(self):
        params = []
        for layer in self.layers:
            params.extend(layer.parameters())
        params.extend(self.lm_head.parameters())
        return params

    def __call__(
        self,
        input_ids,
        attention_mask=None,
        output_hidden_states=False,
        use_cache=False,
        **kwargs,
    ):
        hidden = input_ids.float().unsqueeze(-1) / 100.0
        states = [hidden.clone()] if output_hidden_states else None
        for layer in self.layers:
            hidden = layer(hidden)[0]
            if output_hidden_states:
                states.append(hidden.clone())
        return SimpleNamespace(logits=self.lm_head(hidden), hidden_states=tuple(states) if states else None)


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


def test_eos_token_ids_preserve_model_generation_config(monkeypatch):
    mock_tokenizer = _MockTokenizerWithTerminators()
    mock_tokenizer.eos_token_id = 128009
    mock_model = _MockModel()
    mock_model.config.eos_token_id = 128009
    mock_model.generation_config.eos_token_id = [128001, 128009]

    fake_transformers = SimpleNamespace(
        AutoTokenizer=SimpleNamespace(
            from_pretrained=lambda *args, **kwargs: mock_tokenizer,
        ),
        AutoModelForCausalLM=SimpleNamespace(
            from_pretrained=lambda *args, **kwargs: mock_model,
        ),
    )
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    model = HFModel("mock-model", device="cpu")

    assert model._eos_token_id_list() == [128001, 128009]
    assert model._generation_eos_token_id() == [128001, 128009]


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


def test_generate_single_preserves_leading_whitespace(monkeypatch):
    mock_tokenizer = _MockTokenizer()
    mock_tokenizer.decoded_text = "    return 1\n\n"

    fake_transformers = SimpleNamespace(
        AutoTokenizer=SimpleNamespace(
            from_pretrained=lambda *args, **kwargs: mock_tokenizer,
        ),
        AutoModelForCausalLM=SimpleNamespace(
            from_pretrained=lambda *args, **kwargs: _MockGenerativeModel(),
        ),
        GenerationConfig=lambda **kwargs: SimpleNamespace(**kwargs),
    )
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    model = HFModel("mock-model", device="cpu")

    assert model._generate_single("prompt", {"max_new_tokens": 8}) == "    return 1"


def test_manualskip_bypasses_configured_transformer_layers(monkeypatch):
    import torch

    mock_model = _MockLayerBypassModel()
    fake_transformers = SimpleNamespace(
        AutoTokenizer=SimpleNamespace(
            from_pretrained=lambda *args, **kwargs: _MockTokenizer(),
        ),
        AutoModelForCausalLM=SimpleNamespace(
            from_pretrained=lambda *args, **kwargs: mock_model,
        ),
    )
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    model = HFModel(
        "mock-model",
        strategy=ManualSkipStrategy(skip_layers=[2]),
        device="cpu",
    )
    input_ids = torch.tensor([[1, 2]], dtype=torch.long)

    outputs = model._forward_model(
        input_ids=input_ids,
        attention_mask=torch.ones_like(input_ids),
        output_hidden_states=True,
        use_cache=False,
    )

    assert [layer.calls for layer in mock_model.layers] == [1, 0, 1]
    assert outputs.hidden_states[-1].squeeze(-1).tolist() == [[3.0, 4.0]]

    restored = mock_model.layers[1](torch.zeros(1, 1, 1))[0]
    assert restored.item() == 1.0


def test_compute_layer_calibration_metrics(monkeypatch):
    mock_model = _MockCalibrationModel()
    fake_transformers = SimpleNamespace(
        AutoTokenizer=SimpleNamespace(
            from_pretrained=lambda *args, **kwargs: _MockTokenizer(),
        ),
        AutoModelForCausalLM=SimpleNamespace(
            from_pretrained=lambda *args, **kwargs: mock_model,
        ),
    )
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    model = HFModel("mock-model", device="cpu")
    metrics = model.compute_layer_calibration_metrics(
        requests=[("prompt", " answer"), ("other", " label")],
        metrics=["activation_ratio", "gradient_trace"],
        batch_size=1,
    )

    assert metrics["num_layers"] == 2
    assert metrics["num_samples"] == 2
    assert len(metrics["layers"]) == 2
    for layer in metrics["layers"]:
        assert 0.0 <= layer["activation_ratio"] <= 1.0
        assert layer["gradient_trace"] >= 0.0
