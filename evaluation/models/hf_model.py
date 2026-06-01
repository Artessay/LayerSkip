"""
HuggingFace-based language model wrapper with layer-skipping strategy support.

Supports any decoder-only causal LM available on the HuggingFace Hub, with
first-class support for:
    - meta-llama/Meta-Llama-3-8B-Instruct
    - meta-llama/Llama-3.2-1B-Instruct

Layer skipping strategies are applied by running the forward pass with
``output_hidden_states=True`` and selecting the exit layer according to the
chosen strategy. This allows zero-shot comparison of strategy quality without
modifying model weights or retraining.
"""

from __future__ import annotations

import inspect
import logging
from contextlib import contextmanager
from collections.abc import Mapping
from typing import Dict, Iterator, List, Optional, Sequence, Tuple, get_origin

import torch
import torch.nn.functional as F

from evaluation.models.base_model import BaseLM
from evaluation.strategies.base_strategy import BaseLayerSkipStrategy
from evaluation.utils.progress import progress

logger = logging.getLogger(__name__)

# Supported backbone model identifiers
SUPPORTED_MODELS = [
    "meta-llama/Meta-Llama-3-8B-Instruct",
    "meta-llama/Llama-3.2-1B-Instruct",
]


@torch.no_grad()
def calculate_shapley_value(param: torch.nn.Parameter) -> torch.Tensor:
    """Calculate row-level Shapley values for a 2D parameter tensor."""
    assert param.grad is not None
    assert param.ndim == 2, (
        "Shapley value calculation now is only supported for 2D parameters."
    )

    weight = param.detach().float()
    gradient = param.grad.detach().float()
    hessian_matrix = torch.matmul(gradient, gradient.T)
    individual_importance = -torch.sum(gradient * weight, dim=1)
    cooperative_interactions = -0.5 * torch.sum(
        weight * torch.matmul(hessian_matrix, weight),
        dim=1,
    )
    return individual_importance + cooperative_interactions


class HFModel(BaseLM):
    """
    HuggingFace causal LM wrapper with optional layer-skipping strategy.

    Args:
        model_name: HuggingFace model identifier or local path.
        strategy: A :class:`~evaluation.strategies.base_strategy.BaseLayerSkipStrategy`
            instance, or ``None`` to use the full model.
        device: Target device string (e.g. ``"cuda:0"`` or ``"cpu"``).
        batch_size: Batch size used during loglikelihood evaluation.
        dtype: Model dtype (``"auto"``, ``"float16"``, ``"bfloat16"``, etc.).
        trust_remote_code: Passed to ``AutoModel.from_pretrained``.
    """

    def __init__(
        self,
        model_name: str,
        strategy: Optional[BaseLayerSkipStrategy] = None,
        device: str = "auto",
        batch_size: int = 1,
        dtype: str = "auto",
        trust_remote_code: bool = False,
    ) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model_name = model_name
        self.strategy = strategy
        self._device = device
        self._batch_size = batch_size

        logger.info("Loading tokenizer for %s …", model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=trust_remote_code,
            use_fast=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        self._use_chat_template = self._should_use_chat_template()

        logger.info("Loading model %s to %s …", model_name, device)
        torch_dtype = (
            {
                "auto": "auto",
                "float16": torch.float16,
                "bfloat16": torch.bfloat16,
                "float32": torch.float32,
            }.get(dtype, "auto")
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            device_map=device,
            trust_remote_code=trust_remote_code,
        )
        self.model.eval()
        if getattr(self.model, "generation_config", None) is not None:
            self.model.generation_config.max_length = None

        self._num_layers: int = self.model.config.num_hidden_layers
        self._bypass_layer_indices = self._get_strategy_bypass_layer_indices()
        self._use_strategy = self._should_use_strategy()
        self._transformer_layers: Optional[Sequence[torch.nn.Module]] = None
        if self._bypass_layer_indices:
            self._transformer_layers = self._resolve_transformer_layers()
            logger.info(
                "Bypassing transformer layers: %s",
                [idx + 1 for idx in self._bypass_layer_indices],
            )
        logger.info(
            "Model loaded. Layers: %d | Strategy: %s",
            self._num_layers,
            strategy.name if strategy else "none",
        )
        if self._use_chat_template:
            logger.info("Using tokenizer chat template for prompts.")

    # ------------------------------------------------------------------ #
    # Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    def _get_strategy_bypass_layer_indices(self) -> Tuple[int, ...]:
        if self.strategy is None:
            return ()
        return tuple(self.strategy.get_skipped_layer_indices(self._num_layers))

    def _should_use_strategy(self) -> bool:
        if self.strategy is None:
            return False
        if self.strategy.is_noop(self._num_layers):
            return False
        return not self.strategy.uses_full_model_logits(self._num_layers)

    @property
    def num_layers(self) -> int:
        return self._num_layers

    def set_strategy(self, strategy: Optional[BaseLayerSkipStrategy]) -> None:
        """Update the active strategy and refresh layer-bypass bookkeeping."""
        self.strategy = strategy
        self._bypass_layer_indices = self._get_strategy_bypass_layer_indices()
        self._use_strategy = self._should_use_strategy()
        self._transformer_layers = None
        if self._bypass_layer_indices:
            self._transformer_layers = self._resolve_transformer_layers()
            logger.info(
                "Bypassing transformer layers: %s",
                [idx + 1 for idx in self._bypass_layer_indices],
            )

    def _resolve_transformer_layers(self) -> Sequence[torch.nn.Module]:
        """Return the model's transformer block list for manual layer bypass."""
        inner = getattr(self.model, "model", None)
        transformer = getattr(self.model, "transformer", None)
        gpt_neox = getattr(self.model, "gpt_neox", None)
        decoder = getattr(inner, "decoder", None) if inner is not None else None

        candidates = [
            getattr(inner, "layers", None),
            getattr(inner, "h", None),
            getattr(inner, "blocks", None),
            getattr(decoder, "layers", None),
            getattr(transformer, "h", None),
            getattr(transformer, "blocks", None),
            getattr(gpt_neox, "layers", None),
        ]

        for layers in candidates:
            if layers is None:
                continue
            try:
                if len(layers) == self._num_layers:
                    return layers
            except TypeError:
                continue

        raise ValueError(
            f"Strategy '{self.strategy.name}' requires direct access to the "
            "transformer layer list, but this model architecture was not recognized."
        )

    @staticmethod
    def _layer_forward_returns_tuple(layer: torch.nn.Module) -> bool:
        try:
            return_annotation = inspect.signature(layer.forward).return_annotation
        except (TypeError, ValueError):
            return True

        if return_annotation is inspect.Signature.empty:
            return True
        if return_annotation is torch.Tensor:
            return False
        if get_origin(return_annotation) is tuple:
            return True

        annotation_text = str(return_annotation).replace("typing.", "").lower()
        if "tuple" in annotation_text:
            return True
        if annotation_text in {"torch.tensor", "tensor", "<class 'torch.tensor'>"}:
            return False
        if annotation_text.endswith(".tensor"):
            return False
        return True

    @staticmethod
    def _make_bypass_forward(returns_tuple: bool):
        def bypass_forward(*args, **kwargs):
            hidden_states = args[0] if args else kwargs.get("hidden_states")
            if hidden_states is None:
                raise ValueError("Cannot bypass a layer without hidden_states input")

            if not returns_tuple:
                return hidden_states

            output_attentions = bool(kwargs.get("output_attentions", False))
            use_cache = bool(kwargs.get("use_cache", False))

            outputs = (hidden_states,)
            if output_attentions:
                outputs += (None,)
            if use_cache:
                outputs += (kwargs.get("past_key_value", None),)
            return outputs

        return bypass_forward

    @contextmanager
    def _bypass_transformer_layers(self) -> Iterator[None]:
        if not self._bypass_layer_indices:
            yield
            return

        if self._transformer_layers is None:
            raise RuntimeError("Transformer layers were not resolved for bypassing")

        originals = []
        for idx in self._bypass_layer_indices:
            layer = self._transformer_layers[idx]
            originals.append((layer, layer.forward))
            returns_tuple = self._layer_forward_returns_tuple(layer)
            layer.forward = self._make_bypass_forward(returns_tuple)

        try:
            yield
        finally:
            for layer, original_forward in originals:
                layer.forward = original_forward

    def _forward_model(self, **kwargs):
        with self._bypass_transformer_layers():
            return self.model(**kwargs)

    def _zero_model_grad(self) -> None:
        try:
            self.model.zero_grad(set_to_none=True)
        except TypeError:
            self.model.zero_grad()

    def _build_teacher_forced_batch(
        self,
        requests: List[Tuple[str, str]],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ctx_encodings = [self._encode_prompt_ids(context) for context, _ in requests]
        cont_encodings = [
            self.tokenizer.encode(continuation, add_special_tokens=False)
            for _, continuation in requests
        ]

        full_ids = []
        context_lengths = []
        for ctx_ids, cont_ids in zip(ctx_encodings, cont_encodings):
            full_ids.append(ctx_ids + cont_ids)
            context_lengths.append(len(ctx_ids))

        max_len = max(len(token_ids) for token_ids in full_ids)
        pad_id = self.tokenizer.pad_token_id

        input_ids_list = []
        attention_mask_list = []
        labels_list = []
        for token_ids, context_len in zip(full_ids, context_lengths):
            pad_len = max_len - len(token_ids)
            padded = [pad_id] * pad_len + token_ids
            attention_mask = [0] * pad_len + [1] * len(token_ids)
            labels = list(padded)
            label_start = pad_len + context_len
            for label_idx in range(label_start):
                labels[label_idx] = -100
            for label_idx, mask_value in enumerate(attention_mask):
                if mask_value == 0:
                    labels[label_idx] = -100

            input_ids_list.append(padded)
            attention_mask_list.append(attention_mask)
            labels_list.append(labels)

        input_ids = torch.tensor(input_ids_list, dtype=torch.long, device=self._device)
        attention_mask = torch.tensor(
            attention_mask_list,
            dtype=torch.long,
            device=self._device,
        )
        labels = torch.tensor(labels_list, dtype=torch.long, device=self._device)
        return input_ids, attention_mask, labels

    def compute_layer_calibration_metrics(
        self,
        requests: List[Tuple[str, str]],
        metrics: Sequence[str],
        batch_size: Optional[int] = None,
    ) -> Dict[str, object]:
        """
        Compute layer-level activation, gradient, gradient-trajectory, and Shapley metrics.

        ``activation_ratio`` is the fraction of positive hidden-state values in
        each layer output, averaged over non-padding tokens. ``gradient_value``
        is the layer-level sum of ``abs(loss_gradient)`` over that layer's
        parameters. ``gradient_trace`` is the layer-level sum of
        ``abs(weight * loss_gradient)`` over that layer's parameters.
        ``shapley_value`` sums the row-level Shapley values for each 2D
        parameter in the layer. Gradient-based metrics are averaged over
        calibration batches.
        """
        metric_set = set(metrics)
        supported_metrics = {
            "activation_ratio",
            "gradient_value",
            "gradient_trace",
            "shapley_value",
        }
        unsupported = metric_set - supported_metrics
        if unsupported:
            raise ValueError(f"Unsupported calibration metrics: {sorted(unsupported)}")
        if not requests:
            raise ValueError("At least one calibration request is required")

        effective_batch_size = max(1, int(batch_size or self._batch_size))
        needs_activation = "activation_ratio" in metric_set
        needs_gradient = bool(
            metric_set & {"gradient_value", "gradient_trace", "shapley_value"}
        )
        needs_gradient_value = "gradient_value" in metric_set
        needs_gradient_trace = "gradient_trace" in metric_set
        needs_shapley = "shapley_value" in metric_set

        activation_active = torch.zeros(self._num_layers, dtype=torch.float64)
        activation_total = torch.zeros(self._num_layers, dtype=torch.float64)
        gradient_value = torch.zeros(self._num_layers, dtype=torch.float64)
        gradient_trace = torch.zeros(self._num_layers, dtype=torch.float64)
        shapley_value = torch.zeros(self._num_layers, dtype=torch.float64)
        gradient_batches = 0
        transformer_layers = self._resolve_transformer_layers() if needs_gradient else None

        was_training = bool(getattr(self.model, "training", False))
        self.model.eval()

        batch_offsets = range(0, len(requests), effective_batch_size)
        try:
            for offset in progress(
                batch_offsets,
                desc="calibration: layer metrics",
                total=(len(requests) + effective_batch_size - 1) // effective_batch_size,
                unit="batch",
            ):
                batch_requests = requests[offset : offset + effective_batch_size]
                input_ids, attention_mask, labels = self._build_teacher_forced_batch(
                    batch_requests
                )

                if input_ids.shape[1] < 2:
                    continue

                if needs_gradient:
                    self._zero_model_grad()

                with torch.set_grad_enabled(needs_gradient):
                    outputs = self._forward_model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        output_hidden_states=needs_activation,
                        use_cache=False,
                    )

                    if needs_activation:
                        token_mask = attention_mask.unsqueeze(-1)
                        for layer_idx in range(self._num_layers):
                            hidden = outputs.hidden_states[layer_idx + 1]
                            hidden_mask = token_mask.to(device=hidden.device, dtype=hidden.dtype)
                            active = ((hidden > 0).to(hidden.dtype) * hidden_mask).sum()
                            total = hidden_mask.sum() * hidden.shape[-1]
                            activation_active[layer_idx] += float(active.detach().cpu())
                            activation_total[layer_idx] += float(total.detach().cpu())

                    if needs_gradient:
                        shift_logits = outputs.logits[:, :-1, :].contiguous()
                        shift_labels = labels[:, 1:].contiguous()
                        valid_labels = shift_labels.ne(-100).sum()
                        if int(valid_labels.item()) == 0:
                            continue
                        loss = F.cross_entropy(
                            shift_logits.view(-1, shift_logits.size(-1)),
                            shift_labels.view(-1),
                            ignore_index=-100,
                            reduction="sum",
                        )
                        loss = loss / max(1, len(batch_requests))
                        loss.backward()

                        for layer_idx, layer in enumerate(transformer_layers or []):
                            layer_gradient = 0.0
                            layer_score = 0.0
                            layer_shapley = 0.0
                            for parameter in layer.parameters():
                                if parameter.grad is None:
                                    continue
                                if needs_gradient_value:
                                    score = torch.abs(
                                        parameter.grad.detach().float()
                                    ).sum()
                                    layer_gradient += float(score.cpu())
                                if needs_gradient_trace:
                                    score = torch.abs(
                                        parameter.detach().float()
                                        * parameter.grad.detach().float()
                                    ).sum()
                                    layer_score += float(score.cpu())
                                if needs_shapley and parameter.ndim == 2:
                                    score = calculate_shapley_value(parameter).sum()
                                    layer_shapley += float(score.cpu())
                            if needs_gradient_value:
                                gradient_value[layer_idx] += layer_gradient
                            if needs_gradient_trace:
                                gradient_trace[layer_idx] += layer_score
                            if needs_shapley:
                                shapley_value[layer_idx] += layer_shapley
                        gradient_batches += 1
        finally:
            if needs_gradient:
                self._zero_model_grad()
            if was_training:
                self.model.train()

        layers = []
        for layer_idx in range(self._num_layers):
            layer_metrics: Dict[str, object] = {"layer": layer_idx + 1}
            if needs_activation:
                total = activation_total[layer_idx].item()
                layer_metrics["activation_ratio"] = (
                    activation_active[layer_idx].item() / total if total > 0 else 0.0
                )
            if needs_gradient:
                if needs_gradient_value:
                    layer_metrics["gradient_value"] = (
                        gradient_value[layer_idx].item() / gradient_batches
                        if gradient_batches
                        else 0.0
                    )
                if needs_gradient_trace:
                    layer_metrics["gradient_trace"] = (
                        gradient_trace[layer_idx].item() / gradient_batches
                        if gradient_batches
                        else 0.0
                    )
                if needs_shapley:
                    layer_metrics["shapley_value"] = (
                        shapley_value[layer_idx].item() / gradient_batches
                        if gradient_batches
                        else 0.0
                    )
            layers.append(layer_metrics)

        return {
            "num_layers": self._num_layers,
            "num_samples": len(requests),
            "num_batches": (len(requests) + effective_batch_size - 1) // effective_batch_size,
            "metrics": sorted(metric_set),
            "layers": layers,
        }

    def _get_layer_norm(self):
        """Return the model's final layer-norm module (architecture-agnostic)."""
        inner = self.model.model
        for attr in ("norm", "ln_f", "final_layer_norm"):
            if hasattr(inner, attr):
                return getattr(inner, attr)
        return None

    def _apply_strategy_to_hidden_states(
        self,
        hidden_states: Tuple[torch.Tensor, ...],
    ) -> torch.Tensor:
        """
        Select the exit hidden state according to the current strategy.

        If no strategy is set, returns the last hidden state (full model).
        """
        if not self._use_strategy:
            return hidden_states[-1]

        layer_norm = self._get_layer_norm()
        lm_head = self.model.lm_head
        return self.strategy.get_exit_hidden_state(
            hidden_states=hidden_states,
            num_layers=self._num_layers,
            lm_head=lm_head,
            layer_norm=layer_norm,
        )

    def _strategy_logits_from_outputs(self, outputs) -> torch.Tensor:
        if self.strategy is None:
            return outputs.logits

        layer_norm = self._get_layer_norm()
        lm_head = self.model.lm_head
        exit_layer = self.strategy.select_exit_layer(
            hidden_states=outputs.hidden_states,
            num_layers=self._num_layers,
            lm_head=lm_head,
            layer_norm=layer_norm,
        )
        if exit_layer == self._num_layers:
            return outputs.logits
        return self._logits_from_hidden(outputs.hidden_states[exit_layer])

    def _logits_from_hidden(self, hidden: torch.Tensor) -> torch.Tensor:
        """Apply layer norm + LM head to a hidden state tensor."""
        layer_norm = self._get_layer_norm()
        if layer_norm is not None:
            hidden = layer_norm(hidden)
        return self.model.lm_head(hidden)

    def _should_use_chat_template(self) -> bool:
        """Return whether prompts should be wrapped with the tokenizer chat template."""
        has_template = bool(getattr(self.tokenizer, "chat_template", None))
        has_apply = callable(getattr(self.tokenizer, "apply_chat_template", None))
        return has_template and has_apply

    @staticmethod
    def _flatten_token_ids(token_ids) -> List[int]:
        """Normalise tokenizer outputs to a flat Python token-id list."""
        if isinstance(token_ids, Mapping):
            token_ids = token_ids["input_ids"]
        if isinstance(token_ids, torch.Tensor):
            if token_ids.ndim == 2:
                return token_ids[0].tolist()
            return token_ids.tolist()
        if token_ids and isinstance(token_ids[0], list):
            return list(token_ids[0])
        return list(token_ids)

    def _encode_prompt_ids(self, prompt: str) -> List[int]:
        """Encode a task prompt, applying the tokenizer chat template when available."""
        if self._use_chat_template:
            token_ids = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=True,
                add_generation_prompt=True,
            )
            if isinstance(token_ids, str):
                return self.tokenizer.encode(token_ids, add_special_tokens=False)
            return self._flatten_token_ids(token_ids)

        return self.tokenizer.encode(prompt, add_special_tokens=True)

    def _prepare_prompt_inputs(self, prompt: str) -> Dict[str, torch.Tensor]:
        """Build model inputs for a generation prompt."""
        input_ids = torch.tensor(
            [self._encode_prompt_ids(prompt)],
            dtype=torch.long,
            device=self._device,
        )
        return {
            "input_ids": input_ids,
            "attention_mask": torch.ones_like(input_ids, device=self._device),
        }

    @staticmethod
    def _append_token_ids(token_ids: List[int], value) -> None:
        """Append one or more token ids while preserving order and uniqueness."""
        if value is None:
            return
        if isinstance(value, torch.Tensor):
            value = value.tolist()
        if isinstance(value, int):
            values = [value]
        else:
            try:
                values = list(value)
            except TypeError:
                values = [value]

        for token_id in values:
            if token_id is None:
                continue
            token_id = int(token_id)
            if token_id not in token_ids:
                token_ids.append(token_id)

    def _eos_token_id_list(self) -> List[int]:
        """Return all known EOS / chat-turn terminator token ids for the model."""
        token_ids: List[int] = []

        generation_config = getattr(self.model, "generation_config", None)
        if generation_config is not None:
            self._append_token_ids(
                token_ids,
                getattr(generation_config, "eos_token_id", None),
            )

        model_config = getattr(self.model, "config", None)
        if model_config is not None:
            self._append_token_ids(token_ids, getattr(model_config, "eos_token_id", None))

        self._append_token_ids(token_ids, getattr(self.tokenizer, "eos_token_id", None))

        convert = getattr(self.tokenizer, "convert_tokens_to_ids", None)
        if callable(convert):
            unk_token_id = getattr(self.tokenizer, "unk_token_id", None)
            for token in ("<|eot_id|>", "<|end_of_text|>"):
                token_id = convert(token)
                if token_id is not None and token_id != unk_token_id:
                    self._append_token_ids(token_ids, token_id)

        return token_ids

    def _generation_eos_token_id(self):
        eos_token_ids = self._eos_token_id_list()
        if not eos_token_ids:
            return None
        if len(eos_token_ids) == 1:
            return eos_token_ids[0]
        return eos_token_ids

    # ------------------------------------------------------------------ #
    # BaseLM interface                                                     #
    # ------------------------------------------------------------------ #

    @property
    def device(self) -> str:
        return self._device

    @property
    def batch_size(self) -> int:
        return self._batch_size

    def loglikelihood(
        self, requests: List[Tuple[str, str]]
    ) -> List[Tuple[float, bool]]:
        """
        Compute log P(continuation | context) for each request.

        Tokenises context+continuation together, masks out the context tokens,
        and sums the log-probabilities of the continuation tokens.
        """
        results: List[Tuple[float, bool]] = []

        batch_offsets = range(0, len(requests), self._batch_size)
        for i in progress(
            batch_offsets,
            desc="model: loglikelihood",
            total=len(batch_offsets),
            unit="batch",
        ):
            batch = requests[i : i + self._batch_size]
            batch_results = self._loglikelihood_batch(batch)
            results.extend(batch_results)

        return results

    def _loglikelihood_batch(
        self, batch: List[Tuple[str, str]]
    ) -> List[Tuple[float, bool]]:
        """Process a single batch of loglikelihood requests."""
        # Tokenise context and continuation separately to know the boundary
        contexts, continuations = zip(*batch)

        ctx_encodings = [self._encode_prompt_ids(c) for c in contexts]
        cont_encodings = [
            self.tokenizer.encode(c, add_special_tokens=False) for c in continuations
        ]

        # Build full input_ids for each example
        full_ids = []
        cont_lengths = []
        for ctx_ids, cont_ids in zip(ctx_encodings, cont_encodings):
            full = ctx_ids + cont_ids
            full_ids.append(full)
            cont_lengths.append(len(cont_ids))

        # Pad to the same length
        max_len = max(len(ids) for ids in full_ids)
        pad_id = self.tokenizer.pad_token_id

        input_ids_list = []
        attention_mask_list = []
        for ids in full_ids:
            pad_len = max_len - len(ids)
            padded = [pad_id] * pad_len + ids
            mask = [0] * pad_len + [1] * len(ids)
            input_ids_list.append(padded)
            attention_mask_list.append(mask)

        input_ids = torch.tensor(input_ids_list, dtype=torch.long).to(self._device)
        attention_mask = torch.tensor(attention_mask_list, dtype=torch.long).to(
            self._device
        )

        with torch.no_grad():
            outputs = self._forward_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=self._use_strategy,
                use_cache=False,
            )

            if self._use_strategy:
                logits = self._strategy_logits_from_outputs(outputs)
            else:
                logits = outputs.logits

        # Shift: logits[t] predicts token at position t+1
        shift_logits = logits[:, :-1, :]  # (batch, seq-1, vocab)
        shift_labels = input_ids[:, 1:]  # (batch, seq-1)

        log_probs = F.log_softmax(shift_logits, dim=-1)

        results: List[Tuple[float, bool]] = []
        for b_idx, cont_len in enumerate(cont_lengths):
            # The continuation tokens start at position (seq_len - cont_len)
            seq_len = input_ids_list[b_idx].__len__()
            # After padding, the actual sequence starts at offset (max_len - seq_len)
            actual_len = sum(attention_mask_list[b_idx])
            start_pos = max_len - cont_len - 1  # position in shift_logits

            lp = 0.0
            is_greedy = True
            for t in range(cont_len):
                pos = start_pos + t
                if pos < 0 or pos >= shift_logits.shape[1]:
                    continue
                token_id = shift_labels[b_idx, pos].item()
                token_lp = log_probs[b_idx, pos, token_id].item()
                lp += token_lp
                # Check if this token is greedy
                greedy_token = shift_logits[b_idx, pos].argmax().item()
                if greedy_token != token_id:
                    is_greedy = False

            results.append((lp, is_greedy))

        return results

    def generate_until(
        self,
        requests: List[Tuple[str, dict]],
    ) -> List[str]:
        """
        Generate text for each (prompt, gen_kwargs) pair.

        Supported ``gen_kwargs`` keys:
            * ``max_new_tokens`` (int, default 256)
            * ``temperature`` (float, default 1.0)
            * ``top_p`` (float, default 1.0)
            * ``do_sample`` (bool, default False)
            * ``stop_sequences`` (list of str, default [])
        """
        results: List[str] = []
        for prompt, gen_kwargs in progress(
            requests,
            desc="model: generate",
            total=len(requests),
            unit="sample",
        ):
            output = self._generate_single(prompt, gen_kwargs)
            results.append(output)
        return results

    def _generate_single(self, prompt: str, gen_kwargs: dict) -> str:
        """Generate a single completion."""
        max_new_tokens = gen_kwargs.get("max_new_tokens", 256)
        temperature = gen_kwargs.get("temperature", 1.0)
        top_p = gen_kwargs.get("top_p", 1.0)
        do_sample = gen_kwargs.get("do_sample", False)
        stop_sequences = gen_kwargs.get("stop_sequences", [])

        inputs = self._prepare_prompt_inputs(prompt)

        input_len = inputs["input_ids"].shape[1]

        if self._use_strategy:
            # For layer-skipping generation: generate token-by-token, applying
            # the strategy at each step.
            generated_ids = self._generate_with_strategy(
                inputs["input_ids"],
                inputs["attention_mask"],
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample,
                stop_sequences=stop_sequences,
            )
        else:
            generated_ids = self._generate_with_native_model(
                inputs=inputs,
                input_len=input_len,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample,
            )

        text = self.tokenizer.decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        # Apply stop sequences
        for stop in stop_sequences:
            if stop in text:
                text = text[: text.index(stop)]

        return text.rstrip()

    def _generate_with_native_model(
        self,
        inputs: Dict[str, torch.Tensor],
        input_len: int,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        do_sample: bool,
    ) -> torch.Tensor:
        from transformers import GenerationConfig

        config_kwargs = {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature if do_sample else 1.0,
            "top_p": top_p if do_sample else 1.0,
            "do_sample": do_sample,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self._generation_eos_token_id(),
        }
        if self._bypass_layer_indices:
            config_kwargs["use_cache"] = False

        gen_config = GenerationConfig(**config_kwargs)
        generate_kwargs = {**inputs, "generation_config": gen_config}

        with torch.no_grad():
            with self._bypass_transformer_layers():
                output_ids = self.model.generate(**generate_kwargs)
        return output_ids[0, input_len:]

    def _generate_with_strategy(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        do_sample: bool,
        stop_sequences: List[str],
    ) -> torch.Tensor:
        """
        Greedy/sampling generation loop that applies the layer-skipping strategy
        at each token step.
        """
        generated = []
        past_key_values = None
        cur_input_ids = input_ids
        cur_attention_mask = attention_mask
        use_cache = not self._bypass_layer_indices

        eos_token_ids = set(self._eos_token_id_list())

        for _ in range(max_new_tokens):
            with torch.no_grad():
                if use_cache:
                    outputs = self._forward_model(
                        input_ids=cur_input_ids,
                        attention_mask=cur_attention_mask,
                        past_key_values=past_key_values,
                        output_hidden_states=True,
                        use_cache=True,
                    )
                else:
                    outputs = self._forward_model(
                        input_ids=cur_input_ids,
                        attention_mask=cur_attention_mask,
                        output_hidden_states=True,
                        use_cache=False,
                    )

            logits = self._strategy_logits_from_outputs(outputs)
            next_logits = logits[:, -1, :]  # (1, vocab)

            if do_sample:
                # Temperature + top-p sampling
                if temperature != 1.0:
                    next_logits = next_logits / temperature
                probs = F.softmax(next_logits, dim=-1)
                if top_p < 1.0:
                    sorted_probs, sorted_idx = torch.sort(probs, descending=True)
                    cum_probs = sorted_probs.cumsum(dim=-1)
                    mask = cum_probs - sorted_probs > top_p
                    sorted_probs[mask] = 0.0
                    probs = torch.zeros_like(probs).scatter_(1, sorted_idx, sorted_probs)
                    probs = probs / probs.sum(dim=-1, keepdim=True)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = next_logits.argmax(dim=-1, keepdim=True)

            token_id = next_token.item()
            generated.append(token_id)

            if token_id in eos_token_ids:
                break

            # Check stop sequences
            decoded_so_far = self.tokenizer.decode(
                generated,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            if any(s in decoded_so_far for s in stop_sequences):
                break

            if use_cache:
                past_key_values = outputs.past_key_values
                cur_input_ids = next_token
            else:
                cur_input_ids = torch.cat([cur_input_ids, next_token], dim=1)
            cur_attention_mask = torch.cat(
                [cur_attention_mask, torch.ones(1, 1, device=self._device, dtype=torch.long)],
                dim=1,
            )

        return torch.tensor(generated, dtype=torch.long)
