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

import logging
from collections.abc import Mapping
from typing import Dict, List, Optional, Tuple, Union

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
            {"auto": "auto", "float16": torch.float16, "bfloat16": torch.bfloat16}.get(
                dtype, "auto"
            )
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
        self._use_strategy: bool = strategy is not None
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
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=self._use_strategy,
                use_cache=False,
            )

            if self._use_strategy:
                hidden = self._apply_strategy_to_hidden_states(outputs.hidden_states)
                logits = self._logits_from_hidden(hidden)
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
            from transformers import GenerationConfig

            gen_config = GenerationConfig(
                max_new_tokens=max_new_tokens,
                temperature=temperature if do_sample else 1.0,
                top_p=top_p if do_sample else 1.0,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self._generation_eos_token_id(),
            )
            with torch.no_grad():
                output_ids = self.model.generate(
                    **inputs,
                    generation_config=gen_config,
                )
            generated_ids = output_ids[0, input_len:]

        text = self.tokenizer.decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        # Apply stop sequences
        for stop in stop_sequences:
            if stop in text:
                text = text[: text.index(stop)]

        return text.strip()

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

        eos_token_ids = set(self._eos_token_id_list())

        for _ in range(max_new_tokens):
            with torch.no_grad():
                outputs = self.model(
                    input_ids=cur_input_ids,
                    attention_mask=cur_attention_mask,
                    past_key_values=past_key_values,
                    output_hidden_states=True,
                    use_cache=True,
                )

            hidden = self._apply_strategy_to_hidden_states(outputs.hidden_states)
            logits = self._logits_from_hidden(hidden)
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

            # Update KV cache and attention mask
            past_key_values = outputs.past_key_values
            cur_input_ids = next_token
            cur_attention_mask = torch.cat(
                [cur_attention_mask, torch.ones(1, 1, device=self._device, dtype=torch.long)],
                dim=1,
            )

        return torch.tensor(generated, dtype=torch.long)
