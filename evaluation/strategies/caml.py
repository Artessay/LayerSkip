"""CALM adaptive early exit (keeps the historical ``caml`` CLI alias).

Implements the inference rule and confidence measures from Schuster et al.,
2022 (arXiv:2207.07061): token-wise exits, top-two softmax response, hidden
state saturation, and the optional Eq. (5) generation-step threshold decay.
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn.functional as F

from evaluation.strategies.base_strategy import BaseLayerSkipStrategy


class CAMLStrategy(BaseLayerSkipStrategy):
    """Token-wise CALM inference using shared intermediate LM heads.

    The backbone must have been trained with intermediate LM losses for a
    faithful paper reproduction. The released ordinary Llama checkpoints do
    not meet that requirement; callers must explicitly opt in to evaluating
    such a checkpoint with ``allow_untrained_exits=True``.
    """

    SUPPORTED_MEASURES = {"softmax", "hidden_state"}

    def __init__(
        self,
        confidence_threshold: float = 0.9,
        min_layers: int = 4,
        check_every: int = 1,
        confidence_measure: str = "softmax",
        use_decaying_threshold: bool = False,
        decay_factor: float = 4.0,
        max_steps: int = 256,
        allow_untrained_exits: bool = False,
    ) -> None:
        measure = confidence_measure.lower()
        config = {
            "confidence_threshold": confidence_threshold,
            "min_layers": min_layers,
            "check_every": check_every,
            "confidence_measure": measure,
            "use_decaying_threshold": use_decaying_threshold,
            "decay_factor": decay_factor,
            "max_steps": max_steps,
            "allow_untrained_exits": allow_untrained_exits,
        }
        super().__init__(config)
        if not 0.0 < confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be in (0, 1]")
        if min_layers < 1 or check_every < 1 or max_steps < 1:
            raise ValueError("min_layers, check_every and max_steps must be positive")
        if measure not in self.SUPPORTED_MEASURES:
            raise ValueError(f"confidence_measure must be one of {sorted(self.SUPPORTED_MEASURES)}")
        if decay_factor < 0:
            raise ValueError("decay_factor must be non-negative")
        self.confidence_threshold = confidence_threshold
        self.min_layers = min_layers
        self.check_every = check_every
        self.confidence_measure = measure
        self.use_decaying_threshold = use_decaying_threshold
        self.decay_factor = decay_factor
        self.max_steps = max_steps
        self.allow_untrained_exits = allow_untrained_exits
        self.generation_step = 0

    @property
    def name(self) -> str:
        return "caml"

    def set_generation_step(self, step: int) -> None:
        self.generation_step = max(0, int(step))

    def current_threshold(self) -> float:
        if not self.use_decaying_threshold:
            return self.confidence_threshold
        # CALM Eq. (5): clip(.9 lambda + .1 exp(-tau*t/N), 0, 1)
        value = 0.9 * self.confidence_threshold + 0.1 * math.exp(
            -self.decay_factor * self.generation_step / self.max_steps
        )
        return min(1.0, max(0.0, value))

    def _confidence_tensor(
        self,
        hidden: torch.Tensor,
        previous_hidden: torch.Tensor,
        lm_head: torch.nn.Module,
        layer_norm: Optional[torch.nn.Module],
    ) -> torch.Tensor:
        if self.confidence_measure == "hidden_state":
            # State saturation from the paper: cosine similarity to the
            # preceding layer, independently for every token.
            return F.cosine_similarity(hidden, previous_hidden, dim=-1)
        if layer_norm is not None:
            hidden = layer_norm(hidden)
        probabilities = F.softmax(lm_head(hidden), dim=-1)
        top_two = probabilities.topk(k=2, dim=-1).values
        # CALM softmax response is the gap between the two most likely tokens.
        return top_two[..., 0] - top_two[..., 1]

    def _compute_confidence(self, hidden, lm_head, layer_norm) -> float:
        """Compatibility helper returning mean softmax-response confidence."""
        confidence = self._confidence_tensor(hidden, hidden, lm_head, layer_norm)
        return float(confidence.mean().item())

    def get_exit_hidden_state(
        self,
        hidden_states: Tuple[torch.Tensor, ...],
        num_layers: int,
        lm_head: Optional[torch.nn.Module] = None,
        layer_norm: Optional[torch.nn.Module] = None,
    ) -> torch.Tensor:
        if lm_head is None:
            return hidden_states[-1]
        selected = hidden_states[-1].clone()
        unresolved = torch.ones(selected.shape[:2], dtype=torch.bool, device=selected.device)
        threshold = self.current_threshold()
        for layer_idx in range(self.min_layers, num_layers + 1):
            if (layer_idx - self.min_layers) % self.check_every and layer_idx != num_layers:
                continue
            confidence = self._confidence_tensor(
                hidden_states[layer_idx], hidden_states[layer_idx - 1], lm_head, layer_norm
            )
            exits = unresolved & (confidence >= threshold)
            selected = torch.where(exits.unsqueeze(-1), hidden_states[layer_idx], selected)
            unresolved &= ~exits
            if not unresolved.any():
                break
        return selected

    def select_exit_layer(self, hidden_states, num_layers, lm_head=None, layer_norm=None) -> int:
        """Return the deepest token exit for compatibility with scalar callers."""
        if lm_head is None:
            return num_layers
        threshold = self.current_threshold()
        deepest = self.min_layers
        unresolved = torch.ones(hidden_states[0].shape[:2], dtype=torch.bool, device=hidden_states[0].device)
        for layer_idx in range(self.min_layers, num_layers + 1):
            if (layer_idx - self.min_layers) % self.check_every and layer_idx != num_layers:
                continue
            confidence = self._confidence_tensor(
                hidden_states[layer_idx], hidden_states[layer_idx - 1], lm_head, layer_norm
            )
            exits = unresolved & (confidence >= threshold)
            if exits.any():
                deepest = layer_idx
            unresolved &= ~exits
            if not unresolved.any():
                return deepest
        return num_layers if unresolved.any() else deepest
