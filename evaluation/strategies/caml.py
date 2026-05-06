"""
CAML strategy – Confidence-Adaptive Multi-Layer inference.

Exits at the earliest layer where the model's per-token prediction confidence
(maximum softmax probability) averaged over the sequence exceeds a configurable
threshold. If no layer reaches the threshold the full model is used.

The idea of confidence-based early exit is explored in several works including:
    "Confident Adaptive Language Modeling" (Schuster et al., 2022)
    https://arxiv.org/abs/2207.07061
"""

from typing import Optional, Tuple

import torch
import torch.nn.functional as F

from evaluation.strategies.base_strategy import BaseLayerSkipStrategy


class CAMLStrategy(BaseLayerSkipStrategy):
    """
    Confidence-Adaptive Multi-Layer (CAML) layer-skipping strategy.

    For each candidate exit layer the strategy applies the model's own layer norm
    and LM head to get a vocabulary distribution, then checks whether the *mean*
    maximum probability across the batch and sequence exceeds
    ``confidence_threshold``. The first layer that satisfies this condition
    becomes the exit layer.

    Args:
        confidence_threshold: Exit when mean max-softmax exceeds this value
            (0 < threshold ≤ 1). Higher values require more processing.
        min_layers: Minimum number of layers to execute before considering exit.
        check_every: Check the confidence criterion every this many layers to
            reduce overhead (default: 1 = check every layer).
    """

    def __init__(
        self,
        confidence_threshold: float = 0.9,
        min_layers: int = 4,
        check_every: int = 1,
    ) -> None:
        super().__init__(
            {
                "confidence_threshold": confidence_threshold,
                "min_layers": min_layers,
                "check_every": check_every,
            }
        )
        if not 0.0 < confidence_threshold <= 1.0:
            raise ValueError(
                f"confidence_threshold must be in (0, 1], got {confidence_threshold}"
            )
        if min_layers < 1:
            raise ValueError(f"min_layers must be >= 1, got {min_layers}")
        if check_every < 1:
            raise ValueError(f"check_every must be >= 1, got {check_every}")
        self.confidence_threshold = confidence_threshold
        self.min_layers = min_layers
        self.check_every = check_every

    @property
    def name(self) -> str:
        return "caml"

    def _compute_confidence(
        self,
        hidden: torch.Tensor,
        lm_head: torch.nn.Module,
        layer_norm: torch.nn.Module,
    ) -> float:
        """
        Compute mean max-softmax confidence for the given hidden state.

        Args:
            hidden: Tensor of shape ``(batch, seq, hidden_dim)``.
            lm_head: Linear projection to vocabulary logits.
            layer_norm: Final layer norm applied before the LM head.

        Returns:
            Scalar confidence value in ``[0, 1]``.
        """
        with torch.no_grad():
            normed = layer_norm(hidden)
            logits = lm_head(normed)  # (batch, seq, vocab)
            probs = F.softmax(logits, dim=-1)
            max_prob = probs.max(dim=-1).values  # (batch, seq)
        return float(max_prob.mean().item())

    def select_exit_layer(
        self,
        hidden_states: Tuple[torch.Tensor, ...],
        num_layers: int,
        lm_head: Optional[torch.nn.Module] = None,
        layer_norm: Optional[torch.nn.Module] = None,
    ) -> int:
        """
        Scan layers from ``min_layers`` onward and return the first layer index
        where confidence exceeds the threshold.

        Requires ``lm_head`` and ``layer_norm`` to be provided; falls back to
        ``num_layers`` (full model) otherwise.
        """
        if lm_head is None or layer_norm is None:
            return num_layers

        for layer_idx in range(self.min_layers, num_layers + 1):
            # Only check at multiples of check_every (or the last layer)
            if (layer_idx - self.min_layers) % self.check_every != 0 and layer_idx != num_layers:
                continue
            confidence = self._compute_confidence(
                hidden_states[layer_idx], lm_head, layer_norm
            )
            if confidence >= self.confidence_threshold:
                return layer_idx

        return num_layers
