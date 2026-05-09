"""
LayerSkip strategy – static early exit.

Based on the paper:
    "LayerSkip: Enabling Early Exit Inference and Self-Speculative Decoding"
    Elhoushi et al., Meta AI (2024). https://arxiv.org/abs/2404.16710

The strategy exits after a fixed fraction of the model's transformer layers.
All positions in a batch use the same exit layer, which makes this the simplest
and most predictable of the three strategies.
"""

from typing import Tuple

import torch

from evaluation.strategies.base_strategy import BaseLayerSkipStrategy


class LayerSkipStrategy(BaseLayerSkipStrategy):
    """
    Static layer skipping via a fixed early-exit layer.

    The exit layer is determined by ``exit_ratio * num_layers`` (clamped to
    ``[min_layers, num_layers]``). Setting ``exit_ratio=1.0`` is equivalent to
    using the full model.

    Args:
        exit_ratio: Fraction of layers to execute before exiting (0 < ratio ≤ 1).
        min_layers: Minimum number of layers that are always executed.
    """

    def __init__(self, exit_ratio: float = 0.75, min_layers: int = 4) -> None:
        super().__init__({"exit_ratio": exit_ratio, "min_layers": min_layers})
        if not 0.0 < exit_ratio <= 1.0:
            raise ValueError(f"exit_ratio must be in (0, 1], got {exit_ratio}")
        if min_layers < 1:
            raise ValueError(f"min_layers must be >= 1, got {min_layers}")
        self.exit_ratio = exit_ratio
        self.min_layers = min_layers

    @property
    def name(self) -> str:
        return "layerskip"

    def compute_exit_layer(self, num_layers: int) -> int:
        """Return the 1-based exit layer index (index into ``hidden_states``)."""
        raw = int(num_layers * self.exit_ratio)
        return max(self.min_layers, min(raw, num_layers))

    def select_exit_layer(
        self,
        hidden_states: Tuple[torch.Tensor, ...],
        num_layers: int,
        lm_head=None,
        layer_norm=None,
    ) -> int:
        """Return the fixed exit layer index (same for all positions)."""
        return self.compute_exit_layer(num_layers)
