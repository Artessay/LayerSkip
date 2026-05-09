"""
GateSkip strategy – gate/change-magnitude-based layer selection.

Inspired by the paper:
    "What Layers When: Learning to Skip Compute in LLMs with Residual Gates"
    Laitenberger et al. (2024). https://arxiv.org/abs/2510.13876

The strategy uses the *relative change* in hidden-state norms across consecutive
layers as a proxy for each layer's "gate value". Layers whose relative change
falls below ``gate_threshold`` are deemed low-importance and counted as skipped.
The exit layer is the last layer before the budget of skippable layers is
exhausted, ensuring that the most informative prefix of layers is used.

This is a training-free approximation of the learned-gate approach described in
the original GateSkip paper, suitable for zero-shot comparison studies.
"""

from typing import Tuple

import torch

from evaluation.strategies.base_strategy import BaseLayerSkipStrategy


class GateSkipStrategy(BaseLayerSkipStrategy):
    """
    Gate-based layer skipping using hidden-state change magnitude.

    For each layer the *relative change* is computed as:

        ``gate_value = ||h_l - h_{l-1}|| / (||h_{l-1}|| + eps)``

    where norms are averaged over batch and sequence dimensions.  Layers with
    ``gate_value < gate_threshold`` are treated as skippable.  Skipping is
    stopped once ``skip_budget`` fraction of the total layers has been reached.
    The strategy returns the index of the last *non-skipped* layer.

    Args:
        gate_threshold: Relative-change threshold below which a layer is
            considered skippable (default: 0.01).
        skip_budget: Maximum fraction of layers that may be skipped (default: 0.3).
        min_layers: Minimum number of layers to always execute (default: 4).
    """

    def __init__(
        self,
        gate_threshold: float = 0.01,
        skip_budget: float = 0.3,
        min_layers: int = 4,
    ) -> None:
        super().__init__(
            {
                "gate_threshold": gate_threshold,
                "skip_budget": skip_budget,
                "min_layers": min_layers,
            }
        )
        if gate_threshold < 0:
            raise ValueError(f"gate_threshold must be >= 0, got {gate_threshold}")
        if not 0.0 <= skip_budget <= 1.0:
            raise ValueError(f"skip_budget must be in [0, 1], got {skip_budget}")
        if min_layers < 1:
            raise ValueError(f"min_layers must be >= 1, got {min_layers}")
        self.gate_threshold = gate_threshold
        self.skip_budget = skip_budget
        self.min_layers = min_layers

    @property
    def name(self) -> str:
        return "gateskip"

    def _compute_gate_values(
        self, hidden_states: Tuple[torch.Tensor, ...], num_layers: int
    ) -> list[float]:
        """
        Compute per-layer gate values (relative hidden-state change).

        Returns a list of ``num_layers`` floats, one per transformer layer.
        ``gate_values[i]`` corresponds to the change from ``hidden_states[i]``
        to ``hidden_states[i+1]``.
        """
        gate_values = []
        with torch.no_grad():
            for i in range(1, num_layers + 1):
                h_prev = hidden_states[i - 1]  # (batch, seq, hidden)
                h_curr = hidden_states[i]
                diff_norm = (h_curr - h_prev).norm(dim=-1).mean()  # scalar
                prev_norm = h_prev.norm(dim=-1).mean()
                gate_val = float((diff_norm / (prev_norm + 1e-8)).item())
                gate_values.append(gate_val)
        return gate_values

    def select_exit_layer(
        self,
        hidden_states: Tuple[torch.Tensor, ...],
        num_layers: int,
        lm_head=None,
        layer_norm=None,
    ) -> int:
        """
        Return the exit layer index based on gate values.

        Layers before ``min_layers`` are always executed.  Starting from
        ``min_layers`` onward, layers with ``gate_value < gate_threshold`` are
        counted as skipped.  The first layer that would exceed ``skip_budget``
        stops the skipping process and the *previous* active layer is used as
        the exit.

        Falls back to ``num_layers`` (full model) if no layers qualify for
        skipping.
        """
        gate_values = self._compute_gate_values(hidden_states, num_layers)
        max_skip = int(num_layers * self.skip_budget)
        skipped = 0
        last_active = num_layers  # default: use full model

        for layer_1based in range(self.min_layers, num_layers + 1):
            gate_val = gate_values[layer_1based - 1]
            if gate_val < self.gate_threshold and skipped < max_skip:
                skipped += 1
            else:
                last_active = layer_1based

        return last_active
