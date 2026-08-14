"""GateSkip residual gating from Laitenberger et al. (arXiv:2510.13876).

This module implements the paper's sigmoid-linear vector gates, mean gate
importance, per-layer linear-interpolation quantile threshold, and token-wise
residual copying. Gate weights must come from a jointly fine-tuned GateSkip
checkpoint; hidden-change magnitude is intentionally not used as a substitute.
"""

from pathlib import Path
from typing import Optional, Tuple

import torch

from evaluation.strategies.base_strategy import BaseLayerSkipStrategy


class GateSkipStrategy(BaseLayerSkipStrategy):
    def __init__(
        self,
        gate_threshold: float = 0.01,  # deprecated compatibility argument
        skip_budget: float = 0.3,
        min_layers: int = 1,
        gate_state_path: Optional[str] = None,
    ) -> None:
        if not 0.0 <= skip_budget < 1.0:
            raise ValueError("skip_budget must be in [0, 1)")
        if min_layers < 1:
            raise ValueError("min_layers must be positive")
        super().__init__({
            "skip_budget": skip_budget,
            "min_layers": min_layers,
            "gate_state_path": gate_state_path,
        })
        self.skip_budget = skip_budget
        self.min_layers = min_layers
        self.gate_state_path = gate_state_path
        self._gate_state = None
        if gate_state_path:
            loaded = torch.load(Path(gate_state_path), map_location="cpu", weights_only=True)
            self._gate_state = loaded.get("state_dict", loaded)

    @property
    def name(self) -> str:
        return "gateskip"

    @staticmethod
    def quantile_threshold(scores: torch.Tensor, skip_ratio: float) -> torch.Tensor:
        """Algorithm 3: linearly interpolated empirical quantile."""
        flat = scores.flatten().float()
        if flat.numel() <= 1 or bool(torch.all(flat == flat[0])):
            # Match the official ThresholdFinder: constant gates must not
            # accidentally cause every token to be masked.
            return flat.new_tensor(float("-inf"))
        return torch.quantile(flat, min(max(skip_ratio, 0.0), 1.0), interpolation="linear")

    def _gate_parameters(self, layer_idx: int, hidden_size: int, device, dtype):
        if self._gate_state is None:
            raise RuntimeError(
                "GateSkip requires jointly fine-tuned sigmoid gate weights. "
                "Pass --gateskip_gate_state_path; random or hidden-change gates "
                "are not a reproduction of the paper."
            )
        prefixes = (f"gates.{layer_idx}", f"layers.{layer_idx}.gate", str(layer_idx))
        for prefix in prefixes:
            weight = self._gate_state.get(f"{prefix}.weight")
            bias = self._gate_state.get(f"{prefix}.bias")
            if weight is not None and bias is not None:
                if tuple(weight.shape) != (hidden_size, hidden_size):
                    raise ValueError(f"Gate {layer_idx} weight has shape {tuple(weight.shape)}")
                return weight.to(device=device, dtype=dtype), bias.to(device=device, dtype=dtype)
        raise KeyError(f"No gate parameters found for layer {layer_idx}")

    def _apply_gate(self, previous: torch.Tensor, module_output: torch.Tensor, layer_idx: int):
        hidden_size = previous.shape[-1]
        weight, bias = self._gate_parameters(
            layer_idx, hidden_size, previous.device, previous.dtype
        )
        gate = torch.sigmoid(torch.nn.functional.linear(previous, weight, bias))
        importance = gate.mean(dim=-1)
        threshold = self.quantile_threshold(importance, self.skip_budget)
        # Paper Eq./Algorithm 2: low-ranked tokens copy h_l; retained tokens
        # receive the gated residual module output.
        skip = importance <= threshold if self.skip_budget > 0 else torch.zeros_like(importance, dtype=torch.bool)
        gated = previous + gate * module_output
        return torch.where(skip.unsqueeze(-1), previous, gated)

    def get_exit_hidden_state(
        self,
        hidden_states: Tuple[torch.Tensor, ...],
        num_layers: int,
        lm_head=None,
        layer_norm=None,
    ) -> torch.Tensor:
        current = hidden_states[0]
        for layer_idx in range(num_layers):
            # The stored full-forward delta is the block residual contribution.
            # Exact attention/MLP branch execution requires an official
            # GateSkip-instrumented checkpoint; this evaluator preserves the
            # paper's gating/ranking equations for checkpoint comparison.
            module_output = hidden_states[layer_idx + 1] - hidden_states[layer_idx]
            if layer_idx + 1 < self.min_layers:
                current = current + module_output
            else:
                current = self._apply_gate(current, module_output, layer_idx)
        return current

    def select_exit_layer(self, hidden_states, num_layers, lm_head=None, layer_norm=None) -> int:
        return num_layers
