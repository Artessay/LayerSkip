"""ManualSkip strategy - user-selected transformer layer bypass."""

from typing import Iterable, Tuple

import torch

from evaluation.strategies.base_strategy import BaseLayerSkipStrategy


class ManualSkipStrategy(BaseLayerSkipStrategy):
    """
    Bypass user-specified transformer layers during the model forward pass.

    ``skip_layers`` uses 1-based layer numbers as exposed in the CLI. When a
    listed layer is reached, its transformer block is not executed; the hidden
    state from the previous layer is fed directly to the next layer.

    Args:
        skip_layers: 1-based layer numbers to bypass.
    """

    def __init__(self, skip_layers: Iterable[int]) -> None:
        normalized = self._normalize_skip_layers(skip_layers)
        super().__init__({"skip_layers": list(normalized)})
        self.skip_layers = normalized

    @property
    def name(self) -> str:
        return "manualskip"

    @staticmethod
    def _normalize_skip_layers(skip_layers: Iterable[int]) -> Tuple[int, ...]:
        layers = []
        for layer in skip_layers:
            layer_num = int(layer)
            if layer_num < 1:
                raise ValueError(
                    f"skip_layers must contain positive layer numbers, got {layer_num}"
                )
            layers.append(layer_num)

        if not layers:
            raise ValueError("skip_layers must contain at least one layer number")

        return tuple(sorted(set(layers)))

    def get_skipped_layer_indices(self, num_layers: int) -> Tuple[int, ...]:
        invalid_layers = [layer for layer in self.skip_layers if layer > num_layers]
        if invalid_layers:
            raise ValueError(
                f"skip_layers must be within [1, {num_layers}], got {invalid_layers}"
            )
        return tuple(layer - 1 for layer in self.skip_layers)

    def uses_full_model_logits(self, num_layers: int) -> bool:
        return True

    def select_exit_layer(
        self,
        hidden_states: Tuple[torch.Tensor, ...],
        num_layers: int,
        lm_head=None,
        layer_norm=None,
    ) -> int:
        return num_layers
