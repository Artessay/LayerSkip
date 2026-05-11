"""Base class for layer skipping strategies."""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, Tuple

import torch


class BaseLayerSkipStrategy(ABC):
    """
    Abstract base class for layer skipping strategies.

    A strategy determines which transformer layer's hidden state to use as the
    final representation, effectively controlling how many (or which) layers
    contribute to the model's prediction. Strategies operate on the full set of
    intermediate hidden states produced by a forward pass with
    ``output_hidden_states=True``.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config = config or {}

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable strategy name."""

    @abstractmethod
    def select_exit_layer(
        self,
        hidden_states: Tuple[torch.Tensor, ...],
        num_layers: int,
        lm_head: Optional[torch.nn.Module] = None,
        layer_norm: Optional[torch.nn.Module] = None,
    ) -> int:
        """
        Decide which layer index to use as the exit point.

        Args:
            hidden_states: Tuple of tensors with shape ``(batch, seq, hidden)``.
                ``hidden_states[0]`` is the embedding output; ``hidden_states[k]``
                for ``k >= 1`` is the output of transformer layer ``k-1``.
            num_layers: Total number of transformer layers in the model.
            lm_head: The language-model head (``nn.Linear``). Required by
                confidence-based strategies.
            layer_norm: The final layer-norm module. Required by confidence-based
                strategies.

        Returns:
            Index into ``hidden_states`` to use as the exit representation.
            Returned values are in the range ``[1, num_layers]``.
        """

    def get_exit_hidden_state(
        self,
        hidden_states: Tuple[torch.Tensor, ...],
        num_layers: int,
        lm_head: Optional[torch.nn.Module] = None,
        layer_norm: Optional[torch.nn.Module] = None,
    ) -> torch.Tensor:
        """
        Return the hidden-state tensor at the selected exit layer.

        Convenience wrapper around :meth:`select_exit_layer`.
        """
        idx = self.select_exit_layer(hidden_states, num_layers, lm_head, layer_norm)
        return hidden_states[idx]

    def get_skipped_layer_indices(self, num_layers: int) -> Tuple[int, ...]:
        """
        Return zero-based transformer-layer indices that should be bypassed.

        Most strategies only choose an exit hidden state after a normal forward
        pass. Strategies that want to avoid computing specific transformer
        blocks can override this hook and return the layer module indices that
        should pass their input hidden state through unchanged.
        """
        return ()

    def is_noop(self, num_layers: int) -> bool:
        """Return whether this strategy leaves the full model path unchanged."""
        return False

    def uses_full_model_logits(self, num_layers: int) -> bool:
        """Return whether the strategy should use the model's native final logits."""
        return False

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(config={self.config})"
