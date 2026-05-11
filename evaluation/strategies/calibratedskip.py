"""CalibratedSkip metadata strategy for calibration-only layer scoring."""

from typing import Iterable, Optional, Tuple

import torch

from evaluation.strategies.base_strategy import BaseLayerSkipStrategy


class CalibratedSkipStrategy(BaseLayerSkipStrategy):
    """Carry calibration metadata without automatically bypassing layers."""

    def __init__(
        self,
        skip_layers: Optional[Iterable[int]] = None,
        calibration_metrics: Optional[Iterable[str]] = None,
        metrics_path: Optional[str] = None,
        calibration_split: Optional[str] = None,
    ) -> None:
        self.skip_layers = self._normalize_skip_layers(skip_layers or [])
        self.calibration_metrics = tuple(calibration_metrics or ())
        self.metrics_path = metrics_path
        self.calibration_split = calibration_split

        super().__init__(
            {
                "skip_layers": list(self.skip_layers),
                "calibration_metrics": list(self.calibration_metrics),
                "metrics_path": metrics_path,
                "calibration_split": calibration_split,
            }
        )

    @property
    def name(self) -> str:
        return "calibratedskip"

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
        return tuple(sorted(set(layers)))

    def get_skipped_layer_indices(self, num_layers: int) -> Tuple[int, ...]:
        invalid_layers = [layer for layer in self.skip_layers if layer > num_layers]
        if invalid_layers:
            raise ValueError(
                f"skip_layers must be within [1, {num_layers}], got {invalid_layers}"
            )
        return tuple(layer - 1 for layer in self.skip_layers)

    def is_noop(self, num_layers: int) -> bool:
        return not self.get_skipped_layer_indices(num_layers)

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
