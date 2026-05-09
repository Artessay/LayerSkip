"""Strategy registry and convenience helpers."""

from evaluation.strategies.base_strategy import BaseLayerSkipStrategy
from evaluation.strategies.layerskip import LayerSkipStrategy
from evaluation.strategies.caml import CAMLStrategy
from evaluation.strategies.gateskip import GateSkipStrategy
from evaluation.strategies.calibratedskip import CalibratedSkipStrategy
from evaluation.strategies.manualskip import ManualSkipStrategy

STRATEGY_REGISTRY = {
    "none": None,
    "layerskip": LayerSkipStrategy,
    "caml": CAMLStrategy,
    "gateskip": GateSkipStrategy,
    "calibratedskip": CalibratedSkipStrategy,
    "manualskip": ManualSkipStrategy,
}

_STRATEGY_DEFAULTS = {
    "layerskip": {"exit_ratio": 0.75, "min_layers": 4},
    "caml": {"confidence_threshold": 0.9, "min_layers": 4, "check_every": 1},
    "gateskip": {"gate_threshold": 0.01, "skip_budget": 0.3, "min_layers": 4},
    "calibratedskip": {
        "skip_layers": [],
        "calibration_metrics": ["activation_ratio", "gradient_trace"],
        "metrics_path": None,
        "calibration_split": None,
    },
}


def get_strategy(name: str, **kwargs) -> "BaseLayerSkipStrategy | None":
    """
    Instantiate a strategy by name with optional overrides.

    Args:
        name: One of ``"none"``, ``"layerskip"``, ``"caml"``, ``"gateskip"``.
        **kwargs: Keyword arguments forwarded to the strategy constructor,
            overriding the defaults.

    Returns:
        A strategy instance, or ``None`` when ``name == "none"``.

    Raises:
        ValueError: If ``name`` is not registered.
    """
    name_lower = name.lower()
    if name_lower not in STRATEGY_REGISTRY:
        raise ValueError(
            f"Unknown strategy '{name}'. "
            f"Available: {list(STRATEGY_REGISTRY.keys())}"
        )
    cls = STRATEGY_REGISTRY[name_lower]
    if cls is None:
        return None

    params = dict(_STRATEGY_DEFAULTS.get(name_lower, {}))
    params.update(kwargs)
    return cls(**params)


__all__ = [
    "BaseLayerSkipStrategy",
    "LayerSkipStrategy",
    "CAMLStrategy",
    "GateSkipStrategy",
    "CalibratedSkipStrategy",
    "ManualSkipStrategy",
    "STRATEGY_REGISTRY",
    "get_strategy",
]
