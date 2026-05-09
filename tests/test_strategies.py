"""Tests for layer-skipping strategy implementations."""

import pytest
import torch

from evaluation.strategies import get_strategy, STRATEGY_REGISTRY
from evaluation.strategies.layerskip import LayerSkipStrategy
from evaluation.strategies.caml import CAMLStrategy
from evaluation.strategies.gateskip import GateSkipStrategy
from evaluation.strategies.manualskip import ManualSkipStrategy


# ------------------------------------------------------------------ #
# Fixtures                                                             #
# ------------------------------------------------------------------ #

def _make_hidden_states(num_layers: int, batch: int = 2, seq: int = 8, hidden: int = 64):
    """Create synthetic hidden-state tuples as would be returned by a forward pass."""
    states = []
    h = torch.randn(batch, seq, hidden)
    for i in range(num_layers + 1):  # +1 for embedding output
        # Each subsequent layer subtly transforms h
        h = h + 0.01 * torch.randn_like(h)
        states.append(h.clone())
    return tuple(states)


def _make_lm_head_and_norm(hidden: int = 64, vocab: int = 128):
    """Return a minimal LM head and layer norm for confidence-based strategies."""
    lm_head = torch.nn.Linear(hidden, vocab, bias=False)
    layer_norm = torch.nn.LayerNorm(hidden)
    return lm_head, layer_norm


# ------------------------------------------------------------------ #
# Registry                                                             #
# ------------------------------------------------------------------ #

def test_strategy_registry_keys():
    assert "none" in STRATEGY_REGISTRY
    assert "layerskip" in STRATEGY_REGISTRY
    assert "caml" in STRATEGY_REGISTRY
    assert "gateskip" in STRATEGY_REGISTRY
    assert "manualskip" in STRATEGY_REGISTRY


def test_get_strategy_none():
    assert get_strategy("none") is None


def test_get_strategy_unknown():
    with pytest.raises(ValueError, match="Unknown strategy"):
        get_strategy("nonexistent_strategy")


# ------------------------------------------------------------------ #
# LayerSkipStrategy                                                    #
# ------------------------------------------------------------------ #

class TestLayerSkipStrategy:

    def test_exit_layer_default(self):
        s = LayerSkipStrategy(exit_ratio=0.75, min_layers=4)
        assert s.compute_exit_layer(32) == 24

    def test_exit_layer_min_clamp(self):
        s = LayerSkipStrategy(exit_ratio=0.01, min_layers=6)
        # 0.01 * 32 = 0 → clamped to min_layers = 6
        assert s.compute_exit_layer(32) == 6

    def test_exit_layer_max_clamp(self):
        s = LayerSkipStrategy(exit_ratio=1.0, min_layers=4)
        # Should not exceed num_layers
        assert s.compute_exit_layer(32) == 32

    def test_select_exit_layer_returns_int(self):
        s = LayerSkipStrategy(exit_ratio=0.75)
        hs = _make_hidden_states(32)
        layer_idx = s.select_exit_layer(hs, 32)
        assert isinstance(layer_idx, int)
        assert 1 <= layer_idx <= 32

    def test_get_exit_hidden_state_shape(self):
        batch, seq, hidden = 2, 8, 64
        s = LayerSkipStrategy(exit_ratio=0.5)
        hs = _make_hidden_states(16, batch=batch, seq=seq, hidden=hidden)
        result = s.get_exit_hidden_state(hs, 16)
        assert result.shape == (batch, seq, hidden)

    def test_invalid_exit_ratio(self):
        with pytest.raises(ValueError):
            LayerSkipStrategy(exit_ratio=0.0)
        with pytest.raises(ValueError):
            LayerSkipStrategy(exit_ratio=1.5)

    def test_invalid_min_layers(self):
        with pytest.raises(ValueError):
            LayerSkipStrategy(min_layers=0)

    def test_strategy_name(self):
        assert LayerSkipStrategy().name == "layerskip"

    def test_repr_contains_config(self):
        s = LayerSkipStrategy(exit_ratio=0.8)
        assert "LayerSkipStrategy" in repr(s)

    @pytest.mark.parametrize("exit_ratio", [0.25, 0.5, 0.75, 1.0])
    def test_exit_layer_monotone(self, exit_ratio):
        """Higher exit_ratio → later (or equal) exit layer."""
        s1 = LayerSkipStrategy(exit_ratio=exit_ratio, min_layers=1)
        s2 = LayerSkipStrategy(exit_ratio=min(exit_ratio + 0.1, 1.0), min_layers=1)
        assert s1.compute_exit_layer(32) <= s2.compute_exit_layer(32)

    def test_get_strategy_factory(self):
        s = get_strategy("layerskip", exit_ratio=0.6)
        assert isinstance(s, LayerSkipStrategy)
        assert s.exit_ratio == 0.6


# ------------------------------------------------------------------ #
# CAMLStrategy                                                         #
# ------------------------------------------------------------------ #

class TestCAMLStrategy:

    def test_select_exit_layer_without_lm_head(self):
        """Falls back to full model when lm_head/layer_norm are not provided."""
        s = CAMLStrategy(confidence_threshold=0.9, min_layers=4)
        hs = _make_hidden_states(16)
        idx = s.select_exit_layer(hs, 16, lm_head=None, layer_norm=None)
        assert idx == 16

    def test_select_exit_layer_with_high_threshold(self):
        """Very high threshold → model uses all layers."""
        lm_head, layer_norm = _make_lm_head_and_norm()
        s = CAMLStrategy(confidence_threshold=0.9999, min_layers=1)
        hs = _make_hidden_states(16)
        idx = s.select_exit_layer(hs, 16, lm_head=lm_head, layer_norm=layer_norm)
        assert 1 <= idx <= 16

    def test_select_exit_layer_with_low_threshold(self):
        """Very low threshold → exits early."""
        lm_head, layer_norm = _make_lm_head_and_norm()
        # Using a very low threshold, model should exit early
        s = CAMLStrategy(confidence_threshold=0.01, min_layers=1)
        hs = _make_hidden_states(16)
        idx = s.select_exit_layer(hs, 16, lm_head=lm_head, layer_norm=layer_norm)
        assert idx <= 16

    def test_exit_layer_respects_min_layers(self):
        """Exit layer should always be >= min_layers."""
        lm_head, layer_norm = _make_lm_head_and_norm()
        s = CAMLStrategy(confidence_threshold=0.0001, min_layers=8)
        hs = _make_hidden_states(16)
        idx = s.select_exit_layer(hs, 16, lm_head=lm_head, layer_norm=layer_norm)
        assert idx >= 8

    def test_invalid_confidence_threshold(self):
        with pytest.raises(ValueError):
            CAMLStrategy(confidence_threshold=0.0)
        with pytest.raises(ValueError):
            CAMLStrategy(confidence_threshold=1.5)

    def test_invalid_check_every(self):
        with pytest.raises(ValueError):
            CAMLStrategy(check_every=0)

    def test_strategy_name(self):
        assert CAMLStrategy().name == "caml"

    def test_compute_confidence_range(self):
        lm_head, layer_norm = _make_lm_head_and_norm()
        s = CAMLStrategy()
        hs = _make_hidden_states(8)
        conf = s._compute_confidence(hs[4], lm_head, layer_norm)
        assert 0.0 <= conf <= 1.0

    def test_check_every_skips_layers(self):
        """check_every=4 should only check at multiples of 4."""
        lm_head, layer_norm = _make_lm_head_and_norm()
        s = CAMLStrategy(confidence_threshold=0.5, min_layers=4, check_every=4)
        hs = _make_hidden_states(16)
        idx = s.select_exit_layer(hs, 16, lm_head=lm_head, layer_norm=layer_norm)
        assert idx in {4, 8, 12, 16}

    def test_get_strategy_factory(self):
        s = get_strategy("caml", confidence_threshold=0.85)
        assert isinstance(s, CAMLStrategy)
        assert s.confidence_threshold == 0.85


# ------------------------------------------------------------------ #
# GateSkipStrategy                                                     #
# ------------------------------------------------------------------ #

class TestGateSkipStrategy:

    def test_select_exit_layer_range(self):
        s = GateSkipStrategy(gate_threshold=0.01, skip_budget=0.3, min_layers=4)
        hs = _make_hidden_states(32)
        idx = s.select_exit_layer(hs, 32)
        assert 1 <= idx <= 32

    def test_zero_budget_no_skip(self):
        """skip_budget=0 means no layers are ever skipped."""
        s = GateSkipStrategy(gate_threshold=1e6, skip_budget=0.0, min_layers=1)
        hs = _make_hidden_states(16)
        idx = s.select_exit_layer(hs, 16)
        # With zero budget, last_active ends at num_layers
        assert idx == 16

    def test_gate_values_computed_correctly(self):
        s = GateSkipStrategy()
        hs = _make_hidden_states(8)
        gate_values = s._compute_gate_values(hs, 8)
        assert len(gate_values) == 8
        assert all(isinstance(v, float) for v in gate_values)
        assert all(v >= 0 for v in gate_values)

    def test_invalid_gate_threshold(self):
        with pytest.raises(ValueError):
            GateSkipStrategy(gate_threshold=-0.1)

    def test_invalid_skip_budget(self):
        with pytest.raises(ValueError):
            GateSkipStrategy(skip_budget=1.5)
        with pytest.raises(ValueError):
            GateSkipStrategy(skip_budget=-0.1)

    def test_invalid_min_layers(self):
        with pytest.raises(ValueError):
            GateSkipStrategy(min_layers=0)

    def test_strategy_name(self):
        assert GateSkipStrategy().name == "gateskip"

    def test_high_threshold_exits_last_layer(self):
        """If gate_threshold is very high, all layers may be skippable (up to budget)."""
        s = GateSkipStrategy(gate_threshold=1e6, skip_budget=1.0, min_layers=1)
        hs = _make_hidden_states(16)
        idx = s.select_exit_layer(hs, 16)
        # All layers pass the threshold → last_active remains 16 (num_layers default)
        assert 1 <= idx <= 16

    def test_get_strategy_factory(self):
        s = get_strategy("gateskip", gate_threshold=0.05, skip_budget=0.2)
        assert isinstance(s, GateSkipStrategy)
        assert s.gate_threshold == 0.05
        assert s.skip_budget == 0.2


# ------------------------------------------------------------------ #
# ManualSkipStrategy                                                   #
# ------------------------------------------------------------------ #

class TestManualSkipStrategy:

    def test_normalizes_unique_sorted_layers(self):
        s = ManualSkipStrategy(skip_layers=[4, 2, 4, 1])
        assert s.skip_layers == (1, 2, 4)
        assert s.config == {"skip_layers": [1, 2, 4]}

    def test_get_skipped_layer_indices_zero_based(self):
        s = ManualSkipStrategy(skip_layers=[1, 3, 8])
        assert s.get_skipped_layer_indices(8) == (0, 2, 7)

    def test_select_exit_layer_uses_full_depth(self):
        s = ManualSkipStrategy(skip_layers=[2, 4])
        hs = _make_hidden_states(8)
        assert s.select_exit_layer(hs, 8) == 8

    def test_invalid_empty_layers(self):
        with pytest.raises(ValueError):
            ManualSkipStrategy(skip_layers=[])

    def test_invalid_non_positive_layers(self):
        with pytest.raises(ValueError):
            ManualSkipStrategy(skip_layers=[0])
        with pytest.raises(ValueError):
            ManualSkipStrategy(skip_layers=[-1])

    def test_invalid_out_of_range_layers(self):
        s = ManualSkipStrategy(skip_layers=[2, 9])
        with pytest.raises(ValueError, match="within"):
            s.get_skipped_layer_indices(8)

    def test_strategy_name(self):
        assert ManualSkipStrategy(skip_layers=[2]).name == "manualskip"

    def test_get_strategy_factory(self):
        s = get_strategy("manualskip", skip_layers=[2, 4])
        assert isinstance(s, ManualSkipStrategy)
        assert s.skip_layers == (2, 4)


# ------------------------------------------------------------------ #
# Cross-strategy consistency                                           #
# ------------------------------------------------------------------ #

class TestStrategyCrossConsistency:

    def test_all_strategies_return_valid_index(self):
        """All strategies must return an index in [1, num_layers]."""
        num_layers = 16
        hs = _make_hidden_states(num_layers)
        lm_head, layer_norm = _make_lm_head_and_norm()

        strategies = [
            LayerSkipStrategy(exit_ratio=0.75),
            CAMLStrategy(confidence_threshold=0.9),
            GateSkipStrategy(gate_threshold=0.01),
            ManualSkipStrategy(skip_layers=[2, 4]),
        ]
        for strat in strategies:
            idx = strat.select_exit_layer(hs, num_layers, lm_head=lm_head, layer_norm=layer_norm)
            assert 1 <= idx <= num_layers, f"{strat.name} returned out-of-range index {idx}"

    def test_all_strategies_return_correct_hidden_shape(self):
        batch, seq, hidden = 2, 8, 64
        num_layers = 16
        hs = _make_hidden_states(num_layers, batch=batch, seq=seq, hidden=hidden)
        lm_head, layer_norm = _make_lm_head_and_norm(hidden=hidden)

        strategies = [
            LayerSkipStrategy(),
            CAMLStrategy(),
            GateSkipStrategy(),
            ManualSkipStrategy(skip_layers=[2, 4]),
        ]
        for strat in strategies:
            h = strat.get_exit_hidden_state(hs, num_layers, lm_head=lm_head, layer_norm=layer_norm)
            assert h.shape == (batch, seq, hidden), (
                f"{strat.name}: expected shape {(batch, seq, hidden)}, got {h.shape}"
            )
