"""CPU-only tests for reversible physical transformer-layer pruning."""

from types import SimpleNamespace

import pytest
import torch

from evaluation.pruning.structural import structurally_pruned


class _IndexedAttention(torch.nn.Module):
    def __init__(self, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx


class _Block(torch.nn.Module):
    def __init__(self, layer_idx):
        super().__init__()
        self.self_attn = _IndexedAttention(layer_idx)

    def forward(self, hidden):
        return hidden + self.self_attn.layer_idx + 1


class _ToyBackbone(torch.nn.Module):
    def __init__(self, num_layers):
        super().__init__()
        self.layers = torch.nn.ModuleList(
            [_Block(layer_idx) for layer_idx in range(num_layers)]
        )

    def forward(self, hidden):
        for layer in self.layers:
            hidden = layer(hidden)
        return hidden


class _ToyCausalLM(torch.nn.Module):
    def __init__(self, num_layers=4):
        super().__init__()
        self.model = _ToyBackbone(num_layers)
        self.config = SimpleNamespace(
            num_hidden_layers=num_layers,
            n_layer=num_layers,
            # A count that demonstrably belongs to something else must remain
            # untouched.
            num_layers=99,
            text_config=SimpleNamespace(num_hidden_layers=num_layers),
        )

    def forward(self, hidden):
        return self.model(hidden)


class _ToyWrapper:
    def __init__(self, num_layers=4):
        self.model = _ToyCausalLM(num_layers)
        self._num_layers = num_layers
        self.strategy = object()
        self._bypass_layer_indices = (1,)
        self._use_strategy = True
        self._transformer_layers = self.model.model.layers

    @property
    def num_layers(self):
        return self._num_layers

    def _resolve_transformer_layers(self):
        if len(self.model.model.layers) != self._num_layers:
            raise ValueError("inconsistent test wrapper")
        return self.model.model.layers


def test_structurally_pruned_compacts_and_restores_all_state():
    wrapper = _ToyWrapper()
    container = wrapper.model.model.layers
    registry = container._modules
    original_layers = tuple(container)
    original_strategy = wrapper.strategy
    original_transformer_layers = wrapper._transformer_layers

    with structurally_pruned(wrapper, [3, 1]) as pruning:
        assert pruning.original_num_layers == 4
        assert pruning.removed_layer_ids == (1, 3)
        assert pruning.kept_layer_ids == (0, 2)
        assert pruning.original_to_compact == (0, None, 1, None)
        assert pruning.num_layers == 2

        assert wrapper.model.model.layers is container
        assert wrapper._resolve_transformer_layers() is container
        assert tuple(container) == (original_layers[0], original_layers[2])
        assert wrapper.num_layers == 2
        assert wrapper.model.config.num_hidden_layers == 2
        assert wrapper.model.config.n_layer == 2
        assert wrapper.model.config.text_config.num_hidden_layers == 2
        assert wrapper.model.config.num_layers == 99
        assert [layer.self_attn.layer_idx for layer in container] == [0, 1]
        assert wrapper.strategy is None
        assert wrapper._bypass_layer_indices == ()
        assert wrapper._use_strategy is False
        assert wrapper._transformer_layers is None
        assert wrapper.model(torch.tensor(0)).item() == 3

    assert wrapper.model.model.layers is container
    assert container._modules is registry
    assert tuple(container) == original_layers
    assert wrapper.num_layers == 4
    assert wrapper.model.config.num_hidden_layers == 4
    assert wrapper.model.config.n_layer == 4
    assert wrapper.model.config.text_config.num_hidden_layers == 4
    assert [layer.self_attn.layer_idx for layer in container] == [0, 1, 2, 3]
    assert wrapper.strategy is original_strategy
    assert wrapper._bypass_layer_indices == (1,)
    assert wrapper._use_strategy is True
    assert wrapper._transformer_layers is original_transformer_layers
    assert not hasattr(wrapper, "_structural_pruning_active")


def test_structurally_pruned_restores_after_evaluation_error():
    wrapper = _ToyWrapper()
    original_layers = tuple(wrapper.model.model.layers)
    original_strategy = wrapper.strategy

    with pytest.raises(RuntimeError, match="evaluation failed"):
        with structurally_pruned(wrapper, [0, 2]):
            assert wrapper.num_layers == 2
            raise RuntimeError("evaluation failed")

    assert tuple(wrapper.model.model.layers) == original_layers
    assert wrapper.num_layers == 4
    assert wrapper.strategy is original_strategy
    assert [
        layer.self_attn.layer_idx for layer in wrapper.model.model.layers
    ] == [0, 1, 2, 3]


def test_structurally_pruned_preserves_none_layer_index():
    wrapper = _ToyWrapper()
    wrapper.model.model.layers[2].self_attn.layer_idx = None

    with structurally_pruned(wrapper, [1, 3]):
        assert wrapper.model.model.layers[1].self_attn.layer_idx is None

    assert wrapper.model.model.layers[2].self_attn.layer_idx is None


def test_structurally_pruned_rejects_nested_context_without_disturbing_outer():
    wrapper = _ToyWrapper()

    with structurally_pruned(wrapper, [1]):
        compact_layers = tuple(wrapper.model.model.layers)
        with pytest.raises(RuntimeError, match="nested"):
            with structurally_pruned(wrapper, [2]):
                pass
        assert tuple(wrapper.model.model.layers) == compact_layers
        assert wrapper.num_layers == 3

    assert wrapper.num_layers == 4


@pytest.mark.parametrize(
    "removed,error,match",
    [
        ([1, 1], ValueError, "unique"),
        ([-1], ValueError, "outside"),
        ([4], ValueError, "outside"),
        ([True], TypeError, "integers"),
        ([1.0], TypeError, "integers"),
        ("1", TypeError, "iterable"),
        ([0, 1, 2, 3], ValueError, "at least one"),
    ],
)
def test_structurally_pruned_validates_original_layer_ids(removed, error, match):
    wrapper = _ToyWrapper()

    with pytest.raises(error, match=match):
        with structurally_pruned(wrapper, removed):
            pass

    assert wrapper.num_layers == 4
    assert len(wrapper.model.model.layers) == 4


def test_structurally_pruned_rejects_non_module_list_container():
    wrapper = _ToyWrapper()
    wrapper.model.model.layers = torch.nn.Sequential(
        *tuple(wrapper.model.model.layers)
    )

    with pytest.raises(TypeError, match="ModuleList"):
        with structurally_pruned(wrapper, [1]):
            pass


def test_structurally_pruned_rejects_inconsistent_config_count():
    wrapper = _ToyWrapper()
    wrapper.model.config.num_hidden_layers = 7
    wrapper.model.config.n_layer = 7

    with pytest.raises(ValueError, match="no supported layer-count field"):
        with structurally_pruned(wrapper, [1]):
            pass

    assert len(wrapper.model.model.layers) == 4
    assert wrapper.num_layers == 4


def test_structurally_pruned_rejects_non_zero_based_layer_index():
    wrapper = _ToyWrapper()
    wrapper.model.model.layers[2].self_attn.layer_idx = 3

    with pytest.raises(ValueError, match="zero-based"):
        with structurally_pruned(wrapper, [1]):
            pass

    assert len(wrapper.model.model.layers) == 4
    assert wrapper.num_layers == 4
