"""Reversible, physical transformer-block pruning for Hugging Face models.

The layer-selection baselines operate on *original*, zero-based block IDs.
This module turns such a selection into a physically shorter ``ModuleList``
for the duration of an evaluation.  Unlike forwarding through no-op blocks,
this exercises the same decoder depth (and cache layout) as an exported
pruned checkpoint.

Only decoder-style models whose wrapper resolves a ``torch.nn.ModuleList``
are supported.  The wrapper is expected to follow :class:`HFModel`'s private
bookkeeping conventions (``_num_layers`` and, when present, the strategy
attributes).  Ambiguous custom containers and inconsistent layer counts are
rejected rather than modified heuristically.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import torch


_CONFIG_LAYER_COUNT_FIELDS = (
    "num_hidden_layers",
    "n_layer",
    "n_layers",
    "num_layers",
    "decoder_layers",
    "num_decoder_layers",
)
_NESTED_CONFIG_FIELDS = (
    "text_config",
    "language_config",
    "llm_config",
    "decoder_config",
    "decoder",
)
_MODEL_CONFIG_HOLDERS = ("model", "transformer", "gpt_neox", "decoder")
_WRAPPER_STRATEGY_FIELDS = (
    "strategy",
    "_bypass_layer_indices",
    "_use_strategy",
    "_transformer_layers",
)
_ACTIVE_FIELD = "_structural_pruning_active"
_MISSING = object()


@dataclass(frozen=True)
class StructuralPruningInfo:
    """The original-to-compact layer mapping active inside the context."""

    original_num_layers: int
    removed_layer_ids: tuple[int, ...]
    kept_layer_ids: tuple[int, ...]

    @property
    def num_layers(self) -> int:
        """Number of transformer blocks in the compact model."""

        return len(self.kept_layer_ids)

    @property
    def original_to_compact(self) -> tuple[int | None, ...]:
        """Map each original block ID to its compact ID, or ``None``."""

        compact_ids = {
            original: compact
            for compact, original in enumerate(self.kept_layer_ids)
        }
        return tuple(
            compact_ids.get(original)
            for original in range(self.original_num_layers)
        )


@dataclass(frozen=True)
class _AttributeState:
    owner: Any
    name: str
    value: Any


def _snapshot_attribute(owner: Any, name: str) -> _AttributeState:
    try:
        value = getattr(owner, name)
    except AttributeError:
        value = _MISSING
    return _AttributeState(owner=owner, name=name, value=value)


def _restore_attribute(state: _AttributeState) -> None:
    if state.value is _MISSING:
        try:
            delattr(state.owner, state.name)
        except AttributeError:
            pass
        return
    setattr(state.owner, state.name, state.value)


def _normalise_removed_ids(
    removed_layer_ids: Iterable[int],
    *,
    num_layers: int,
) -> tuple[int, ...]:
    if isinstance(removed_layer_ids, (str, bytes)):
        raise TypeError("removed_layer_ids must be an iterable of integers")
    try:
        raw_ids = tuple(removed_layer_ids)
    except TypeError as exc:
        raise TypeError("removed_layer_ids must be an iterable of integers") from exc

    for layer_id in raw_ids:
        if isinstance(layer_id, bool) or not isinstance(layer_id, int):
            raise TypeError("removed layer IDs must be integers (bool is not accepted)")
        if layer_id < 0 or layer_id >= num_layers:
            raise ValueError(
                f"removed layer ID {layer_id} is outside [0, {num_layers})"
            )
    if len(set(raw_ids)) != len(raw_ids):
        raise ValueError("removed layer IDs must be unique")

    removed = tuple(sorted(raw_ids))
    if len(removed) == num_layers:
        raise ValueError(
            "structural pruning must retain at least one transformer layer"
        )
    return removed


def _config_objects(model: torch.nn.Module) -> tuple[Any, ...]:
    root_config = getattr(model, "config", None)
    if root_config is None:
        raise TypeError("wrapper.model must expose a Hugging Face-style config")

    configs = [root_config]
    for field in _NESTED_CONFIG_FIELDS:
        nested = getattr(root_config, field, None)
        if nested is not None:
            configs.append(nested)

    for holder_name in _MODEL_CONFIG_HOLDERS:
        holder = getattr(model, holder_name, None)
        holder_config = getattr(holder, "config", None) if holder is not None else None
        if holder_config is not None:
            configs.append(holder_config)

    unique = []
    seen = set()
    for config in configs:
        if id(config) not in seen:
            seen.add(id(config))
            unique.append(config)
    return tuple(unique)


def _config_layer_count_states(
    model: torch.nn.Module,
    *,
    original_num_layers: int,
) -> tuple[_AttributeState, ...]:
    configs = _config_objects(model)
    states = []
    root_has_count = False

    for config_index, config in enumerate(configs):
        for field in _CONFIG_LAYER_COUNT_FIELDS:
            try:
                value = getattr(config, field)
            except AttributeError:
                continue
            except Exception as exc:
                raise TypeError(f"could not inspect config field {field!r}") from exc

            # Only mutate fields whose current value proves that they describe
            # this decoder stack.  For example, an unrelated encoder count is
            # intentionally left alone.
            if (
                isinstance(value, int)
                and not isinstance(value, bool)
                and value == original_num_layers
            ):
                states.append(_AttributeState(config, field, value))
                if config_index == 0:
                    root_has_count = True

    if not root_has_count:
        fields = ", ".join(_CONFIG_LAYER_COUNT_FIELDS)
        raise ValueError(
            "wrapper.model.config has no supported layer-count field equal to "
            f"the resolved depth ({original_num_layers}); supported fields: {fields}"
        )
    return tuple(states)


def _indexed_module_states(
    original_layers: tuple[torch.nn.Module, ...],
    kept_layer_ids: tuple[int, ...],
) -> tuple[tuple[_AttributeState, int], ...]:
    """Find zero-based ``layer_idx`` attributes and their compact values."""

    indexed: dict[int, tuple[_AttributeState, int]] = {}
    for compact_id, original_id in enumerate(kept_layer_ids):
        layer = original_layers[original_id]
        for module in layer.modules():
            try:
                old_index = getattr(module, "layer_idx")
            except AttributeError:
                continue
            except Exception as exc:
                raise TypeError("could not inspect a module's layer_idx") from exc

            # ``None`` is frequently used by attention implementations that
            # deliberately opt out of indexed caching.  Preserve that mode.
            if old_index is None:
                continue
            if isinstance(old_index, bool) or not isinstance(old_index, int):
                raise TypeError("module layer_idx values must be integers or None")
            if old_index != original_id:
                raise ValueError(
                    "cannot safely renumber a module whose layer_idx does not "
                    "match its original zero-based block ID "
                    f"({old_index} != {original_id})"
                )

            key = id(module)
            previous = indexed.get(key)
            if previous is not None and previous[1] != compact_id:
                raise ValueError(
                    "an indexed submodule is shared by multiple transformer blocks; "
                    "safe structural renumbering is ambiguous"
                )
            indexed[key] = (_AttributeState(module, "layer_idx", old_index), compact_id)
    return tuple(indexed.values())


def _replacement_registry(
    original_registry: dict[str, torch.nn.Module],
    kept_layers: tuple[torch.nn.Module, ...],
) -> dict[str, torch.nn.Module]:
    items = ((str(index), layer) for index, layer in enumerate(kept_layers))
    try:
        return original_registry.__class__(items)
    except (TypeError, ValueError):
        # PyTorch currently uses either dict or OrderedDict.  The fallback is
        # mainly for subclasses with a non-standard constructor.
        return OrderedDict(
            (str(index), layer) for index, layer in enumerate(kept_layers)
        )


@contextmanager
def structurally_pruned(
    wrapper: Any,
    removed_layer_ids: Iterable[int],
) -> Iterator[StructuralPruningInfo]:
    """Temporarily execute ``wrapper.model`` with transformer blocks removed.

    Args:
        wrapper: An ``HFModel``-like wrapper.  It must expose ``model``,
            ``_num_layers``, and ``_resolve_transformer_layers()``.
        removed_layer_ids: Original-model, zero-based block IDs.  IDs may be in
            any order but must be unique and in range.

    Yields:
        A :class:`StructuralPruningInfo` describing the active compact mapping.

    The existing layer ``ModuleList`` is shortened in place, so all references
    held by the model continue to point at the active container.  Known
    ``HFModel`` strategy bookkeeping is neutralized while the compact model is
    active and restored exactly on exit.  Configuration counts and integer,
    zero-based ``layer_idx`` attributes are likewise restored, including when
    evaluation raises an exception.

    This is a reversible evaluation utility: removed modules remain referenced
    for restoration, so it provides real depth/cache-speed behavior but does
    not claim to release their parameter memory.  Exporting a smaller
    checkpoint should copy the compact state into a separate model.
    """

    if bool(getattr(wrapper, _ACTIVE_FIELD, False)):
        raise RuntimeError("nested structural-pruning contexts are not supported")

    model = getattr(wrapper, "model", None)
    if not isinstance(model, torch.nn.Module):
        raise TypeError("wrapper.model must be a torch.nn.Module")
    if not hasattr(wrapper, "_num_layers"):
        raise TypeError(
            "wrapper must expose HFModel-compatible _num_layers bookkeeping"
        )
    original_num_layers = getattr(wrapper, "_num_layers")
    if (
        isinstance(original_num_layers, bool)
        or not isinstance(original_num_layers, int)
        or original_num_layers <= 0
    ):
        raise ValueError("wrapper._num_layers must be a positive integer")

    resolver = getattr(wrapper, "_resolve_transformer_layers", None)
    if not callable(resolver):
        raise TypeError("wrapper must define _resolve_transformer_layers()")
    try:
        layers = resolver()
    except Exception as exc:
        raise ValueError("could not resolve the transformer's layer container") from exc
    if not isinstance(layers, torch.nn.ModuleList):
        raise TypeError(
            "structural pruning supports only torch.nn.ModuleList layer containers"
        )
    if len(layers) != original_num_layers:
        raise ValueError(
            "resolved ModuleList length does not match wrapper._num_layers "
            f"({len(layers)} != {original_num_layers})"
        )

    original_layers = tuple(layers)
    if not all(isinstance(layer, torch.nn.Module) for layer in original_layers):
        raise TypeError("every transformer layer must be a torch.nn.Module")
    removed = _normalise_removed_ids(
        removed_layer_ids,
        num_layers=original_num_layers,
    )
    removed_set = set(removed)
    kept_ids = tuple(
        layer_id
        for layer_id in range(original_num_layers)
        if layer_id not in removed_set
    )
    kept_layers = tuple(original_layers[layer_id] for layer_id in kept_ids)
    info = StructuralPruningInfo(
        original_num_layers=original_num_layers,
        removed_layer_ids=removed,
        kept_layer_ids=kept_ids,
    )

    config_states = _config_layer_count_states(
        model,
        original_num_layers=original_num_layers,
    )
    indexed_states = _indexed_module_states(original_layers, kept_ids)
    wrapper_states = tuple(
        _snapshot_attribute(wrapper, field)
        for field in (_ACTIVE_FIELD, "_num_layers", *_WRAPPER_STRATEGY_FIELDS)
    )
    original_registry = layers._modules
    compact_registry = _replacement_registry(original_registry, kept_layers)

    caught_error: BaseException | None = None
    try:
        setattr(wrapper, _ACTIVE_FIELD, True)
        if hasattr(wrapper, "strategy"):
            wrapper.strategy = None
        if hasattr(wrapper, "_bypass_layer_indices"):
            wrapper._bypass_layer_indices = ()
        if hasattr(wrapper, "_use_strategy"):
            wrapper._use_strategy = False
        if hasattr(wrapper, "_transformer_layers"):
            wrapper._transformer_layers = None

        layers._modules = compact_registry
        wrapper._num_layers = info.num_layers
        for state in config_states:
            setattr(state.owner, state.name, info.num_layers)
        for state, compact_id in indexed_states:
            setattr(state.owner, state.name, compact_id)

        yield info
    except BaseException as exc:
        caught_error = exc
        raise
    finally:
        restore_errors = []

        for state, _ in reversed(indexed_states):
            try:
                _restore_attribute(state)
            except Exception as exc:  # pragma: no cover - hostile custom setters
                restore_errors.append(exc)
        for state in reversed(config_states):
            try:
                _restore_attribute(state)
            except Exception as exc:  # pragma: no cover - hostile custom setters
                restore_errors.append(exc)
        try:
            layers._modules = original_registry
        except Exception as exc:  # pragma: no cover - ModuleList internals changed
            restore_errors.append(exc)
        for state in reversed(wrapper_states):
            try:
                _restore_attribute(state)
            except Exception as exc:  # pragma: no cover - hostile custom setters
                restore_errors.append(exc)

        if restore_errors:
            message = (
                "structural pruning could not completely restore model state: "
                + "; ".join(str(error) for error in restore_errors)
            )
            if caught_error is not None:
                caught_error.add_note(message)
            else:
                raise RuntimeError(message) from restore_errors[0]


__all__ = ["StructuralPruningInfo", "structurally_pruned"]
