"""Versioned, resumable persistence for pruning-search traces."""

from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
from typing import Any, Dict, Mapping, Optional, Union

from evaluation.utils.result_io import config_hash, model_basename, slugify, to_jsonable


SEARCH_SCHEMA_VERSION = 1


def search_trace_path(
    results_dir: Union[str, Path],
    *,
    model_name: str,
    method: str,
    search_config: Mapping[str, Any],
    task_name: Optional[str] = None,
) -> Path:
    """Return a stable trace path for one exact search configuration."""

    scope = task_name or "global"
    identity = {
        "schema_version": SEARCH_SCHEMA_VERSION,
        "model": str(model_name),
        "method": method,
        "scope": scope,
        "search_config": to_jsonable(dict(search_config)),
    }
    return (
        Path(results_dir)
        / slugify(model_basename(model_name))
        / "pruning"
        / slugify(method)
        / slugify(scope)
        / f"{config_hash(identity)}.json"
    )


def make_search_envelope(
    *,
    model_name: str,
    method: str,
    search_config: Mapping[str, Any],
    algorithm_trace: Mapping[str, Any],
    provenance: Mapping[str, Any],
    task_name: Optional[str] = None,
) -> Dict[str, Any]:
    return {
        "schema_version": SEARCH_SCHEMA_VERSION,
        "model": str(model_name),
        "method": method,
        "scope": task_name or "global",
        "search_config": to_jsonable(dict(search_config)),
        "provenance": to_jsonable(dict(provenance)),
        "algorithm_trace": to_jsonable(dict(algorithm_trace)),
    }


def load_search_envelope(
    path: Union[str, Path],
    *,
    model_name: str,
    method: str,
    search_config: Mapping[str, Any],
    task_name: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Load a matching trace, rejecting stale or manually mixed checkpoints."""

    input_path = Path(path)
    if not input_path.exists():
        return None
    with input_path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Search trace {input_path} must contain a JSON object")

    expected = {
        "schema_version": SEARCH_SCHEMA_VERSION,
        "model": str(model_name),
        "method": method,
        "scope": task_name or "global",
        "search_config": to_jsonable(dict(search_config)),
    }
    for field, expected_value in expected.items():
        if value.get(field) != expected_value:
            raise ValueError(
                f"Search trace {input_path} has mismatched {field}: "
                f"{value.get(field)!r} != {expected_value!r}"
            )
    if not isinstance(value.get("algorithm_trace"), dict):
        raise ValueError(f"Search trace {input_path} has no algorithm_trace object")
    return value


def save_search_envelope(
    envelope: Mapping[str, Any],
    path: Union[str, Path],
) -> None:
    """Atomically replace a trace so interrupted writes remain recoverable."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = to_jsonable(dict(envelope))
    temporary_name: Optional[str] = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=output_path.parent,
            prefix=f".{output_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_name = handle.name
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, output_path)
        temporary_name = None
    finally:
        if temporary_name is not None:
            try:
                os.unlink(temporary_name)
            except FileNotFoundError:
                pass


__all__ = [
    "SEARCH_SCHEMA_VERSION",
    "load_search_envelope",
    "make_search_envelope",
    "save_search_envelope",
    "search_trace_path",
]
