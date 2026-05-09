"""Utilities for result file paths and JSON/JSONL persistence."""

import hashlib
import json
import logging
from pathlib import Path
import re
from typing import Any, Dict, List, Union

logger = logging.getLogger(__name__)


def slugify(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value)).strip("._-")
    return slug or "unknown"


def model_basename(model_name: str) -> str:
    normalized = str(model_name).strip().rstrip("/\\").replace("\\", "/")
    if not normalized:
        return "unknown"
    return Path(normalized).name or slugify(normalized)


def config_hash(config: Dict[str, Any]) -> str:
    encoded = json.dumps(config, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:12]


def task_version(task: Any) -> Union[int, str]:
    version = getattr(type(task), "VERSION", None)
    if version is None:
        version = getattr(task, "VERSION", 0)
    if isinstance(version, (int, str)):
        return version
    return 0


def build_task_evaluation_config(
    *,
    model_name: str,
    strategy_name: str,
    strategy_kwargs: Dict[str, Any],
    strategy_config: Dict[str, Any],
    task_name: str,
    task: Any,
    task_kwargs: Dict[str, Dict[str, Any]],
    batch_size: int,
    device: str,
    requested_device: str,
    dtype: str,
    trust_remote_code: bool,
) -> Dict[str, Any]:
    return {
        "model": model_basename(model_name),
        "strategy": {
            "name": strategy_name,
            "kwargs": strategy_kwargs,
            "config": strategy_config,
        },
        "task": {
            "name": task_name,
            "version": task_version(task),
            "kwargs": task_kwargs.get(task_name, {}),
            "resolved_kwargs": {
                "num_fewshot": task.num_fewshot,
                "max_samples": task.max_samples,
                "seed": task.seed,
            },
        },
        "runtime": {
            "batch_size": batch_size,
            "device": device,
            "requested_device": requested_device,
            "dtype": dtype,
            "trust_remote_code": trust_remote_code,
        },
    }


def task_result_path(results_dir: Union[str, Path], config: Dict[str, Any]) -> Path:
    model_slug = slugify(config["model"])
    task_slug = slugify(config["task"]["name"])
    strategy_slug = slugify(config["strategy"]["name"])
    return (
        Path(results_dir)
        / model_slug
        / task_slug
        / strategy_slug
        / f"{config_hash(config)}.json"
    )


def task_samples_path(results_dir: Union[str, Path], config: Dict[str, Any]) -> Path:
    return task_result_path(results_dir, config).with_suffix(".jsonl")


def to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return [to_jsonable(v) for v in value]
    if isinstance(value, list):
        return [to_jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass
    if hasattr(value, "tolist"):
        try:
            return to_jsonable(value.tolist())
        except (TypeError, ValueError):
            pass
    return value


def save_json(data: Dict[str, Any], path: Union[str, Path]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(to_jsonable(data), f, indent=2)
    logger.info("Results saved to %s", output_path)


def append_jsonl(record: Dict[str, Any], path: Union[str, Path]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "a") as f:
        json.dump(to_jsonable(record), f)
        f.write("\n")


def load_jsonl(path: Union[str, Path]) -> List[Dict[str, Any]]:
    input_path = Path(path)
    if not input_path.exists():
        return []

    records: List[Dict[str, Any]] = []
    with open(input_path) as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                logger.warning(
                    "Skipping invalid JSONL line %d in %s",
                    line_number,
                    input_path,
                )
                continue
            if isinstance(record, dict):
                records.append(record)
    return records


def save_task_result(
    *,
    results_dir: Union[str, Path],
    task_results: Dict[str, Any],
    task_elapsed: float,
    config: Dict[str, Any],
    samples_path: Union[str, Path, None] = None,
) -> str:
    path = task_result_path(results_dir, config)
    output = {
        "model": config["model"],
        "strategy": config["strategy"]["name"],
        "task": config["task"]["name"],
        "evaluation_config": config,
        "results": task_results,
        "elapsed_seconds": round(task_elapsed, 2),
    }
    if samples_path is not None:
        output["samples_file"] = str(samples_path)
    save_json(output, path)
    return str(path)