"""Utilities for result file paths and JSON persistence."""

import hashlib
import json
import logging
from pathlib import Path
import re
from typing import Any, Dict, Union

logger = logging.getLogger(__name__)


def slugify(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value)).strip("._-")
    return slug or "unknown"


def config_hash(config: Dict[str, Any]) -> str:
    encoded = json.dumps(config, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:12]


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
    max_length: int,
    trust_remote_code: bool,
) -> Dict[str, Any]:
    return {
        "model": model_name,
        "strategy": {
            "name": strategy_name,
            "kwargs": strategy_kwargs,
            "config": strategy_config,
        },
        "task": {
            "name": task_name,
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
            "max_length": max_length,
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


def save_json(data: Dict[str, Any], path: Union[str, Path]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
    logger.info("Results saved to %s", output_path)


def save_task_result(
    *,
    results_dir: Union[str, Path],
    task_results: Dict[str, Any],
    task_elapsed: float,
    config: Dict[str, Any],
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
    save_json(output, path)
    return str(path)