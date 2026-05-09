"""Layer-importance calibration utilities."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple, Union

from evaluation.models.hf_model import HFModel
from evaluation.tasks.base_task import BaseTask
from evaluation.utils.result_io import config_hash, model_basename, save_json, slugify

logger = logging.getLogger(__name__)

SUPPORTED_CALIBRATION_METRICS = ("activation_ratio", "gradient_trace")


def normalize_calibration_metrics(metrics: Iterable[str]) -> Tuple[str, ...]:
    normalized = tuple(dict.fromkeys(str(metric).lower() for metric in metrics))
    if not normalized:
        raise ValueError("At least one calibration metric must be requested")
    unsupported = sorted(set(normalized) - set(SUPPORTED_CALIBRATION_METRICS))
    if unsupported:
        raise ValueError(f"Unsupported calibration metrics: {unsupported}")
    return normalized


def _calibration_config(
    *,
    model_name: str,
    task_name: str,
    task: BaseTask,
    task_kwargs: Dict[str, Dict[str, Any]],
    metrics: Sequence[str],
    max_samples: Optional[int],
    batch_size: int,
    device: str,
    requested_device: str,
    dtype: str,
    trust_remote_code: bool,
) -> Dict[str, Any]:
    return {
        "model": model_basename(model_name),
        "task": {
            "name": task_name,
            "calibration_split": task.calibration_split_name,
            "kwargs": task_kwargs.get(task_name, {}),
            "resolved_kwargs": {
                "num_fewshot": task.num_fewshot,
                "max_samples": task.max_samples,
                "seed": task.seed,
            },
        },
        "calibration": {
            "metrics": list(metrics),
            "max_samples": max_samples,
        },
        "runtime": {
            "batch_size": batch_size,
            "device": device,
            "requested_device": requested_device,
            "dtype": dtype,
            "trust_remote_code": trust_remote_code,
        },
    }


def calibration_result_path(
    results_dir: Union[str, Path],
    config: Dict[str, Any],
) -> Path:
    model_slug = slugify(config["model"])
    task_slug = slugify(config["task"]["name"])
    return (
        Path(results_dir)
        / model_slug
        / task_slug
        / "calibration"
        / f"{config_hash(config)}.json"
    )


def calibrate_task_layers(
    *,
    model: HFModel,
    task: BaseTask,
    task_name: str,
    model_name: str,
    task_kwargs: Dict[str, Dict[str, Any]],
    strategy_kwargs: Dict[str, Any],
    batch_size: int,
    device: str,
    requested_device: str,
    dtype: str,
    trust_remote_code: bool,
    results_dir: Union[str, Path],
) -> Dict[str, Any]:
    """Compute and persist layer calibration metrics for one task."""
    metrics = normalize_calibration_metrics(
        strategy_kwargs.get("calibration_metrics", SUPPORTED_CALIBRATION_METRICS)
    )
    max_samples = strategy_kwargs.get("calibration_max_samples")

    _, requests = task.build_calibration_requests(
        max_samples=max_samples,
        seed=task.seed,
    )
    logger.info(
        "Calibrating task '%s' on %d labeled samples from split '%s'.",
        task_name,
        len(requests),
        task.calibration_split_name,
    )
    metric_output = model.compute_layer_calibration_metrics(
        requests=requests,
        metrics=metrics,
        batch_size=batch_size,
    )
    layers = metric_output["layers"]

    config = _calibration_config(
        model_name=model_name,
        task_name=task_name,
        task=task,
        task_kwargs=task_kwargs,
        metrics=metrics,
        max_samples=max_samples,
        batch_size=batch_size,
        device=device,
        requested_device=requested_device,
        dtype=dtype,
        trust_remote_code=trust_remote_code,
    )
    output_path = calibration_result_path(results_dir, config)
    output = {
        "model": config["model"],
        "task": task_name,
        "calibration_config": config,
        "calibration_split": task.calibration_split_name,
        "num_layers": metric_output["num_layers"],
        "num_samples": metric_output["num_samples"],
        "num_batches": metric_output["num_batches"],
        "layers": layers,
    }
    save_json(output, output_path)
    return {
        "metrics_file": str(output_path),
        "skip_layers": [],
        "calibration_split": task.calibration_split_name,
        "layers": layers,
        "config": config,
    }
