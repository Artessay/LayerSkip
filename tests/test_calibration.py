"""Tests for layer calibration helpers."""

import json

import pytest

from evaluation.calibration import calibrate_task_layers


def _make_layers():
    return [
        {"layer": 1, "activation_ratio": 0.1, "gradient_trace": 9.0},
        {"layer": 2, "activation_ratio": 0.4, "gradient_trace": 1.0},
        {"layer": 3, "activation_ratio": 0.2, "gradient_trace": 3.0},
        {"layer": 4, "activation_ratio": 0.8, "gradient_trace": 2.0},
    ]


class _FakeModel:
    def compute_layer_calibration_metrics(self, requests, metrics, batch_size=None):
        return {
            "num_layers": 4,
            "num_samples": len(requests),
            "num_batches": 1,
            "metrics": list(metrics),
            "layers": _make_layers(),
        }


class _FakeTask:
    num_fewshot = 0
    max_samples = None
    seed = 123
    calibration_split_name = "validation"

    def build_calibration_requests(self, max_samples=None, seed=None):
        return [{"id": 1}, {"id": 2}], [("prompt", " answer"), ("prompt", " other")]


def test_calibrate_task_layers_saves_all_layer_metrics(tmp_path):
    result = calibrate_task_layers(
        model=_FakeModel(),
        task=_FakeTask(),
        task_name="mocktask",
        model_name="org/mock-model",
        task_kwargs={"mocktask": {"seed": 123}},
        strategy_kwargs={
            "calibration_metrics": ["activation_ratio", "gradient_trace"],
            "calibration_max_samples": 2,
        },
        batch_size=2,
        device="cpu",
        requested_device="cpu",
        dtype="auto",
        trust_remote_code=False,
        results_dir=tmp_path,
    )

    assert result["skip_layers"] == []
    with open(result["metrics_file"]) as metrics_file:
        saved = json.load(metrics_file)

    assert "selected_skip_layers" not in saved
    assert len(saved["layers"]) == 4
    assert "selected_for_skip" not in saved["layers"][1]
    assert saved["layers"][0]["activation_ratio"] == pytest.approx(0.1)
