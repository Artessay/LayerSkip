#!/usr/bin/env python3
"""Visualize calibration layer metrics across tasks.

The script scans calibration JSON files written by the calibratedskip strategy,
groups them by base model and calibration metric, writes one line chart for
each (model, metric) pair, and prints per-task layer rankings to the terminal.

Examples
--------
python scripts/visualize_calibration.py

python scripts/visualize_calibration.py \
    --models Meta-Llama-3-8B-Instruct \
    --metrics gradient_trace shapley_value \
    --normalize minmax
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


SUPPORTED_CALIBRATION_METRICS = (
    "activation_ratio",
    "gradient_value",
    "gradient_trace",
    "shapley_value",
)

SHAPLEY_PLOT_MIN_LAYER = 5

PALETTE = (
    "#2563eb",
    "#dc2626",
    "#16a34a",
    "#9333ea",
    "#ea580c",
    "#0891b2",
    "#be123c",
    "#4f46e5",
    "#65a30d",
    "#c026d3",
)


@dataclass(frozen=True)
class CalibrationRun:
    path: Path
    model: str
    task: str
    layers: Tuple[Dict[str, float], ...]
    num_samples: Optional[int]
    num_layers: int

    def has_metric(self, metric: str) -> bool:
        return any(metric in layer for layer in self.layers)

    def layer_values(self, metric: str) -> List[Tuple[int, float]]:
        values = []
        for layer in self.layers:
            if metric not in layer:
                continue
            values.append((int(layer["layer"]), float(layer[metric])))
        return sorted(values, key=lambda item: item[0])


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot calibrated layer-importance metrics by model and metric, "
            "with different tasks overlaid as colored lines."
        )
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results"),
        help="Directory containing results/<model>/<task>/calibration/*.json.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/calibration_plots"),
        help="Directory where plots will be written.",
    )
    parser.add_argument(
        "--files",
        nargs="+",
        type=Path,
        default=None,
        help="Optional explicit calibration JSON files to visualize.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Optional model names or slugs to include.",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=None,
        help="Optional task names to include.",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        choices=SUPPORTED_CALIBRATION_METRICS,
        default=None,
        help="Calibration metrics to plot and rank. Defaults to all present metrics.",
    )
    parser.add_argument(
        "--normalize",
        choices=("none", "minmax"),
        default="none",
        help="Plot raw metric values or min-max normalize each task line.",
    )
    parser.add_argument(
        "--rank-mode",
        choices=("auto", "value", "abs"),
        default="auto",
        help=(
            "How to sort layer importance. Auto ranks shapley_value by absolute "
            "magnitude and other metrics by raw value."
        ),
    )
    parser.add_argument(
        "--include-all-runs",
        action="store_true",
        help="Include every calibration JSON. By default, only the newest run per model/task is used.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=0,
        help="Print only the top K layers per task and metric. The default 0 prints every layer.",
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        choices=("svg", "png"),
        default=["png"],
        # default=["svg", "png"],
        help="Plot formats to save. Defaults to svg png.",
    )
    parser.add_argument(
        "--png-dpi",
        type=int,
        default=200,
        help="PNG output resolution in dots per inch.",
    )
    return parser.parse_args(argv)


def slugify(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value)).strip("._-")
    return slug or "unknown"


def discover_calibration_files(args: argparse.Namespace) -> List[Path]:
    if args.files:
        return sorted(path.expanduser() for path in args.files)
    return sorted(args.results_dir.expanduser().rglob("calibration/*.json"))


def load_calibration_run(path: Path) -> CalibrationRun:
    with open(path) as file_obj:
        data = json.load(file_obj)

    raw_layers = data.get("layers")
    if not isinstance(raw_layers, list) or not raw_layers:
        raise ValueError("missing non-empty 'layers' list")

    inferred_model = path.parents[2].name if len(path.parents) >= 3 else "unknown"
    inferred_task = path.parents[1].name if len(path.parents) >= 2 else "unknown"
    config = data.get("calibration_config") if isinstance(data, dict) else {}
    task_config = config.get("task", {}) if isinstance(config, dict) else {}

    model = str(data.get("model") or config.get("model") or inferred_model)
    task = str(data.get("task") or task_config.get("name") or inferred_task)

    layers = []
    for raw_layer in raw_layers:
        if not isinstance(raw_layer, dict) or "layer" not in raw_layer:
            continue
        try:
            layer_num = int(raw_layer["layer"])
        except (TypeError, ValueError):
            continue

        layer_data: Dict[str, float] = {"layer": float(layer_num)}
        for key, value in raw_layer.items():
            if key == "layer":
                continue
            try:
                numeric_value = float(value)
            except (TypeError, ValueError):
                continue
            if math.isfinite(numeric_value):
                layer_data[str(key)] = numeric_value
        layers.append(layer_data)

    if not layers:
        raise ValueError("no valid layer metric entries found")

    num_samples = data.get("num_samples")
    try:
        num_samples = int(num_samples) if num_samples is not None else None
    except (TypeError, ValueError):
        num_samples = None

    num_layers = data.get("num_layers")
    try:
        num_layers = int(num_layers)
    except (TypeError, ValueError):
        num_layers = len(layers)

    return CalibrationRun(
        path=path,
        model=model,
        task=task,
        layers=tuple(layers),
        num_samples=num_samples,
        num_layers=num_layers,
    )


def filter_runs(runs: Iterable[CalibrationRun], args: argparse.Namespace) -> List[CalibrationRun]:
    model_filters = {slugify(model) for model in args.models or []}
    raw_model_filters = set(args.models or [])
    task_filters = set(args.tasks or [])

    filtered = []
    for run in runs:
        if model_filters and run.model not in raw_model_filters and slugify(run.model) not in model_filters:
            continue
        if task_filters and run.task not in task_filters:
            continue
        filtered.append(run)
    return filtered


def select_latest_per_task(runs: Iterable[CalibrationRun]) -> List[CalibrationRun]:
    selected: Dict[Tuple[str, str], CalibrationRun] = {}
    duplicate_counts: Dict[Tuple[str, str], int] = {}
    for run in runs:
        key = (run.model, run.task)
        duplicate_counts[key] = duplicate_counts.get(key, 0) + 1
        current = selected.get(key)
        if current is None or run.path.stat().st_mtime >= current.path.stat().st_mtime:
            selected[key] = run

    duplicate_keys = [key for key, count in duplicate_counts.items() if count > 1]
    if duplicate_keys:
        print(
            "Note: multiple calibration files found for "
            f"{len(duplicate_keys)} model/task pair(s); using the newest file. "
            "Pass --include-all-runs to plot every file.",
            file=sys.stderr,
        )

    return sorted(selected.values(), key=lambda run: (run.model, run.task, str(run.path)))


def present_metrics(runs: Iterable[CalibrationRun]) -> List[str]:
    found = set()
    for run in runs:
        for layer in run.layers:
            found.update(key for key in layer if key != "layer")
    ordered = [metric for metric in SUPPORTED_CALIBRATION_METRICS if metric in found]
    ordered.extend(sorted(found - set(ordered)))
    return ordered


def normalize_values(values: Sequence[float], mode: str) -> List[float]:
    if mode == "none":
        return list(values)
    min_value = min(values)
    max_value = max(values)
    span = max_value - min_value
    if span == 0:
        return [0.5 for _ in values]
    return [(value - min_value) / span for value in values]


def plot_value_description(normalize: str) -> str:
    if normalize == "none":
        return "raw metric values"
    return "min-max normalized values per task"


def display_metric_name(metric: str) -> str:
    if metric == "shapley_value":
        return f"abs({metric})"
    return metric


def display_layer_values(run: CalibrationRun, metric: str) -> List[Tuple[int, float]]:
    values = run.layer_values(metric)
    if metric == "shapley_value":
        return [(layer, abs(value)) for layer, value in values]
    return values


def load_matplotlib():
    try:
        import matplotlib

        matplotlib.use("Agg")
        from matplotlib import pyplot as plt
        from matplotlib.ticker import FuncFormatter
    except ImportError as exc:
        raise RuntimeError(
            "Saving plots requires matplotlib. Install dependencies with "
            "`pip install -r requirements.txt`."
        ) from exc
    return plt, FuncFormatter


def y_bounds(values: Sequence[float]) -> Tuple[float, float]:
    min_value = min(values)
    max_value = max(values)
    if min_value == max_value:
        padding = abs(min_value) * 0.05 or 1.0
        return min_value - padding, max_value + padding
    padding = (max_value - min_value) * 0.06
    return min_value - padding, max_value + padding


def format_number(value: float) -> str:
    abs_value = abs(value)
    if abs_value >= 1_000_000_000:
        return f"{value / 1_000_000_000:.2f}B"
    if abs_value >= 1_000_000:
        return f"{value / 1_000_000:.2f}M"
    if abs_value >= 1_000:
        return f"{value / 1_000:.2f}K"
    if abs_value >= 1:
        return f"{value:.4g}"
    return f"{value:.6f}"


def format_tick(value: float) -> str:
    abs_value = abs(value)
    if abs_value >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    if abs_value >= 1_000:
        return f"{value / 1_000:.1f}K"
    if abs_value >= 1:
        return f"{value:.2f}"
    return f"{value:.3f}"


def x_ticks(min_layer: int, max_layer: int) -> List[int]:
    if min_layer == max_layer:
        return [min_layer]
    ticks = {min_layer, max_layer}
    step = max(1, round((max_layer - min_layer) / 8))
    for layer in range(min_layer, max_layer + 1, step):
        ticks.add(layer)
    return sorted(ticks)


def x_bounds(min_layer: int, max_layer: int) -> Tuple[float, float]:
    span = max_layer - min_layer
    padding = max(0.5, span * 0.03)
    return min_layer - padding, max_layer + padding


def line_label(run: CalibrationRun, include_run_id: bool) -> str:
    if include_run_id:
        return f"{run.task} [{run.path.stem}]"
    return run.task


def build_plot_series(
    *,
    metric: str,
    runs: Sequence[CalibrationRun],
    normalize: str,
    include_run_id: bool,
) -> List[Dict[str, object]]:
    series = []
    for index, run in enumerate(sorted(runs, key=lambda item: (item.task, str(item.path)))):
        layer_values = display_layer_values(run, metric)
        if metric == "shapley_value":
            layer_values = [item for item in layer_values if item[0] >= SHAPLEY_PLOT_MIN_LAYER]
        if not layer_values:
            continue
        layers = [layer for layer, _ in layer_values]
        raw_values = [value for _, value in layer_values]
        plot_values = normalize_values(raw_values, normalize)
        series.append(
            {
                "label": line_label(run, include_run_id),
                "layers": layers,
                "values": plot_values,
                "samples": run.num_samples,
                "color": PALETTE[index % len(PALETTE)],
            }
        )
    return series


def write_line_plots(
    *,
    model: str,
    metric: str,
    runs: Sequence[CalibrationRun],
    output_stem: Path,
    formats: Sequence[str],
    normalize: str,
    include_run_id: bool,
    png_dpi: int,
) -> List[Path]:
    series = build_plot_series(
        metric=metric,
        runs=runs,
        normalize=normalize,
        include_run_id=include_run_id,
    )

    if not series:
        return []

    all_layers = [layer for item in series for layer in item["layers"]]
    all_values = [value for item in series for value in item["values"]]
    min_layer = int(min(all_layers))
    max_layer = int(max(all_layers))
    min_y, max_y = y_bounds(all_values)

    plt, FuncFormatter = load_matplotlib()
    fig, ax = plt.subplots(figsize=(12, 7.2), dpi=png_dpi)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    for item in series:
        sample_text = f' (n={item["samples"]})' if item["samples"] is not None else ""
        label = str(item["label"]) + sample_text
        ax.plot(
            item["layers"],
            item["values"],
            color=str(item["color"]),
            linewidth=2.2,
            marker="o",
            markersize=4,
            label=label,
        )

    metric_label = display_metric_name(metric)
    ax.set_title(f"{model} - {metric_label}", loc="left", fontsize=15, fontweight="bold", pad=24)
    ax.text(
        0,
        1.01,
        f"Tasks overlaid by color; plotting {plot_value_description(normalize)}.",
        transform=ax.transAxes,
        fontsize=9,
        color="#6b7280",
    )
    ax.set_xlabel("Layer")
    ylabel = metric_label if normalize == "none" else f"{metric_label} (min-max)"
    ax.set_ylabel(ylabel)
    ax.set_xlim(*x_bounds(min_layer, max_layer))
    ax.set_ylim(min_y, max_y)
    ax.set_xticks(x_ticks(min_layer, max_layer))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: format_tick(value)))
    ax.grid(True, color="#e5e7eb", linewidth=0.9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False, fontsize=9)
    fig.subplots_adjust(right=0.78)

    output_stem.parent.mkdir(parents=True, exist_ok=True)
    written_paths = []
    for fmt in tuple(dict.fromkeys(formats)):
        output_path = output_stem.with_suffix(f".{fmt}")
        save_kwargs = {
            "format": fmt,
            "bbox_inches": "tight",
            "facecolor": "white",
        }
        if fmt == "png":
            save_kwargs["dpi"] = png_dpi
        fig.savefig(output_path, **save_kwargs)
        written_paths.append(output_path)

    plt.close(fig)
    return written_paths


def effective_rank_mode(metric: str, rank_mode: str) -> str:
    if metric == "shapley_value":
        return "abs"
    if rank_mode == "auto":
        return "value"
    return rank_mode


def ranked_layers(run: CalibrationRun, metric: str, rank_mode: str) -> List[Tuple[int, float]]:
    mode = effective_rank_mode(metric, rank_mode)
    values = display_layer_values(run, metric)
    if mode == "abs":
        return sorted(values, key=lambda item: abs(item[1]), reverse=True)
    return sorted(values, key=lambda item: item[1], reverse=True)


def print_rankings(
    runs_by_model: Dict[str, List[CalibrationRun]],
    metrics: Sequence[str],
    rank_mode: str,
    top_k: int,
    include_run_id: bool,
) -> None:
    print("\nLayer importance rankings")
    print("=" * 88)
    if rank_mode == "auto":
        print("Rank mode: auto (shapley_value uses abs(value); other metrics use value)")
    else:
        print(f"Rank mode: {rank_mode} (shapley_value uses abs(value))")

    for model in sorted(runs_by_model):
        print("\n" + "=" * 88)
        print(f"Model: {model}")
        runs = sorted(runs_by_model[model], key=lambda run: (run.task, str(run.path)))
        for metric in metrics:
            metric_runs = [run for run in runs if run.has_metric(metric)]
            if not metric_runs:
                continue
            mode = effective_rank_mode(metric, rank_mode)
            score_name = "abs(value)" if mode == "abs" else "value"
            print(f"\nMetric: {metric} | sorted by {score_name} descending")
            for run in metric_runs:
                ranked = ranked_layers(run, metric, rank_mode)
                if top_k > 0:
                    ranked = ranked[:top_k]
                entries = [f"L{layer}({format_number(value)})" for layer, value in ranked]
                print(f"  Task: {line_label(run, include_run_id)}")
                for start in range(0, len(entries), 8):
                    print("    " + ", ".join(entries[start : start + 8]))


def group_by_model(runs: Iterable[CalibrationRun]) -> Dict[str, List[CalibrationRun]]:
    grouped: Dict[str, List[CalibrationRun]] = {}
    for run in runs:
        grouped.setdefault(run.model, []).append(run)
    return grouped


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)

    calibration_files = discover_calibration_files(args)
    if not calibration_files:
        print("No calibration JSON files found.", file=sys.stderr)
        return 1

    runs = []
    for path in calibration_files:
        try:
            runs.append(load_calibration_run(path))
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            print(f"Skipping {path}: {exc}", file=sys.stderr)

    runs = filter_runs(runs, args)
    if not runs:
        print("No calibration runs matched the requested filters.", file=sys.stderr)
        return 1

    if not args.include_all_runs:
        runs = select_latest_per_task(runs)

    metrics = list(args.metrics or present_metrics(runs))
    metrics = [metric for metric in metrics if any(run.has_metric(metric) for run in runs)]
    if not metrics:
        print("No requested calibration metrics were present in the selected files.", file=sys.stderr)
        return 1

    runs_by_model = group_by_model(runs)
    written_plots = []
    for model, model_runs in sorted(runs_by_model.items()):
        for metric in metrics:
            metric_runs = [run for run in model_runs if run.has_metric(metric)]
            if not metric_runs:
                continue
            output_stem = args.output_dir / slugify(model) / slugify(metric)
            written = write_line_plots(
                model=model,
                metric=metric,
                runs=metric_runs,
                output_stem=output_stem,
                formats=args.formats,
                normalize=args.normalize,
                include_run_id=args.include_all_runs,
                png_dpi=args.png_dpi,
            )
            written_plots.extend(written)

    print(f"Saved {len(written_plots)} plot(s):")
    for path in written_plots:
        print(f"  {path}")

    print_rankings(
        runs_by_model=runs_by_model,
        metrics=metrics,
        rank_mode=args.rank_mode,
        top_k=args.top_k,
        include_run_id=args.include_all_runs,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())