#!/usr/bin/env python3
"""
LayerSkip Evaluation Framework – Command-Line Interface

Evaluate language models with different layer-skipping strategies on standard
NLP benchmarks.

Examples
--------
# Evaluate Llama-3.2-1B with no layer skipping on MMLU and HellaSwag:
python eval.py \\
    --model meta-llama/Llama-3.2-1B-Instruct \\
    --strategy none \\
    --tasks mmlu hellaswag \\
    --max_samples 100

# Compare all strategies on WinoGrande with Llama-3-8B:
python eval.py \\
    --model meta-llama/Meta-Llama-3-8B-Instruct \\
    --strategy layerskip caml gateskip manualskip \\
    --manualskip_layers 2 4 8 \\
    --tasks winogrande \\
    --batch_size 4 \\
    --output results

# LayerSkip with a custom exit ratio:
python eval.py \\
    --model meta-llama/Llama-3.2-1B-Instruct \\
    --strategy layerskip \\
    --layerskip_exit_ratio 0.5 \\
    --tasks mmlu hellaswag winogrande gsm8k humaneval

# CAML with a custom confidence threshold:
python eval.py \\
    --model meta-llama/Llama-3.2-1B-Instruct \\
    --strategy caml \\
    --caml_confidence_threshold 0.85 \\
    --tasks mmlu

# ManualSkip with user-selected layers bypassed:
python eval.py \\
    --model meta-llama/Llama-3.2-1B-Instruct \\
    --strategy manualskip \\
    --manualskip_layers 2 4 8 \\
    --tasks mmlu

# CalibratedSkip: compute layer metrics on the task calibration split and save
# them for manual inspection:
python eval.py \\
    --model meta-llama/Llama-3.2-1B-Instruct \\
    --strategy calibratedskip \\
    --calibratedskip_metrics activation_ratio gradient_value gradient_trace shapley_value \\
    --calibration_max_samples 64 \\
    --tasks mmlu
"""

import argparse
import logging
from pathlib import Path
from typing import Any, Dict, List

from evaluation.evaluator import Evaluator
from evaluation.strategies import STRATEGY_REGISTRY
from evaluation.tasks import TASK_REGISTRY
from evaluation.models.hf_model import SUPPORTED_MODELS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

LOCAL_ROOT = Path("/data")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="LayerSkip LLM Evaluation Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # ------------------------------------------------------------------ #
    # Model arguments                                                      #
    # ------------------------------------------------------------------ #
    model_group = parser.add_argument_group("Model")
    model_group.add_argument(
        "--model",
        type=str,
        default="meta-llama/Meta-Llama-3-8B-Instruct",
        # default="meta-llama/Llama-3.2-1B-Instruct",
        help=(
            "HuggingFace model identifier or local path. "
            f"Officially supported backbones: {SUPPORTED_MODELS}"
        ),
    )
    model_group.add_argument(
        "--dtype",
        type=str,
        default="auto",
        choices=["auto", "float16", "bfloat16", "float32"],
        help="Model dtype (default: auto).",
    )
    model_group.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Target device, e.g. 'cuda', 'cuda:0', 'cpu' (default: auto).",
    )
    model_group.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for loglikelihood evaluation (default: 1).",
    )
    model_group.add_argument(
        "--trust_remote_code",
        action="store_true",
        help="Allow executing remote code when loading the model.",
    )

    # ------------------------------------------------------------------ #
    # Strategy arguments                                                   #
    # ------------------------------------------------------------------ #
    strategy_group = parser.add_argument_group("Layer Skipping Strategy")
    strategy_group.add_argument(
        "--strategy",
        nargs="+",
        default=["none"],
        choices=list(STRATEGY_REGISTRY.keys()),
        help=(
            "One or more layer-skipping strategies to evaluate. "
            "When multiple strategies are specified all are run and results "
            "are compared. (default: none)"
        ),
    )
    # LayerSkip-specific
    strategy_group.add_argument(
        "--layerskip_exit_ratio",
        type=float,
        default=0.75,
        metavar="RATIO",
        help="LayerSkip: fraction of layers to execute (default: 0.75).",
    )
    strategy_group.add_argument(
        "--layerskip_min_layers",
        type=int,
        default=4,
        metavar="N",
        help="LayerSkip: minimum number of layers to always execute (default: 4).",
    )
    # CAML-specific
    strategy_group.add_argument(
        "--caml_confidence_threshold",
        type=float,
        default=0.9,
        metavar="THRESH",
        help="CAML: confidence threshold for early exit (default: 0.9).",
    )
    strategy_group.add_argument(
        "--caml_min_layers",
        type=int,
        default=4,
        metavar="N",
        help="CAML: minimum layers before considering exit (default: 4).",
    )
    strategy_group.add_argument(
        "--caml_check_every",
        type=int,
        default=1,
        metavar="N",
        help="CAML: check confidence every N layers (default: 1).",
    )
    # GateSkip-specific
    strategy_group.add_argument(
        "--gateskip_gate_threshold",
        type=float,
        default=0.01,
        metavar="THRESH",
        help="GateSkip: relative-change threshold for skippable layers (default: 0.01).",
    )
    strategy_group.add_argument(
        "--gateskip_skip_budget",
        type=float,
        default=0.3,
        metavar="BUDGET",
        help="GateSkip: max fraction of layers to skip (default: 0.3).",
    )
    strategy_group.add_argument(
        "--gateskip_min_layers",
        type=int,
        default=4,
        metavar="N",
        help="GateSkip: minimum layers before skipping is considered (default: 4).",
    )
    # CalibratedSkip-specific
    strategy_group.add_argument(
        "--calibratedskip_metrics",
        nargs="+",
        default=["activation_ratio", "gradient_trace"],
        choices=[
            "activation_ratio",
            "gradient_value",
            "gradient_trace",
            "shapley_value",
        ],
        metavar="METRIC",
        help=(
            "CalibratedSkip: layer metrics to compute and save "
            "(default: activation_ratio gradient_trace)."
        ),
    )
    strategy_group.add_argument(
        "--calibration_max_samples",
        type=int,
        default=None,
        metavar="N",
        help="CalibratedSkip: cap calibration examples per task (default: all).",
    )
    # ManualSkip-specific
    strategy_group.add_argument(
        "--manualskip_layers",
        nargs="+",
        default=[],
        metavar="LAYER",
        help=(
            "ManualSkip: 1-based layer numbers to bypass, e.g. "
            "'--manualskip_layers 2 4 8' or '--manualskip_layers 2,4,8'."
        ),
    )

    # ------------------------------------------------------------------ #
    # Task arguments                                                       #
    # ------------------------------------------------------------------ #
    task_group = parser.add_argument_group("Tasks")
    task_group.add_argument(
        "--tasks",
        nargs="+",
        default=["mmlu"],
        choices=list(TASK_REGISTRY.keys()),
        help=(
            "One or more tasks to evaluate on. "
            f"Available: {list(TASK_REGISTRY.keys())} (default: mmlu)"
        ),
    )
    task_group.add_argument(
        "--max_samples",
        type=int,
        default=None,
        metavar="N",
        help="Cap on the number of evaluation examples per task (default: all).",
    )
    task_group.add_argument(
        "--num_fewshot",
        type=int,
        default=None,
        metavar="K",
        help=(
            "Override the default number of few-shot examples for all tasks. "
            "When not set, task-specific defaults are used."
        ),
    )
    task_group.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42).",
    )
    task_group.add_argument(
        "--local",
        action="store_true",
        help="Use /data/<model_or_dataset_id> paths for the model and datasets.",
    )

    # ------------------------------------------------------------------ #
    # Output arguments                                                     #
    # ------------------------------------------------------------------ #
    out_group = parser.add_argument_group("Output")
    out_group.add_argument(
        "--output",
        type=str,
        default="results",
        metavar="DIR",
        help=(
            "Directory for per-task result JSON files. Each model/task/strategy/"
            "config setting is saved separately. (default: results)."
        ),
    )
    out_group.add_argument(
        "--verbosity",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: INFO).",
    )

    return parser


def _parse_manualskip_layers(values: List[str]) -> List[int]:
    """Parse ManualSkip CLI values into a flat list of 1-based layer numbers."""
    if not values:
        raise ValueError("--manualskip_layers must include at least one layer")

    layers = []
    for value in values:
        cleaned = value.strip().strip("[]")
        for part in cleaned.split(","):
            part = part.strip()
            if not part:
                continue
            try:
                layer_num = int(part)
            except ValueError as exc:
                raise ValueError(
                    f"--manualskip_layers values must be integers, got '{part}'"
                ) from exc
            layers.append(layer_num)

    if not layers:
        raise ValueError("--manualskip_layers must include at least one layer")
    return layers


def _build_strategy_kwargs(args: argparse.Namespace, strategy_name: str) -> Dict[str, Any]:
    """Extract strategy-specific kwargs from parsed args."""
    if strategy_name == "layerskip":
        return {
            "exit_ratio": args.layerskip_exit_ratio,
            "min_layers": args.layerskip_min_layers,
        }
    if strategy_name == "caml":
        return {
            "confidence_threshold": args.caml_confidence_threshold,
            "min_layers": args.caml_min_layers,
            "check_every": args.caml_check_every,
        }
    if strategy_name == "gateskip":
        return {
            "gate_threshold": args.gateskip_gate_threshold,
            "skip_budget": args.gateskip_skip_budget,
            "min_layers": args.gateskip_min_layers,
        }
    if strategy_name == "calibratedskip":
        return {
            "calibration_metrics": args.calibratedskip_metrics,
            "calibration_max_samples": args.calibration_max_samples,
        }
    if strategy_name == "manualskip":
        return {"skip_layers": _parse_manualskip_layers(args.manualskip_layers)}
    return {}


def _build_task_kwargs(args: argparse.Namespace) -> Dict[str, Dict[str, Any]]:
    """Build per-task kwargs dict from CLI args."""
    kwargs: Dict[str, Any] = {}
    if args.max_samples is not None:
        kwargs["max_samples"] = args.max_samples
    if args.num_fewshot is not None:
        kwargs["num_fewshot"] = args.num_fewshot
    kwargs["seed"] = args.seed
    return {task: kwargs for task in args.tasks}


def _as_local_path(identifier: str) -> str:
    path = Path(identifier)
    if path.is_absolute():
        return identifier
    return str(LOCAL_ROOT / identifier)


def _apply_local_dataset_paths(task_names: List[str]) -> Dict[str, str]:
    original_paths: Dict[str, str] = {}
    for task_name in task_names:
        if task_name in original_paths:
            continue
        task_cls = TASK_REGISTRY[task_name]
        dataset_path = task_cls.DATASET_PATH
        original_paths[task_name] = dataset_path
        task_cls.DATASET_PATH = _as_local_path(dataset_path)
        logger.info(
            "Using local dataset path for task '%s': %s",
            task_name,
            task_cls.DATASET_PATH,
        )
    return original_paths


def _restore_dataset_paths(original_paths: Dict[str, str]) -> None:
    for task_name, dataset_path in original_paths.items():
        TASK_REGISTRY[task_name].DATASET_PATH = dataset_path


def main(argv: List[str] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    logging.getLogger().setLevel(getattr(logging, args.verbosity))

    if args.local:
        args.model = _as_local_path(args.model)
        logger.info("Using local model path: %s", args.model)

    strategies = args.strategy
    task_kwargs = _build_task_kwargs(args)
    original_dataset_paths = _apply_local_dataset_paths(args.tasks) if args.local else {}

    all_run_results = []

    try:
        for strategy_name in strategies:
            try:
                strategy_kwargs = _build_strategy_kwargs(args, strategy_name)
            except ValueError as exc:
                parser.error(str(exc))

            logger.info(
                "Running evaluation: model=%s | strategy=%s | tasks=%s",
                args.model,
                strategy_name,
                args.tasks,
            )

            evaluator = Evaluator(
                model_name=args.model,
                strategy_name=strategy_name,
                strategy_kwargs=strategy_kwargs,
                tasks=args.tasks,
                task_kwargs=task_kwargs,
                batch_size=args.batch_size,
                device=args.device,
                dtype=args.dtype,
                trust_remote_code=args.trust_remote_code,
                results_dir=args.output,
            )

            run_results = evaluator.run()
            Evaluator.print_results(run_results)
            all_run_results.append(run_results)
    finally:
        _restore_dataset_paths(original_dataset_paths)

    if len(all_run_results) > 1:
        comparison = Evaluator.compare_results(all_run_results)
        print("\n--- Strategy Comparison ---")
        Evaluator.print_comparison(comparison)


if __name__ == "__main__":
    main()
