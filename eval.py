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

# CalibratedSkip: compute layer metrics on the task calibration split, save
# them for manual inspection, and exit without task evaluation:
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
from evaluation.pruning import PRUNING_METHODS, TALE_VARIANTS
from evaluation.strategies import STRATEGY_REGISTRY
from evaluation.tasks import TASK_REGISTRY
from evaluation.models.hf_model import SUPPORTED_MODELS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# README download commands mirror Hugging Face identifiers directly below
# /data (for example /data/meta-llama/... and /data/cais/...).  Keep --local
# consistent with that documented on-disk layout.
MODEL_LOCAL_ROOT = Path("/data")
DATASET_LOCAL_ROOT = Path("/data")


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
    model_group.add_argument(
        "--apply_chat_template",
        action="store_true",
        help="Wrap prompts in the tokenizer chat template (off by default for canonical benchmarks).",
    )

    # ------------------------------------------------------------------ #
    # Strategy arguments                                                   #
    # ------------------------------------------------------------------ #
    strategy_group = parser.add_argument_group("Layer Skipping Strategy")
    strategy_group.add_argument(
        "--strategy",
        nargs="+",
        default=["none"],
        choices=list(STRATEGY_REGISTRY.keys()) + list(PRUNING_METHODS),
        help=(
            "One or more layer-skipping or layer-pruning methods to evaluate. "
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
    strategy_group.add_argument(
        "--caml_confidence_measure",
        choices=["softmax", "hidden_state"],
        default="softmax",
        help="CALM confidence measure (default: softmax response).",
    )
    strategy_group.add_argument(
        "--caml_use_decaying_threshold",
        action="store_true",
        help="Use CALM Eq. (5) generation-step threshold decay.",
    )
    strategy_group.add_argument(
        "--caml_decay_factor", type=float, default=4.0,
        help="CALM threshold decay temperature tau (default: 4).",
    )
    strategy_group.add_argument(
        "--caml_allow_untrained_exits", action="store_true",
        help="Allow CALM heads on a checkpoint not trained with intermediate LM loss.",
    )
    # GateSkip-specific
    strategy_group.add_argument(
        "--gateskip_skip_budget",
        type=float,
        default=0.3,
        metavar="BUDGET",
        help="GateSkip: target fraction of tokens skipped per gated module (default: 0.3).",
    )
    strategy_group.add_argument(
        "--gateskip_min_layers",
        type=int,
        default=1,
        metavar="N",
        help="GateSkip: number of initial transformer layers left ungated (default: 1).",
    )
    strategy_group.add_argument(
        "--gateskip_gate_state_path",
        type=str,
        default=None,
        help="GateSkip fine-tuned gate state dict (required for GateSkip).",
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
    # ShortGPT-specific
    strategy_group.add_argument(
        "--shortgpt_prune_ratio",
        type=float,
        default=0.25,
        metavar="RATIO",
        help="ShortGPT: fraction of transformer layers to remove (default: 0.25).",
    )
    strategy_group.add_argument(
        "--shortgpt_num_remove",
        type=int,
        default=None,
        metavar="N",
        help="ShortGPT: exact number of layers to remove (overrides prune ratio).",
    )
    strategy_group.add_argument(
        "--shortgpt_dataset",
        type=str,
        default="emozilla/pg19",
        metavar="PATH",
        help="ShortGPT: PG19 dataset identifier or local path (default: emozilla/pg19).",
    )
    strategy_group.add_argument(
        "--shortgpt_split",
        type=str,
        default="validation",
        help="ShortGPT: calibration corpus split (default: validation).",
    )
    strategy_group.add_argument(
        "--shortgpt_max_samples",
        type=int,
        default=None,
        metavar="N",
        help="ShortGPT: cap source documents used for block-influence scoring.",
    )
    strategy_group.add_argument(
        "--shortgpt_sequence_length",
        type=int,
        default=256,
        metavar="TOKENS",
        help="ShortGPT: non-overlapping calibration chunk length (default: 256).",
    )
    strategy_group.add_argument(
        "--shortgpt_search_batch_size",
        type=int,
        default=1,
        metavar="N",
        help="ShortGPT: calibration forward-pass batch size (default: 1).",
    )
    # SLEB-specific
    strategy_group.add_argument(
        "--sleb_prune_ratio",
        type=float,
        default=0.2,
        metavar="RATIO",
        help="SLEB: fraction of transformer layers to remove (default: 0.2).",
    )
    strategy_group.add_argument(
        "--sleb_num_remove",
        type=int,
        default=None,
        metavar="N",
        help="SLEB: exact number of layers to remove (overrides prune ratio).",
    )
    strategy_group.add_argument(
        "--sleb_dataset",
        type=str,
        default="wikitext",
        metavar="PATH",
        help="SLEB: calibration dataset identifier or local path (default: wikitext).",
    )
    strategy_group.add_argument(
        "--sleb_dataset_name",
        type=str,
        default="wikitext-2-raw-v1",
        metavar="NAME",
        help="SLEB: dataset configuration name (default: wikitext-2-raw-v1).",
    )
    strategy_group.add_argument(
        "--sleb_split",
        type=str,
        default="train",
        help="SLEB: calibration corpus split (default: train).",
    )
    strategy_group.add_argument(
        "--sleb_max_samples",
        type=int,
        default=128,
        metavar="N",
        help="SLEB: number of source documents to sample (default: 128).",
    )
    strategy_group.add_argument(
        "--sleb_sequence_length",
        type=int,
        default=2048,
        metavar="TOKENS",
        help="SLEB: calibration sequence length (default: 2048).",
    )
    strategy_group.add_argument(
        "--sleb_search_batch_size",
        type=int,
        default=1,
        metavar="N",
        help="SLEB: candidate-scoring batch size (default: 1).",
    )
    strategy_group.add_argument(
        "--sleb_seed",
        type=int,
        default=0,
        metavar="N",
        help="SLEB: corpus sampling seed (paper default: 0).",
    )
    strategy_group.add_argument(
        "--sleb_early_barrier",
        type=int,
        default=0,
        metavar="N",
        help="SLEB: protect this many initial layers (paper default: 0).",
    )
    strategy_group.add_argument(
        "--sleb_latter_barrier",
        type=int,
        default=0,
        metavar="N",
        help="SLEB: protect this many final layers (paper default: 0).",
    )
    # TALE-specific
    strategy_group.add_argument(
        "--tale_threshold",
        type=float,
        default=0.08,
        metavar="DELTA",
        help="TALE: permitted accuracy drop from the dense model (default: 0.08).",
    )
    strategy_group.add_argument(
        "--tale_search_max_samples",
        type=int,
        default=None,
        metavar="N",
        help="TALE: cap labeled search examples per task (default: all).",
    )
    strategy_group.add_argument(
        "--tale_max_remove",
        type=int,
        default=None,
        metavar="N",
        help="TALE: maximum greedy search depth (default: all but one layer).",
    )
    strategy_group.add_argument(
        "--tale_target_remove",
        type=int,
        default=None,
        metavar="N",
        help="TALE: optional exact removal budget used by the budget variant.",
    )
    strategy_group.add_argument(
        "--tale_variant",
        choices=list(TALE_VARIANTS),
        default="threshold_final",
        help="TALE checkpoint to evaluate (default: threshold_final).",
    )
    strategy_group.add_argument(
        "--tale_continue_below_threshold",
        action="store_true",
        help="TALE: continue greedy search after accuracy falls below the threshold.",
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
            "confidence_measure": args.caml_confidence_measure,
            "use_decaying_threshold": args.caml_use_decaying_threshold,
            "decay_factor": args.caml_decay_factor,
            "allow_untrained_exits": args.caml_allow_untrained_exits,
        }
    if strategy_name == "gateskip":
        return {
            "skip_budget": args.gateskip_skip_budget,
            "min_layers": args.gateskip_min_layers,
            "gate_state_path": args.gateskip_gate_state_path,
        }
    if strategy_name == "calibratedskip":
        return {
            "calibration_metrics": args.calibratedskip_metrics,
            "calibration_max_samples": args.calibration_max_samples,
        }
    if strategy_name == "manualskip":
        return {"skip_layers": _parse_manualskip_layers(args.manualskip_layers)}
    if strategy_name == "shortgpt":
        return {
            "prune_ratio": args.shortgpt_prune_ratio,
            "num_remove": args.shortgpt_num_remove,
            "dataset_path": args.shortgpt_dataset,
            "split": args.shortgpt_split,
            "max_samples": args.shortgpt_max_samples,
            "sequence_length": args.shortgpt_sequence_length,
            "search_batch_size": args.shortgpt_search_batch_size,
            "seed": args.seed,
        }
    if strategy_name == "sleb":
        return {
            "prune_ratio": args.sleb_prune_ratio,
            "num_remove": args.sleb_num_remove,
            "dataset_path": args.sleb_dataset,
            "dataset_name": args.sleb_dataset_name,
            "split": args.sleb_split,
            "max_samples": args.sleb_max_samples,
            "sequence_length": args.sleb_sequence_length,
            "search_batch_size": args.sleb_search_batch_size,
            "early_barrier": args.sleb_early_barrier,
            "latter_barrier": args.sleb_latter_barrier,
            "seed": args.sleb_seed,
        }
    if strategy_name == "tale":
        return {
            "threshold": args.tale_threshold,
            "search_max_samples": args.tale_search_max_samples,
            "max_remove": args.tale_max_remove,
            "target_remove": args.tale_target_remove,
            "variant": args.tale_variant,
            "stop_at_threshold": not args.tale_continue_below_threshold,
            "seed": args.seed,
        }
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


def _as_local_model_path(identifier: str) -> str:
    path = Path(identifier)
    if path.is_absolute():
        return identifier
    return str(MODEL_LOCAL_ROOT / identifier)


def _as_local_dataset_path(identifier: str) -> str:
    path = Path(identifier)
    if path.is_absolute():
        return identifier
    return str(DATASET_LOCAL_ROOT / identifier)


def _apply_local_dataset_paths(task_names: List[str]) -> Dict[str, str]:
    original_paths: Dict[str, str] = {}
    for task_name in task_names:
        if task_name in original_paths:
            continue
        task_cls = TASK_REGISTRY[task_name]
        dataset_path = task_cls.DATASET_PATH
        original_paths[task_name] = dataset_path
        task_cls.DATASET_PATH = _as_local_dataset_path(dataset_path)
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
        args.model = _as_local_model_path(args.model)
        logger.info("Using local model path: %s", args.model)
        args.shortgpt_dataset = _as_local_dataset_path(args.shortgpt_dataset)
        args.sleb_dataset = _as_local_dataset_path(args.sleb_dataset)
        logger.info("Using local ShortGPT dataset path: %s", args.shortgpt_dataset)
        logger.info("Using local SLEB dataset path: %s", args.sleb_dataset)

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
                apply_chat_template=args.apply_chat_template,
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
