"""
Main evaluation orchestrator.

The :class:`Evaluator` ties together models, strategies, and tasks into a
single evaluation run, and produces a structured results dict that can be
printed, saved to JSON, or compared across runs.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict, List, Optional, Union

from evaluation.models.hf_model import HFModel
from evaluation.strategies import get_strategy
from evaluation.tasks import get_task
from evaluation.tasks.base_task import BaseTask

logger = logging.getLogger(__name__)


class Evaluator:
    """
    Orchestrates multi-task evaluation of a language model with a layer-skipping
    strategy.

    Example usage::

        evaluator = Evaluator(
            model_name="meta-llama/Llama-3.2-1B-Instruct",
            strategy_name="layerskip",
            strategy_kwargs={"exit_ratio": 0.75},
            tasks=["mmlu", "hellaswag"],
            batch_size=4,
        )
        results = evaluator.run()
        evaluator.print_results(results)

    Args:
        model_name: HuggingFace model identifier or local path.
        strategy_name: Name of the layer-skipping strategy (``"none"``,
            ``"layerskip"``, ``"caml"``, ``"gateskip"``).
        strategy_kwargs: Extra keyword arguments forwarded to the strategy
            constructor, overriding defaults.
        tasks: List of task names to evaluate.  Each name must be registered
            in :data:`evaluation.tasks.TASK_REGISTRY`.
        task_kwargs: Optional dict mapping task name → kwargs dict, used to
            override per-task defaults (e.g. ``{"mmlu": {"max_samples": 100}}``).
        batch_size: Batch size for loglikelihood evaluation.
        device: Target device (``"cuda"``, ``"cuda:0"``, ``"cpu"``, …).
        dtype: Model dtype string (``"auto"``, ``"float16"``, ``"bfloat16"``).
        max_length: Maximum token sequence length.
        trust_remote_code: Passed to ``AutoModel.from_pretrained``.
    """

    def __init__(
        self,
        model_name: str,
        strategy_name: str = "none",
        strategy_kwargs: Optional[Dict[str, Any]] = None,
        tasks: Optional[List[str]] = None,
        task_kwargs: Optional[Dict[str, Dict[str, Any]]] = None,
        batch_size: int = 1,
        device: str = "auto",
        dtype: str = "auto",
        max_length: int = 2048,
        trust_remote_code: bool = False,
    ) -> None:
        self.model_name = model_name
        self.strategy_name = strategy_name
        self.strategy_kwargs = strategy_kwargs or {}
        self.task_names = tasks or []
        self.task_kwargs = task_kwargs or {}
        self.batch_size = batch_size
        self.dtype = dtype
        self.max_length = max_length
        self.trust_remote_code = trust_remote_code

        import torch

        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

    # ------------------------------------------------------------------ #
    # Model / strategy construction                                        #
    # ------------------------------------------------------------------ #

    def _build_model(self) -> HFModel:
        strategy = get_strategy(self.strategy_name, **self.strategy_kwargs)
        return HFModel(
            model_name=self.model_name,
            strategy=strategy,
            device=self.device,
            batch_size=self.batch_size,
            dtype=self.dtype,
            max_length=self.max_length,
            trust_remote_code=self.trust_remote_code,
        )

    def _build_task(self, name: str) -> BaseTask:
        kwargs = self.task_kwargs.get(name, {})
        return get_task(name, **kwargs)

    # ------------------------------------------------------------------ #
    # Evaluation                                                           #
    # ------------------------------------------------------------------ #

    def run(self) -> Dict[str, Any]:
        """
        Execute the evaluation and return a results dictionary.

        Returns:
            A nested dict with the structure::

                {
                    "model": "<model_name>",
                    "strategy": "<strategy_name>",
                    "strategy_config": {...},
                    "results": {
                        "<task_name>": {
                            "<metric>": <value>,
                            ...
                        },
                        ...
                    },
                    "elapsed_seconds": <float>,
                }
        """
        logger.info(
            "Building model %s with strategy '%s' …",
            self.model_name,
            self.strategy_name,
        )
        model = self._build_model()

        strategy_config: Dict[str, Any] = {}
        if model.strategy is not None:
            strategy_config = model.strategy.config

        results: Dict[str, Any] = {}
        t0 = time.time()

        for task_name in self.task_names:
            logger.info("Evaluating task '%s' …", task_name)
            task = self._build_task(task_name)
            t_task = time.time()
            task_results = task.evaluate(model)
            elapsed = time.time() - t_task
            logger.info(
                "Task '%s' done in %.1f s: %s",
                task_name,
                elapsed,
                task_results,
            )
            results[task_name] = task_results

        total_elapsed = time.time() - t0

        return {
            "model": self.model_name,
            "strategy": self.strategy_name,
            "strategy_config": strategy_config,
            "results": results,
            "elapsed_seconds": round(total_elapsed, 2),
        }

    # ------------------------------------------------------------------ #
    # Reporting                                                            #
    # ------------------------------------------------------------------ #

    @staticmethod
    def print_results(eval_output: Dict[str, Any]) -> None:
        """Pretty-print evaluation results to stdout."""
        print("\n" + "=" * 60)
        print(f"Model    : {eval_output['model']}")
        print(f"Strategy : {eval_output['strategy']}")
        if eval_output.get("strategy_config"):
            print(f"Config   : {eval_output['strategy_config']}")
        print("=" * 60)

        results = eval_output.get("results", {})
        for task_name, metrics in results.items():
            print(f"\n  Task: {task_name}")
            for metric, value in metrics.items():
                pct = f"{value * 100:.2f}%" if isinstance(value, float) else str(value)
                print(f"    {metric:20s}: {pct}")

        elapsed = eval_output.get("elapsed_seconds", 0)
        print(f"\nTotal elapsed: {elapsed:.1f} s")
        print("=" * 60 + "\n")

    @staticmethod
    def save_results(eval_output: Dict[str, Any], path: str) -> None:
        """Save evaluation results to a JSON file."""
        with open(path, "w") as f:
            json.dump(eval_output, f, indent=2)
        logger.info("Results saved to %s", path)

    @staticmethod
    def compare_results(
        results_list: List[Dict[str, Any]],
    ) -> Dict[str, Dict[str, Dict[str, float]]]:
        """
        Compare multiple evaluation runs side-by-side.

        Args:
            results_list: List of dicts as returned by :meth:`run`.

        Returns:
            Nested dict: ``{task → {metric → {strategy → value}}}``.
        """
        comparison: Dict[str, Dict[str, Dict[str, float]]] = {}
        for run in results_list:
            strat = run.get("strategy", "unknown")
            for task, metrics in run.get("results", {}).items():
                comparison.setdefault(task, {})
                for metric, value in metrics.items():
                    comparison[task].setdefault(metric, {})
                    comparison[task][metric][strat] = value
        return comparison

    @staticmethod
    def print_comparison(comparison: Dict[str, Dict[str, Dict[str, float]]]) -> None:
        """Print a comparison table produced by :meth:`compare_results`."""
        for task, metrics in comparison.items():
            print(f"\nTask: {task}")
            for metric, strategy_scores in metrics.items():
                print(f"  {metric}:")
                for strategy, score in sorted(strategy_scores.items()):
                    pct = (
                        f"{score * 100:.2f}%"
                        if isinstance(score, float)
                        else str(score)
                    )
                    print(f"    {strategy:20s}: {pct}")
