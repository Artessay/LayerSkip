"""
LayerSkip Evaluation Framework

A lm-evaluation-harness style framework for comparing layer skipping strategies
(CAML, LayerSkip, GateSkip) across benchmarks (MMLU, HellaSwag, WinoGrande,
GSM8K, HumanEval) with Llama-3 backbone models.
"""

__version__ = "0.1.0"


def __getattr__(name):
    # Lazy imports to avoid requiring heavy dependencies at import time
    if name == "Evaluator":
        from evaluation.evaluator import Evaluator
        return Evaluator
    if name == "get_strategy":
        from evaluation.strategies import get_strategy
        return get_strategy
    if name == "STRATEGY_REGISTRY":
        from evaluation.strategies import STRATEGY_REGISTRY
        return STRATEGY_REGISTRY
    if name == "get_task":
        from evaluation.tasks import get_task
        return get_task
    if name == "TASK_REGISTRY":
        from evaluation.tasks import TASK_REGISTRY
        return TASK_REGISTRY
    raise AttributeError(f"module 'evaluation' has no attribute {name!r}")


__all__ = ["Evaluator", "get_strategy", "STRATEGY_REGISTRY", "get_task", "TASK_REGISTRY"]
