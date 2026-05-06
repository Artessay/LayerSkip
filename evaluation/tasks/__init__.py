"""Task registry and convenience helpers."""

from evaluation.tasks.base_task import BaseTask
from evaluation.tasks.mmlu import MMLUTask
from evaluation.tasks.hellaswag import HellaSwagTask
from evaluation.tasks.winogrande import WinoGrandeTask
from evaluation.tasks.gsm8k import GSM8KTask
from evaluation.tasks.humaneval import HumanEvalTask

TASK_REGISTRY: dict = {
    "mmlu": MMLUTask,
    "hellaswag": HellaSwagTask,
    "winogrande": WinoGrandeTask,
    "gsm8k": GSM8KTask,
    "humaneval": HumanEvalTask,
}

_TASK_DEFAULTS: dict = {
    "mmlu": {"num_fewshot": 5},
    "hellaswag": {"num_fewshot": 0},
    "winogrande": {"num_fewshot": 0},
    "gsm8k": {"num_fewshot": 8},
    "humaneval": {"num_fewshot": 0},
}


def get_task(name: str, **kwargs) -> BaseTask:
    """
    Instantiate a task by name with optional parameter overrides.

    Args:
        name: One of the keys in :data:`TASK_REGISTRY`.
        **kwargs: Keyword arguments forwarded to the task constructor,
            overriding the defaults.

    Returns:
        An instantiated :class:`~evaluation.tasks.base_task.BaseTask`.

    Raises:
        ValueError: If ``name`` is not registered.
    """
    name_lower = name.lower()
    if name_lower not in TASK_REGISTRY:
        raise ValueError(
            f"Unknown task '{name}'. Available: {list(TASK_REGISTRY.keys())}"
        )
    cls = TASK_REGISTRY[name_lower]
    params = dict(_TASK_DEFAULTS.get(name_lower, {}))
    params.update(kwargs)
    return cls(**params)


__all__ = [
    "BaseTask",
    "MMLUTask",
    "HellaSwagTask",
    "WinoGrandeTask",
    "GSM8KTask",
    "HumanEvalTask",
    "TASK_REGISTRY",
    "get_task",
]
