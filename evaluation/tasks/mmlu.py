"""
MMLU (Massive Multitask Language Understanding) evaluation task.

Dataset: cais/mmlu (HuggingFace)
Metric: Accuracy (0-shot or few-shot multiple choice)

Reference:
    Hendrycks et al., 2020. https://arxiv.org/abs/2009.03300
"""

import logging
from typing import Any, Dict, List, Optional

from evaluation.tasks.base_task import BaseTask
from evaluation.utils.progress import progress

_CHOICES = ["A", "B", "C", "D"]
logger = logging.getLogger(__name__)

# Subjects grouped for convenience; all 57 subjects are evaluated by default
_ALL_SUBJECTS = [
    "abstract_algebra", "anatomy", "astronomy", "business_ethics",
    "clinical_knowledge", "college_biology", "college_chemistry",
    "college_computer_science", "college_mathematics", "college_medicine",
    "college_physics", "computer_security", "conceptual_physics",
    "econometrics", "electrical_engineering", "elementary_mathematics",
    "formal_logic", "global_facts", "high_school_biology",
    "high_school_chemistry", "high_school_computer_science",
    "high_school_european_history", "high_school_geography",
    "high_school_government_and_politics", "high_school_macroeconomics",
    "high_school_mathematics", "high_school_microeconomics",
    "high_school_physics", "high_school_psychology", "high_school_statistics",
    "high_school_us_history", "high_school_world_history", "human_aging",
    "human_sexuality", "international_law", "jurisprudence",
    "logical_fallacies", "machine_learning", "management", "marketing",
    "medical_genetics", "miscellaneous", "moral_disputes", "moral_scenarios",
    "nutrition", "philosophy", "prehistory", "professional_accounting",
    "professional_law", "professional_medicine", "professional_psychology",
    "public_relations", "security_studies", "sociology", "us_foreign_policy",
    "virology", "world_religions",
]


class MMLUTask(BaseTask):
    """
    MMLU evaluation task.

    Evaluates the model on all 57 subjects (or a configurable subset) using
    log-likelihood scoring over the four answer choices (A/B/C/D).

    Args:
        subjects: List of subject names to evaluate. Defaults to all 57.
        num_fewshot: Number of few-shot examples (0–5 as in the original paper).
        max_samples: Per-subject sample cap. ``None`` evaluates all examples.
        seed: Random seed.
    """

    VERSION = 1
    DATASET_PATH = "cais/mmlu"

    def __init__(
        self,
        subjects: Optional[List[str]] = None,
        num_fewshot: int = 5,
        max_samples: Optional[int] = None,
        seed: int = 42,
    ) -> None:
        super().__init__(num_fewshot=num_fewshot, max_samples=max_samples, seed=seed)
        self.subjects = subjects if subjects is not None else _ALL_SUBJECTS

    def _load_subject_split(self, split: str, desc: str):
        from datasets import load_dataset, concatenate_datasets

        splits = []
        for subject in progress(
            self.subjects,
            desc=desc,
            total=len(self.subjects),
            unit="subject",
        ):
            ds = load_dataset(self.DATASET_PATH, subject, split=split)
            splits.append(ds)
        return concatenate_datasets(splits) if len(splits) > 1 else splits[0]

    def _load_dataset(self):
        return self._load_subject_split("test", "mmlu: load test subjects")

    def _load_fewshot_dataset(self):
        return self._load_subject_split("dev", "mmlu: load dev subjects")

    def _load_calibration_dataset(self):
        last_error = None
        for split in ("validation", "train", "dev", "test"):
            try:
                dataset = self._load_subject_split(
                    split,
                    f"mmlu: load calibration {split} subjects",
                )
            except Exception as exc:
                last_error = exc
                logger.debug("MMLU calibration split '%s' unavailable: %s", split, exc)
                continue
            self._calibration_split_name = split
            return dataset
        raise RuntimeError("No MMLU calibration split could be loaded") from last_error

    @property
    def calibration_split_name(self) -> str:
        return getattr(self, "_calibration_split_name", "validation")

    def fewshot_examples(self, k: int, rng) -> List[Dict[str, Any]]:
        if k == 0:
            return []
        dev_set = self._load_fewshot_dataset()
        examples = list(dev_set)
        rng.shuffle(examples)
        return examples[:k]

    def doc_to_text(self, doc: Dict[str, Any]) -> str:
        choices_str = "\n".join(
            f"{_CHOICES[i]}. {choice}"
            for i, choice in enumerate(doc["choices"])
        )
        return (
            f"Question: {doc['question']}\n"
            f"{choices_str}\n"
            "Answer:"
        )

    def doc_to_target(self, doc: Dict[str, Any]) -> str:
        return f" {_CHOICES[doc['answer']]}"

    def construct_requests(
        self, doc: Dict[str, Any], ctx: str
    ) -> List[tuple]:
        """Return one loglikelihood request per answer choice."""
        return [(ctx, f" {c}") for c in _CHOICES]

    def process_results(
        self, doc: Dict[str, Any], results: List[Any]
    ) -> Dict[str, Any]:
        log_likelihoods = [r[0] for r in results]
        predicted = int(max(range(len(log_likelihoods)), key=log_likelihoods.__getitem__))
        correct = int(doc["answer"])
        return {"accuracy": int(predicted == correct)}

    def aggregation(self) -> Dict[str, Any]:
        return {"accuracy": _mean}

    def higher_is_better(self) -> Dict[str, bool]:
        return {"accuracy": True}

    @property
    def name(self) -> str:
        return "mmlu"


def _mean(values: list) -> float:
    return sum(values) / len(values) if values else 0.0
