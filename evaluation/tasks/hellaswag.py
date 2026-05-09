"""
HellaSwag commonsense reasoning task.

Dataset: Rowan/hellaswag (HuggingFace)
Metric: Accuracy (normalised log-likelihood of correct ending)

Reference:
    Zellers et al., 2019. https://arxiv.org/abs/1905.07830
"""

import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from evaluation.tasks.base_task import BaseTask


def _preprocess(text: str) -> str:
    """Clean HellaSwag text by removing HTML-like tags and normalising spaces."""
    text = text.strip()
    text = re.sub(r"\[header\]", "\n", text)
    text = re.sub(r"\[step\]", "\n", text)
    text = re.sub(r"\[substeps\]", "\n", text)
    text = re.sub(r"\[title\]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


class HellaSwagTask(BaseTask):
    """
    HellaSwag commonsense NLI task.

    Given an activity description and a partial sentence, the model must choose
    which of four endings best completes the sentence.  Evaluation uses
    *length-normalised* log-likelihood (dividing by the number of continuation
    tokens) so that shorter candidates are not unfairly favoured.

    Args:
        num_fewshot: Number of in-context examples (default 0).
        max_samples: Maximum number of evaluation examples (``None`` = all).
        seed: Random seed.
    """

    VERSION = 1
    DATASET_PATH = "Rowan/hellaswag"

    def __init__(
        self,
        num_fewshot: int = 0,
        max_samples: Optional[int] = None,
        seed: int = 42,
    ) -> None:
        super().__init__(num_fewshot=num_fewshot, max_samples=max_samples, seed=seed)

    def _load_dataset(self):
        from datasets import load_dataset

        local_path = Path(self.DATASET_PATH)
        if local_path.exists():
            validation_files = sorted(local_path.rglob("validation-*.parquet"))
            if validation_files:
                return load_dataset(
                    "parquet",
                    data_files={"validation": [str(path) for path in validation_files]},
                    split="validation",
                )

        return load_dataset(self.DATASET_PATH, split="validation")

    def _load_calibration_dataset(self):
        return self._load_dataset()

    @property
    def calibration_split_name(self) -> str:
        return "validation"

    def doc_to_text(self, doc: Dict[str, Any]) -> str:
        ctx = _preprocess(doc["ctx"])
        return f"{ctx}"

    def doc_to_target(self, doc: Dict[str, Any]) -> str:
        label = int(doc["label"])
        return " " + _preprocess(doc["endings"][label])

    def construct_requests(
        self, doc: Dict[str, Any], ctx: str
    ) -> List[tuple]:
        return [
            (ctx, " " + _preprocess(ending)) for ending in doc["endings"]
        ]

    def process_results(
        self, doc: Dict[str, Any], results: List[Any]
    ) -> Dict[str, Any]:
        # Length-normalised log-likelihood
        log_likelihoods = [r[0] for r in results]
        endings = doc["endings"]
        from evaluation.utils.metrics import length_normalise

        # Tokenisation happens in the model so we use raw char length as proxy
        normalised = [
            ll / max(len(e.split()), 1)
            for ll, e in zip(log_likelihoods, endings)
        ]
        predicted = int(max(range(len(normalised)), key=normalised.__getitem__))
        correct = int(doc["label"])
        return {"accuracy": int(predicted == correct)}

    def aggregation(self) -> Dict[str, Any]:
        return {"accuracy": _mean}

    def higher_is_better(self) -> Dict[str, bool]:
        return {"accuracy": True}

    @property
    def name(self) -> str:
        return "hellaswag"


def _mean(values: list) -> float:
    return sum(values) / len(values) if values else 0.0
