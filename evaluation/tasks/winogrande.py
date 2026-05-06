"""
WinoGrande commonsense pronoun resolution task.

Dataset: allenai/winogrande (HuggingFace)
Metric: Accuracy

Reference:
    Sakaguchi et al., 2019. https://arxiv.org/abs/1907.10641
"""

from typing import Any, Dict, List, Optional

from evaluation.tasks.base_task import BaseTask


class WinoGrandeTask(BaseTask):
    """
    WinoGrande Winograd-schema task.

    Each example provides a sentence with a ``_`` placeholder and two
    candidate completions (``option1``, ``option2``).  The model must choose
    the correct option by comparing the log-likelihood of the full sentence
    with each option substituted in.

    Args:
        size: WinoGrande dataset size variant (``"xs"``, ``"s"``, ``"m"``,
            ``"l"``, ``"xl"``).  Defaults to ``"xl"`` (largest split).
        num_fewshot: Number of in-context examples (default 0).
        max_samples: Maximum number of evaluation examples (``None`` = all).
        seed: Random seed.
    """

    VERSION = 1
    DATASET_PATH = "allenai/winogrande"

    def __init__(
        self,
        size: str = "winogrande_xl",
        num_fewshot: int = 0,
        max_samples: Optional[int] = None,
        seed: int = 42,
    ) -> None:
        super().__init__(num_fewshot=num_fewshot, max_samples=max_samples, seed=seed)
        self.size = size

    def _load_dataset(self):
        from datasets import load_dataset

        return load_dataset(
            self.DATASET_PATH, self.size, split="validation", trust_remote_code=True
        )

    def _partial_context(self, doc: Dict[str, Any], option: str) -> str:
        """Replace ``_`` with the given option in the sentence."""
        return doc["sentence"].replace("_", option)

    def doc_to_text(self, doc: Dict[str, Any]) -> str:
        # We return the sentence with placeholder as the "context"
        sentence = doc["sentence"]
        idx = sentence.index("_")
        return sentence[:idx]

    def doc_to_target(self, doc: Dict[str, Any]) -> str:
        answer_idx = int(doc["answer"]) - 1  # 1-indexed → 0-indexed
        return doc[f"option{answer_idx + 1}"]

    def construct_requests(
        self, doc: Dict[str, Any], ctx: str
    ) -> List[tuple]:
        """
        Build two loglikelihood requests: one per option.

        We compare:
            P(option1 text after ``_`` | text before ``_``)
            P(option2 text after ``_`` | text before ``_``)
        """
        sentence = doc["sentence"]
        idx = sentence.index("_")
        prefix = sentence[:idx]
        suffix = sentence[idx + 1 :]  # text after the blank

        requests = []
        for opt_key in ("option1", "option2"):
            option = doc[opt_key]
            continuation = option + suffix
            requests.append((prefix, continuation))
        return requests

    def process_results(
        self, doc: Dict[str, Any], results: List[Any]
    ) -> Dict[str, Any]:
        log_likelihoods = [r[0] for r in results]
        predicted = int(max(range(len(log_likelihoods)), key=log_likelihoods.__getitem__))
        correct = int(doc["answer"]) - 1  # 1-indexed → 0-indexed
        return {"accuracy": int(predicted == correct)}

    def aggregation(self) -> Dict[str, Any]:
        return {"accuracy": _mean}

    def higher_is_better(self) -> Dict[str, bool]:
        return {"accuracy": True}

    @property
    def name(self) -> str:
        return "winogrande"


def _mean(values: list) -> float:
    return sum(values) / len(values) if values else 0.0
