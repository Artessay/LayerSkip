"""
GSM8K grade-school math word problem evaluation task.

Dataset: openai/gsm8k (HuggingFace)
Metric: Exact match on the final numerical answer

Reference:
    Cobbe et al., 2021. https://arxiv.org/abs/2110.14168
"""

import re
from typing import Any, Dict, List, Optional

from evaluation.tasks.base_task import BaseTask

_ANSWER_RE = re.compile(r"####\s*([\-\d,\.]+)")
_NUMBER_RE = re.compile(r"[\-\d,]+(?:\.\d+)?")


def _extract_answer(text: str) -> Optional[str]:
    """
    Extract the final numerical answer from a GSM8K-formatted string.

    Looks for the ``#### <number>`` pattern used in the dataset.  If not
    found, falls back to extracting the last number in the string.
    """
    match = _ANSWER_RE.search(text)
    if match:
        return match.group(1).replace(",", "").strip()
    # Fallback: last number in text
    numbers = _NUMBER_RE.findall(text)
    if numbers:
        return numbers[-1].replace(",", "").strip()
    return None


def _extract_generated_answer(text: str) -> Optional[str]:
    """
    Extract an answer from a model-generated response.

    Handles several common patterns:
      - ``#### 42``
      - ``The answer is 42.``
      - ``= 42``
      - Plain last number
    """
    # #### pattern
    match = _ANSWER_RE.search(text)
    if match:
        return match.group(1).replace(",", "").strip()

    # "the answer is X" pattern
    ans_match = re.search(
        r"(?:the answer is|answer:|=)\s*([\-\d,]+(?:\.\d+)?)", text, re.IGNORECASE
    )
    if ans_match:
        return ans_match.group(1).replace(",", "").strip()

    # Last number
    numbers = _NUMBER_RE.findall(text)
    if numbers:
        return numbers[-1].replace(",", "").strip()

    return None


class GSM8KTask(BaseTask):
    """
    GSM8K math word problem task.

    The model is prompted with a problem and must generate a step-by-step
    solution ending with ``#### <answer>``.  Evaluation compares the
    extracted numerical answer against the reference.

    Args:
        num_fewshot: Number of in-context examples (8 as in the original paper).
        max_samples: Maximum number of evaluation examples (``None`` = all).
        seed: Random seed.
        max_new_tokens: Token budget for generation.
        chain_of_thought: Whether to include chain-of-thought prompting.
    """

    VERSION = 1
    DATASET_PATH = "openai/gsm8k"
    DATASET_NAME = "main"

    _COT_PREAMBLE = "Solve the following math problem step by step."
    _COT_INSTRUCTION = (
        f"{_COT_PREAMBLE} "
        "At the end, write the final answer after #### as a numeric value.\n\n"
    )

    def __init__(
        self,
        num_fewshot: int = 8,
        max_samples: Optional[int] = None,
        seed: int = 42,
        max_new_tokens: int = 512,
        chain_of_thought: bool = True,
    ) -> None:
        super().__init__(num_fewshot=num_fewshot, max_samples=max_samples, seed=seed)
        self.max_new_tokens = max_new_tokens
        self.chain_of_thought = chain_of_thought

    def _load_dataset(self):
        from datasets import load_dataset

        return load_dataset(self.DATASET_PATH, self.DATASET_NAME, split="test")

    def _load_fewshot_dataset(self):
        from datasets import load_dataset

        return load_dataset(self.DATASET_PATH, self.DATASET_NAME, split="train")

    def fewshot_examples(self, k: int, rng) -> List[Dict[str, Any]]:
        if k == 0:
            return []
        train = self._load_fewshot_dataset()
        examples = list(train)
        rng.shuffle(examples)
        return examples[:k]

    def doc_to_text(self, doc: Dict[str, Any]) -> str:
        prefix = self._COT_INSTRUCTION if self.chain_of_thought else ""
        return f"{prefix}Question: {doc['question']}\nAnswer:"

    def doc_to_target(self, doc: Dict[str, Any]) -> str:
        return " " + doc["answer"]

    def construct_requests(
        self, doc: Dict[str, Any], ctx: str
    ) -> List[tuple]:
        stop_sequences = ["\n\nQuestion:", "\nQuestion:"]
        if self.chain_of_thought:
            stop_sequences.append(f"\n{self._COT_PREAMBLE}")

        return [
            (
                ctx,
                {
                    "max_new_tokens": self.max_new_tokens,
                    "do_sample": False,
                    "stop_sequences": stop_sequences,
                },
            )
        ]

    def process_results(
        self, doc: Dict[str, Any], results: List[Any]
    ) -> Dict[str, Any]:
        generated = results[0]  # single generation result
        ref_answer = _extract_answer(doc["answer"])
        pred_answer = _extract_generated_answer(generated)
        correct = ref_answer is not None and pred_answer == ref_answer
        return {"exact_match": int(correct)}

    def aggregation(self) -> Dict[str, Any]:
        return {"exact_match": _mean}

    def higher_is_better(self) -> Dict[str, bool]:
        return {"exact_match": True}

    @property
    def name(self) -> str:
        return "gsm8k"


def _mean(values: list) -> float:
    return sum(values) / len(values) if values else 0.0
