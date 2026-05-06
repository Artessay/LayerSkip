"""Abstract base class and common utilities for evaluation tasks."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from evaluation.models.base_model import BaseLM


class BaseTask(ABC):
    """
    Abstract base class for evaluation tasks.

    Each task defines how to:
    1. Load its dataset.
    2. Format examples into prompts and answer choices.
    3. Submit requests to the model (loglikelihood or generation).
    4. Evaluate model responses and aggregate metrics.

    This design mirrors the task interface used in EleutherAI's
    lm-evaluation-harness.
    """

    VERSION: int = 0

    # HuggingFace ``datasets`` identifier
    DATASET_PATH: str = ""
    DATASET_NAME: Optional[str] = None

    def __init__(
        self,
        num_fewshot: int = 0,
        max_samples: Optional[int] = None,
        seed: int = 42,
    ) -> None:
        """
        Args:
            num_fewshot: Number of in-context examples to prepend to each prompt.
            max_samples: Cap on the number of evaluation examples. ``None`` uses
                the full split.
            seed: Random seed used when sampling few-shot examples and
                subsampling the dataset.
        """
        self.num_fewshot = num_fewshot
        self.max_samples = max_samples
        self.seed = seed
        self._dataset = None

    # ------------------------------------------------------------------ #
    # Dataset loading                                                      #
    # ------------------------------------------------------------------ #

    def load_dataset(self) -> None:
        """Load and cache the dataset. Called lazily before evaluation."""
        if self._dataset is None:
            self._dataset = self._load_dataset()

    @abstractmethod
    def _load_dataset(self):
        """Return the raw HuggingFace dataset split used for evaluation."""

    # ------------------------------------------------------------------ #
    # Example formatting                                                   #
    # ------------------------------------------------------------------ #

    @abstractmethod
    def doc_to_text(self, doc: Dict[str, Any]) -> str:
        """Convert a dataset example to a text prompt (without the answer)."""

    @abstractmethod
    def doc_to_target(self, doc: Dict[str, Any]) -> str:
        """Return the expected answer/completion for an example."""

    def fewshot_examples(self, k: int, rng) -> List[Dict[str, Any]]:
        """
        Return ``k`` few-shot examples sampled from the training split.

        Override this method if the task uses a different split for few-shot
        examples, or if special formatting is required.
        """
        return []

    def fewshot_context(self, doc: Dict[str, Any]) -> str:
        """
        Build a prompt that includes ``num_fewshot`` in-context examples
        followed by the test example.
        """
        import random

        rng = random.Random(self.seed)
        examples = self.fewshot_examples(self.num_fewshot, rng)
        prompt_parts = []
        for ex in examples:
            prompt_parts.append(
                self.doc_to_text(ex) + self.doc_to_target(ex) + "\n"
            )
        prompt_parts.append(self.doc_to_text(doc))
        return "".join(prompt_parts)

    # ------------------------------------------------------------------ #
    # Request construction                                                 #
    # ------------------------------------------------------------------ #

    @abstractmethod
    def construct_requests(self, doc: Dict[str, Any], ctx: str) -> List[Any]:
        """
        Build the list of model requests for a single example.

        Returns a list of (context, continuation) pairs for loglikelihood
        tasks, or [(prompt, gen_kwargs)] for generation tasks.
        """

    # ------------------------------------------------------------------ #
    # Result processing                                                    #
    # ------------------------------------------------------------------ #

    @abstractmethod
    def process_results(
        self, doc: Dict[str, Any], results: List[Any]
    ) -> Dict[str, Any]:
        """
        Process model outputs for a single example.

        Args:
            doc: The original dataset example.
            results: Model outputs corresponding to :meth:`construct_requests`.

        Returns:
            Dict mapping metric name to its value for this example.
        """

    @abstractmethod
    def aggregation(self) -> Dict[str, Any]:
        """
        Return a dict mapping metric names to aggregation functions.

        Aggregation functions take a list of per-example metric values and
        return a scalar.  Example: ``{"accuracy": mean}``.
        """

    @abstractmethod
    def higher_is_better(self) -> Dict[str, bool]:
        """Return a dict indicating whether higher metric values are better."""

    # ------------------------------------------------------------------ #
    # Full evaluation pipeline                                             #
    # ------------------------------------------------------------------ #

    def evaluate(self, model: BaseLM) -> Dict[str, float]:
        """
        Run the full evaluation pipeline and return aggregated metrics.

        Args:
            model: An instance of :class:`~evaluation.models.base_model.BaseLM`.

        Returns:
            Dict mapping metric names to their aggregated scalar values.
        """
        self.load_dataset()
        docs = list(self._dataset)
        if self.max_samples is not None:
            import random

            rng = random.Random(self.seed)
            docs = rng.sample(docs, min(self.max_samples, len(docs)))

        all_requests = []
        request_doc_map = []  # (doc_idx, request_slice)

        for doc_idx, doc in enumerate(docs):
            ctx = self.fewshot_context(doc)
            requests = self.construct_requests(doc, ctx)
            start = len(all_requests)
            all_requests.extend(requests)
            request_doc_map.append((doc_idx, start, start + len(requests)))

        # Determine request type: loglikelihood vs generation
        request_type = self._infer_request_type(all_requests)

        if request_type == "loglikelihood":
            raw_results = model.loglikelihood(all_requests)
        else:
            raw_results = model.generate_until(all_requests)

        # Aggregate per-example metrics
        metric_lists: Dict[str, list] = {k: [] for k in self.aggregation()}
        for doc_idx, start, end in request_doc_map:
            doc = docs[doc_idx]
            doc_results = raw_results[start:end]
            per_example = self.process_results(doc, doc_results)
            for metric, value in per_example.items():
                if metric in metric_lists:
                    metric_lists[metric].append(value)

        agg_fns = self.aggregation()
        return {metric: agg_fns[metric](vals) for metric, vals in metric_lists.items()}

    @staticmethod
    def _infer_request_type(requests: list) -> str:
        """Detect whether requests are loglikelihood or generation."""
        if not requests:
            return "loglikelihood"
        first = requests[0]
        if isinstance(first, tuple) and isinstance(first[1], dict):
            return "generation"
        return "loglikelihood"

    @property
    def name(self) -> str:
        """Human-readable task name."""
        return self.__class__.__name__.lower()

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"num_fewshot={self.num_fewshot}, max_samples={self.max_samples})"
        )
