"""Abstract base class for language models used in evaluation."""

from abc import ABC, abstractmethod
from typing import List, Optional, Tuple


class BaseLM(ABC):
    """
    Minimal interface that every language model wrapper must implement.

    The two core operations mirror those used by lm-evaluation-harness:

    * :meth:`loglikelihood` – compute log-probability of a continuation given a
      context.  Used by multiple-choice tasks (MMLU, HellaSwag, WinoGrande).
    * :meth:`generate` – produce a text completion given a prompt.  Used by
      open-ended generation tasks (GSM8K, HumanEval).
    """

    # ------------------------------------------------------------------ #
    # Required interface                                                   #
    # ------------------------------------------------------------------ #

    @abstractmethod
    def loglikelihood(
        self, requests: List[Tuple[str, str]]
    ) -> List[Tuple[float, bool]]:
        """
        Compute log-likelihood of continuations given contexts.

        Args:
            requests: List of ``(context, continuation)`` string pairs.

        Returns:
            List of ``(log_likelihood, is_greedy)`` tuples, one per request.
            ``is_greedy`` is ``True`` when the continuation matches the greedy
            decode of the context.
        """

    @abstractmethod
    def generate_until(
        self,
        requests: List[Tuple[str, dict]],
    ) -> List[str]:
        """
        Generate text until stop conditions are met.

        Args:
            requests: List of ``(prompt, gen_kwargs)`` pairs.
                ``gen_kwargs`` may contain keys such as ``max_new_tokens``,
                ``temperature``, ``stop_sequences``.

        Returns:
            List of generated strings (one per request).
        """

    # ------------------------------------------------------------------ #
    # Optional helpers                                                     #
    # ------------------------------------------------------------------ #

    @property
    def device(self) -> str:
        """Target device (e.g. ``"cuda:0"`` or ``"cpu"``)."""
        return "cpu"

    @property
    def batch_size(self) -> int:
        """Preferred evaluation batch size."""
        return 1

    def __repr__(self) -> str:
        return self.__class__.__name__
