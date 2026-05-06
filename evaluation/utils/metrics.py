"""Shared metric helper functions."""

from typing import List, Optional


def mean(values: List[float]) -> float:
    """Arithmetic mean of a list of floats. Returns 0.0 for empty lists."""
    return sum(values) / len(values) if values else 0.0


def accuracy(correct: List[int]) -> float:
    """Fraction of correct predictions (values are 0 or 1)."""
    return mean([float(c) for c in correct])


def length_normalise(log_likelihood: float, num_tokens: int) -> float:
    """Divide log-likelihood by sequence length for normalised scoring."""
    return log_likelihood / max(num_tokens, 1)


def pass_at_k(n: int, c: int, k: int) -> float:
    """
    Unbiased pass@k estimator from Chen et al. (2021).

    Args:
        n: Total number of samples generated per problem.
        c: Number of correct samples.
        k: k in pass@k.

    Returns:
        pass@k estimate in [0, 1].
    """
    if n - c < k:
        return 1.0
    from math import comb

    return 1.0 - comb(n - c, k) / comb(n, k)
