from __future__ import annotations
import numpy as np


def gini(x: np.ndarray) -> float:
    """
    Compute Gini coefficient of non-negative array x.
    Returns 0 for all-zeros.
    """
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return 0.0
    if np.any(x < 0):
        raise ValueError("Gini requires non-negative values")
    if np.all(x == 0):
        return 0.0
    sorted_x = np.sort(x)
    n = x.size
    cumx = np.cumsum(sorted_x)
    g = (n + 1 - 2 * np.sum(cumx) / cumx[-1]) / n
    return float(g)


def wealth_histogram(x: np.ndarray, bins: int = 50):
    hist, bin_edges = np.histogram(x, bins=bins)
    return hist, bin_edges