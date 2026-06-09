"""
signatures/entropy.py -- complexity measures for the richer-feature robustness
probe (Phase 4 follow-up). Both are standard, cheap nonlinear descriptors that
complement the Lyapunov / dimension / recurrence signatures.

  permutation_entropy : Bandt-Pompe ordinal-pattern entropy (normalised 0..1).
  sample_entropy      : regularity statistic (lower = more self-similar/regular).
"""

import numpy as np
from math import factorial
from itertools import permutations


def permutation_entropy(x, order=3, delay=1):
    """Normalised Bandt-Pompe permutation entropy in [0, 1]."""
    x = np.asarray(x, float)
    n = len(x) - (order - 1) * delay
    if n <= 1:
        return np.nan
    patterns = {p: 0 for p in permutations(range(order))}
    for i in range(n):
        window = x[i: i + order * delay: delay]
        patterns[tuple(np.argsort(window))] += 1
    counts = np.array([c for c in patterns.values() if c > 0], float)
    p = counts / counts.sum()
    H = -np.sum(p * np.log(p))
    return float(H / np.log(factorial(order)))


def sample_entropy(x, m=2, r=0.2, max_n=1500):
    """Sample entropy SampEn(m, r*std). Subsamples to max_n (O(N^2))."""
    x = np.asarray(x, float)
    if len(x) > max_n:
        x = x[:max_n]
    N = len(x)
    tol = r * np.std(x)

    n_templates = N - m
    if n_templates <= 1:
        return np.nan

    def phi(mm):
        templates = np.array([x[i:i + mm] for i in range(n_templates)])
        count = 0
        for i in range(n_templates):
            d = np.max(np.abs(templates - templates[i]), axis=1)
            count += np.sum(d <= tol) - 1
        return count

    B = phi(m)
    A = phi(m + 1)
    if B == 0 or A == 0:
        return np.nan
    return float(-np.log(A / B))