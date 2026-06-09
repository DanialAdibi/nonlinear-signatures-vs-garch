"""
signatures/bds.py -- BDS test statistic (Brock, Dechert, Scheinkman, LeBaron).

Brock, W. A., Dechert, W. D., Scheinkman, J. A. & LeBaron, B. (1996). A test for
independence based on the correlation dimension. Econometric Reviews 15, 197-235.

BDS tests the null that a series is i.i.d. Under H0 the m-dimensional
correlation integral factorises, C_m(eps) -> C_1(eps)^m, and the standardised
discrepancy is asymptotically N(0,1). Large |BDS| indicates dependence -- of any
kind (linear, nonlinear, or chaotic), so BDS is a general dependence detector,
not a chaos test. Distance uses the sup (max) norm, as in the original test.
"""

import numpy as np


def _indicator(x, eps):
    """A[i, j] = 1 if |x_i - x_j| < eps (m=1 sup-norm indicator)."""
    d = np.abs(np.subtract.outer(x, x))
    return (d < eps).astype(np.float64)


def bds_statistic(x, m=2, eps_mult=1.0, max_n=2000):
    """
    Returns dict: bds (standardised statistic at dimension m), C1, K, C_m.
    eps = eps_mult * std(x). Subsamples to max_n.
    """
    x = np.asarray(x, float)
    if len(x) > max_n:
        x = x[:max_n]
    N = len(x)
    eps = eps_mult * np.std(x)
    A = _indicator(x, eps)
    np.fill_diagonal(A, 0.0)

    C1 = A.sum() / (N * (N - 1))

    h = A.sum(axis=1) / (N - 1) 
    K = np.mean(h * h)

    Nm = N - (m - 1)
    prod = np.ones((Nm, Nm))
    for s in range(m):
        prod *= A[s:s + Nm, s:s + Nm]
    np.fill_diagonal(prod, 0.0)
    Cm = prod.sum() / (Nm * (Nm - 1))

    var = K ** m + (m - 1) ** 2 * C1 ** (2 * m) - m ** 2 * K * C1 ** (2 * m - 2)
    for j in range(1, m):
        var += 2 * (K ** (m - j)) * (C1 ** (2 * j))
    var *= 4.0
    sigma = np.sqrt(max(var, 1e-300))

    bds = np.sqrt(N) * (Cm - C1 ** m) / sigma if sigma > 0 else np.nan
    return {"bds": float(bds), "C1": float(C1), "K": float(K),
            "Cm": float(Cm), "m": m, "eps": float(eps)}
