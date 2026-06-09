"""
signatures/lyapunov.py -- largest Lyapunov exponent by the Rosenstein method.

Rosenstein, M. T., Collins, J. J. & De Luca, C. J. (1993). A practical method
for calculating largest Lyapunov exponents from small data sets. Physica D,
65(1-2), 117-134.

For each reference point we find its nearest neighbour (excluding temporally
close points via a Theiler window) and track the mean log distance between the
two trajectories over a number of forward steps. The slope of the (initially
linear) mean log-divergence curve over a fixed fit window estimates the largest
Lyapunov exponent lambda_1 (per sample). We also return the R^2 of that linear
fit -- the "scaling-region quality" -- which is itself a discriminating feature.
"""

import numpy as np
from scipy.spatial import cKDTree

from signatures.embedding import embed


def rosenstein(x, m, tau, theiler=None, n_steps=20, fit_start=0, fit_end=None):
    """
    Returns dict: lambda1 (slope), r2 (fit quality), divergence (mean log-div
    curve), and the parameters used. NaN-safe.
    """
    x = np.asarray(x, float)
    emb = embed(x, m, tau)
    M = len(emb)
    if theiler is None:
        theiler = tau * m
    if fit_end is None:
        fit_end = n_steps

    def _nan_result(divergence):
        return {"lambda1": np.nan, "r2": np.nan, "divergence": divergence,
                "m": m, "tau": tau, "theiler": theiler,
                "fit_start": fit_start, "fit_end": fit_end}

    n_ref = M - n_steps
    if n_ref <= 1:
        return _nan_result(None)

    tree = cKDTree(emb)
    
    nbr = np.full(n_ref, -1)
    ref = np.arange(n_ref)
    k_query = min(M, 2 * theiler + 5)
    todo = ref
    while len(todo) > 0:
        _, idx = tree.query(emb[todo], k=k_query)
        if idx.ndim == 1:
            idx = idx[:, None]
        for row, i in enumerate(todo):
            for jj in range(1, idx.shape[1]):
                j = int(idx[row, jj])
                if abs(j - i) > theiler and j < n_ref:
                    nbr[i] = j
                    break
        todo = ref[nbr < 0]
        if k_query >= M or len(todo) == 0:
            break
        k_query = min(M, k_query * 2)

    div = np.full(n_steps, np.nan)
    for k in range(n_steps):
        logs = []
        for i in range(n_ref):
            j = nbr[i]
            if j < 0:
                continue
            d = np.linalg.norm(emb[i + k] - emb[j + k])
            if d > 0:
                logs.append(np.log(d))
        if logs:
            div[k] = np.mean(logs)

    steps = np.arange(n_steps)
    mask = (steps >= fit_start) & (steps < fit_end) & np.isfinite(div)
    if mask.sum() < 2:
        return _nan_result(div)

    xs, ys = steps[mask], div[mask]
    slope, intercept = np.polyfit(xs, ys, 1)
    yhat = slope * xs + intercept
    ss_res = np.sum((ys - yhat) ** 2)
    ss_tot = np.sum((ys - ys.mean()) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    return {
        "lambda1": float(slope),
        "r2": float(r2),
        "divergence": div,
        "m": m, "tau": tau, "theiler": theiler,
        "fit_start": fit_start, "fit_end": fit_end,
    }