"""
signatures/rqa.py -- recurrence quantification analysis (RQA).

Marwan, N. et al. (2007). Recurrence plots for the analysis of complex systems.
Physics Reports 438, 237-329.

The recurrence matrix R_ij = 1 if ||x_i - x_j|| <= eps (Theiler-excluded near
the diagonal). The threshold eps is set to achieve a target recurrence rate RR.
Determinism DET is the fraction of recurrence points lying on diagonal lines of
length >= l_min: deterministic systems produce long diagonals (high DET);
stochastic systems produce isolated points (low DET). LAM is the analogous
quantity for vertical lines (laminar states).
"""

import numpy as np
from scipy.spatial.distance import pdist, squareform

from signatures.embedding import embed


def _line_points(binary_lines, l_min):
    """Count points that belong to runs of length >= l_min in a 1-D 0/1 array."""
    total = 0
    run = 0
    for v in binary_lines:
        if v:
            run += 1
        else:
            if run >= l_min:
                total += run
            run = 0
    if run >= l_min:
        total += run
    return total


def recurrence_quant(x, m, tau, rr=0.05, theiler=None, l_min=3, max_n=2000):
    """
    Returns dict: det, lam, rr_actual, eps. Subsamples to max_n for the O(N^2)
    recurrence matrix. l_min=3 (rather than the conventional 2) gives a cleaner
    separation between deterministic and stochastic series, since chance 2-point
    diagonals inflate DET for noise.
    """
    x = np.asarray(x, float)
    emb = embed(x, m, tau)
    N = len(emb)
    if theiler is None:
        theiler = tau * m
    if N > max_n:
        emb = emb[:max_n]
        N = max_n

    D = squareform(pdist(emb))

    band = np.abs(np.subtract.outer(np.arange(N), np.arange(N))) <= theiler
    offband = ~band
    eps = np.quantile(D[offband], rr)          # threshold for target recurrence rate
    R = (D <= eps) & offband
    rr_actual = R.sum() / offband.sum()
    total_points = R.sum()
    if total_points == 0:
        return {"det": np.nan, "lam": np.nan, "rr_actual": 0.0, "eps": float(eps)}

    diag_points = 0
    for k in range(1, N):                  
        diag = np.diag(R, k)
        diag_points += _line_points(diag, l_min)
    diag_points *= 2                     
    det = diag_points / total_points

    vert_points = 0
    for j in range(N):
        vert_points += _line_points(R[:, j], l_min)
    lam = vert_points / total_points

    return {"det": float(det), "lam": float(lam),
            "rr_actual": float(rr_actual), "eps": float(eps),
            "m": m, "tau": tau, "theiler": theiler, "l_min": l_min}
