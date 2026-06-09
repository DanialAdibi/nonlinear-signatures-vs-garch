"""
signatures/dimension.py -- Grassberger-Procaccia correlation dimension d2.

Grassberger, P. & Procaccia, I. (1983). Characterization of strange attractors.
Phys. Rev. Lett. 50(5), 346-349.

The correlation integral C(r) is the fraction of point pairs closer than r
(excluding temporally close pairs via a Theiler window). For a low-dimensional
attractor, C(r) ~ r^{d2} over a scaling region, so d2 is the slope of
log C(r) vs log r there. For stochastic data the points fill the embedding
space and d2 grows with the embedding dimension m (no low finite value).
"""

import numpy as np
from scipy.spatial import cKDTree

from signatures.embedding import embed


def correlation_dimension(x, m, tau, theiler=None, n_radii=20, max_n=2000,
                          c_lo=0.005, c_hi=0.2, rng=None):
    """
    Returns dict: d2 (scaling-region slope), r2 (fit quality), and the log-log
    correlation-integral curve. Subsamples to max_n points (contiguously) for
    the O(N^2) neighbour counts.
    """
    x = np.asarray(x, float)
    emb = embed(x, m, tau)
    N = len(emb)
    if theiler is None:
        theiler = tau * m
    if N > max_n:                  
        emb = emb[:max_n]
        N = max_n

    tree = cKDTree(emb)

    d_all = tree.query(emb, k=2)[0][:, 1]      
    r_min = max(np.percentile(d_all, 5), 1e-9)
    r_max = np.linalg.norm(emb.max(0) - emb.min(0))
    radii = np.logspace(np.log10(r_min), np.log10(r_max), n_radii)

    total_pairs = N * (N - 1) / 2
    band_pairs = sum(N - k for k in range(1, theiler + 1))   
    valid_pairs = total_pairs - band_pairs


    if theiler >= 1:
        band_d = np.concatenate([np.linalg.norm(emb[k:] - emb[:-k], axis=1)
                                 for k in range(1, theiler + 1)])
    else:
        band_d = np.empty(0)

    C = np.empty(n_radii)
    for q, r in enumerate(radii):
        full = tree.count_neighbors(tree, r)            
        within = (full - N) / 2.0                       
        band = np.count_nonzero(band_d <= r)    
        C[q] = max(within - band, 0) / valid_pairs

    mask = (C >= c_lo) & (C <= c_hi) & (C > 0)
    if mask.sum() < 3:
        mask = C > 0
    lr, lc = np.log(radii[mask]), np.log(C[mask])
    if len(lr) < 2:
        return {"d2": np.nan, "r2": np.nan, "radii": radii, "C": C}
    slope, intercept = np.polyfit(lr, lc, 1)
    yhat = slope * lr + intercept
    ss_res = np.sum((lc - yhat) ** 2)
    ss_tot = np.sum((lc - lc.mean()) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return {"d2": float(slope), "r2": float(r2), "radii": radii, "C": C,
            "m": m, "tau": tau, "theiler": theiler}