"""
signatures/features.py -- assemble the full feature vector for one return
series, on a single FIXED embedding (m, tau) shared across all models.

Two named groups so the discrimination engine can train on each separately:

  REFEREE 1 (standard moments)  : the conventional stylized-fact checklist.
  REFEREE 2 (nonlinear signatures): Lyapunov slope + scaling R^2, correlation
                                    dimension d2 + its R^2, recurrence DET/LAM,
                                    and the BDS statistic.

The study question is whether REFEREE 2 separates the DCA/TPA pair better than
REFEREE 1 does.
"""

import numpy as np

from signatures.moments import excess_kurtosis, autocorr, hill_tail_index
from signatures.lyapunov import rosenstein
from signatures.dimension import correlation_dimension
from signatures.rqa import recurrence_quant
from signatures.bds import bds_statistic

REFEREE_1 = ["std", "excess_kurtosis", "hill_alpha",
             "acf_abs_1", "acf_abs_5", "acf_abs_10", "acf_ret_max"]
REFEREE_2 = ["lambda1", "lyap_r2", "d2", "d2_r2", "det", "lam", "bds"]
ALL_FEATURES = REFEREE_1 + REFEREE_2


def feature_vector(returns, m=4, tau=1, rr=0.05, n_steps=20, fit_end=10,
                   max_n_d2=1800, max_n_rqa=1400, max_n_bds=2000):
    """Return an ordered dict of all features for one return series."""
    r = np.asarray(returns, float)

    # Referee 1 -- moments
    acf_abs = autocorr(np.abs(r), (1, 5, 10))
    acf_ret = autocorr(r, (1, 2, 3, 5, 10))
    f = {
        "std": float(np.std(r)),
        "excess_kurtosis": excess_kurtosis(r),
        "hill_alpha": float(hill_tail_index(r)),
        "acf_abs_1": float(acf_abs[0]),
        "acf_abs_5": float(acf_abs[1]),
        "acf_abs_10": float(acf_abs[2]),
        "acf_ret_max": float(np.nanmax(np.abs(acf_ret))),
    }

    # Referee 2 -- nonlinear signatures (fixed embedding)
    ly = rosenstein(r, m=m, tau=tau, n_steps=n_steps, fit_start=0, fit_end=fit_end)
    d2 = correlation_dimension(r, m=m, tau=tau, max_n=max_n_d2)
    rq = recurrence_quant(r, m=m, tau=tau, rr=rr, max_n=max_n_rqa)
    bd = bds_statistic(r, m=2, max_n=max_n_bds)
    f.update({
        "lambda1": ly["lambda1"],
        "lyap_r2": ly["r2"],
        "d2": d2["d2"],
        "d2_r2": d2["r2"],
        "det": rq["det"],
        "lam": rq["lam"],
        "bds": bd["bds"],
    })
    return f


def feature_matrix(series_list, **kwargs):
    """Stack feature vectors into an (n_series, n_features) array + name list.

    Each series' features are computed once, then the row is built by indexing
    that dict with ALL_FEATURES by name, so a column always corresponds to its
    named feature regardless of dict insertion order; a missing feature fails
    loudly (KeyError) rather than silently mislabelling the matrix.
    """
    rows = []
    for r in series_list:
        fv = feature_vector(r, **kwargs)
        rows.append([fv[name] for name in ALL_FEATURES])
    return np.asarray(rows), ALL_FEATURES