"""
signatures/moments.py -- "Referee 1": standard stylized-fact moments.

These are the conventional summary statistics used to validate behavioural
market models: fat tails (excess kurtosis, tail index), absence of linear
autocorrelation in raw returns, and slow-decaying autocorrelation in squared /
absolute returns (volatility clustering).
"""

import numpy as np
from scipy import stats


def excess_kurtosis(r):
    """Fisher (excess) kurtosis; 0 for a Gaussian, >0 for fat tails."""
    return float(stats.kurtosis(r, fisher=True, bias=False))


def autocorr(x, lags):
    """Sample autocorrelation of x at the given lags."""
    x = np.asarray(x, float)
    x = x - x.mean()
    denom = np.dot(x, x)
    out = []
    for k in lags:
        if k == 0:
            out.append(1.0)
        elif denom > 0 and k < len(x):
            out.append(float(np.dot(x[:-k], x[k:]) / denom))
        else:
            out.append(np.nan)
    return np.array(out)


def hill_tail_index(r, tail_frac=0.05):
    """
    Hill estimator of the tail index alpha on the upper tail of |r|.
    Lower alpha => heavier tails. ~2-4 is typical for financial returns;
    a Gaussian gives a (much) larger / effectively infinite alpha.
    """
    a = np.sort(np.abs(np.asarray(r, float)))[::-1]
    k = min(max(10, int(tail_frac * len(a))), len(a) - 1)
    if k < 2:
        return np.nan
    xi = float(np.mean(np.log(a[:k]) - np.log(a[k])))   # = 1/alpha
    return 1.0 / xi if xi > 0 else np.nan


def stylized_facts(r, acf_lags=(1, 2, 3, 5, 10), vol_lags=(1, 5, 10, 25)):
    """Compute the Referee-1 stylized-fact summary for a return series r."""
    r = np.asarray(r, float)
    return {
        "n": int(len(r)),
        "mean": float(np.mean(r)),
        "std": float(np.std(r)),
        "excess_kurtosis": excess_kurtosis(r),
        "hill_alpha": float(hill_tail_index(r)),
        "acf_returns": autocorr(r, acf_lags),
        "acf_sq_returns": autocorr(r ** 2, vol_lags),
        "acf_abs_returns": autocorr(np.abs(r), vol_lags),
        "acf_lags": acf_lags,
        "vol_lags": vol_lags,
    }
