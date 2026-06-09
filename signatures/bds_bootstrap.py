"""
signatures/bds_bootstrap.py -- BDS with a bootstrap (permutation) null.

WHY THIS EXISTS
The asymptotic BDS calibration (standardised statistic ~ N(0,1)) is known to be
unreliable in finite samples with heavy tails -- exactly the regime of financial
returns -- so a BDS rejection that relies on the asymptotic normal can be
spurious. This module replaces the asymptotic null with an empirical one built by
resampling, giving a p-value and critical values that do not depend on the N(0,1)
approximation.

THE NULL
BDS tests independence (i.i.d.). The natural finite-sample null that preserves the
marginal distribution but destroys all temporal dependence is a RANDOM PERMUTATION
of the series. Under permutation the values (and hence the tails, variance, every
order statistic) are exactly preserved, while any serial structure -- linear,
nonlinear, volatility clustering -- is removed. Recomputing the BDS discrepancy on
many permutations yields its distribution under H0 for THIS sample's marginal,
heavy tails and all. This is the appropriate calibration for the heavy-tailed case.

WHAT IS REPORTED
For the observed series we compute the raw discrepancy D = C_m - C_1^m and the
asymptotically-standardised statistic (from bds_statistic, for reference). The
bootstrap p-value is the two-sided fraction of permutation discrepancies at least
as extreme as the observed D. We also report bootstrap critical values and a
bootstrap-standardised z (observed D minus permutation mean, over permutation sd),
which is the heavy-tail-robust analogue of the asymptotic statistic.

USAGE
    from signatures.bds_bootstrap import bds_bootstrap
    res = bds_bootstrap(returns, m=2, n_boot=999, rng=get_rng(seed, "bds", tag))
    res["p_value"], res["z_boot"], res["bds_asymp"]

NOTE
Permutation (sampling without replacement) is preferred over the iid bootstrap
(sampling with replacement) here because it preserves the marginal exactly; with
replacement it would only preserve it in expectation. For BDS independence testing
both are valid; permutation is the cleaner statement.
"""

import numpy as np
from signatures.bds import bds_statistic, _indicator


def _bds_discrepancy(x, m, eps):
    """Raw BDS discrepancy D = C_m - C_1^m at a FIXED eps (no standardisation).

    eps is passed in (not recomputed) so that every permutation uses the same
    threshold as the observed series -- otherwise the permutation null would
    also absorb sampling variation in std(x), which we do not want since
    permutation preserves std(x) exactly anyway.
    """
    x = np.asarray(x, float)
    N = len(x)
    A = _indicator(x, eps)
    np.fill_diagonal(A, 0.0)
    C1 = A.sum() / (N * (N - 1))
    Nm = N - (m - 1)
    prod = np.ones((Nm, Nm))
    for s in range(m):
        prod *= A[s:s + Nm, s:s + Nm]
    np.fill_diagonal(prod, 0.0)
    Cm = prod.sum() / (Nm * (Nm - 1))
    return Cm - C1 ** m


def bds_bootstrap(x, m=2, eps_mult=1.0, max_n=2000, n_boot=999, rng=None):
    """
    BDS test with a permutation null (heavy-tail robust).

    Returns dict with:
      bds_asymp : asymptotically-standardised BDS statistic (for reference)
      D_obs     : observed raw discrepancy C_m - C_1^m
      p_value   : two-sided permutation p-value
      z_boot    : bootstrap-standardised statistic (D_obs vs permutation null)
      crit_lo, crit_hi : 2.5%/97.5% permutation critical values for D
      n_boot, m, eps
    """
    if rng is None:
        rng = np.random.default_rng()
    x = np.asarray(x, float)
    if len(x) > max_n:
        x = x[:max_n]
    N = len(x)
    eps = eps_mult * np.std(x)

    # observed statistics (asymptotic version kept for reference/comparison)
    asymp = bds_statistic(x, m=m, eps_mult=eps_mult, max_n=max_n)
    D_obs = _bds_discrepancy(x, m, eps)

    # permutation null distribution of the discrepancy
    null = np.empty(n_boot)
    for b in range(n_boot):
        xp = rng.permutation(x)
        null[b] = _bds_discrepancy(xp, m, eps)

    mu, sd = null.mean(), null.std()
    z_boot = (D_obs - mu) / sd if sd > 0 else np.nan
    # two-sided p-value, centred on the permutation mean; +1 smoothing
    extreme = np.abs(null - mu) >= np.abs(D_obs - mu)
    p_value = (extreme.sum() + 1) / (n_boot + 1)
    crit_lo, crit_hi = np.percentile(null, [2.5, 97.5])

    return {"bds_asymp": asymp["bds"], "D_obs": float(D_obs),
            "p_value": float(p_value), "z_boot": float(z_boot),
            "crit_lo": float(crit_lo), "crit_hi": float(crit_hi),
            "null_mean": float(mu), "null_sd": float(sd),
            "n_boot": n_boot, "m": m, "eps": float(eps)}