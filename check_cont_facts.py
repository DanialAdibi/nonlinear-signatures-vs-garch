"""
check_cont_facts.py -- benchmark FW output against Cont (2001) stylized facts.

Cont, R. (2001). Empirical properties of asset returns: stylized facts and
statistical issues. Quantitative Finance 1, 223-236.

Checks the UNIVARIATE facts relevant to a single-asset daily return generator
(the multivariate / volume / multifractal facts in Cont sections 6-7 are out of
scope for this study). For each, we report the FW value averaged over many paths
and whether it matches Cont's stated behaviour:

  Fact 1  No linear autocorrelation of returns        -> max|ACF(r)| ~ 0
  Fact 2  Heavy tails, tail index alpha in (2,5)       -> Hill alpha in (2,5)
  Fact 3  Gain/loss asymmetry (skewness)               -> FW expected ~0 (symmetric)
  Fact 4  Aggregational Gaussianity                    -> excess kurtosis falls with horizon
  Fact 6/8 Volatility clustering, slow |r|-ACF decay   -> ACF(|r|) positive, slow
  Fact 9  Leverage effect L(tau)=corr(r_t, r_{t+tau}^2) -> FW expected ~0 (symmetric)

Reading: FW is a SYMMETRIC speculative-market model, so it is expected to PASS the
core facts (1,2,4,6,8) and NOT to reproduce the equity asymmetries (3,9) -- Cont
himself notes Fact 3 fails for exchange rates, so a symmetric generator is a
defensible market model; the asymmetry facts are simply out of FW's scope (and,
note, the GJR-GARCH null DOES carry a leverage term).

Run from the repo root:   python check_cont_facts.py
"""

import numpy as np
from scipy import stats

from models.fw_ssv import FWParams, simulate_fw
from signatures.moments import excess_kurtosis, autocorr, hill_tail_index
from seeds import get_rng, MASTER_SEED


def fw_returns(K, T, burn_in=1500, alpha_lev=0.0):
    out = []
    for s in range(K):
        r = simulate_fw(FWParams(switching="dca", alpha_lev=alpha_lev), n=T,
                        burn_in=burn_in, rng=get_rng(MASTER_SEED, "cont", alpha_lev, s))
        if r["finite"] and not r["blew_up"]:
            out.append(r["returns"])
    return out


def aggregate(r, h):
    """Non-overlapping h-period (summed log) returns."""
    n = (len(r) // h) * h
    return r[:n].reshape(-1, h).sum(axis=1)


def leverage_curve(r, taus=(1, 2, 3, 5, 10)):
    """L(tau) = corr(r_t, r_{t+tau}^2). Negative => leverage effect."""
    r = r - r.mean()
    out = []
    for tau in taus:
        a, b = r[:-tau], r[tau:] ** 2
        out.append(float(np.corrcoef(a, b)[0, 1]))
    return np.array(out)


def mean_sd(vals):
    v = np.array([x for x in vals if np.isfinite(x)])
    return float(v.mean()), float(v.std())


def run(K=40, T=10000, alpha_lev=0.0):
    print("=" * 66)
    print("CONT (2001) STYLIZED-FACT CHECK on Franke-Westerhoff (DCA) returns")
    variant = "SYMMETRIC (alpha_lev=0)" if alpha_lev == 0 else f"ASYMMETRIC (alpha_lev={alpha_lev})"
    print(f"variant={variant}")
    print(f"K={K} paths, T={T}, seed={MASTER_SEED}")
    print("=" * 66)

    R = fw_returns(K, T, alpha_lev=alpha_lev)
    print(f"\ngenerated {len(R)} FW return paths\n")

    # Fact 1 -- no linear autocorrelation of returns
    ac_max = [np.nanmax(np.abs(autocorr(r, (1, 2, 3, 5, 10)))) for r in R]
    m, s = mean_sd(ac_max)
    print(f"[Fact 1] max|ACF(returns)| lags1-10 : {m:.3f} +/- {s:.3f}   "
          f"({'PASS ~0' if m < 0.10 else 'CHECK'})")

    # Fact 2 -- heavy tails, tail index in (2,5)
    alpha = [hill_tail_index(r) for r in R]
    m, s = mean_sd(alpha)
    print(f"[Fact 2] Hill tail index alpha      : {m:.2f} +/- {s:.2f}   "
          f"({'PASS in (2,5)' if 2 < m < 5 else 'CHECK'})")

    # Fact 3 -- gain/loss asymmetry (skewness); FW expected ~0
    sk = [float(stats.skew(r)) for r in R]
    m, s = mean_sd(sk)
    print(f"[Fact 3] skewness                   : {m:+.3f} +/- {s:.3f}   "
          f"({'~0 => SYMMETRIC (no gain/loss asymmetry, as expected)' if abs(m) < 0.2 else 'asymmetric'})")

    # Fact 4 -- aggregational Gaussianity (excess kurtosis falls with horizon)
    print(f"[Fact 4] aggregational Gaussianity (excess kurtosis by horizon):")
    prev = None
    monotone = True
    for h in (1, 5, 20, 40):
        kv = [excess_kurtosis(aggregate(r, h)) for r in R]
        mh, _ = mean_sd(kv)
        print(f"           horizon {h:>3}: excess kurtosis = {mh:+.2f}")
        if prev is not None and mh > prev + 0.05:
            monotone = False
        prev = mh
    print(f"           -> {'PASS (kurtosis declines toward 0)' if monotone else 'CHECK (not monotone)'}")

    # Fact 6/8 -- volatility clustering, slow-decaying |r| ACF
    print(f"[Fact 6/8] ACF(|returns|) by lag (positive, slow decay):")
    lags = (1, 5, 10, 25)
    means = []
    for lag in lags:
        vals = [autocorr(np.abs(r), (lag,))[0] for r in R]
        mh, _ = mean_sd(vals)
        means.append(mh)
        print(f"           lag {lag:>2}: ACF(|r|) = {mh:+.3f}")
    ok = means[0] > 0.05 and means[-1] > 0 and means[-1] < means[0]
    print(f"           -> {'PASS (positive, slowly decaying)' if ok else 'CHECK'}")

    # Fact 9 -- leverage effect; FW expected ~0
    lev = np.array([leverage_curve(r) for r in R])
    lev_m = lev.mean(axis=0)
    print(f"[Fact 9] leverage L(tau)=corr(r_t, r_(t+tau)^2), tau=1,2,3,5,10:")
    print("           " + "  ".join(f"{v:+.3f}" for v in lev_m))
    has_lev = np.any(lev_m < -0.03)
    if alpha_lev == 0:
        print(f"           -> {'~0 => NO leverage effect (symmetric model, as expected)' if not has_lev else 'unexpected leverage'}")
    else:
        print(f"           -> {'NEGATIVE => leverage effect PRESENT (as intended for the asymmetric variant)' if has_lev else 'CHECK: leverage not detected -- raise alpha_lev'}")

    if alpha_lev == 0:
        print("\nSummary: symmetric FW is expected to PASS Facts 1,2,4,6,8 (the core")
        print("univariate market facts) and NOT reproduce the equity asymmetries")
        print("(Facts 3,9). Facts 10,11 and cross-asset / multifractal facts are out")
        print("of scope for this study.")
    else:
        print("\nSummary: the asymmetric variant is intended to ADD a leverage effect")
        print("(Fact 9, conditional/volatility asymmetry) on top of the core facts")
        print("(1,2,4,6,8), while the unconditional return distribution stays")
        print("symmetric (Fact 3 ~ 0) -- leverage is a volatility-response asymmetry,")
        print("distinct from distributional skewness. Facts 10,11 and cross-asset /")
        print("multifractal facts remain out of scope.")


if __name__ == "__main__":
    import sys
    lev = float(sys.argv[1]) if len(sys.argv) > 1 else 0.0
    run(alpha_lev=lev)