"""
check_bh98.py -- hand-run provenance diagnostic for the (shelved) plain
two-type Brock-Hommes generator.

Runs the Brock-Hommes generator across a few intensity-of-choice (beta) values
and prints the stylized-fact diagnostics. The purpose of this step is to LOOK
at what the model actually produces, not to assume it matches markets. We are
checking for: positive excess kurtosis (fat tails), near-zero autocorrelation
of raw returns, and positive, slowly decaying autocorrelation of squared /
absolute returns (volatility clustering).

This script is not part of run_all.sh; it documents why the plain model was
shelved (it explodes for all but the lowest intensity, and where it is bounded
it is thin-tailed with no clustering).

Run from the repo root:  python check_bh98.py
"""

import numpy as np
from models.bh98 import BHParams, simulate_bh98
from signatures.moments import stylized_facts
from seeds import get_rng, MASTER_SEED


def fmt(arr):
    return "[" + ", ".join(f"{v:+.3f}" for v in arr) + "]"


def report(name, r, sf):
    print(f"\n=== {name} ===")
    print(f"  n={sf['n']}  mean={sf['mean']:+.4f}  std={sf['std']:.4f}")
    print(f"  excess kurtosis : {sf['excess_kurtosis']:+.3f}   (>0 = fat tails)")
    print(f"  Hill tail alpha : {sf['hill_alpha']:.3f}        (~2-4 = heavy)")
    print(f"  ACF returns      lags {sf['acf_lags']}: {fmt(sf['acf_returns'])}  (~0 expected)")
    print(f"  ACF sq-returns   lags {sf['vol_lags']}: {fmt(sf['acf_sq_returns'])}  (>0 = vol cluster)")
    print(f"  ACF abs-returns  lags {sf['vol_lags']}: {fmt(sf['acf_abs_returns'])}")


def verdict(sf):
    fat = sf["excess_kurtosis"] > 0.5
    no_ac = np.nanmax(np.abs(sf["acf_returns"])) < 0.1
    volclust = np.nanmean(sf["acf_abs_returns"]) > 0.05
    tags = []
    tags.append("fat-tails OK" if fat else "tails ~thin")
    tags.append("returns uncorrelated OK" if no_ac else "returns autocorrelated")
    tags.append("vol-clustering OK" if volclust else "no vol-clustering")
    return "  -> " + " | ".join(tags)


if __name__ == "__main__":
    N = 5000
    BURN = 1000
    print(f"Brock-Hommes (1998) two-type model -- stylized-fact diagnostic")
    print(f"n={N}, burn_in={BURN}, master_seed={MASTER_SEED}")

    for beta in [1.0, 5.0, 10.0, 50.0, 100.0]:
        params = BHParams(beta=beta)
        rng = get_rng(MASTER_SEED, "bh98", "beta", beta)
        out = simulate_bh98(params, n=N, burn_in=BURN, rng=rng)
        if out["blew_up"] or not out["finite"]:
            print(f"\n=== BH98 beta={beta} ===\n  EXPLODED / non-finite -- "
                  f"this parameter regime is unusable.")
            continue
        r = out["returns"]
        sf = stylized_facts(r)
        report(f"BH98 beta={beta}", r, sf)
        print(verdict(sf))
