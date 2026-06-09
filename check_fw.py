"""
check_fw.py -- gate for the Franke-Westerhoff generator (plan "Phase 2").

Mirrors check_bh98.py: runs the FW DCA-HPM model and prints the same
stylized-fact diagnostics, so the two generators are inspected identically.
We expect FW (unlike the toy 2-type BH) to produce bounded returns with fat
tails, near-zero return autocorrelation, and slowly decaying volatility
clustering.

Run from the repo root:  python check_fw.py
"""

import numpy as np
from models.fw_ssv import FWParams, simulate_fw
from signatures.moments import stylized_facts
from seeds import get_rng, MASTER_SEED


def fmt(arr):
    return "[" + ", ".join(f"{v:+.3f}" for v in arr) + "]"


def report(name, sf):
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
    tags = [
        "fat-tails OK" if fat else "tails ~thin",
        "returns uncorrelated OK" if no_ac else "returns autocorrelated",
        "vol-clustering OK" if volclust else "no vol-clustering",
    ]
    return "  -> " + " | ".join(tags)


if __name__ == "__main__":
    N, BURN = 5000, 1000
    print("Franke-Westerhoff DCA-HPM model -- stylized-fact diagnostic")
    print(f"n={N}, burn_in={BURN}, master_seed={MASTER_SEED}")
    print("(parameters are the verified FW 2012 Table 1 DCA-HPM MSM estimates; see fw_ssv.py)")

    for mu in [0.001, 0.01, 0.05]:
        params = FWParams(mu=mu)
        rng = get_rng(MASTER_SEED, "fw", "mu", mu)
        out = simulate_fw(params, n=N, burn_in=BURN, rng=rng)
        if out["blew_up"] or not out["finite"]:
            print(f"\n=== FW mu={mu} ===\n  EXPLODED / non-finite -- unusable regime.")
            continue
        sf = stylized_facts(out["returns"])
        tag = "  <-- default, market-like" if mu == 0.01 else ""
        report(f"FW mu={mu}{tag}", sf)
        print(verdict(sf))