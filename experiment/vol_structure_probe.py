"""
experiment/vol_structure_probe.py -- Step 3 of the EGARCH-puzzle dig:
NAME the volatility statistic the laminarity feature is detecting.

CONTEXT
Step 1 ruled out a simulation artifact; Step 2 localised the FW-vs-EGARCH gap to the
recurrence/geometry cluster (laminarity, correlation dimension, determinism), i.e.
volatility-regime structure, not trajectory divergence. This step tries to reduce that
to a concrete, named volatility statistic.

THE SELECTIVITY DESIGN
The discrimination uses RANK-MATCHED nulls, so FW, the GJR null and the EGARCH null all
share an identical marginal distribution (same values, reordered). Any difference in a
volatility-STRUCTURE statistic is therefore purely temporal arrangement -- exactly the
higher-order structure the moments cannot see. The puzzle is that FW-vs-GJR collapses
(GJR matches FW) while FW-vs-EGARCH separates (EGARCH does not). So the statistic we want
is the one where:
    EGARCH sits FAR from FW   AND   GJR sits CLOSE to FW.
That statistic is what laminarity detects: the structure GJR reproduces but EGARCH breaks.
A second FW batch (FWb) is carried as a calibration column -- its distance from FW should
be ~0 for every statistic, or that statistic is too noisy to trust.

STATISTICS (computed per path on rank-matched paths; local vol = trailing rolling std,
window 22):
  volvol        std of log local-vol            (how much volatility itself fluctuates)
  volvol_kurt   excess kurtosis of log local-vol (tail of the vol-of-vol)
  long_mem_abs  sum of ACF(|r|) over lags 11..100 (long memory beyond the moment lags)
  acf_abs_50    ACF(|r|) at lag 50              (a representative long lag)
  hi_vol_runlen mean run-length above median local vol  (turbulent-spell dwell time)
  lo_vol_runlen mean run-length below median local vol  (quiet-spell dwell time)

READOUT
For each statistic: FW/GJR/EGARCH medians, then signed Cohen's d of each null vs FW, plus
selectivity = |d(FW,EGARCH)| - |d(FW,GJR)|. The carrier is the statistic with the largest
positive selectivity and clean calibration (small |d(FW,FWb)|). If one stands out, the
puzzle has a named mechanism (decisive follow-up: match EGARCH to FW on that statistic and
confirm the +0.146 edge drops). If none separates EGARCH from FW more than GJR does, the
laminar difference is not reducible to these named statistics -> honest limitation.

Run:  python -u -m experiment.vol_structure_probe          # K=300
Output: results/vol_structure_probe.txt
"""

import os
import sys
import warnings
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.stats import kurtosis as _kurt

from experiment.harder_pair_lev import build_or_load
from experiment.egarch_null import egarch_fit_and_simulate
from signatures.moments import autocorr
from seeds import MASTER_SEED

warnings.filterwarnings("ignore")
RESULT_FILE = "results/vol_structure_probe.txt"
OUT = []

STATS = ["volvol", "volvol_kurt", "long_mem_abs", "acf_abs_50",
         "hi_vol_runlen", "lo_vol_runlen"]


def log(s=""):
    print(s, flush=True)
    OUT.append(s)
    os.makedirs("results", exist_ok=True)
    with open(RESULT_FILE, "w") as f:
        f.write("\n".join(OUT))


def local_vol(r, w=22):
    r = np.asarray(r, float)
    if len(r) < w:
        return np.array([np.std(r)])
    return sliding_window_view(r, w).std(axis=1)


def mean_run_length(indicator):
    """Mean length of consecutive True runs in a boolean array."""
    idx = np.asarray(indicator).astype(int)
    runs, c = [], 0
    for b in idx:
        if b:
            c += 1
        elif c:
            runs.append(c)
            c = 0
    if c:
        runs.append(c)
    return float(np.mean(runs)) if runs else 0.0


def vol_stats(r):
    v = local_vol(r, 22)
    lv = np.log(v + 1e-12)
    ar = np.abs(np.asarray(r, float))
    med = np.median(v)
    hi = v > med
    long_lags = list(range(11, 101))
    return {
        "volvol": float(np.std(lv)),
        "volvol_kurt": float(_kurt(lv, fisher=True, bias=False)),
        "long_mem_abs": float(np.nansum(autocorr(ar, long_lags))),
        "acf_abs_50": float(autocorr(ar, [50])[0]),
        "hi_vol_runlen": mean_run_length(hi),
        "lo_vol_runlen": mean_run_length(~hi),
    }


def stat_matrix(paths):
    rows = [vol_stats(r) for r in paths]
    return {s: np.array([row[s] for row in rows]) for s in STATS}


def cohend(a, b):
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    p = np.sqrt((a.var() + b.var()) / 2.0) + 1e-12
    return float((a.mean() - b.mean()) / p)


def run(K=300, T=2500, alpha_lev=50.0):
    log("=" * 92)
    log("STEP 3 -- volatility-structure probe (name the statistic EGARCH misses)")
    log(f"K={K}  T={T}  alpha_lev={alpha_lev}  rank-matched nulls (marginals identical)")
    log("=" * 92)

    fw, fwb, gc_gjr, _gam = build_or_load(K, T, alpha_lev)
    log(f"loaded {len(fw)} FW + {len(fwb)} FWb + {len(gc_gjr)} GJR (cached)")
    log("fitting EGARCH null (one-time, rank_match=True, slow) ...")
    gc_eg, eg_gam = egarch_fit_and_simulate(fw, MASTER_SEED, rank_match=True)
    n = min(len(fw), len(fwb), len(gc_gjr), len(gc_eg))
    fw, fwb, gc_gjr, gc_eg = fw[:n], fwb[:n], gc_gjr[:n], gc_eg[:n]
    log(f"  aligned {n} paths per group"
        + (f"; EGARCH gamma mean={np.mean(eg_gam):+.4f}" if len(eg_gam) else ""))

    log("\ncomputing volatility-structure statistics per path ...")
    SFW = stat_matrix(fw)
    SFWb = stat_matrix(fwb)
    SGJR = stat_matrix(gc_gjr)
    SEG = stat_matrix(gc_eg)

    log("\n" + "-" * 92)
    log(f"  {'statistic':14}{'FW med':>10}{'GJR med':>10}{'EG med':>10}"
        f"{'d(FW,FWb)':>11}{'d(FW,GJR)':>11}{'d(FW,EG)':>11}{'select':>9}")
    log("-" * 92)
    rows = []
    for s in STATS:
        d_cal = cohend(SFWb[s], SFW[s])
        d_gjr = cohend(SGJR[s], SFW[s])
        d_eg = cohend(SEG[s], SFW[s])
        select = abs(d_eg) - abs(d_gjr)
        rows.append((s, d_cal, d_gjr, d_eg, select))
        log(f"  {s:14}{np.median(SFW[s]):>10.3f}{np.median(SGJR[s]):>10.3f}"
            f"{np.median(SEG[s]):>10.3f}{d_cal:>+11.2f}{d_gjr:>+11.2f}{d_eg:>+11.2f}{select:>+9.2f}")

    max_cal = max(abs(r[1]) for r in rows)
    rows_sorted = sorted(rows, key=lambda r: r[4], reverse=True)
    top = rows_sorted[0]
    log("\n" + "=" * 92)
    log(f"calibration: largest |d(FW,FWb)| across statistics = {max_cal:.2f}"
        f" ({'clean, A-vs-A near zero' if max_cal < 0.25 else 'WARNING: noisy, interpret with care'})")
    found = (top[4] > 0.30 and abs(top[3]) > 0.40
             and abs(top[3]) > abs(top[2]) + 0.30 and abs(top[1]) < 0.25)
    if found:
        direction = "higher" if top[3] > 0 else "lower"
        log(f"VERDICT: MECHANISM FOUND -- '{top[0]}' carries it.")
        log(f"  EGARCH departs from FW (d={top[3]:+.2f}, the null runs {direction} than FW)")
        log(f"  while GJR stays close (d={top[2]:+.2f}); selectivity {top[4]:+.2f}.")
        log(f"  This is the volatility-regime property laminarity detects: GJR's additive")
        log(f"  variance recursion reproduces FW's '{top[0]}', EGARCH's log-variance recursion")
        log(f"  does not. DECISIVE follow-up: match the EGARCH null to FW on '{top[0]}'")
        log(f"  (or condition on it) and confirm the +0.146 edge collapses; if it does, the")
        log(f"  puzzle becomes a finding -- the nonlinear tools catch a regime-persistence")
        log(f"  mis-specification that standard GARCH diagnostics (clustering, leverage) miss.")
    else:
        log("VERDICT: NO single named statistic separates EGARCH from FW more than GJR does.")
        log(f"  (best candidate '{top[0]}': selectivity {top[4]:+.2f}, d(FW,EG)={top[3]:+.2f},")
        log(f"   d(FW,GJR)={top[2]:+.2f}.)  The laminar/recurrence difference is real (Steps 1-2)")
        log("  but not reducible to these standard volatility summaries. Honest reporting:")
        log("  the EGARCH edge survives for a structural reason the recurrence features capture")
        log("  but conventional volatility diagnostics do not -- a limitation, stated plainly,")
        log("  not a mechanism. (Could extend the probe, e.g. vol asymmetry persistence or")
        log("  multifractal/long-memory exponents, but diminishing returns past this point.)")
    log("=" * 92)
    print(f"\n[saved to {RESULT_FILE}]")


if __name__ == "__main__":
    K = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    run(K=K)
