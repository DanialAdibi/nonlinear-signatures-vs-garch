"""
experiment/null_fit_quality.py -- WHY does the nonlinear edge collapse against
the GJR null but SURVIVE (and grow) against the EGARCH null?

BACKGROUND
The EGARCH alternative-null check (egarch_null.py) found that the leverage-arm
collapse is NOT robust to the asymmetric-null functional form. At K=300:
  - FW vs GJR (leverage engaged):   dAUC ~ 0 across m   (collapses, as claimed)
  - FW vs EGARCH (leverage engaged): dAUC RISES to ~+0.15 at m=8 -- larger even
    than FW vs the symmetric GARCH null (~+0.07).
So a null with a leverage TERM is not automatically enough to kill the edge.

THE THESIS PREDICTS A SPECIFIC EXPLANATION
The project's conclusion is that the nonlinear edge measures how badly the null
captures FW's dynamics, not whether the null carries an asymmetry label. That
thesis makes a falsifiable prediction about THIS surprise: the EGARCH null must
reproduce FW's volatility dynamics WORSE than the GJR null. If it does, the
surviving FW-vs-EGARCH edge is explained (EGARCH stays mis-specified, so the
nonlinear features detect the gap) and the thesis holds in a sharpened form. If
EGARCH instead matches FW's dynamics as WELL as GJR yet the edge survives, the
thesis is challenged and the EGARCH result is a genuine problem.

WHAT THIS SCRIPT DOES
Fits GJR and EGARCH to the SAME leverage FW paths, simulates each model's NATIVE
(non-rank-matched) null, and compares both nulls against FW on the two dynamics
the nonlinear features key on -- the same quantities check_cont_facts validates:
  - volatility clustering : ACF(|r|) at lags 1, 5, 10, 25   (signatures.moments.autocorr)
  - leverage              : L(tau) = corr(r_t, r_{t+tau}^2) at tau 1,2,3,5,10

Native (not rank-matched) simulations are used on purpose: rank-matching only
forces the MARGINAL to match FW and cannot repair a temporal/leverage mismatch,
which is precisely what we are measuring. (The discrimination experiment rank-
matches the marginal, so any gap this diagnostic finds is one the rank-match
leaves untouched -- i.e. exactly what Referee-2 features can still see.)

DECISIVE READ (printed at the end)
  dist(null, FW) = mean |null_profile - FW_profile| over the lags / taus.
  If dist(EGARCH,FW) > dist(GJR,FW) on leverage and/or clustering -> EGARCH is the
  poorer null, explaining the surviving/growing edge: SUPPORTS the thesis.
  If the two distances are comparable -> the EGARCH survival is NOT a fit-quality
  artefact and must be confronted on its own terms.

Run:  python -u -m experiment.null_fit_quality        # K=300, matches egarch_null
      python -u -m experiment.null_fit_quality 100    # quicker, K=100
Output: results/null_fit_quality.txt
"""

import os
import sys
import warnings
import numpy as np

from experiment.harder_pair_lev import build_or_load
from experiment.egarch_null import egarch_fit_and_simulate
from experiment.harder_pair import garch_match
from signatures.moments import autocorr
from seeds import MASTER_SEED

warnings.filterwarnings("ignore")
RESULT_FILE = "results/null_fit_quality.txt"
OUT = []
LAGS = (1, 5, 10, 25)
TAUS = (1, 2, 3, 5, 10)


def log(s=""):
    print(s, flush=True)
    OUT.append(s)
    os.makedirs("results", exist_ok=True)
    with open(RESULT_FILE, "w") as f:
        f.write("\n".join(OUT))


def acf_abs_profile(r):
    """ACF(|r|) at LAGS -- same estimator check_cont_facts uses (Fact 6/8)."""
    return np.asarray(autocorr(np.abs(np.asarray(r, float)), LAGS), float)


def leverage_profile(r):
    """L(tau)=corr(r_t, r_{t+tau}^2) at TAUS -- same as check_cont_facts (Fact 9)."""
    r = np.asarray(r, float)
    out = []
    for tau in TAUS:
        a, b = r[:-tau], r[tau:] ** 2
        c = np.corrcoef(a, b)[0, 1] if np.std(a) > 0 and np.std(b) > 0 else np.nan
        out.append(c)
    return np.asarray(out, float)


def cloud_mean(paths, fn):
    """Average a per-path profile across the cloud (paths may be any length)."""
    return np.nanmean(np.array([fn(r) for r in paths]), axis=0)


def fmt_row(label, profile, ref=None):
    vals = "".join(f"{v:>+9.3f}" for v in profile)
    if ref is None:
        return f"  {label:18}{vals}"
    dist = np.nanmean(np.abs(profile - ref))
    return f"  {label:18}{vals}   | dist {dist:>6.3f}"


def run(K=300, T=2500, alpha_lev=50.0):
    log("=" * 78)
    log("NULL FIT-QUALITY DIAGNOSTIC -- GJR vs EGARCH reproduction of FW dynamics")
    log(f"K={K}  T={T}  alpha_lev={alpha_lev}  seed={MASTER_SEED}")
    log("=" * 78)

    # FW leverage paths (reuse the cached set the experiment used)
    fw, _fwb, _gc, gammas = build_or_load(K, T, alpha_lev)
    log(f"loaded {len(fw)} FW leverage paths (cached); "
        f"reference GJR gamma mean={np.mean(gammas):+.4f}")

    # NATIVE nulls: each model's own simulation, NOT rank-matched (see header).
    log("\nsimulating native GJR null (rank_match=False) ...")
    gjr = [s for s in garch_match(fw, MASTER_SEED, rank_match=False) if s is not None]
    log(f"  got {len(gjr)} GJR null paths")
    log("fitting + simulating native EGARCH null (rank_match=False, slow) ...")
    eg, eg_g = egarch_fit_and_simulate(fw, MASTER_SEED, rank_match=False)
    log(f"  got {len(eg)} EGARCH null paths; "
        f"EGARCH gamma mean={np.mean(eg_g):+.4f} "
        f"({100 * np.mean(eg_g < 0):.0f}% negative)")

    # ---- volatility clustering: ACF(|r|) ----
    fw_acf, gjr_acf, eg_acf = (cloud_mean(p, acf_abs_profile) for p in (fw, gjr, eg))
    log("\n" + "-" * 78)
    log("VOLATILITY CLUSTERING  ACF(|r|)   lags = " + str(LAGS))
    log("-" * 78)
    log("  " + " " * 18 + "".join(f"{f'lag{L}':>9}" for L in LAGS))
    log(fmt_row("FW (target)", fw_acf))
    log(fmt_row("GJR null", gjr_acf, fw_acf))
    log(fmt_row("EGARCH null", eg_acf, fw_acf))
    acf_d_gjr = np.nanmean(np.abs(gjr_acf - fw_acf))
    acf_d_eg = np.nanmean(np.abs(eg_acf - fw_acf))

    # ---- leverage: L(tau) ----
    fw_lev, gjr_lev, eg_lev = (cloud_mean(p, leverage_profile) for p in (fw, gjr, eg))
    log("\n" + "-" * 78)
    log("LEVERAGE  L(tau) = corr(r_t, r_(t+tau)^2)   tau = " + str(TAUS))
    log("-" * 78)
    log("  " + " " * 18 + "".join(f"{f'tau{t}':>9}" for t in TAUS))
    log(fmt_row("FW (target)", fw_lev))
    log(fmt_row("GJR null", gjr_lev, fw_lev))
    log(fmt_row("EGARCH null", eg_lev, fw_lev))
    lev_d_gjr = np.nanmean(np.abs(gjr_lev - fw_lev))
    lev_d_eg = np.nanmean(np.abs(eg_lev - fw_lev))

    # ---- verdict ----
    log("\n" + "=" * 78)
    log("FIT-QUALITY DISTANCE TO FW  (mean |null - FW|, smaller = better fit)")
    log(f"  clustering ACF(|r|):   GJR {acf_d_gjr:.3f}   EGARCH {acf_d_eg:.3f}")
    log(f"  leverage   L(tau)  :   GJR {lev_d_gjr:.3f}   EGARCH {lev_d_eg:.3f}")
    eg_worse_clust = acf_d_eg > acf_d_gjr
    eg_worse_lev = lev_d_eg > lev_d_gjr
    log("")
    if eg_worse_clust or eg_worse_lev:
        worse = " and ".join(
            w for w, f in [("clustering", eg_worse_clust), ("leverage", eg_worse_lev)] if f)
        log(f"VERDICT: EGARCH reproduces FW's {worse} WORSE than GJR.")
        log("  -> The surviving/growing FW-vs-EGARCH nonlinear edge is consistent with")
        log("     the thesis: EGARCH stays mis-specified for FW's dynamics, so the")
        log("     nonlinear features detect the gap. The leverage-arm collapse against")
        log("     GJR reflects GJR's adequacy for FW, not the mere presence of an")
        log("     asymmetry term. (Sharpened claim, not a contradiction.)")
    else:
        log("VERDICT: EGARCH matches FW's dynamics about as well as GJR on these")
        log("  curves, yet the FW-vs-EGARCH edge still grows. This is NOT explained by")
        log("  fit quality and must be confronted directly -- it would challenge the")
        log("  'edge = under-specified null' reading. Investigate further before")
        log("  asserting robustness in the manuscript.")
    log("=" * 78)
    print(f"\n[saved to {RESULT_FILE}]")


if __name__ == "__main__":
    K = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    run(K=K)
