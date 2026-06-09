"""
experiment/null_pathology_check.py -- Step 1 of the EGARCH-puzzle dig:
is the surviving FW-vs-EGARCH nonlinear edge a SIMULATION ARTIFACT?

CONTEXT
null_fit_quality.py (K=300) found GJR and EGARCH reproduce FW's clustering and
leverage equally, yet FW-vs-EGARCH separates (dAUC +0.146 at m=8) while FW-vs-GJR
collapses. Before reading that gap as a real higher-order effect, rule out the dull
explanation: EGARCH models the LOG variance, which can blow up into freak variance
spikes if the fitted persistence (the beta on lagged log-variance) sits near the
stability edge. A handful of pathological paths would inflate the recurrence and
dimension features and fake the whole gap.

WHAT IT CHECKS, per fitted null (GJR and EGARCH), against FW:
  - fitted persistence vs the stability edge
      EGARCH: |beta[1]| -> 1  (log-variance AR root near unity = near-explosive)
      GJR   : alpha[1] + beta[1] + gamma[1]/2 -> 1
  - simulated-path health: excess kurtosis and max|r|/std (the largest standardised
    move on a path), plus a count of pathological paths (non-finite, or a spike far
    beyond anything FW produces)
Same paths, seeds, scaling and rank_match=False as the experiment, so the paths are
the ones the discrimination actually scored.

DECISIVE READ (printed at the end)
  CLEAN  : EGARCH paths are well-behaved -- kurtosis and max-move comparable to FW
           and GJR, beta safely below 1, no/few pathological paths. The edge is NOT a
           simulation artifact; the puzzle is real -> proceed to Step 2 (localise the
           feature) and Step 3 (higher-order volatility statistics).
  FLAGGED: EGARCH shows near-explosive beta and/or freak spikes that FW and GJR lack.
           The edge is (partly) a simulation artifact -> fix or downweight the EGARCH
           arm; the puzzle dissolves.

Run:  python -u -m experiment.null_pathology_check        # K=300, T=2500
      python -u -m experiment.null_pathology_check 100    # quicker
Output: results/null_pathology_check.txt
"""

import os
import sys
import warnings
import numpy as np
from scipy.stats import kurtosis as _kurt

from experiment.harder_pair_lev import build_or_load
from seeds import get_rng, MASTER_SEED

warnings.filterwarnings("ignore")
RESULT_FILE = "results/null_pathology_check.txt"
OUT = []


def log(s=""):
    print(s, flush=True)
    OUT.append(s)
    os.makedirs("results", exist_ok=True)
    with open(RESULT_FILE, "w") as f:
        f.write("\n".join(OUT))


def fit_null(fw, vol, seed_tag):
    """Fit GJR (vol='GARCH', o=1) or EGARCH per path, simulate native (no rank
    match), and record the persistence. Mirrors garch_match / egarch_fit_and_simulate
    exactly (scale, seeds), so the simulated paths match the experiment's."""
    from arch import arch_model
    from arch.univariate import StudentsT
    paths, persistence, beta = [], [], []
    for i, r in enumerate(fw):
        scale = 10.0 / (np.std(r) + 1e-12)
        y = r * scale
        try:
            am = arch_model(y, vol=vol, p=1, o=1, q=1, dist="t",
                            mean="Constant", rescale=False)
            res = am.fit(disp="off", show_warning=False)
            p = res.params
            a = float(p.get("alpha[1]", np.nan))
            b = float(p.get("beta[1]", np.nan))
            g = float(p.get("gamma[1]", np.nan))
            if vol == "EGARCH":
                persistence.append(abs(b))       # log-variance AR root
            else:
                persistence.append(a + b + 0.5 * g)  # GJR variance persistence
            beta.append(b)
            seed = int(get_rng(MASTER_SEED, seed_tag, i).integers(0, 2 ** 31))
            try:
                am.distribution = StudentsT(seed=seed)
            except Exception:
                pass
            sd = am.simulate(res.params, nobs=len(y))
            paths.append(sd["data"].values / scale)
        except Exception:
            continue
    return paths, np.array(persistence), np.array(beta)


def path_health(paths):
    """Per-path excess kurtosis and largest standardised move; count pathologies."""
    kurt, maxmove, nonfinite = [], [], 0
    for s in paths:
        s = np.asarray(s, float)
        if not np.all(np.isfinite(s)):
            nonfinite += 1
            continue
        sd = np.std(s)
        if sd <= 0:
            nonfinite += 1
            continue
        kurt.append(float(_kurt(s, fisher=True, bias=False)))
        maxmove.append(float(np.max(np.abs(s)) / sd))
    return np.array(kurt), np.array(maxmove), nonfinite


def summarise(name, kurt, maxmove, nonfinite, n):
    log(f"  {name:13} excess kurtosis  mean={np.mean(kurt):7.2f}  med={np.median(kurt):7.2f}  max={np.max(kurt):8.2f}")
    log(f"  {'':13} max|r|/std       mean={np.mean(maxmove):7.2f}  med={np.median(maxmove):7.2f}  max={np.max(maxmove):8.2f}")
    log(f"  {'':13} non-finite/degenerate paths: {nonfinite} of {n}")


def run(K=300, T=2500, alpha_lev=50.0):
    log("=" * 78)
    log("STEP 1 -- EGARCH null pathology check (is the edge a simulation artifact?)")
    log(f"K={K}  T={T}  alpha_lev={alpha_lev}  seed={MASTER_SEED}")
    log("=" * 78)

    fw, _fwb, _gc, _gam = build_or_load(K, T, alpha_lev)
    log(f"loaded {len(fw)} FW leverage paths (cached)\n")

    log("fitting GJR null (native, rank_match=False) ...")
    gjr, gjr_pers, gjr_beta = fit_null(fw, "GARCH", "garch_seed")
    log(f"  got {len(gjr)} GJR paths")
    log("fitting EGARCH null (native, rank_match=False, slow) ...")
    eg, eg_pers, eg_beta = fit_null(fw, "EGARCH", "egarch_seed")
    log(f"  got {len(eg)} EGARCH paths")

    # ---- fitted persistence vs the stability edge ----
    log("\n" + "-" * 78)
    log("FITTED PERSISTENCE  (stationarity needs < 1; near 1 = near-explosive)")
    log("-" * 78)
    log(f"  GJR   alpha+beta+gamma/2 : mean={np.mean(gjr_pers):.3f}  med={np.median(gjr_pers):.3f}  "
        f"max={np.max(gjr_pers):.3f}  frac>0.98={np.mean(gjr_pers > 0.98):.2f}")
    log(f"  EGARCH |beta[1]| (logvar): mean={np.mean(eg_pers):.3f}  med={np.median(eg_pers):.3f}  "
        f"max={np.max(eg_pers):.3f}  frac>0.98={np.mean(eg_pers > 0.98):.2f}")

    # ---- simulated-path health vs FW ----
    fw_k, fw_m, fw_nf = path_health(fw)
    gjr_k, gjr_m, gjr_nf = path_health(gjr)
    eg_k, eg_m, eg_nf = path_health(eg)
    log("\n" + "-" * 78)
    log("SIMULATED-PATH HEALTH  (compare EGARCH to FW and GJR)")
    log("-" * 78)
    summarise("FW (target)", fw_k, fw_m, fw_nf, len(fw))
    summarise("GJR null", gjr_k, gjr_m, gjr_nf, len(gjr))
    summarise("EGARCH null", eg_k, eg_m, eg_nf, len(eg))

    # spike count: paths whose largest standardised move exceeds FW's worst
    fw_worst = np.max(fw_m)
    gjr_spikes = int(np.sum(gjr_m > fw_worst))
    eg_spikes = int(np.sum(eg_m > fw_worst))
    log(f"\n  FW's largest single max|r|/std across all paths: {fw_worst:.2f}")
    log(f"  GJR null paths exceeding it   : {gjr_spikes} of {len(gjr_m)}")
    log(f"  EGARCH null paths exceeding it: {eg_spikes} of {len(eg_m)}")

    # freak-path count (the EGARCH-specific failure mode): non-finite, or a single
    # path with absurd kurtosis from a near-explosive log-variance excursion.
    EXTREME_K = 15.0
    gjr_freak = gjr_nf + int(np.sum(gjr_k > EXTREME_K))
    eg_freak = eg_nf + int(np.sum(eg_k > EXTREME_K))
    log(f"\n  freak paths (non-finite or excess kurtosis > {EXTREME_K:.0f}):"
        f"   GJR {gjr_freak} of {len(gjr)}   EGARCH {eg_freak} of {len(eg)}")
    log("  NOTE: persistence railing to the 0.98-1.0 edge afflicts BOTH nulls at"
        " short T (a")
    log("  small-sample fitting effect); it is only meaningful if EGARCH is railed"
        " MORE than GJR.")

    # ---- verdict: comparative. GJR is the control whose edge DOES collapse, so a
    # pathology shared with GJR cannot explain why only EGARCH separates. Flag EGARCH
    # only where it is worse than GJR. ----
    log("\n" + "=" * 78)
    eg_more_freak = eg_freak > gjr_freak and (eg_freak / max(len(eg), 1)) > 0.02
    eg_more_explosive = np.mean(eg_pers > 0.98) > np.mean(gjr_pers > 0.98) + 0.10
    eg_systematic = (np.median(eg_m) > 1.3 * np.median(gjr_m)) or \
                    (np.median(eg_k) > 1.5 * np.median(gjr_k))
    if eg_more_freak or eg_more_explosive or eg_systematic:
        flags = ", ".join(f for f, c in [
            (f"more freak paths than GJR ({eg_freak} vs {gjr_freak})", eg_more_freak),
            ("more near-explosive fits than GJR", eg_more_explosive),
            ("systematically heavier paths than GJR", eg_systematic)] if c)
        log(f"VERDICT: FLAGGED -- relative to the GJR control, EGARCH shows {flags}.")
        log("  The surviving FW-vs-EGARCH edge may be partly a SIMULATION artifact driven")
        log("  by a few pathological EGARCH paths. DECISIVE follow-up: re-run the")
        log("  FW-vs-EGARCH discrimination with those freak paths removed (or with")
        log("  near-explosive fits rejected) and see whether the +0.146 edge survives. If")
        log("  it collapses, the puzzle was an artifact; if it holds, the effect is real.")
    else:
        log("VERDICT: CLEAN relative to GJR -- EGARCH is no more pathological than the GJR")
        log("  control (comparable freak-path count, persistence, and path health), yet")
        log("  only EGARCH's edge survives. So the edge is NOT a simulation artifact; the")
        log("  puzzle is real -> proceed to Step 2 (localise which feature carries the gap)")
        log("  and Step 3 (higher-order volatility statistics).")
    log("=" * 78)
    print(f"\n[saved to {RESULT_FILE}]")


if __name__ == "__main__":
    K = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    run(K=K)
