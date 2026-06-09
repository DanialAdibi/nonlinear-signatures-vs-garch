"""
experiment/aparch_null.py -- isolating the additive-vs-log axis behind the EGARCH puzzle.

WHY THIS EXISTS
In the leverage arm, GJR (additive-variance, leverage-aware) TIES with FW (the edge
collapses, dAUC at m=8 = +0.016) but EGARCH (LOG-variance, leverage-aware) SEPARATES and
grows (dAUC at m=8 = +0.146). Those two nulls differ on the additive-vs-log axis but also
in parameterisation and news-impact form, so that single contrast is confounded. This adds
a THIRD leverage-aware null, APARCH (asymmetric power ARCH), which is additive-FAMILY (it
models a power of the conditional standard deviation additively) but a different
parameterisation from GJR.

  GJR variance:   sigma2 = omega + alpha e^2 + gamma e^2 1{e<0} + beta sigma2   (additive, power 2)
  APARCH:         sigma^delta = omega + alpha(|e| - gamma e)^delta + beta sigma^delta (additive power, delta free)
  EGARCH log-var: log sigma2 = omega + alpha(|z|-E|z|) + gamma z + beta log sigma2 (LOG)

HYPOTHESIS TEST
  If the axis is additive-vs-log, APARCH (additive) should TIE like GJR.
  If APARCH SEPARATES like EGARCH, additive-vs-log is REFUTED -- the EGARCH separation is
  not about the log-variance functional form, and the structural difference the recurrence
  features detect is something else again.

WHAT IT DOES
Scores FW(leverage) vs APARCH over m exactly as the EGARCH arm (rank_match=True). In the
SAME run it re-scores FW vs the cached GJR null as the additive anchor (should reproduce
+0.016, a faithfulness check) and carries an A-vs-A column. EGARCH (+0.146) is cited from
the canonical run as the log anchor. Reports the fitted APARCH delta and gamma to confirm
the model engaged, and the per-feature Cohen's d at the peak m, so we see whether
laminarity/d2 stay flat (GJR-like, additive) or light up (EGARCH-like).

Run:  python -u -m experiment.aparch_null
Output: results/aparch_null.txt
"""

import os
import sys
import numpy as np

from experiment.harder_pair_lev import build_or_load
from experiment.embedding_diagnostics import evaluate
from signatures.features import REFEREE_2
from seeds import get_rng, MASTER_SEED

RESULT_FILE = "results/aparch_null.txt"
OUT = []

GJR_REF_M8 = +0.016    # canonical leverage-arm GJR dAUC at m=8 (additive anchor)
EGARCH_REF_M8 = +0.146  # canonical EGARCH dAUC at m=8 (log anchor)
# EGARCH per-feature Cohen's d at m=8 (from Step 2, localise_egarch_gap.py, K=300):
EG_REF_D = {"lam": 0.88, "d2": 0.82, "det": 0.60, "d2_r2": 0.26,
            "bds": 0.30, "lyap_r2": 0.13, "lambda1": 0.14}


def log(s=""):
    print(s, flush=True)
    OUT.append(s)
    os.makedirs("results", exist_ok=True)
    with open(RESULT_FILE, "w") as f:
        f.write("\n".join(OUT))


def aparch_fit_and_simulate(fw_series, seed0, rank_match=True):
    """Fit APARCH(1,1,1)-t per path and simulate a (rank-matched) path. Mirrors
    egarch_fit_and_simulate but with vol='APARCH'. Returns (paths, gammas, deltas)."""
    from arch import arch_model
    from arch.univariate import StudentsT
    out, gammas, deltas = [], [], []
    for i, r in enumerate(fw_series):
        scale = 10.0 / (np.std(r) + 1e-12)
        y = r * scale
        try:
            am = arch_model(y, vol="APARCH", p=1, o=1, q=1, dist="t",
                            mean="Constant", rescale=False)
            res = am.fit(disp="off", show_warning=False)
            if "gamma[1]" in res.params.index:
                gammas.append(float(res.params["gamma[1]"]))
            if "delta" in res.params.index:
                deltas.append(float(res.params["delta"]))
            seed = int(get_rng(seed0, "aparch_seed", i).integers(0, 2 ** 31))
            try:
                am.distribution = StudentsT(seed=seed)
            except Exception:
                pass
            sd = am.simulate(res.params, nobs=len(y))
            s = sd["data"].values / scale
            if rank_match and len(s) == len(r):
                s = np.sort(r)[np.argsort(np.argsort(s))]
            out.append(s)
        except Exception:
            continue
    return out, np.array(gammas), np.array(deltas)


def run(K=300, T=2500, alpha_lev=50.0, m_grid=(3, 5, 8), tau=1):
    log("=" * 80)
    log("APARCH null -- isolating the additive-vs-log axis (EGARCH-puzzle follow-up)")
    log(f"K={K}  T={T}  alpha_lev={alpha_lev}  null=APARCH(1,1,1)-t  (additive power family)")
    log("=" * 80)

    fw, fwb, gc_gjr, _gjr_gam = build_or_load(K, T, alpha_lev)
    log(f"loaded {len(fw)} FW + {len(fwb)} FWb + {len(gc_gjr)} GJR (cached)")
    log("fitting APARCH null (one-time, rank_match=True, slow) ...")
    gc_ap, ap_gam, ap_delta = aparch_fit_and_simulate(fw, MASTER_SEED, rank_match=True)
    n = min(len(fw), len(fwb), len(gc_gjr), len(gc_ap))
    fw, fwb, gc_gjr, gc_ap = fw[:n], fwb[:n], gc_gjr[:n], gc_ap[:n]
    log(f"  aligned {n} paths per group")
    if len(ap_delta):
        log(f"  APARCH delta (power): mean={np.mean(ap_delta):.3f}  med={np.median(ap_delta):.3f}"
            f"  (2 = GJR-like variance; 1 = std-dev/TARCH)")
    if len(ap_gam):
        log(f"  APARCH gamma (asymmetry): mean={np.mean(ap_gam):+.4f}  "
            f"({100 * np.mean(ap_gam > 0):.0f}% positive -> equity leverage sign engaged)")

    log("\nFW(leverage) vs nulls, by embedding dimension m  (dAUC = AUC_R2 - AUC_R1):")
    log(f"  {'m':>3}{'GJR dAUC':>11}{'APARCH dAUC':>13}{'(R1':>8}{'R2)':>8}{'A-vs-A':>10}")
    cache = {}
    ap_dauc, gjr_dauc, r2d_peak, gjr_r2d_peak = {}, {}, None, None
    peak = max(m_grid)
    for m in m_grid:
        a1g, a2g, _, r2dg = evaluate(fw, "afw", gc_gjr, "agjr", m, tau, cache)
        a1a, a2a, _clusta, r2da = evaluate(fw, "afw", gc_ap, "aap", m, tau, cache)
        a1n, a2n, _, _ = evaluate(fw, "afw", fwb, "afwb", m, tau, cache)
        gjr_dauc[m], ap_dauc[m] = a2g - a1g, a2a - a1a
        log(f"  {m:>3}{a2g - a1g:>+11.3f}{a2a - a1a:>+13.3f}{a1a:>8.3f}{a2a:>8.3f}{a2n - a1n:>+10.3f}")
        if m == peak:
            r2d_peak, gjr_r2d_peak = r2da, r2dg

    # per-feature separation at the peak: does APARCH track GJR (flat) or EGARCH (raised)?
    log(f"\nPer-feature Cohen's d at m={peak} -- APARCH vs the GJR (additive, in-run) and")
    log("EGARCH (log, cited from Step 2) anchors.  APARCH near GJR = additive ties;")
    log("APARCH near EGARCH = additive separates:")
    log(f"  {'feature':10}{'APARCH':>9}{'GJR':>9}{'EGARCH':>9}")
    for f in sorted(REFEREE_2, key=lambda f: r2d_peak.get(f, 0), reverse=True):
        log(f"  {f:10}{r2d_peak.get(f, float('nan')):>9.2f}"
            f"{gjr_r2d_peak.get(f, float('nan')):>9.2f}{EG_REF_D.get(f, float('nan')):>9.2f}")

    # ---- verdict ----
    apk = ap_dauc[peak]
    log("\n" + "=" * 80)
    log(f"ANCHORS at m={peak}:  GJR (additive) {GJR_REF_M8:+.3f} ties"
        f"   |   EGARCH (log) {EGARCH_REF_M8:+.3f} separates")
    log(f"  in-run GJR re-score: {gjr_dauc[peak]:+.3f}  (faithfulness check vs canonical +0.016)")
    log(f"  APARCH (additive power): {apk:+.3f}")
    near_gjr = abs(apk - GJR_REF_M8)
    near_eg = abs(apk - EGARCH_REF_M8)
    if apk < 0.05 and near_gjr < near_eg:
        log("VERDICT: APARCH TIES like GJR -> the additive family collapses the edge; only the")
        log("  LOG-variance null (EGARCH) separates. This SUPPORTS the additive-vs-log axis: the")
        log("  structural difference the recurrence features detect tracks the variance recursion")
        log("  form (additive vs log), not leverage-awareness per se. Two additive leverage-aware")
        log("  nulls (GJR, APARCH) tie; one log-variance leverage-aware null (EGARCH) separates.")
        log("  NB this is now a positive, named characterisation -- worth confirming with a second")
        log("  log-family null (e.g. FIEGARCH/log-GARCH) before stating it strongly in the paper.")
    elif apk > 0.10 and near_eg < near_gjr:
        log("VERDICT: APARCH SEPARATES like EGARCH -> additive-vs-log is REFUTED. An additive")
        log("  null also separates, so the EGARCH separation is not about the log-variance form.")
        log("  The structural difference is something both APARCH and EGARCH share but GJR does")
        log("  not (e.g. a free power/news-impact shape). Back to a genuinely unnamed higher-order")
        log("  difference; report as a limitation. (Next probe: what do APARCH+EGARCH share vs GJR?)")
    else:
        log(f"VERDICT: INTERMEDIATE (APARCH {apk:+.3f}, between the GJR and EGARCH anchors).")
        log("  The additive-vs-log axis is not cleanly decisive here. Inspect the per-feature d")
        log("  above (lam/d2 raised = EGARCH-like; flat = GJR-like) and consider a larger K or a")
        log("  matched log/additive pair before drawing a conclusion.")
    log("=" * 80)
    print(f"\n[saved to {RESULT_FILE}]")


if __name__ == "__main__":
    K = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    run(K=K)
