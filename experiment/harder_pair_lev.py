"""
experiment/harder_pair_lev.py -- the ASYMMETRIC (leverage) discrimination arm.

Parallel to the symmetric harder-pair / embedding study, but the behavioural
model now carries a leverage effect (FWParams.alpha_lev > 0), and the GJR-GARCH
null's asymmetry term is therefore ENGAGED (gamma_hat ~ +0.07, vs ~0 on symmetric
FW). So the statistical null can now match FW on the distribution (via tight
rank-matching) AND on leverage (via the fitted GJR gamma).

THE SHARP QUESTION
------------------
With the null able to reproduce BOTH the return distribution AND the leverage,
does the behavioural switching mechanism still leave a nonlinear fingerprint that
the standard moments miss? Two informative outcomes:

  (a) the m-growing nonlinear advantage seen in the SYMMETRIC arm persists here
      -> the behavioural mechanism leaves structure beyond distribution+leverage
         that GJR cannot capture (strong positive for the nonlinear toolkit);
  (b) the advantage shrinks/vanishes vs the symmetric arm -> what laminarity was
      detecting was leverage-like structure, and giving GARCH a (now-engaged)
      leverage term closes the gap (the nonlinear edge was capturable by a richer
      statistical null all along).

Either outcome is a clean, publishable contrast against the symmetric Section.

DESIGN: generate asymmetric FW (alpha_lev) paths; fit GJR-GARCH(1,1)-t (o=1),
rank-match to FW's exact marginal (TIGHT); sweep m and report DeltaAUC and
laminarity for FW-vs-GJR and for an FW-vs-FW null (artifact guard). Also reports
the fitted gamma to confirm the leverage term engaged. Paths are cached (they do
not depend on m), so re-runs at other m are fast.

Run from the repo root:   python -m experiment.harder_pair_lev
Prints and saves to results/harder_pair_lev.txt.
"""

import os
import warnings
import numpy as np

from models.fw_ssv import FWParams, simulate_fw
from experiment.harder_pair import feats, auc
from experiment.garch_spec_check import fit_and_simulate
from experiment.embedding_diagnostics import evaluate
from signatures.features import feature_vector, REFEREE_1, REFEREE_2
from seeds import get_rng, MASTER_SEED

warnings.filterwarnings("ignore")
OUT = []
CACHE_DIR = "results/cache"
RESULT_FILE = "results/harder_pair_lev.txt"


def log(s=""):
    print(s, flush=True)
    OUT.append(s)
    os.makedirs("results", exist_ok=True)
    with open(RESULT_FILE, "w") as f:
        f.write("\n".join(OUT))


def fw_lev_paths(K, T, alpha_lev, tag):
    out = []
    for s in range(K):
        r = simulate_fw(FWParams(switching="dca", alpha_lev=alpha_lev), n=T,
                        burn_in=1500, rng=get_rng(MASTER_SEED, tag, alpha_lev, s))
        if r["finite"] and not r["blew_up"]:
            out.append(r["returns"])
    return out


def build_or_load(K, T, alpha_lev):
    suf = f"K{K}_T{T}_lev{alpha_lev}"
    fwf = os.path.join(CACHE_DIR, f"lev_fw_{suf}.npy")
    fwbf = os.path.join(CACHE_DIR, f"lev_fwb_{suf}.npy")
    gcf = os.path.join(CACHE_DIR, f"lev_gc_{suf}.npy")
    gammaf = os.path.join(CACHE_DIR, f"lev_gamma_{suf}.npy")
    if all(os.path.exists(p) for p in (fwf, fwbf, gcf, gammaf)):
        log(f"loaded cached leverage paths for {suf} from results/cache/")
        return (list(np.load(fwf)), list(np.load(fwbf)),
                list(np.load(gcf)), np.load(gammaf))
    log(f"no cache for {suf} -- generating asymmetric FW + fitting GJR (one-time)...")
    fw = fw_lev_paths(K, T, alpha_lev, "hpl_fw")
    fwb = fw_lev_paths(K, T, alpha_lev, "hpl_fwb")
    gc, _ok, gammas, _ = fit_and_simulate(fw, MASTER_SEED, o=1, rank_match=True)
    n = min(len(fw), len(fwb), len(gc))
    fw, fwb, gc = fw[:n], fwb[:n], gc[:n]
    os.makedirs(CACHE_DIR, exist_ok=True)
    np.save(fwf, np.array(fw))
    np.save(fwbf, np.array(fwb))
    np.save(gcf, np.array(gc))
    np.save(gammaf, np.array(gammas))
    log(f"cached {n} paths per group (+ {len(gammas)} fitted gammas) for {suf}")
    return fw, fwb, gc, gammas


def decompose(fw, gc, m=4, tau=1):
    XF, _ = feats(fw, m, tau)
    XG, _ = feats(gc, m, tau)
    full = list(feature_vector(fw[0]).keys())
    ds = []
    for i, nm in enumerate(full):
        a, b = XF[:, i], XG[:, i]
        a, b = a[np.isfinite(a)], b[np.isfinite(b)]
        if len(a) < 2 or len(b) < 2:
            continue
        p = np.sqrt((a.var() + b.var()) / 2) + 1e-12
        grp = "R1" if nm in REFEREE_1 else "R2"
        ds.append((abs(a.mean() - b.mean()) / p, nm, grp))
    return sorted(ds, reverse=True)


def run(K=300, T=2500, alpha_lev=50.0, m_grid=(4, 5, 6, 7, 8, 3), tau=1):
    log("=" * 70)
    log("ASYMMETRIC (LEVERAGE) ARM -- leverage FW vs GJR-GARCH (leverage engaged)")
    log(f"K={K}, T={T}, alpha_lev={alpha_lev}, tau={tau}, seed={MASTER_SEED}")
    log("=" * 70)

    fw, fwb, gc, gammas = build_or_load(K, T, alpha_lev)
    if len(gammas):
        log(f"\nfitted GJR gamma over {len(gammas)} fits: mean={gammas.mean():+.4f}  "
            f"median={np.median(gammas):+.4f}  frac>0={np.mean(gammas > 0):.2f}")
        log("  (systematically positive => the GJR null's leverage term ENGAGED,")
        log("   i.e. the statistical null is matching FW's leverage, not just its")
        log("   distribution. Contrast: on SYMMETRIC FW this was ~0, sign-mixed.)")

    log(f"\nn={len(fw)} per group; sweeping m={list(m_grid)} at tau={tau}")
    log("Compare the FW-vs-GJR Delta column against the SYMMETRIC arm (Sec 6.4):")
    log("  if it tracks the symmetric rise -> behavioural fingerprint survives even")
    log("     when the null matches leverage; if it is LOWER -> the engaged GJR")
    log("     leverage term has absorbed part of what the nonlinear measures saw.\n")
    log(f"{'m':>4}{'AUC_R1':>9}{'AUC_R2':>9}{'Delta':>8}{'clust_d':>9} | {'NULL Delta':>11}")
    log("-" * 60)
    cache = {}
    rows_g, rows_n = {}, {}
    for m in m_grid:
        a1g, a2g, clustg, r2dg = evaluate(fw, "lfw", gc, "lgc", m, tau, cache)
        a1n, a2n, _clustn, r2dn = evaluate(fw, "lfw", fwb, "lfwb", m, tau, cache)
        rows_g[m], rows_n[m] = r2dg, r2dn
        log(f"{m:>4}{a1g:>9.3f}{a2g:>9.3f}{a2g - a1g:>+8.3f}{clustg:>9.2f} | {a2n - a1n:>+11.3f}")

    def r2_table(rows, title):
        log(f"\n   {title} -- R2 per-feature Cohen's d by m:")
        ms = sorted(rows)
        log("   " + f"{'feature':>14}" + "".join(f"{'m='+str(m):>8}" for m in ms))
        for f in sorted(REFEREE_2, key=lambda f: max(rows[m][f] for m in ms), reverse=True):
            log("   " + f"{f:>14}" + "".join(f"{rows[m][f]:>8.2f}" for m in ms))

    r2_table(rows_g, "FW-vs-GJR (leverage)")
    r2_table(rows_n, "NULL (FW vs FW)")

    # decomposition at m=4 for the record
    log("\nper-feature separation at m=4 (Cohen's d, sorted):")
    for d, nm, grp in decompose(fw, gc, m=4, tau=tau):
        log(f"   {grp}  {nm:16} d = {d:.2f}")

    log("\nReading: distribution is rank-matched (R1 distributional d ~ 0); the GJR")
    log("null now also carries leverage (gamma>0). Any surviving FW-vs-GJR Delta is")
    log("behavioural structure beyond BOTH distribution and leverage. Compare its")
    log("m-profile to the symmetric arm to see whether matching leverage shrinks it.")
    print(f"\n[saved to {RESULT_FILE}]")


if __name__ == "__main__":
    run()