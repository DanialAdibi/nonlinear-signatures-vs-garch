"""
experiment/egarch_null.py -- robustness of the leverage-arm collapse to the
choice of asymmetric null: EGARCH instead of GJR-GARCH.

WHY THIS EXISTS
The central claim is that the apparent nonlinear advantage on leverage data
collapses once the statistical NULL can represent the leverage (Section on the
leverage arm). That result uses a GJR-GARCH null. A natural referee question is
whether the collapse is specific to the GJR functional form. EGARCH captures the
leverage effect through a DIFFERENT mechanism -- asymmetry in the LOG conditional
variance via the standardised innovation -- rather than GJR's indicator term on
squared residuals. If the nonlinear advantage also collapses against an EGARCH
null, the conclusion is not an artefact of the GJR specification.

  GJR variance:    sigma2_t = omega + alpha e^2 + gamma e^2 1{e<0} + beta sigma2
  EGARCH log-var:  log sigma2_t = omega + alpha(|z|-E|z|) + gamma z + beta log sigma2

The gamma term in EGARCH (the coefficient on the signed standardised residual z)
is the leverage parameter; gamma < 0 is the usual equity sign (negative shocks
raise volatility). We report the fitted EGARCH gamma to confirm the asymmetry
term engages on leverage data, mirroring the GJR gamma diagnostic.

WHAT IT DOES
Re-runs the leverage arm exactly as harder_pair_lev.py, but with an EGARCH null in
place of GJR. Reuses the cached leverage FW paths (they are null-independent), so
this only refits the null. Reports, per embedding dimension m, AUC_R1, AUC_R2,
Delta AUC, plus an A-vs-A null column -- the same table as the GJR leverage arm,
for side-by-side comparison.

Run:  python -u -m experiment.egarch_null
Output: results/egarch_null.txt

EXPECTED RESULT (to be confirmed by the run): Delta AUC near zero across m, as in
the GJR leverage arm -- i.e. the collapse is robust to the asymmetric-null form.
"""

import os
import numpy as np

from experiment.harder_pair_lev import build_or_load
from experiment.embedding_diagnostics import evaluate
from seeds import get_rng, MASTER_SEED

RESULT_FILE = "results/egarch_null.txt"
OUT = []


def log(s=""):
    print(s, flush=True)
    OUT.append(s)
    os.makedirs("results", exist_ok=True)
    with open(RESULT_FILE, "w") as f:
        f.write("\n".join(OUT))


def egarch_fit_and_simulate(fw_series, seed0, rank_match=True):
    """Fit EGARCH(1,1,1)-t per path and simulate a (rank-matched) path.

    Mirrors garch_spec_check.fit_and_simulate but with vol='EGARCH'. Returns
    (simulated_paths, fitted_gammas). The EGARCH asymmetry coefficient is the
    leverage parameter; we collect it as the engagement diagnostic.
    """
    from arch import arch_model
    from arch.univariate import StudentsT
    out, gammas = [], []
    for i, r in enumerate(fw_series):
        scale = 10.0 / (np.std(r) + 1e-12)
        y = r * scale
        try:
            am = arch_model(y, vol="EGARCH", p=1, o=1, q=1, dist="t",
                            mean="Constant", rescale=False)
            res = am.fit(disp="off", show_warning=False)
            if "gamma[1]" in res.params.index:
                gammas.append(float(res.params["gamma[1]"]))
            seed = int(get_rng(seed0, "egarch_seed", i).integers(0, 2 ** 31))
            try:
                am.distribution = StudentsT(seed=seed)
            except Exception:
                pass
            sd = am.simulate(res.params, nobs=len(y))
            s = sd["data"].values / scale
            if rank_match and len(s) == len(r):
                target = np.sort(r)
                s = target[np.argsort(np.argsort(s))]
            out.append(s)
        except Exception:
            continue
    return out, np.array(gammas)


def run(K=300, T=2500, alpha_lev=50.0, m_grid=(4, 5, 6, 7, 8, 3), tau=1):
    log("=" * 64)
    log("EGARCH alternative-null robustness check (leverage arm)")
    log(f"K={K}  T={T}  alpha_lev={alpha_lev}  null=EGARCH(1,1,1)-t")
    log("=" * 64)

    # Reuse cached leverage FW paths (null-independent); refit only the null.
    fw, fwb, _gc_gjr, gammas_gjr = build_or_load(K, T, alpha_lev)
    log(f"loaded {len(fw)} leverage FW paths (alpha_lev={alpha_lev})")
    if len(gammas_gjr):
        log(f"reference GJR gamma on these paths: mean={np.mean(gammas_gjr):+.4f}")

    log("\nfitting EGARCH null (one-time) ...")
    gc, eg_gammas = egarch_fit_and_simulate(fw, MASTER_SEED, rank_match=True)
    n = min(len(fw), len(fwb), len(gc))
    fw, fwb, gc = fw[:n], fwb[:n], gc[:n]
    if len(eg_gammas):
        frac_neg = float(np.mean(eg_gammas < 0))
        log(f"EGARCH asymmetry gamma: mean={np.mean(eg_gammas):+.4f}  "
            f"({100*frac_neg:.0f}% negative -> equity leverage sign)")

    log("\nFW(leverage) vs EGARCH null, by embedding dimension m:")
    log(f"{'m':>3}{'AUC_R1':>9}{'AUC_R2':>9}{'dAUC':>9}{'clust_d':>9}{'A-vs-A dAUC':>13}")
    cache = {}
    for m in m_grid:
        a1g, a2g, clustg, _ = evaluate(fw, "efw", gc, "egc", m, tau, cache)
        a1n, a2n, _clustn, _ = evaluate(fw, "efw", fwb, "efwb", m, tau, cache)
        log(f"{m:>3}{a1g:>9.3f}{a2g:>9.3f}{a2g - a1g:>+9.3f}{clustg:>9.2f}{a2n - a1n:>+13.3f}")

    log("\nInterpretation: dAUC near zero across m (matching the GJR leverage")
    log("arm) indicates the collapse is robust to the asymmetric-null form;")
    log("the A-vs-A column should stay near zero (calibration).")
    log("=" * 64)


if __name__ == "__main__":
    run()