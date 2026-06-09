"""
experiment/leverage_sweep.py -- does the nonlinear edge decline SMOOTHLY as
leverage increases? (Makes the "mis-specified null" claim airtight.)

The two-arm result (Sections 6.4, 6.6) contrasts two endpoints: symmetric FW
(alpha_lev=0), where the nonlinear advantage grows to ~+0.10 at high m, and
strongly-asymmetric FW (alpha_lev=50), where it collapses to ~0 because the GJR
null's leverage term engages and the standard moments catch up. This script
fills in the MIDDLE: it sweeps alpha_lev and, at fixed high embeddings (m=6,8),
reports the fitted GJR gamma and the FW-vs-GJR DeltaAUC at each leverage level.

Expectation (the airtight version of the claim): as alpha_lev rises, gamma_hat
rises monotonically (the null progressively engages its leverage term) and
DeltaAUC declines monotonically from the symmetric ceiling toward zero. A smooth
monotone decline shows the nonlinear edge is governed by HOW MIS-SPECIFIED the
null is for the asymmetry -- not an artefact of the two particular endpoints.

Each alpha_lev's paths are cached (keyed by K,T,alpha_lev), reusing the
harder_pair_lev cache where it already exists (e.g. alpha_lev=50). So re-runs and
added grid points are cheap; only new alpha_lev values pay the GARCH fit.

NOTE: the alpha_lev=0 point here is an INDEPENDENT draw of the symmetric case
(same machinery/tagging as the other sweep points), so it should land NEAR the
Section 6.4 ceiling (~+0.08-0.10 at high m) but need not be byte-identical to it.

Run from the repo root:   python -m experiment.leverage_sweep
Prints and saves to results/leverage_sweep.txt.  Long run (one GARCH fit set per
new alpha_lev); trim ALPHA_GRID or Ctrl-C between points (cache persists).
"""

import os
import warnings
import numpy as np

from experiment.harder_pair_lev import build_or_load
from experiment.embedding_diagnostics import evaluate

warnings.filterwarnings("ignore")
OUT = []
RESULT_FILE = "results/leverage_sweep.txt"
ALPHA_GRID = (0.0, 10.0, 20.0, 30.0, 50.0)
M_REPORT = (6, 8)


def log(s=""):
    print(s, flush=True)
    OUT.append(s)
    os.makedirs("results", exist_ok=True)
    with open(RESULT_FILE, "w") as f:
        f.write("\n".join(OUT))


def run(K=300, T=2500, alpha_grid=ALPHA_GRID, m_report=M_REPORT, tau=1):
    log("=" * 72)
    log("LEVERAGE SWEEP -- does the nonlinear edge decline smoothly with alpha_lev?")
    log(f"K={K}, T={T}, m_report={list(m_report)}, tau={tau}, seed=master")
    log("=" * 72)
    log("Expect: as alpha_lev rises, gamma_hat rises (null engages leverage) and")
    log("DeltaAUC falls from the symmetric ceiling toward ~0 (edge was mis-spec).")
    log("")

    header = (f"{'alpha_lev':>9}{'gamma_hat':>11}{'g>0':>6}"
              + "".join(f"{f'dAUC(m{m})':>11}" for m in m_report)
              + "".join(f"{f'lam(m{m})':>9}" for m in m_report)
              + "".join(f"{f'd2(m{m})':>9}" for m in m_report)
              + f"{'null(m' + str(m_report[-1]) + ')':>12}")
    log(header)
    log("-" * len(header))

    for lev in alpha_grid:
        fw, fwb, gc, gammas = build_or_load(K, T, lev)
        cache = {}
        gmean = gammas.mean() if len(gammas) else float("nan")
        gpos = float(np.mean(gammas > 0)) if len(gammas) else float("nan")
        dauc, lam, d2 = {}, {}, {}
        for m in m_report:
            a1, a2, _clust, r2d = evaluate(fw, "lfw", gc, "lgc", m, tau, cache)
            dauc[m] = a2 - a1
            lam[m] = r2d.get("lam", float("nan"))   # laminarity separation
            d2[m] = r2d.get("d2", float("nan"))      # correlation-dimension separation (co-driver, Sec 6.4)
        # null guard at the top m only
        a1n, a2n, _, _ = evaluate(fw, "lfw", fwb, "lfwb", m_report[-1], tau, cache)
        row = (f"{lev:>9.0f}{gmean:>+11.4f}{gpos:>6.2f}"
               + "".join(f"{dauc[m]:>+11.3f}" for m in m_report)
               + "".join(f"{lam[m]:>9.2f}" for m in m_report)
               + "".join(f"{d2[m]:>9.2f}" for m in m_report)
               + f"{a2n - a1n:>+12.3f}")
        log(row)

    log("")
    log("Reading: a monotone rise in gamma_hat alongside a monotone fall in dAUC")
    log("(toward 0) across alpha_lev demonstrates the nonlinear edge is governed by")
    log("the null's mis-specification for asymmetry -- the airtight version of the")
    log("two-endpoint result. The null column must stay ~0 throughout (no artefact).")
    print(f"\n[saved to {RESULT_FILE}]")


if __name__ == "__main__":
    import sys
    # Resolve K from: CLI arg, then env var LEV_SWEEP_K, then default 300.
    # The env var is the robust path (some shells drop positional args to
    # `python -m`). We PRINT the resolved K loudly so a silent fallback to the
    # default can never pass unnoticed before a ~100-minute run.
    if len(sys.argv) > 1:
        K = int(sys.argv[1])
    elif os.environ.get("LEV_SWEEP_K"):
        K = int(os.environ["LEV_SWEEP_K"])
    else:
        K = 300
    if len(sys.argv) > 2:
        T = int(sys.argv[2])
    else:
        T = int(os.environ.get("LEV_SWEEP_T", 2500))
    print(f"*** leverage_sweep resolved K={K}, T={T} "
          f"(argv={sys.argv[1:]}, env LEV_SWEEP_K={os.environ.get('LEV_SWEEP_K')}) ***",
          flush=True)
    run(K=K, T=T)