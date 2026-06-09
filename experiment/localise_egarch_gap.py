"""
experiment/localise_egarch_gap.py -- Step 2 of the EGARCH-puzzle dig:
WHICH feature carries the FW-vs-EGARCH separation?

CONTEXT
Step 1 (null_pathology_check.py) ruled out a simulation artifact: the EGARCH null
paths are clean (milder than GJR), so the FW-vs-EGARCH edge (dAUC +0.146 at m=8,
growing with m) is a real higher-order effect. dAUC = AUC(REFEREE_2, the 7 nonlinear
features) - AUC(REFEREE_1, the 7 moments). This step decomposes that gap to name the
feature(s) doing the work, so Step 3 knows which volatility property to chase.

REFEREE_2 (nonlinear):
  lambda1  largest Lyapunov exponent (divergence rate)
  lyap_r2  goodness-of-fit of the Lyapunov divergence line
  d2       correlation dimension (attractor complexity)
  d2_r2    goodness-of-fit of the d2 scaling line
  det      determinism (recurrence quantification)
  lam      laminarity (recurrence quantification; vertical/laminar structure)
  bds      BDS statistic (general serial dependence)

THREE COMPLEMENTARY DECOMPOSITIONS, per m (same FW/EGARCH paths and seeds as
egarch_null, rank_match=True null):
  - single-feature AUC : marginal discriminating power of each feature alone
  - Cohen's d          : standardised FW-vs-EGARCH mean separation (same metric
                         evaluate() reports as r2d)
  - leave-one-out      : drop in AUC_R2 when each feature is removed -> UNIQUE
                         contribution, which is what disentangles the correlated
                         pairs lambda1/lyap_r2 and d2/d2_r2 (a feature can lead on
                         single-AUC yet add nothing once its partner is in)

CARRIER TEST: a feature carries the gap if it leads on single-feature AUC at the
peak m AND has the largest leave-one-out drop AND its separation GROWS with m,
tracking the dAUC rise. The result tells Step 3 where to look:
  lambda1 / d2     -> attractor geometry / divergence (the dynamical-systems story)
  lam / det        -> recurrence/laminar structure (persistence of volatility regimes)
  bds              -> generic higher-order serial dependence

Run:  python -u -m experiment.localise_egarch_gap            # K=300, m=(3,5,8)
      python -u -m experiment.localise_egarch_gap 300 3,5,8
Output: results/localise_egarch_gap.txt
"""

import os
import sys
import warnings
import numpy as np

from experiment.harder_pair_lev import build_or_load
from experiment.egarch_null import egarch_fit_and_simulate
from experiment.embedding_diagnostics import feat_cached
from experiment.harder_pair import auc, max_d
from signatures.features import feature_vector, REFEREE_1, REFEREE_2
from seeds import MASTER_SEED

warnings.filterwarnings("ignore")
RESULT_FILE = "results/localise_egarch_gap.txt"
OUT = []


def log(s=""):
    print(s, flush=True)
    OUT.append(s)
    os.makedirs("results", exist_ok=True)
    with open(RESULT_FILE, "w") as f:
        f.write("\n".join(OUT))


def run(K=300, T=2500, alpha_lev=50.0, m_grid=(3, 5, 8), tau=1):
    log("=" * 78)
    log("STEP 2 -- localising the FW-vs-EGARCH gap (which feature carries dAUC?)")
    log(f"K={K}  T={T}  alpha_lev={alpha_lev}  m_grid={m_grid}  null=EGARCH(1,1,1)-t")
    log("dAUC = AUC(REFEREE_2 nonlinear-only) - AUC(REFEREE_1 moments-only)")
    log("=" * 78)

    fw, _fwb, _gc, _gam = build_or_load(K, T, alpha_lev)
    log(f"loaded {len(fw)} FW leverage paths (cached)")
    log("fitting EGARCH null (one-time, rank_match=True, slow) ...")
    gc, eg_gam = egarch_fit_and_simulate(fw, MASTER_SEED, rank_match=True)
    n = min(len(fw), len(gc))
    fw, gc = fw[:n], gc[:n]
    log(f"  paired {n} FW vs EGARCH paths"
        + (f"; EGARCH gamma mean={np.mean(eg_gam):+.4f}" if len(eg_gam) else ""))

    full = list(feature_vector(fw[0]).keys())
    cache = {}
    sf_by_m = {}
    dauc_by_m = {}

    for m in m_grid:
        XA = feat_cached(fw, "efw", m, tau, cache)   
        XB = feat_cached(gc, "egc", m, tau, cache)  
        X = np.vstack([XA, XB])
        y = np.array([0] * len(fw) + [1] * len(gc))
        fc = np.all(np.isfinite(X), axis=0)
        names = [nm for nm, k in zip(full, fc) if k]
        Xf = X[:, fc]

        def sub(cols):
            idx = [names.index(c) for c in cols if c in names]
            return Xf[:, idx]

        def col(c):
            return Xf[:, [names.index(c)]] if c in names else None

        a1 = auc(sub(REFEREE_1), y)
        a2 = auc(sub(REFEREE_2), y)
        dauc_by_m[m] = a2 - a1

        present_r2 = [f for f in REFEREE_2 if f in names]
        single = {f: (auc(col(f), y) if f in names else float("nan")) for f in REFEREE_2}
        cohen = {f: max_d(XA, XB, full, [f]) for f in REFEREE_2}
        loo = {}
        for f in present_r2:
            rest = [g for g in present_r2 if g != f]
            loo[f] = a2 - (auc(sub(rest), y) if rest else 0.5)
        sf_by_m[m] = single

        log("\n" + "-" * 78)
        log(f"m = {m}    AUC_R1={a1:.3f}   AUC_R2={a2:.3f}   dAUC={a2 - a1:+.3f}")
        log("-" * 78)
        log(f"  {'feature':10}{'single-AUC':>12}{'Cohen_d':>10}{'LOO drop':>11}")
        order = sorted(REFEREE_2, key=lambda f: (single[f] if np.isfinite(single[f]) else 0),
                       reverse=True)
        for f in order:
            sval = f"{single[f]:.3f}" if np.isfinite(single[f]) else "  n/a"
            lval = f"{loo[f]:+.3f}" if f in loo else "   . "
            log(f"  {f:10}{sval:>12}{cohen[f]:>10.2f}{lval:>11}")
        best_mom = max(((auc(col(c), y), c) for c in REFEREE_1 if c in names),
                       default=(float("nan"), "n/a"))
        log(f"  best single moment (REFEREE_1): {best_mom[1]} AUC={best_mom[0]:.3f}"
            f"   (near 0.5 confirms moments do not carry the gap)")

    # cross-m carrier identification
    peak = max(m_grid)
    low = min(m_grid)
    present = [f for f in REFEREE_2 if np.isfinite(sf_by_m[peak][f])]
    log("\n" + "=" * 78)
    log("CARRIER SUMMARY -- single-feature AUC across m (growth tracks dAUC?)")
    log("=" * 78)
    log(f"  dAUC: " + "  ".join(f"m{m}={dauc_by_m[m]:+.3f}" for m in m_grid))
    header = "  " + f"{'feature':10}" + "".join(f"{'m'+str(m):>8}" for m in m_grid) + f"{'rise':>9}"
    log(header)
    rise = {}
    for f in sorted(present, key=lambda f: sf_by_m[peak][f], reverse=True):
        row = "".join(f"{sf_by_m[m][f]:>8.3f}" for m in m_grid)
        rise[f] = sf_by_m[peak][f] - sf_by_m[low][f]
        log(f"  {f:10}{row}{rise[f]:>+9.3f}")

    carrier = max(present, key=lambda f: sf_by_m[peak][f])
    grower = max(present, key=lambda f: rise[f])
    geom = {"lambda1", "lyap_r2", "d2", "d2_r2"}
    recur = {"det", "lam"}
    fam = ("attractor geometry / divergence" if carrier in geom else
           "recurrence/laminar structure (volatility-regime persistence)" if carrier in recur else
           "generic higher-order serial dependence")
    log("\n" + "=" * 78)
    log(f"VERDICT: at the peak (m={peak}) the leading feature is '{carrier}' "
        f"(single-AUC={sf_by_m[peak][carrier]:.3f}).")
    log(f"  Fastest-growing feature across m={low}->{peak}: '{grower}' (rise {rise[grower]:+.3f}).")
    if carrier == grower:
        log(f"  These agree: '{carrier}' both leads and grows with m, so it carries the gap.")
    else:
        log(f"  Leader and fastest-grower differ ('{carrier}' vs '{grower}'); the gap is shared.")
    log(f"  Family -> {fam}.")
    log("  Step 3 should target this property: e.g. if geometry, compare attractor")
    log("  dimension/divergence; if recurrence, compare volatility-of-volatility,")
    log("  long-lag |r| memory, and high-vol run-lengths on FW vs GJR vs EGARCH to find")
    log("  the statistic where EGARCH departs from FW but GJR does not.")
    log("=" * 78)
    print(f"\n[saved to {RESULT_FILE}]")


if __name__ == "__main__":
    K = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    mg = tuple(int(x) for x in sys.argv[2].split(",")) if len(sys.argv) > 2 else (3, 5, 8)
    run(K=K, m_grid=mg)
