"""
experiment/embedding_sweep.py -- robustness of the TIGHT FW-vs-GARCH result to
the embedding dimension m.

m=4 was a hand-chosen modest value (FNN does not converge for stochastic series,
so it could not be read off the data the way tau=1 was). This script re-runs the
decisive TIGHT comparison (distribution-matched GARCH null) at m = 3,4,5,6 with
tau=1 held fixed, and reports AUC_R1, AUC_R2, their gap, and the per-feature
separation of every nonlinear (Referee 2) feature. If the small R2 edge and its
mechanism survive across m, the m=4 choice is not load-bearing.

The FW paths and the rank-matched GARCH null do NOT depend on m, so they are
generated once; only feature extraction is repeated per m.

Run from the repo root:   python -m experiment.embedding_sweep
Prints and saves to results/embedding_sweep.txt.
"""

import os
import warnings
import numpy as np

from experiment.harder_pair import fw_paths, garch_match, feats, auc, max_d
from signatures.features import feature_vector, REFEREE_1, REFEREE_2
from seeds import MASTER_SEED

warnings.filterwarnings("ignore")
OUT = []


def log(s=""):
    print(s)
    OUT.append(s)


def evaluate_at_m(fw, gc, m, tau=1):
    XF, _ = feats(fw, m, tau)
    XG, _ = feats(gc, m, tau)
    X = np.vstack([XF, XG])
    y = np.array([0] * len(fw) + [1] * len(gc))
    fc = np.all(np.isfinite(X), axis=0)
    X, names = X[:, fc], [n for n, k in zip(list(feature_vector(fw[0]).keys()), fc) if k]

    def sub(cols):
        idx = [names.index(c) for c in cols if c in names]
        return X[:, idx]

    a1 = auc(sub(REFEREE_1), y)
    a2 = auc(sub(REFEREE_2), y)
    full = list(feature_vector(fw[0]).keys())
    clust_d = max_d(XF, XG, full, ["acf_abs_1", "acf_abs_5", "acf_abs_10"])
    r2d = {f: max_d(XF, XG, full, [f]) for f in REFEREE_2}
    return a1, a2, clust_d, r2d


def run(K=120, T=2500, m_grid=(3, 4, 5, 6)):
    log("=" * 64)
    log("EMBEDDING-DIMENSION SWEEP (TIGHT FW vs rank-matched GARCH)")
    log(f"K={K}, T={T}, tau=1, seed={MASTER_SEED}")
    log("=" * 64)

    fw = fw_paths(K, T)
    gc = garch_match(fw, MASTER_SEED, rank_match=True)
    n = min(len(fw), len(gc))
    fw, gc = fw[:n], gc[:n]
    log(f"\ngenerated {n} FW + {n} tight-GARCH paths; sweeping m...\n")

    log(f"{'m':>4}{'AUC_R1':>9}{'AUC_R2':>9}{'Delta':>8}{'clust_d':>9}")
    rows = {}
    for m in m_grid:
        a1, a2, clust_d, r2d = evaluate_at_m(fw, gc, m)
        log(f"{m:>4}{a1:>9.3f}{a2:>9.3f}{a2 - a1:>+8.3f}{clust_d:>9.2f}")
        rows[m] = r2d
    log("\n   R2 per-feature Cohen's d by m (which nonlinear feature carries any gap):")
    log("   " + f"{'feature':>14}" + "".join(f"{'m='+str(m):>8}" for m in m_grid))
    for f in sorted(REFEREE_2, key=lambda f: max(rows[m][f] for m in m_grid), reverse=True):
        log("   " + f"{f:>14}" + "".join(f"{rows[m][f]:>8.2f}" for m in m_grid))

    log("\nReading: if Delta (AUC_R2 - AUC_R1) stays positive and similar across m,")
    log("the m=4 choice is not driving the result; the per-feature table shows which")
    log("nonlinear feature carries the gap and whether it grows with m. If Delta")
    log("swings sign with m, the result is embedding-dependent and must be reported")
    log("as such. NOTE: at K=120 the levels are inflated by the small-sample effect")
    log("seen in the hub (Delta shrinks toward zero as K grows to 300); only the")
    log("trend across m is informative here, not the absolute Delta.")

    os.makedirs("results", exist_ok=True)
    with open("results/embedding_sweep.txt", "w") as f:
        f.write("\n".join(OUT))
    print("\n[saved to results/embedding_sweep.txt]")


if __name__ == "__main__":
    run()