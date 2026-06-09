"""
experiment/embedding_diagnostics.py -- is the rising ΔAUC-with-m an artifact?

The plain m-sweep (experiment/embedding_sweep.py) showed AUC_R2 climbing with the
embedding dimension m at tau=1, opening a growing gap over AUC_R1. That could be
either (i) REAL -- higher m gives the nonlinear measures room to resolve a genuine
FW-vs-GARCH timing difference -- or (ii) an ARTIFACT of tau=1 vector overlap: at
tau=1 an m-dimensional embedding vector shares m-1 coordinates with the next one,
which can manufacture apparent recurrence/dimension structure (the same trap that
faked a scaling region for noise in the Phase-3 validation gate).

Three diagnostics settle it, all at K=300 (so the trend is not small-sample noise):

  (1) NULL ARTIFACT TEST. Sweep m on TWO INDEPENDENT FW BATCHES (same DGP). For
      two identical models ΔAUC must be ~0 at every m. If ΔAUC instead CLIMBS with
      m here, the embedding is generating separation where there is none -> the
      FW-vs-GARCH rise is an ARTIFACT.

  (2) tau=1 RE-RUN at K=300. The clean tau=1 trend at full sample size, to compare
      against the earlier K=120 sweep and against tau=2.

  (3) OVERLAP TEST (tau=2). Re-run FW-vs-GARCH with a delay that removes the
      heavy vector overlap. If the rising-ΔAUC pattern SURVIVES at tau=2 it is
      REAL; if it DISAPPEARS it was the overlap.

Reading:
  * NULL flat (~0) across m  AND  FW-vs-GARCH rise survives at tau=2  -> REAL
    (a recovered positive finding: nonlinear measures need m>=5 to see it).
  * NULL climbs with m  OR  the rise vanishes at tau=2                -> ARTIFACT
    (negative result stands; high-m + tau=1 nonlinear measures are untrustworthy
    on this kind of data -- itself a useful methodological point).

Run from the repo root:   python -m experiment.embedding_diagnostics
Prints and saves to results/embedding_diagnostics.txt.
"""

import os
import warnings
import numpy as np

from models.fw_ssv import FWParams, simulate_fw
from experiment.harder_pair import garch_match, feats, auc, max_d
from signatures.features import feature_vector, REFEREE_1, REFEREE_2
from seeds import get_rng, MASTER_SEED

warnings.filterwarnings("ignore")
OUT = []


def log(s=""):
    print(s)
    OUT.append(s)


def fw_paths_tagged(K, T, tag):
    out = []
    for s in range(K):
        r = simulate_fw(FWParams(switching="dca"), n=T, burn_in=1500,
                        rng=get_rng(MASTER_SEED, tag, s))
        if r["finite"] and not r["blew_up"]:
            out.append(r["returns"])
    return out


def feat_cached(series, tag, m, tau, cache):
    """Cache feature matrices by (set tag, m, tau) so each is computed once."""
    key = (tag, m, tau)
    if key not in cache:
        cache[key] = feats(series, m, tau)[0]
    return cache[key]


def evaluate(A, tagA, B, tagB, m, tau, cache):
    XA = feat_cached(A, tagA, m, tau, cache)
    XB = feat_cached(B, tagB, m, tau, cache)
    X = np.vstack([XA, XB])
    y = np.array([0] * len(A) + [1] * len(B))
    fc = np.all(np.isfinite(X), axis=0)
    full = list(feature_vector(A[0]).keys())
    names = [n for n, k in zip(full, fc) if k]
    X = X[:, fc]

    def sub(cols):
        idx = [names.index(c) for c in cols if c in names]
        return X[:, idx]

    a1 = auc(sub(REFEREE_1), y)
    a2 = auc(sub(REFEREE_2), y)
    clust = max_d(XA, XB, full, ["acf_abs_1", "acf_abs_5", "acf_abs_10"])
    r2d = {f: max_d(XA, XB, full, [f]) for f in REFEREE_2}
    return a1, a2, clust, r2d


def msweep(A, tagA, B, tagB, label, tau, cache, m_grid=(3, 4, 5, 6)):
    log(f"\n[{label}]   tau={tau}")
    log(f"{'m':>4}{'AUC_R1':>9}{'AUC_R2':>9}{'Delta':>8}{'clust_d':>9}")
    rows = {}
    for m in m_grid:
        a1, a2, clust, r2d = evaluate(A, tagA, B, tagB, m, tau, cache)
        log(f"{m:>4}{a1:>9.3f}{a2:>9.3f}{a2 - a1:>+8.3f}{clust:>9.2f}")
        rows[m] = r2d
    log("   R2 per-feature Cohen's d by m:")
    log("   " + f"{'feature':>14}" + "".join(f"{'m='+str(m):>8}" for m in m_grid))
    for f in sorted(REFEREE_2, key=lambda f: max(rows[m][f] for m in m_grid), reverse=True):
        log("   " + f"{f:>14}" + "".join(f"{rows[m][f]:>8.2f}" for m in m_grid))


def run(K=300, T=2500):
    log("=" * 64)
    log("EMBEDDING DIAGNOSTICS -- is the rising Delta-with-m real or artifact?")
    log(f"K={K}, T={T}, seed={MASTER_SEED}")
    log("=" * 64)

    fw = fw_paths_tagged(K, T, "hp_fw")
    fw_b = fw_paths_tagged(K, T, "hp_fw_b")
    log(f"\ngenerated {len(fw)} FW + {len(fw_b)} FW(null) paths; fitting tight GARCH...")
    gc = garch_match(fw, MASTER_SEED, rank_match=True)
    n = min(len(fw), len(fw_b), len(gc))
    fw, fw_b, gc = fw[:n], fw_b[:n], gc[:n]
    log(f"got {len(gc)} tight-GARCH paths; running diagnostics (n={n} per group)...")

    cache = {}
    # (1) the decisive artifact test: two identical FW batches, tau=1
    msweep(fw, "fw", fw_b, "fwb", "NULL: FW vs FW (artifact test)", 1, cache)
    # (2) the K=300 tau=1 trend (clean re-run of the original sweep)
    msweep(fw, "fw", gc, "gc", "FW vs tight-GARCH -- tau=1, K=300", 1, cache)
    # (3) the overlap test: same comparison at tau=2 (vectors no longer overlap heavily)
    msweep(fw, "fw", gc, "gc", "FW vs tight-GARCH -- tau=2 (overlap test)", 2, cache)
    # supporting: null at tau=2 as well
    msweep(fw, "fw", fw_b, "fwb", "NULL: FW vs FW -- tau=2", 2, cache)

    log("\nREADING:")
    log("  REAL  if NULL stays ~0 across m AND the FW-vs-GARCH rise survives tau=2.")
    log("  ARTIFACT if NULL climbs with m OR the rise vanishes at tau=2.")
    log("  (AUC_R1 must be constant across m within each block -- it ignores the")
    log("   embedding; any variation flags a bug.)")

    os.makedirs("results", exist_ok=True)
    with open("results/embedding_diagnostics.txt", "w") as f:
        f.write("\n".join(OUT))
    print("\n[saved to results/embedding_diagnostics.txt]")


if __name__ == "__main__":
    run()