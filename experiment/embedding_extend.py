"""
experiment/embedding_extend.py -- extend the embedding-dimension sweep to higher
m WITHOUT refitting GARCH every time, by caching the (m-independent) simulated
paths to disk.

The embedding diagnostics (K=300, tau=1 and tau=2) indicated that the rising
nonlinear advantage with m is real rather than a tau=1 overlap artifact: the
FW-vs-GARCH Delta climbs with m while the FW-vs-FW null stays flat. (Those
figures, including the m=6 Delta around +0.083 and a laminarity-led reading, are
from the pre-regeneration run and are being reconfirmed under the verified code
and pinned environment; the per-feature tables below show which nonlinear feature
actually carries the trend, which need not be laminarity.) The open question is
whether the climb PLATEAUS or keeps rising, which this script answers by
extending to m=7,8(+).

CACHING: the FW paths, FW-null paths, and tight-GARCH paths do NOT depend on m.
They are generated and fitted ONCE, saved to results/cache/, and reloaded on
every subsequent run. Only the (cheap) feature extraction repeats per m. So the
first run pays the GARCH fit once; all later runs (any m) are fast.

The m grid is ordered so m=7,8 are computed FIRST (the conclusion), then m=6 as a
FAITHFULNESS CHECK (the disk-cached paths must reproduce the m=6 result from
embedding_diagnostics in the same regeneration -- a cache-integrity check), then
the rest. Partial results are written after every row, so interrupting after
m=7,8 still leaves the cache and partial output on disk.

Run from the repo root:   python -m experiment.embedding_extend
Prints and saves to results/embedding_extend.txt.
"""

import os
import warnings
import numpy as np

from experiment.harder_pair import garch_match, feats, auc, max_d
from experiment.embedding_diagnostics import fw_paths_tagged, evaluate
from signatures.features import feature_vector, REFEREE_1, REFEREE_2
from seeds import MASTER_SEED

warnings.filterwarnings("ignore")
OUT = []
CACHE_DIR = "results/cache"
RESULT_FILE = "results/embedding_extend.txt"


def log(s=""):
    print(s, flush=True)
    OUT.append(s)
    os.makedirs("results", exist_ok=True)
    with open(RESULT_FILE, "w") as f:
        f.write("\n".join(OUT))


def build_or_load(K, T):
    fwf = os.path.join(CACHE_DIR, f"fw_K{K}_T{T}.npy")
    fwbf = os.path.join(CACHE_DIR, f"fwb_K{K}_T{T}.npy")
    gcf = os.path.join(CACHE_DIR, f"gc_K{K}_T{T}.npy")
    if all(os.path.exists(p) for p in (fwf, fwbf, gcf)):
        log(f"loaded cached paths for K={K},T={T} from results/cache/")
        return list(np.load(fwf)), list(np.load(fwbf)), list(np.load(gcf))
    log("no cache found -- generating paths + fitting tight GARCH (one-time, slow)...")
    fw = fw_paths_tagged(K, T, "hp_fw")
    fwb = fw_paths_tagged(K, T, "hp_fw_b")
    gc = garch_match(fw, MASTER_SEED, rank_match=True)
    n = min(len(fw), len(fwb), len(gc))
    fw, fwb, gc = fw[:n], fwb[:n], gc[:n]
    os.makedirs(CACHE_DIR, exist_ok=True)
    np.save(fwf, np.array(fw))
    np.save(fwbf, np.array(fwb))
    np.save(gcf, np.array(gc))
    log(f"cached {n} paths per group to results/cache/ (future runs are fast)")
    return fw, fwb, gc


def run(K=300, T=2500, m_grid=(7, 8, 6, 5, 4, 3), tau=1):
    log("=" * 70)
    log("EMBEDDING EXTENSION -- does the nonlinear advantage plateau or keep rising?")
    log(f"K={K}, T={T}, tau={tau}, seed={MASTER_SEED}")
    log("=" * 70)

    fw, fwb, gc = build_or_load(K, T)
    log(f"n={len(fw)} per group; sweeping m={list(m_grid)} at tau={tau}")
    log("(m=7,8 first = the conclusion; m=6 = faithfulness check vs prior +0.083)")
    log("")
    log(f"{'m':>4} | {'FW-vs-GARCH Delta':>18} | {'NULL Delta':>12}")
    log("-" * 44)
    cache = {}
    rows_g, rows_n = {}, {}
    for m in m_grid:
        a1g, a2g, _cg, r2dg = evaluate(fw, "fw", gc, "gc", m, tau, cache)
        a1n, a2n, _cn, r2dn = evaluate(fw, "fw", fwb, "fwb", m, tau, cache)
        rows_g[m], rows_n[m] = r2dg, r2dn
        tag = "   <- faithfulness check (must match embedding_diagnostics m=6, same run)" if m == 6 else ""
        log(f"{m:>4} | {a2g - a1g:>+18.3f} | {a2n - a1n:>+12.3f}{tag}")

    def r2_table(rows, title):
        log(f"\n   {title} -- R2 per-feature Cohen's d by m:")
        ms = sorted(rows)
        log("   " + f"{'feature':>14}" + "".join(f"{'m='+str(m):>8}" for m in ms))
        for f in sorted(REFEREE_2, key=lambda f: max(rows[m][f] for m in ms), reverse=True):
            log("   " + f"{f:>14}" + "".join(f"{rows[m][f]:>8.2f}" for m in ms))

    r2_table(rows_g, "FW-vs-GARCH")
    r2_table(rows_n, "NULL (FW vs FW)")

    log("")
    log("Reading: if the FW-vs-GARCH Delta and its leading R2 feature level off at")
    log("high m -> the signal SATURATES (that m is the regime structure's")
    log("dimensionality). If they keep climbing -> unbounded growth, needs a")
    log("mechanism. The NULL Delta and its R2 features must stay flat (~0) at all m,")
    log("confirming high-m does not manufacture separation. The per-feature tables")
    log("show which nonlinear feature carries the trend (not assumed to be laminarity).")
    print(f"\n[saved to {RESULT_FILE}]")


if __name__ == "__main__":
    run()