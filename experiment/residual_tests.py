"""
experiment/residual_tests.py -- two complementary residual-based probes.

The main study runs the toolkits on (rank-matched) RAW RETURNS. The older
chaos-in-finance literature instead FILTERS returns through an AR-GARCH and tests
the standardized residuals (Hsieh 1991; Brock-Hsieh-LeBaron 1991), because raw
returns conflate volatility clustering with genuine nonlinear structure. These
two probes connect to that tradition WITHOUT changing the main study's claim.

PART A (#3) -- DISCRIMINATION ON RESIDUALS (an extension of harder_pair).
  Filter BOTH the FW series and the GARCH series through their own fitted GJR,
  take standardized residuals, and run the SAME classifier (R1 moments vs R2
  nonlinear) on the residuals, swept over m. Fair test: each series is filtered
  identically (its own best-fit GJR), so neither side is advantaged.
  Question: does the nonlinear edge seen on RAW returns survive once the
  volatility/leverage layer is filtered out? If Delta -> 0 on residuals, the
  edge lived entirely in the volatility dynamics (the sharp version of the
  study's central claim). This is still a TOOLKIT-vs-TOOLKIT discrimination test.

PART B (#2) -- BDS ADEQUACY (a bridge to a real-data study, not a discrimination
  test). Run BDS on raw FW returns vs on FW's GJR standardized residuals. Raw
  returns should show strong BDS nonlinearity (driven by clustering); residuals
  should show much less if GJR captured the structure. This is a single-model
  goodness-of-fit question (does GARCH leave leftover nonlinearity?), in the
  language of the classic literature. NOTE: BDS asymptotic SEs are unreliable
  under heavy tails (Cont 5.3); these statistics are indicative only and a
  reported version should bootstrap the null.

Run from the repo root:   python -m experiment.residual_tests
Prints and saves to results/residual_tests.txt (symmetric arm) or
results/residual_tests_lev50.txt (leverage arm).  Uses the symmetric arm (where
the raw-returns edge was largest); set alpha_lev>0 to use the leverage arm.
"""

import os
import warnings
import numpy as np

from models.fw_ssv import FWParams, simulate_fw
from experiment.harder_pair import garch_match
from experiment.embedding_diagnostics import evaluate
from signatures.bds import bds_statistic
from signatures.bds_bootstrap import bds_bootstrap
from seeds import get_rng, MASTER_SEED

warnings.filterwarnings("ignore")
OUT = []
RESULT_FILE = "results/residual_tests.txt"


def log(s=""):
    print(s, flush=True)
    OUT.append(s)
    os.makedirs("results", exist_ok=True)
    with open(RESULT_FILE, "w") as f:
        f.write("\n".join(OUT))


def gjr_residuals(series):
    """Fit a GJR-GARCH(1,1)-t to each series and return its standardized
    residuals (scale-invariant). Each series filtered by its OWN best fit."""
    from arch import arch_model
    out = []
    for r in series:
        y = r * (10.0 / (np.std(r) + 1e-12))
        try:
            res = arch_model(y, vol="GARCH", p=1, o=1, q=1, dist="t",
                             mean="Constant", rescale=False).fit(disp="off",
                                                                 show_warning=False)
            sr = np.asarray(res.std_resid)
            sr = sr[np.isfinite(sr)]
            if len(sr) > 100:
                out.append(sr)
        except Exception:
            continue
    return out


def fw_paths(K, T, alpha_lev, tag):
    out = []
    for s in range(K):
        r = simulate_fw(FWParams(switching="dca", alpha_lev=alpha_lev), n=T,
                        burn_in=1500, rng=get_rng(MASTER_SEED, tag, alpha_lev, s))
        if r["finite"] and not r["blew_up"]:
            out.append(r["returns"])
    return out


def run(K=300, T=2500, alpha_lev=0.0, m_grid=(4, 6, 8), tau=1):

    global RESULT_FILE, OUT
    OUT = []
    RESULT_FILE = ("results/residual_tests.txt" if alpha_lev == 0
                   else f"results/residual_tests_lev{int(alpha_lev)}.txt")
    arm = "SYMMETRIC" if alpha_lev == 0 else f"LEVERAGE (alpha_lev={alpha_lev})"
    log("=" * 70)
    log(f"RESIDUAL-BASED COMPLEMENTARY TESTS -- {arm} arm")
    log(f"K={K}, T={T}, tau={tau}, seed={MASTER_SEED}")
    log("=" * 70)

    fw = fw_paths(K, T, alpha_lev, "rt_fw")
    log(f"\ngenerated {len(fw)} FW paths; fitting tight GARCH null...")
    gc = garch_match(fw, MASTER_SEED, rank_match=True)
    n = min(len(fw), len(gc))
    fw, gc = fw[:n], gc[:n]
    log(f"got {len(gc)} GARCH paths")

    # (A) discrimination on residuals
    log("\n" + "-" * 70)
    log("PART A (#3) -- discrimination on GJR standardized residuals")
    log("-" * 70)
    log("filtering both arms through their own GJR and extracting residuals...")
    fw_res = gjr_residuals(fw)
    gc_res = gjr_residuals(gc)
    n2 = min(len(fw_res), len(gc_res))
    fw_res, gc_res = fw_res[:n2], gc_res[:n2]
    log(f"got {n2} residual series per group; sweeping m={list(m_grid)}\n")
    log("  (compare to the RAW-returns result: symmetric arm had Delta rising to")
    log("   ~+0.10 by m=8. If Delta ~ 0 here, the edge was all volatility layer.)")
    log(f"\n{'m':>4}{'AUC_R1':>9}{'AUC_R2':>9}{'Delta(resid)':>14}")
    cache = {}
    for m in m_grid:
        a1, a2, _, _ = evaluate(fw_res, "rfw", gc_res, "rgc", m, tau, cache)
        log(f"{m:>4}{a1:>9.3f}{a2:>9.3f}{a2 - a1:>+14.3f}")

    # (B) BDS adequacy
    log("\n" + "-" * 70)
    log("PART B (#2) -- BDS adequacy: raw FW returns vs FW GJR residuals")
    log("-" * 70)
    raw_bds = [bds_statistic(r, m=2)["bds"] for r in fw]
    res_bds = [bds_statistic(r, m=2)["bds"] for r in fw_res]
    raw_bds = np.array([b for b in raw_bds if np.isfinite(b)])
    res_bds = np.array([b for b in res_bds if np.isfinite(b)])
    log(f"\n  Asymptotic BDS statistic (m=2), mean +/- sd across paths")
    log(f"  [reference only -- asymptotic SEs unreliable under heavy tails]:")
    log(f"    raw FW returns      : {raw_bds.mean():>7.2f} +/- {raw_bds.std():.2f}")
    log(f"    FW GJR residuals    : {res_bds.mean():>7.2f} +/- {res_bds.std():.2f}")
    drop = 100 * (1 - abs(res_bds.mean()) / (abs(raw_bds.mean()) + 1e-12))
    log(f"    -> filtering removes ~{drop:.0f}% of the raw BDS magnitude")

    log(f"\n  Bootstrap (permutation) BDS test [primary -- heavy-tail robust]:")
    n_bds = min(len(fw), len(fw_res), 60)
    raw_p, res_p = [], []
    for i in range(n_bds):
        rb = bds_bootstrap(fw[i], m=2, n_boot=199,
                           rng=get_rng(MASTER_SEED, "bdsboot_raw", i))
        rs = bds_bootstrap(fw_res[i], m=2, n_boot=199,
                           rng=get_rng(MASTER_SEED, "bdsboot_res", i))
        if np.isfinite(rb["p_value"]):
            raw_p.append(rb["p_value"])
        if np.isfinite(rs["p_value"]):
            res_p.append(rs["p_value"])
    raw_p, res_p = np.array(raw_p), np.array(res_p)
    raw_rej = 100 * np.mean(raw_p < 0.05)
    res_rej = 100 * np.mean(res_p < 0.05)
    log(f"    raw FW returns      : rejects iid in {raw_rej:.0f}% of {len(raw_p)} paths"
        f"  (median p={np.median(raw_p):.3f})")
    log(f"    FW GJR residuals    : rejects iid in {res_rej:.0f}% of {len(res_p)} paths"
        f"  (median p={np.median(res_p):.3f})")
    log("  A high raw rejection rate collapsing to near the 5% nominal level on")
    log("  residuals = GJR captured the dependence (the raw 'nonlinearity' was")
    log("  volatility clustering). A residual rejection rate still well above 5%")
    log("  would indicate leftover structure GARCH missed. This bootstrap result,")
    log("  not the asymptotic statistic above, is the one to report.")

    print(f"\n[saved to {RESULT_FILE}]")


if __name__ == "__main__":
    import sys
    lev = float(sys.argv[1]) if len(sys.argv) > 1 else 0.0
    run(alpha_lev=lev)