"""
check_fw_pair.py -- DCA vs TPA moment-separability diagnostic.

Model pair: Franke-Westerhoff with two switching mechanisms, same demand/price
engine.
    Model A = DCA  (discrete choice; memoryless logit of the attractiveness index)
    Model B = TPA  (transition probabilities; switching with inertia)

WHAT THIS DOCUMENTS (and why it is no longer a "confusability gate")
--------------------------------------------------------------------
With the faithful TPA switching probabilities (pi = min(1, nu*exp(+/-a)), per
Franke & Westerhoff 2012 p.9), the DCA and TPA models separate strongly on
standard moments wherever they are dynamically distinct. They become confusable
only as nu -> 1, where TPA degenerates into the DCA limit (so there is then no
real difference to detect). There is therefore no nu at which the pair is both
confusable on moments AND dynamically distinct.

This is the basis for Finding 1: because the behavioural pair separates on the
standard moments wherever it genuinely differs, the nonlinear signatures have no
gap left to fill -- moments already capture the difference. The study's
matched-moment comparison (where nonlinear features are given a real chance) is
instead constructed explicitly against a GARCH null; see experiment/harder_pair.py.

This script quantifies the separation with a per-moment standardised mean gap
(Cohen's d) and shows its dependence on nu. It is a diagnostic, not a pass/fail
gate. Run from repo root:

    python check_fw_pair.py
"""

import numpy as np
from models.fw_ssv import FWParams, simulate_fw
from signatures.moments import stylized_facts
from seeds import get_rng, MASTER_SEED

MOMENTS = ["std", "excess_kurtosis", "hill_alpha", "acf_abs_mean", "acf_ret_max"]
NU_GRID = [0.05, 0.10, 0.30, 1.00] 


def moment_vector(sf):
    return [
        sf["std"],
        sf["excess_kurtosis"],
        sf["hill_alpha"],
        float(np.nanmean(sf["acf_abs_returns"])),
        float(np.nanmax(np.abs(sf["acf_returns"]))),
    ]


def sample_moments(params, tag, K=80, n=2500, burn_in=1500):
    rows = []
    for s in range(K):
        rng = get_rng(MASTER_SEED, "sepdiag", tag, s)
        out = simulate_fw(params, n=n, burn_in=burn_in, rng=rng)
        if out["blew_up"] or not out["finite"]:
            continue
        rows.append(moment_vector(stylized_facts(out["returns"])))
    return np.asarray(rows)


def cohens_d(a, b):
    """Cohen's d with a nan-guard: drop non-finite entries per group first."""
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return np.nan
    pooled = np.sqrt((a.var() + b.var()) / 2) + 1e-12
    return abs(a.mean() - b.mean()) / pooled


if __name__ == "__main__":
    K, N = 80, 2500
    print("DCA vs TPA moment-separability diagnostic (faithful FW TPA, FW 2012 p.9)")
    print(f"{K} paths each, n={N}, master_seed={MASTER_SEED}\n")

    dca = sample_moments(FWParams(switching="dca"), "dca", K, N)

    tpa_by_nu = {nu: sample_moments(FWParams(switching="tpa", nu=nu), f"tpa{nu}", K, N)
                 for nu in NU_GRID}

    nu0 = 0.05
    tpa0 = tpa_by_nu[nu0]
    print(f"Per-moment Cohen's d at nu = {nu0} (FW 2012 value):")
    print(f"{'moment':16}{'DCA mean+/-sd':>22}{'TPA mean+/-sd':>22}   Cohen-d")
    for i, nm in enumerate(MOMENTS):
        d_, t_ = dca[:, i], tpa0[:, i]
        gap = cohens_d(d_, t_)
        print(f"{nm:16}{np.nanmean(d_):>11.4f} +/-{np.nanstd(d_):<7.4f}"
              f"{np.nanmean(t_):>11.4f} +/-{np.nanstd(t_):<7.4f}   {gap:.2f}")

    print("\nMax standardised gap (Cohen's d) across moments, vs nu:")
    for nu in NU_GRID:
        tpa = tpa_by_nu[nu]
        gaps = [cohens_d(dca[:, i], tpa[:, i]) for i in range(len(MOMENTS))]
        gaps = [g for g in gaps if np.isfinite(g)]
        max_d = max(gaps) if gaps else float("nan")
        tag = "  <- FW 2012 value" if nu == 0.05 else ("  <- TPA -> DCA limit" if nu == 1.00 else "")
        print(f"  nu={nu:<5} max d = {max_d:5.2f}{tag}")

    print("\nReading: the faithful DCA/TPA pair separates strongly on standard")
    print("moments wherever it is dynamically distinct (large d at nu <= 0.30),")
    print("and is confusable only as nu -> 1 where TPA collapses into DCA (d small).")
    print("So moments already capture the behavioural difference; this is the")
    print("premise of Finding 1. The matched-moment test in which nonlinear")
    print("features could matter is constructed against a GARCH null instead")
    print("(experiment/harder_pair.py).")