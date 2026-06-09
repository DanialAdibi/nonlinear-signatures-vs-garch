"""
experiment/loggarch_null.py -- second LOG-family null to confirm the additive-vs-log axis.

WHY THIS EXISTS
The APARCH arm showed two ADDITIVE leverage-aware nulls (GJR power-2, APARCH power-1.7)
tie with FW while the one LOG-variance null (EGARCH) separates (dAUC +0.146). That rests on
a single log null, so the log side needs a second, INDEPENDENT witness. Log-GARCH is the
canonical alternative log-variance model and, crucially, it is NOT a relabelled EGARCH:

  log-GARCH:  log sigma2_t = omega + alpha log eps2_{t-1} + xi sign(eps_{t-1}) + beta log sigma2_{t-1}
  EGARCH:     log sigma2_t = omega + alpha(|z_{t-1}|-E|z|) + gamma z_{t-1}       + beta log sigma2_{t-1}

Both are LOG-variance (multiplicative) with persistence alpha+beta (resp. beta), but the
SHOCK transform differs: log-GARCH is driven by log eps2 (a log|.| news impact), EGARCH by
|z| (a V-shaped news impact). The asymmetry here is a single sign term xi (xi<0 = equity
leverage: a negative lagged return raises log-variance), the log-space analogue of EGARCH's
one gamma.

HYPOTHESIS TEST
  If the axis is additive-vs-log, log-GARCH (log) should SEPARATE like EGARCH -> two log
  nulls separate, two additive nulls tie, axis LOCKED.
  If log-GARCH TIES like GJR/APARCH, then EGARCH's separation was specific to its |z|
  news-impact shape, NOT log-ness generically -> redirect to the news-impact form.

PRACTICAL NOTES
  - Inlier problem: the driver log eps2 -> -inf as a return -> 0, which destabilises the
    fit. We floor log eps2 at its FLOOR_PCT percentile (computed once from demeaned data).
    FLOOR_PCT is exposed so the fit can be checked for floor-sensitivity (persistence should
    be stable as it moves).
  - Custom Student-t QML (arch has no log-GARCH). Standardised-t with nu>2, mean=Constant,
    matching the scaling/seeding of the other null arms (scale=10/(std+eps), rank_match=True).
  - In-run GJR re-score is carried as a faithfulness anchor (should reproduce +0.016).

Run:  python -u -m experiment.loggarch_null
Output: results/loggarch_null.txt
"""

import os
import sys
import numpy as np
from scipy.optimize import minimize
from scipy.special import gammaln

from experiment.harder_pair_lev import build_or_load
from experiment.embedding_diagnostics import evaluate
from signatures.features import REFEREE_2
from seeds import get_rng, MASTER_SEED

RESULT_FILE = "results/loggarch_null.txt"
OUT = []

FLOOR_PCT = 1.0         # percentile to floor log(eps^2) at (inlier guard); vary to check sensitivity
SIM_BURN = 250          # simulation burn-in to wash out the initial log-variance
GJR_REF_M8 = +0.016     # canonical additive anchor (ties)
EGARCH_REF_M8 = +0.146  # canonical log anchor (separates)
# EGARCH per-feature Cohen's d at m=8 (from Step 2, K=300), as the log-side reference column:
EG_REF_D = {"lam": 0.88, "d2": 0.82, "det": 0.60, "d2_r2": 0.26,
            "bds": 0.30, "lyap_r2": 0.13, "lambda1": 0.14}


def log(s=""):
    print(s, flush=True)
    OUT.append(s)
    os.makedirs("results", exist_ok=True)
    with open(RESULT_FILE, "w") as f:
        f.write("\n".join(OUT))


def _std_t_loglik(eps, logh, nu):
    """Standardised Student-t (unit variance, nu>2) log-density of eps given variance exp(logh)."""
    h = np.exp(logh)
    z2 = (eps * eps) / h
    c = gammaln((nu + 1) / 2) - gammaln(nu / 2) - 0.5 * np.log((nu - 2) * np.pi)
    return c - 0.5 * (nu + 1) * np.log1p(z2 / (nu - 2)) - 0.5 * logh


def loggarch_fit(r):
    """Custom Student-t QML for the asymmetric log-GARCH. Returns a param dict."""
    r = np.asarray(r, float)
    n = len(r)
    e0 = r - np.median(r)
    floor = float(np.percentile(np.log(e0 * e0 + 1e-300), FLOOR_PCT))
    v0 = float(np.log(np.var(r) + 1e-12))

    def negll(theta):
        mu, omega, alpha, xi, beta, lognu = theta
        nu = 2.0 + np.exp(lognu)
        eps = r - mu
        le2 = np.maximum(np.log(eps * eps + 1e-300), floor)
        s = np.sign(eps)
        c0 = omega + alpha * le2 + xi * s          # vectorised driver
        logh = np.empty(n)
        logh[0] = v0
        b = np.clip(beta, -0.9999, 0.9999)
        for t in range(1, n):                       # sequential log-variance recursion
            logh[t] = c0[t - 1] + b * logh[t - 1]
        logh = np.clip(logh, -50.0, 50.0)
        ll = _std_t_loglik(eps, logh, nu)
        val = -np.sum(ll)
        return val if np.isfinite(val) else 1e12

    x0 = [float(np.mean(r)), v0 * 0.05, 0.04, -0.02, 0.90, np.log(6.0)]
    bnds = [(None, None), (None, None), (-0.6, 0.6), (-1.0, 1.0), (-0.999, 0.999), (-2.0, 5.0)]
    res = minimize(negll, x0, method="L-BFGS-B", bounds=bnds)
    if not res.success:                              # polish a stubborn fit
        res2 = minimize(negll, res.x, method="Nelder-Mead",
                        options={"maxiter": 4000, "xatol": 1e-6, "fatol": 1e-6})
        if res2.fun < res.fun:
            res = res2
    mu, omega, alpha, xi, beta, lognu = res.x
    return dict(mu=mu, omega=omega, alpha=alpha, xi=xi, beta=beta,
                nu=2.0 + np.exp(lognu), floor=floor, v0=v0, ok=bool(res.success))


def loggarch_simulate(p, n, rng):
    """Simulate n returns from a fitted log-GARCH (with burn-in)."""
    mu, omega, alpha, xi, beta = p["mu"], p["omega"], p["alpha"], p["xi"], p["beta"]
    nu, floor, v0 = p["nu"], p["floor"], p["v0"]
    b = np.clip(beta, -0.9999, 0.9999)
    scale_t = np.sqrt((nu - 2) / nu)               # standardise t to unit variance
    total = n + SIM_BURN
    logh = v0
    out = np.empty(total)
    for t in range(total):
        logh = min(max(logh, -50.0), 50.0)
        z = rng.standard_t(nu) * scale_t
        eps = np.sqrt(np.exp(logh)) * z
        out[t] = mu + eps
        le2 = max(np.log(eps * eps + 1e-300), floor)
        logh = omega + alpha * le2 + xi * np.sign(eps) + b * logh
    return out[SIM_BURN:]


def loggarch_fit_and_simulate(fw_series, seed0, rank_match=True):
    """Per-path fit+simulate, mirroring the other null arms (scale, seed, rank_match)."""
    out, xis, persist, oks = [], [], [], []
    for i, r in enumerate(fw_series):
        scale = 10.0 / (np.std(r) + 1e-12)
        y = r * scale
        try:
            p = loggarch_fit(y)
            xis.append(p["xi"])
            persist.append(p["alpha"] + p["beta"])
            oks.append(p["ok"])
            rng = get_rng(seed0, "loggarch_seed", i)
            s = loggarch_simulate(p, len(y), rng) / scale
            if not np.all(np.isfinite(s)):
                continue
            if rank_match and len(s) == len(r):
                s = np.sort(r)[np.argsort(np.argsort(s))]
            out.append(s)
        except Exception:
            continue
    return out, np.array(xis), np.array(persist), np.array(oks)


def run(K=300, T=2500, alpha_lev=50.0, m_grid=(3, 5, 8), tau=1, floor_pct=None):
    global FLOOR_PCT, RESULT_FILE
    if floor_pct is not None:
        FLOOR_PCT = floor_pct
        RESULT_FILE = f"results/loggarch_null_floor{FLOOR_PCT}.txt"
    log("=" * 80)
    log("log-GARCH null -- second LOG-family witness for the additive-vs-log axis")
    log(f"K={K}  T={T}  alpha_lev={alpha_lev}  null=asymmetric log-GARCH-t  FLOOR_PCT={FLOOR_PCT}")
    log("=" * 80)

    fw, fwb, gc_gjr, _gam = build_or_load(K, T, alpha_lev)
    log(f"loaded {len(fw)} FW + {len(fwb)} FWb + {len(gc_gjr)} GJR (cached)")
    log("fitting log-GARCH null (one-time custom QML, rank_match=True, slow) ...")
    gc_lg, xis, persist, oks = loggarch_fit_and_simulate(fw, MASTER_SEED, rank_match=True)
    n = min(len(fw), len(fwb), len(gc_gjr), len(gc_lg))
    fw, fwb, gc_gjr, gc_lg = fw[:n], fwb[:n], gc_gjr[:n], gc_lg[:n]
    log(f"  aligned {n} paths per group; optimiser-converged fits: {int(np.sum(oks))}/{len(oks)}")
    if len(persist):
        log(f"  persistence alpha+beta: mean={np.mean(persist):.3f}  med={np.median(persist):.3f}"
            f"  frac>0.98={np.mean(persist > 0.98):.2f}  (stationary if < 1)")
    if len(xis):
        log(f"  asymmetry xi: mean={np.mean(xis):+.4f}  "
            f"({100 * np.mean(xis < 0):.0f}% negative -> equity leverage sign engaged)")

    log("\nFW(leverage) vs nulls, by embedding dimension m  (dAUC = AUC_R2 - AUC_R1):")
    log(f"  {'m':>3}{'GJR dAUC':>11}{'logGARCH dAUC':>15}{'(R1':>8}{'R2)':>8}{'A-vs-A':>10}")
    cache = {}
    lg_dauc, gjr_dauc, r2d_peak, gjr_r2d_peak = {}, {}, None, None
    peak = max(m_grid)
    for m in m_grid:
        a1g, a2g, _, r2dg = evaluate(fw, "lfw", gc_gjr, "lgjr", m, tau, cache)
        a1l, a2l, _c, r2dl = evaluate(fw, "lfw", gc_lg, "llg", m, tau, cache)
        a1n, a2n, _, _ = evaluate(fw, "lfw", fwb, "lfwb", m, tau, cache)
        gjr_dauc[m], lg_dauc[m] = a2g - a1g, a2l - a1l
        log(f"  {m:>3}{a2g - a1g:>+11.3f}{a2l - a1l:>+15.3f}{a1l:>8.3f}{a2l:>8.3f}{a2n - a1n:>+10.3f}")
        if m == peak:
            r2d_peak, gjr_r2d_peak = r2dl, r2dg

    log(f"\nPer-feature Cohen's d at m={peak} -- log-GARCH vs GJR (additive, in-run) and")
    log("EGARCH (log, cited Step 2).  log-GARCH near EGARCH = log side confirmed:")
    log(f"  {'feature':10}{'logGARCH':>10}{'GJR':>9}{'EGARCH':>9}")
    for f in sorted(REFEREE_2, key=lambda f: r2d_peak.get(f, 0), reverse=True):
        log(f"  {f:10}{r2d_peak.get(f, float('nan')):>10.2f}"
            f"{gjr_r2d_peak.get(f, float('nan')):>9.2f}{EG_REF_D.get(f, float('nan')):>9.2f}")

    # ---- verdict ----
    lgk = lg_dauc[peak]
    log("\n" + "=" * 80)
    log(f"ANCHORS at m={peak}:  GJR/APARCH (additive) ~{GJR_REF_M8:+.3f} tie"
        f"   |   EGARCH (log) {EGARCH_REF_M8:+.3f} separates")
    log(f"  in-run GJR re-score: {gjr_dauc[peak]:+.3f}  (faithfulness vs canonical +0.016)")
    log(f"  log-GARCH (second log null): {lgk:+.3f}")
    near_eg = abs(lgk - EGARCH_REF_M8)
    near_gjr = abs(lgk - GJR_REF_M8)
    if lgk > 0.10 and near_eg < near_gjr:
        log("VERDICT: log-GARCH SEPARATES like EGARCH -> the LOG side is CONFIRMED by a second,")
        log("  independent log-variance null with a different news-impact transform. The")
        log("  additive-vs-log axis is LOCKED: two additive nulls (GJR, APARCH) tie, two log")
        log("  nulls (EGARCH, log-GARCH) separate. The recurrence features detect conditional-")
        log("  variance FUNCTIONAL-FORM (additive vs log) mis-specification that the moments miss.")
        log("  This can now be stated as a finding, not a conjecture.")
    elif lgk < 0.05 and near_gjr < near_eg:
        log("VERDICT: log-GARCH TIES like the additive nulls -> the log side is NOT confirmed.")
        log("  EGARCH's separation was specific to its |z| (V-shaped) news-impact, not to the")
        log("  log-variance recursion in general (log-GARCH uses a log|.| news-impact and ties).")
        log("  Reframe: the axis is the NEWS-IMPACT SHAPE, not additive-vs-log. EGARCH stands")
        log("  alone; report it as such and, if pursued, probe the news-impact curve directly.")
    else:
        log(f"VERDICT: INTERMEDIATE (log-GARCH {lgk:+.3f}). Not cleanly on either side. Check the")
        log("  per-feature table (lam/d2 near EGARCH = log-like; near GJR = additive-like), the")
        log("  fit health (convergence, persistence, xi sign) and floor-sensitivity before")
        log("  concluding; a fit pathology can mute a real separation.")
    log("=" * 80)
    print(f"\n[saved to {RESULT_FILE}]")


if __name__ == "__main__":
    # usage: python -m experiment.loggarch_null [K] [FLOOR_PCT]
    #   floor-sensitivity sweep: run at 0.5, 1.0, 2.5 -> separate floor-tagged output files
    K = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    floor = float(sys.argv[2]) if len(sys.argv) > 2 else None
    run(K=K, floor_pct=floor)