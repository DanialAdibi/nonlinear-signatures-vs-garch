"""
experiment/harder_pair.py -- FW (behavioural) vs a moment-matched GARCH null.

Run from the repo root:   python -m experiment.harder_pair
Results print AND save to results/harder_pair.txt.

THE QUESTION
------------
Does the behavioural switching mechanism leave a nonlinear fingerprint that a
purely statistical volatility model (GJR-GARCH) lacks? GARCH is the workhorse
null in empirical finance, so this is the externally interesting version of the
study: "behavioural vs statistical, matched on standard moments -- can the
nonlinear signatures tell them apart?"

DESIGN
------
1. Generate FW (DCA) return paths -- fat tails + volatility clustering.
2. Fit a GJR-GARCH(1,1) with Student-t innovations to each FW path and SIMULATE
   a matched-length series from the fitted model. By construction the GARCH
   captures the conditional-variance structure, so the two should be confusable
   on standard moments.
3. Confusability check (per-feature Cohen's d on Referee 1).
4. Discriminate FW vs GARCH: AUC for Referee 1 (moments), Referee 2 (nonlinear),
   Combined. Positive result for the hypothesis = R2 clearly > R1 (and > ~0.5):
   the behavioural mechanism is visible to nonlinear signatures but not moments.
5. (Optional) Paired bootstrap CI on Delta = AUC_R2 - AUC_R1. The point estimate
   of Delta is small and its SIGN is not stable across reseedings, so the honest
   claim is a tie. The bootstrap turns "Delta lands near zero" into a measured
   interval: a CI that brackets zero is positive evidence that FW and GARCH are
   statistically indistinguishable once the marginal is matched. Enable with the
   HP_BOOTSTRAP=1 environment variable (off by default; the main run is unchanged).

Requires the `arch` package:  pip install arch
"""

import os
import warnings
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score

from models.fw_ssv import FWParams, simulate_fw
from signatures.features import feature_vector, REFEREE_1, REFEREE_2
from seeds import get_rng, MASTER_SEED

warnings.filterwarnings("ignore")
OUT = []


def log(s=""):
    print(s)
    OUT.append(s)


def auc(X, y):
    pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=6, random_state=0)
    return float(cross_val_score(pipe, X, y, cv=cv, scoring="roc_auc").mean())


def fw_paths(K, T):
    out = []
    for s in range(K):
        r = simulate_fw(FWParams(switching="dca"), n=T, burn_in=1500,
                        rng=get_rng(MASTER_SEED, "hp_fw", s))
        if r["finite"] and not r["blew_up"]:
            out.append(r["returns"])
    return out


def garch_match(fw_series, seed0, rank_match=False):
    """
    Fit GJR-GARCH(1,1)-t to each FW path and simulate a matched-length path.
    If rank_match=True, remap the simulated series onto the paired FW path's
    empirical distribution (rank-order transform) so the UNCONDITIONAL moments
    match FW exactly, while GARCH's conditional-variance dynamics are preserved.
    This forces Referee-1 distributional features toward parity, isolating any
    purely temporal/nonlinear signal for Referee 2.
    """
    from arch import arch_model
    from arch.univariate import StudentsT
    out = []
    for i, r in enumerate(fw_series):
        scale = 10.0 / (np.std(r) + 1e-12)
        y = r * scale
        try:
            am = arch_model(y, vol="GARCH", p=1, o=1, q=1, dist="t",
                            mean="Constant", rescale=False)
            res = am.fit(disp="off", show_warning=False)
            seed = int(get_rng(seed0, "garch_seed", i).integers(0, 2 ** 31))
            try:
                am.distribution = StudentsT(seed=seed)
            except Exception:
                pass
            sd = am.simulate(res.params, nobs=len(y))
            s = sd["data"].values / scale
            if rank_match and len(s) == len(r):
                s = np.sort(r)[np.argsort(np.argsort(s))]
            out.append(s)
        except Exception:
            out.append(None)
    return out


def feats(series, m=4, tau=1):
    rows = [list(feature_vector(r, m=m, tau=tau).values()) for r in series]
    names = list(feature_vector(series[0], m=m, tau=tau).keys())
    return np.asarray(rows), names


def max_d(XA, XB, names, group):
    best = 0.0
    for i, nm in enumerate(names):
        if nm not in group:
            continue
        a, b = XA[:, i], XB[:, i]
        a, b = a[np.isfinite(a)], b[np.isfinite(b)]
        p = np.sqrt((a.var() + b.var()) / 2) + 1e-12
        best = max(best, abs(a.mean() - b.mean()) / p)
    return best


def _evaluate(fw, gc, m, tau, label, decompose=False):
    n = min(len(fw), len(gc))
    fw, gc = fw[:n], gc[:n]
    XF, names = feats(fw, m, tau)
    XG, _ = feats(gc, m, tau)
    X = np.vstack([XF, XG])
    y = np.array([0] * n + [1] * n)
    fc = np.all(np.isfinite(X), axis=0)
    X, names = X[:, fc], [nm for nm, k in zip(names, fc) if k]
    full_names = list(feature_vector(fw[0]).keys())
    r1d = max_d(XF, XG, full_names, REFEREE_1)
    r2d = max_d(XF, XG, full_names, REFEREE_2)

    def sub(cols):
        idx = [names.index(c) for c in cols if c in names]
        return X[:, idx]

    a1 = auc(sub(REFEREE_1), y)
    a2 = auc(sub(REFEREE_2), y)
    ac = auc(sub(REFEREE_1 + REFEREE_2), y)
    log(f"\n[{label}]  R1 max d = {r1d:.2f}   R2 max d = {r2d:.2f}")
    log(f"   AUC_R1 = {a1:.3f}   AUC_R2 = {a2:.3f}   AUC_combined = {ac:.3f}   "
        f"Delta(R2-R1) = {a2 - a1:+.3f}")
    if decompose:
        log("   per-feature separation (Cohen's d), sorted:")
        ds = []
        for i, nm in enumerate(full_names):
            a, b = XF[:, i], XG[:, i]
            a, b = a[np.isfinite(a)], b[np.isfinite(b)]
            if len(a) < 2 or len(b) < 2:
                continue
            p = np.sqrt((a.var() + b.var()) / 2) + 1e-12
            grp = "R1" if nm in REFEREE_1 else "R2"
            ds.append((abs(a.mean() - b.mean()) / p, nm, grp))
        for d, nm, grp in sorted(ds, reverse=True):
            log(f"      {grp}  {nm:16} d = {d:.2f}")
    return a1, a2, ac


def bootstrap_delta(fw, gc, m=4, tau=1, n_boot=2000, ci=95.0, label="TIGHT"):
    """
    Paired bootstrap CI for Delta = AUC_R2 - AUC_R1 on the FW-vs-GARCH comparison.

    Why this design (so a referee cannot pick it apart):
      * Resampling unit is the PAIR -- FW path i and the GARCH path fit to it --
        so we draw pair-indices with replacement. The two arms are coupled (the
        GARCH is fit to its FW path; the tight path is a reordering of that FW
        path's own values), so resampling pairs propagates that coupling instead
        of pretending the arms are independent samples.
      * To avoid bootstrap-plus-cross-validation leakage (a duplicated row landing
        in both the train and test fold inflates AUC), predicted scores are
        computed OUT-OF-FOLD once via cross_val_predict, then the AUC difference is
        bootstrapped over those fixed scores. No model is refit inside the loop.
      * It is a PAIRED bootstrap on the difference: R1 and R2 scores come from the
        same resampled paths each iteration, which is the correct construction for
        the question "is the difference zero".

    Interpretation note. The point estimate here is the out-of-fold AUC from a
    single 5-fold split, which can differ by a small amount from the main-line
    repeated-CV AUC; the inference is the interval, not the third decimal of the
    point. Holding the cross-validated scores fixed and resampling the evaluation
    set captures evaluation-sample variability (the dominant term) but not the
    extra variability of refitting on each resample; that makes the interval
    mildly conservative for a tie -- if it still brackets zero, the tie is safe.
    """
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.metrics import roc_auc_score

    n = min(len(fw), len(gc))
    fw, gc = fw[:n], gc[:n]
    XF, names = feats(fw, m, tau)
    XG, _ = feats(gc, m, tau)
    fcol = np.all(np.isfinite(np.vstack([XF, XG])), axis=0)
    names_k = [nm for nm, k in zip(names, fcol) if k]
    XF, XG = XF[:, fcol], XG[:, fcol]
    y = np.array([0] * n + [1] * n)

    def oof(cols):
        idx = [names_k.index(c) for c in cols if c in names_k]
        Xc = np.vstack([XF[:, idx], XG[:, idx]])
        pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
        return cross_val_predict(pipe, Xc, y, cv=cv, method="predict_proba")[:, 1]

    s1, s2 = oof(REFEREE_1), oof(REFEREE_2)
    sF1, sG1, sF2, sG2 = s1[:n], s1[n:], s2[:n], s2[n:]

    def auc_pair(idx, sF, sG):
        yy = np.array([0] * len(idx) + [1] * len(idx))
        return roc_auc_score(yy, np.concatenate([sF[idx], sG[idx]]))

    base = np.arange(n)
    a1_pt = auc_pair(base, sF1, sG1)
    a2_pt = auc_pair(base, sF2, sG2)

    rng = get_rng(MASTER_SEED, "hp_boot", label)
    deltas = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        deltas[b] = auc_pair(idx, sF2, sG2) - auc_pair(idx, sF1, sG1)

    lo = (100.0 - ci) / 2.0
    dlo, dhi = np.percentile(deltas, [lo, 100.0 - lo])
    frac_pos = float((deltas > 0).mean())

    log(f"\n[{label}] paired bootstrap on Delta = AUC_R2 - AUC_R1")
    log(f"   (out-of-fold scores, {n_boot} resamples of n={n} pairs)")
    log(f"   AUC_R1(oof) = {a1_pt:.3f}   AUC_R2(oof) = {a2_pt:.3f}   "
        f"Delta(oof) = {a2_pt - a1_pt:+.3f}")
    log(f"   Delta {ci:.0f}% CI = [{dlo:+.3f}, {dhi:+.3f}]   P(Delta>0) = {frac_pos:.3f}")
    if dlo <= 0.0 <= dhi:
        log("   => CI brackets 0: once the marginal is matched, FW and GARCH are")
        log("      statistically indistinguishable on the AUC difference (tie).")
    else:
        side = "R2 (nonlinear)" if dlo > 0 else "R1 (moments)"
        log(f"   => CI excludes 0: the difference favours {side}.")
    return a1_pt, a2_pt, (float(dlo), float(dhi)), frac_pos


def run(K=300, T=2500, m=4, tau=1, bootstrap=False, n_boot=2000, ci=95.0):
    log("=" * 64)
    log("HARDER PAIR: FW (behavioural) vs GJR-GARCH null -- LOOSE vs TIGHT match")
    log(f"K={K}, T={T}, embedding (m={m}, tau={tau}), seed={MASTER_SEED}")
    log("=" * 64)

    fw = fw_paths(K, T)
    log(f"\ngenerated {len(fw)} FW paths; fitting GARCH (loose) + rank-matched (tight)...")
    gc_loose = garch_match(fw, MASTER_SEED, rank_match=False)
    gc_tight = garch_match(fw, MASTER_SEED, rank_match=True)
    ok = [i for i in range(len(fw)) if gc_loose[i] is not None and gc_tight[i] is not None]
    fw = [fw[i] for i in ok]
    gc_loose = [gc_loose[i] for i in ok]
    gc_tight = [gc_tight[i] for i in ok]
    log(f"got {len(gc_loose)} loose, {len(gc_tight)} tight GARCH paths "
        f"({len(fw)} retained after pairing)")

    log("\nLOOSE = GARCH-t fit (marginal only approximately matched).")
    _evaluate(fw, gc_loose, m, tau, "LOOSE")
    log("\nTIGHT = LOOSE + rank-remap onto FW's exact marginal (moments matched).")
    _evaluate(fw, gc_tight, m, tau, "TIGHT", decompose=True)

    if bootstrap:
        bootstrap_delta(fw, gc_tight, m, tau, n_boot=n_boot, ci=ci, label="TIGHT")

    log("\nReading the TIGHT row -- the decisive test:")
    log("  * If AUC_R1 -> ~0.5 and AUC_R2 stays high: the behavioural mechanism")
    log("    leaves a nonlinear fingerprint GARCH lacks (positive result).")
    log("  * If both -> ~0.5: once moments are matched, FW and GARCH are")
    log("    indistinguishable even nonlinearly (clean negative).")
    log("  * R2 max d is itself informative: how separated are the nonlinear")
    log("    features after the marginal is forced to match.")

    os.makedirs("results", exist_ok=True)
    with open("results/harder_pair_tight.txt", "w") as f:
        f.write("\n".join(OUT))
    print("\n[saved to results/harder_pair_tight.txt]")


if __name__ == "__main__":
    boot = os.environ.get("HP_BOOTSTRAP", "0") not in ("0", "", "false", "False")
    nb = int(os.environ.get("HP_NBOOT", "2000"))
    run(bootstrap=boot, n_boot=nb)