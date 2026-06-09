"""
experiment/discriminate.py -- the study's core experiment.

Generate DCA and TPA paths and ask which feature set lets a classifier recover
which model produced each path:

    Referee 1 (standard moments)     -> AUC_1
    Referee 2 (nonlinear signatures) -> AUC_2

Run across a grid of TPA switching speeds nu, because the result depends on it:
small nu = strong switching inertia (structurally distinct), large nu = TPA
collapses toward the memoryless DCA limit. For each nu we report the largest
standardised per-feature gap (Cohen's d) within each referee -- a stable
descriptor of available signal -- alongside cross-validated AUC.

NOTE (Finding 1 framing): with the faithful TPA switching this pair is NOT
confusable on standard moments. It separates easily wherever it is dynamically
distinct (AUC_R1 ~ 0.99 across the nu grid; confusable only as nu -> 1, where TPA
degenerates into the DCA limit and there is no real difference to detect). So
Finding 1 is a baseline negative: the standard moments already capture the
behavioural difference, and the nonlinear signatures add nothing on top. The
matched-moment test in which Referee 2 could in principle matter is the GARCH
null (Finding 2, experiment/harder_pair.py), not a tuned nu.

Includes the A-vs-A null control (two DCA batches must be inseparable, AUC~0.5).

Run from the repo root:  python -m experiment.discriminate
"""

import warnings
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score

from models.fw_ssv import FWParams, simulate_fw
from signatures.features import feature_matrix, REFEREE_1, REFEREE_2, ALL_FEATURES
from seeds import get_rng, MASTER_SEED

warnings.filterwarnings("ignore")


def generate_paths(params, tag, K, T, burn_in=1500):
    out = []
    for s in range(K):
        res = simulate_fw(params, n=T, burn_in=burn_in,
                          rng=get_rng(MASTER_SEED, "exp", tag, T, s))
        if res["finite"] and not res["blew_up"]:
            out.append(res["returns"])
    return out


def max_cohens_d(Xa, Xb, names, group):
    best = 0.0
    for i, nm in enumerate(names):
        if nm not in group:
            continue
        a, b = Xa[:, i], Xb[:, i]
        a, b = a[np.isfinite(a)], b[np.isfinite(b)]
        pooled = np.sqrt((a.var() + b.var()) / 2) + 1e-12
        best = max(best, abs(a.mean() - b.mean()) / pooled)
    return best


def auc(X, y, cols, names, n_splits=5, n_repeats=6):
    idx = [names.index(c) for c in cols if c in names]
    pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=0)
    return float(cross_val_score(pipe, X[:, idx], y, cv=cv, scoring="roc_auc").mean())


def run(T=2500, K=120, m=4, tau=1, nu_grid=(0.02, 0.05, 0.1, 0.3, 0.6)):
    # K=120 gives a clean null (AUC~0.5) and stable AUCs but is compute-bound on
    # a single core (~12 min full sweep). K=60 runs in ~4 min and shows the same
    # SHAPE (the Cohen's d gaps are stable), but its null AUC is noisy (~0.5-0.64).
    print("Phase-4 discrimination experiment: DCA vs TPA")
    print(f"T={T}, K={K}/model, embedding (m={m}, tau={tau}), seed={MASTER_SEED}\n")

    dca = generate_paths(FWParams(switching="dca"), "dca", K, T)
    Xd, names = feature_matrix(dca, m=m, tau=tau)
    print(f"generated {len(dca)} DCA paths; sweeping TPA nu...\n")

    print(f"{'nu':>6}{'maxR1_d':>9}{'maxR2_d':>9}{'AUC_R1':>9}{'AUC_R2':>9}{'AUC_all':>9}")
    for nu in nu_grid:
        tpa = generate_paths(FWParams(switching="tpa", nu=nu), f"tpa{nu}", K, T)
        Xt, _ = feature_matrix(tpa, m=m, tau=tau)
        X = np.vstack([Xd, Xt])
        y = np.array([0] * len(dca) + [1] * len(tpa))
        fc = np.all(np.isfinite(X), axis=0)
        Xf, nm = X[:, fc], [n for n, k in zip(names, fc) if k]
        r1d = max_cohens_d(Xd, Xt, names, REFEREE_1)
        r2d = max_cohens_d(Xd, Xt, names, REFEREE_2)
        a1 = auc(Xf, y, REFEREE_1, nm)
        a2 = auc(Xf, y, REFEREE_2, nm)
        ac = auc(Xf, y, ALL_FEATURES, nm)
        print(f"{nu:>6}{r1d:>9.2f}{r2d:>9.2f}{a1:>9.3f}{a2:>9.3f}{ac:>9.3f}")

    # --- A-vs-A null control (must differ only by seed, not by any setting) -
    dca_b = generate_paths(FWParams(switching="dca"), "dca_b", K, T)
    Xb, _ = feature_matrix(dca_b, m=m, tau=tau)
    Xn = np.vstack([Xd, Xb])
    yn = np.array([0] * len(dca) + [1] * len(dca_b))
    fc = np.all(np.isfinite(Xn), axis=0)
    an = auc(Xn[:, fc], yn, ALL_FEATURES, [n for n, k in zip(names, fc) if k])
    print(f"\nNULL CONTROL (DCA vs DCA): AUC_all = {an:.3f}  "
          f"({'PASS ~0.5' if abs(an - 0.5) < 0.12 else 'WARN'})")

    print("\nReading: AUC ~ 0.5 = inseparable. With the faithful TPA the DCA/TPA "
          "pair separates strongly on the standard moments wherever it differs "
          "(AUC_R1 ~ 0.99), so the nonlinear features have no gap to fill: this is "
          "a baseline negative on an easily-separated pair, not a confusable-pair "
          "result. The matched-moment test where Referee 2 could in principle "
          "matter is Finding 2 (FW vs the GARCH null, experiment/harder_pair.py).")


if __name__ == "__main__":
    run()