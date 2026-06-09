"""
controls/validate_estimators.py -- Phase-3 validation gate.

Runs the Lyapunov estimator on systems with KNOWN behaviour and asserts the
results land in the expected ranges. Nothing downstream is trustworthy unless
this passes.

Key validated principle: the Lyapunov slope alone does NOT separate chaos from
noise (both give a positive finite-sample value); the slope TOGETHER WITH the
scaling-region R^2 does -- chaotic systems show a clean linear scaling region
(high R^2), stochastic ones do not (low R^2).

Run from the repo root:  python -m controls.validate_estimators
"""

import sys
import numpy as np

from models.chaotic import henon, lorenz, iid_noise, ar1
from signatures.lyapunov import rosenstein
from signatures.dimension import correlation_dimension
from signatures.rqa import recurrence_quant
from signatures.bds import bds_statistic
from signatures.embedding import ami_first_min, fnn_dimension
from seeds import get_rng, MASTER_SEED


def _check(name, ok, detail):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name:22} {detail}")
    return ok


def run():
    print("Phase-3 estimator validation gate")
    print(f"master_seed={MASTER_SEED}\n")
    results = []

    # Henon: lambda1 ~ 0.419/iter, clean scaling
    h = henon(6000, rng=get_rng(MASTER_SEED, "val", "henon"))
    r = rosenstein(h, m=3, tau=1, theiler=5, n_steps=20, fit_start=0, fit_end=10)
    ok = (0.35 < r["lambda1"] < 0.50) and (r["r2"] > 0.95)
    results.append(_check("Henon lambda1", ok,
                          f"lambda1={r['lambda1']:+.3f} (exp ~0.42), R2={r['r2']:.3f} (exp >0.95)"))

    L = lorenz(5000, dt=0.01, sample_every=10, rng=get_rng(MASTER_SEED, "val", "lorenz"))
    tau = ami_first_min(L)
    m = fnn_dimension(L, tau, m_max=8)
    r = rosenstein(L, m=m, tau=tau, theiler=tau * m, n_steps=40, fit_start=0, fit_end=20)
    ok = (0.05 < r["lambda1"] < 0.25) and (r["r2"] > 0.90)
    results.append(_check("Lorenz lambda1", ok,
                          f"lambda1={r['lambda1']:+.3f} (per-sample, >0), R2={r['r2']:.3f} (exp >0.90)"))

    for name, x in [("iid_noise", iid_noise(5000, get_rng(MASTER_SEED, "val", "noise"))),
                    ("ar1", ar1(5000, 0.5, rng=get_rng(MASTER_SEED, "val", "ar1")))]:
        r = rosenstein(x, m=3, tau=1, theiler=5, n_steps=20, fit_start=0, fit_end=10)
        ok = r["r2"] < 0.7
        results.append(_check(f"{name} (no scaling)", ok,
                              f"lambda1={r['lambda1']:+.3f} (floor), R2={r['r2']:.3f} (exp <0.7)"))
    systems = {
        "henon":  (henon(4000, rng=get_rng(MASTER_SEED, "val2", "henon")), 3, 1),
        "lorenz": (lorenz(4000, dt=0.01, sample_every=10,
                          rng=get_rng(MASTER_SEED, "val2", "lorenz")), 3, 2),
        "noise":  (iid_noise(4000, get_rng(MASTER_SEED, "val2", "noise")), 3, 1),
        "ar1":    (ar1(4000, 0.5, rng=get_rng(MASTER_SEED, "val2", "ar1")), 3, 1),
    }

    print()
    d2 = {k: correlation_dimension(x, m=m, tau=tau, max_n=2000)["d2"]
          for k, (x, m, tau) in systems.items()}
    ok = (d2["henon"] < 1.6) and (d2["lorenz"] < 2.3) and \
         (d2["noise"] > 2.3) and (d2["noise"] > d2["lorenz"])
    results.append(_check("d2 (chaos<noise)", ok,
                          f"henon={d2['henon']:.2f} lorenz={d2['lorenz']:.2f} "
                          f"noise={d2['noise']:.2f} ar1={d2['ar1']:.2f}"))

    det = {k: recurrence_quant(x, m=m, tau=tau, rr=0.05, max_n=1500)["det"]
           for k, (x, m, tau) in systems.items()}
    ok = (det["henon"] > 0.5) and (det["lorenz"] > 0.5) and \
         (det["noise"] < 0.5) and (det["ar1"] < 0.5)
    results.append(_check("DET (chaos>noise)", ok,
                          f"henon={det['henon']:.2f} lorenz={det['lorenz']:.2f} "
                          f"noise={det['noise']:.2f} ar1={det['ar1']:.2f}"))

    t = np.arange(4000.0)
    sine = np.sin(2 * np.pi * t / 120.0)
    lam_sine = recurrence_quant(sine, m=3, tau=1, rr=0.05, max_n=1500)["lam"]
    lam_noise = recurrence_quant(iid_noise(4000, get_rng(MASTER_SEED, "val", "lamnoise")),
                                 m=3, tau=1, rr=0.05, max_n=1500)["lam"]
    ok = (lam_sine > 0.5) and (lam_noise < 0.2)
    results.append(_check("LAM (laminar>noise)", ok,
                          f"sine={lam_sine:.2f} (exp >0.5), noise={lam_noise:.2f} (exp <0.2)"))

    bds = {k: bds_statistic(x, m=2, max_n=2000)["bds"]
           for k, (x, m, tau) in systems.items()}
    ok = (abs(bds["henon"]) > 3) and (abs(bds["lorenz"]) > 3) and \
         (abs(bds["ar1"]) > 3) and (abs(bds["noise"]) < 3)
    results.append(_check("BDS (iid only for noise)", ok,
                          f"henon={bds['henon']:+.0f} lorenz={bds['lorenz']:+.0f} "
                          f"noise={bds['noise']:+.1f} ar1={bds['ar1']:+.0f}"))

    print()
    if all(results):
        print("GATE PASSES: estimators behave correctly on known systems.")
        print("Note: slope alone does not separate chaos from noise; slope + R^2 does.")
    else:
        print("GATE FAILS: do not trust downstream results until fixed.")
    return all(results)


if __name__ == "__main__":
    sys.exit(0 if run() else 1)