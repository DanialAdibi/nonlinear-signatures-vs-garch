"""
check_ghw.py -- hand-run provenance diagnostic for the (shelved) GHW-conditioned
Brock-Hommes model.

Shows that the GHW-inspired conditioning fixes the explosion of the plain
two-type model and produces fat tails + volatility clustering, BUT that in the
clustering regime the returns are strongly autocorrelated (the deterministic
limit cycle), unlike the noise-dominated Franke-Westerhoff model. The verdict
below is DERIVED from the measured statistics (it is not asserted): the return
autocorrelation is computed and compared against the same near-white tolerance
used in check_bh98.py, and the model is rejected as a market generator when it
exceeds that tolerance.

This script is not part of run_all.sh. See README for the design implication
(why Franke-Westerhoff was chosen as the generator).

Run from the repo root:  python check_ghw.py
"""

import numpy as np
from models.bh_ghw import BHGHWParams, simulate_bh_ghw
from signatures.moments import stylized_facts
from seeds import get_rng, MASTER_SEED

RET_AC_TOL = 0.1   # near-white tolerance, matching check_bh98.py's no_ac test


def fmt(a):
    return "[" + ", ".join(f"{v:+.3f}" for v in a) + "]"


if __name__ == "__main__":
    N, BURN = 8000, 2000
    print("Brock-Hommes + GHW conditioning -- stylized-fact diagnostic")
    print(f"n={N}, burn_in={BURN}, master_seed={MASTER_SEED}")
    print("(parameters are a working regime, not a published calibration)")

    params = BHGHWParams()
    rng = get_rng(MASTER_SEED, "bh_ghw", "default")
    out = simulate_bh_ghw(params, n=N, burn_in=BURN, rng=rng)

    if out["blew_up"] or not out["finite"]:
        print("\n  EXPLODED / non-finite -- unexpected for the conditioned model.")
    else:
        sf = stylized_facts(out["returns"])
        ret_ac_max = float(np.nanmax(np.abs(sf["acf_returns"])))
        vol_cluster = float(np.nanmean(sf["acf_abs_returns"]))
        fat = sf["excess_kurtosis"] > 0.5

        print(f"\n  excess kurtosis : {sf['excess_kurtosis']:+.3f}   (>0 = fat tails)")
        print(f"  Hill tail alpha : {sf['hill_alpha']:.3f}")
        print(f"  ACF returns      lags {sf['acf_lags']}: {fmt(sf['acf_returns'])}")
        print(f"  ACF |returns|    lags {sf['vol_lags']}: {fmt(sf['acf_abs_returns'])}")

        autocorrelated = ret_ac_max >= RET_AC_TOL
        print("\n  derived verdict:")
        print(f"    bounded (no explosion)             : yes")
        print(f"    fat tails (exkurt > 0.5)           : {'yes' if fat else 'no'}")
        print(f"    vol clustering (mean|acf| > 0.05)  : "
              f"{'yes' if vol_cluster > 0.05 else 'no'}  (= {vol_cluster:+.3f})")
        print(f"    max |ACF returns| over lags >= 1   : {ret_ac_max:.3f}")
        if autocorrelated:
            print(f"    -> REJECT as market generator: returns autocorrelated "
                  f"(max|acf| {ret_ac_max:.3f} >= {RET_AC_TOL}), violating near-white returns.")
        else:
            print(f"    -> returns near-white (max|acf| {ret_ac_max:.3f} < {RET_AC_TOL}).")
