"""
models/bh98.py -- Brock & Hommes (1998) two-type adaptive belief system.

Reference: Brock, W. A. & Hommes, C. H. (1998). Heterogeneous beliefs and
routes to chaos in a simple asset pricing model. J. Econ. Dyn. Control,
22(8-9), 1235-1274.

State variable x_t = deviation of price from the (constant) fundamental.
Two belief types use linear predictors of next-period deviation:

    f_{h,t} = g_h * x_{t-1} + b_h            (forecast of x_{t+1}, formed at t)

Equilibrium price (gross risk-free rate R = 1 + r):

    R * x_t = n_{1,t} * f_{1,t} + n_{2,t} * f_{2,t} + eps_t

so x_t = (n_{1,t} f_{1,t} + n_{2,t} f_{2,t} + eps_t) / R. The pricing shock
eps_t therefore enters scaled by 1/R, consistent with the equation above.

Fractions follow a multinomial logit on past fitness (intensity of choice beta):

    n_{h,t} = exp(beta * U_{h,t-1}) / sum_k exp(beta * U_{k,t-1})

Fitness = realized profit (positions taken one step earlier), with memory w:

    excess_t  = x_t - R * x_{t-1}                       (realized excess return)
    z_{h,t-1} = (g_h * x_{t-2} + b_h - R * x_{t-1}) / (a * sigma2)
    pi_{h,t}  = excess_t * z_{h,t-1} - C_h
    U_{h,t}   = pi_{h,t} + w * U_{h,t-1}

Cost convention: the information cost C_h is charged inside the recursion (it is
part of pi_{h,t}), so past costs are geometrically discounted by the memory w.
In steady state a constant cost contributes approximately -C_h / (1 - w) to
fitness rather than a flat -C_h per period. Set w = 0 (the default) for the
undiscounted, single-period-cost case.

Type 1 is the (costly) fundamentalist (g1 = b1 = 0, forecasts return to
fundamental); type 2 is the trend follower (g2 > R is locally destabilising,
bounded globally by evolutionary switching when the fundamentalist is cheap
enough to attract followers).

PROVENANCE / STATUS: this model is SHELVED and is retained for provenance only;
it is not used to generate any result in the study. The defaults below are an
illustrative *explosive* regime (costly fundamentalist C1 = 1 with a
destabilising trend g2 = 1.2 > R), which is why the model was set aside as a
market generator: under these defaults it blows up. The dynamics are bounded in
other regimes (for example g2 < R, or a cheap fundamentalist C1 = 0), where the
switching mechanism pulls price back; those bounded regimes are mean-reverting
(negative return autocorrelation) rather than near-white, which is a further
reason the model is unsuitable here. No calibration is performed for this file.

The output is the additive return series r_t = x_t - x_{t-1}.
"""

from dataclasses import dataclass, asdict
import numpy as np


@dataclass
class BHParams:
    R: float = 1.01          # gross risk-free rate (1 + r)
    a: float = 1.0           # risk aversion
    sigma2: float = 1.0      # conditional variance of excess returns
    # Type 1 -- fundamentalist
    g1: float = 0.0
    b1: float = 0.0
    C1: float = 1.0          # information cost
    # Type 2 -- trend follower
    g2: float = 1.2
    b2: float = 0.0
    C2: float = 0.0
    beta: float = 10.0       # intensity of choice
    w: float = 0.0           # fitness memory in [0, 1)
    sigma_eps: float = 0.05  # pricing-noise standard deviation

    def as_dict(self):
        return asdict(self)


def simulate_bh98(params: BHParams, n: int, burn_in: int = 1000, rng=None):
    """
    Simulate the two-type BH98 model and return n-1 additive returns.

    Returns a dict with the return series and a stability flag so the caller
    can detect explosive / degenerate parameter regimes honestly rather than
    silently propagating NaNs.
    """
    if rng is None:
        rng = np.random.default_rng()

    T = n + burn_in
    x = np.empty(T)
    x[0] = rng.normal(0.0, params.sigma_eps)
    x[1] = rng.normal(0.0, params.sigma_eps)

    U1 = 0.0
    U2 = 0.0
    R = params.R
    inv_var = 1.0 / (params.a * params.sigma2)
    g1, g2, b1, b2 = params.g1, params.g2, params.b1, params.b2
    C1, C2, beta, w = params.C1, params.C2, params.beta, params.w

    blew_up = False
    EXPLODE = 1e8

    for t in range(2, T):
        # fractions from fitness known before x_t is realised (stable softmax)
        bu1, bu2 = beta * U1, beta * U2
        m = bu1 if bu1 > bu2 else bu2
        e1 = np.exp(bu1 - m)
        e2 = np.exp(bu2 - m)
        Z = e1 + e2
        n1 = e1 / Z
        n2 = e2 / Z

        # forecasts of x_{t+1} formed at t (use x_{t-1})
        f1 = g1 * x[t - 1] + b1
        f2 = g2 * x[t - 1] + b2
        eps = rng.normal(0.0, params.sigma_eps)
        # pricing equation: R*x_t = n1*f1 + n2*f2 + eps  (noise enters via 1/R)
        x[t] = (n1 * f1 + n2 * f2 + eps) / R

        if not np.isfinite(x[t]) or abs(x[t]) > EXPLODE:
            blew_up = True
            x[t:] = np.nan
            break

        # realised profits: positions at t-1 used forecasts g_h*x_{t-2}+b_h
        excess = x[t] - R * x[t - 1]
        z1 = (g1 * x[t - 2] + b1 - R * x[t - 1]) * inv_var
        z2 = (g2 * x[t - 2] + b2 - R * x[t - 1]) * inv_var
        pi1 = excess * z1 - C1
        pi2 = excess * z2 - C2
        U1 = pi1 + w * U1
        U2 = pi2 + w * U2

    x = x[burn_in:]
    returns = np.diff(x)
    return {
        "returns": returns,
        "deviation": x,
        "blew_up": bool(blew_up),
        "finite": bool(np.all(np.isfinite(returns))),
    }
