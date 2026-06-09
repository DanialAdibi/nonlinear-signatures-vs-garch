"""
models/bh_ghw.py -- Brock & Hommes adaptive belief system with a
Gaunersdorfer-Hommes-Wagener (GHW) inspired conditioning, intended to bound the
dynamics and generate endogenous volatility clustering.

STATUS: SHELVED, retained for provenance only. It is NOT used to generate any
result in the study; the study's data-generating model is Franke-Westerhoff
(see fw_ssv.py). The conditioning does succeed in bounding the dynamics (unlike
the explosive plain two-type model in bh98.py), but the resulting returns are
strongly autocorrelated, which violates the near-white-returns stylized fact.
That autocorrelation is what disqualified the model as a market generator. It is
measured by the hand-run diagnostic check_ghw.py, which prints a verdict derived
from the measured return autocorrelation (rejecting the model when it exceeds the
near-white tolerance); see that file for the value rather than relying on a
number quoted here.

References:
  Brock, W. A. & Hommes, C. H. (1998). Heterogeneous beliefs and routes to
    chaos in a simple asset pricing model. JEDC 22(8-9), 1235-1274.
  Gaunersdorfer, A., Hommes, C. H. & Wagener, F. O. O. (2008). Bifurcation
    routes to volatility clustering under evolutionary learning. JEBO 67(1),
    27-47. (Also Gaunersdorfer & Hommes, "A nonlinear structural model for
    volatility clustering.")

THE MECHANISM (GHW-INSPIRED, NOT THE PUBLISHED GHW FORM)
--------------------------------------------------------
State x_t = deviation of price from the constant fundamental (= 0).
Type 1 = fundamentalist, forecasts return to fundamental: f_1 = 0.
Type 2 = technical / trend trader, forecast f_2 = g * x_{t-1} + b, with g > R
locally destabilising.

This file applies the distance-to-fundamental conditioning to the chartist
FORECAST: technical traders attenuate their rule as price strays from
fundamental (they expect eventual reversion). Their EFFECTIVE forecast is

    f2_eff_t = exp( -(x_{t-1})^2 / beta_cond ) * (g * x_{t-1} + b)

so far from fundamental (|x| large) f2_eff -> 0 and the price is pulled back;
near fundamental the trend rule is expansive.

This is GHW-INSPIRED but is not the published GHW mechanism: the published
Gaunersdorfer-Hommes form conditions the chartist FRACTION (the population
weight on the strategy), not the point forecast. The two are different models.
The published form is constructed to produce near-white returns TOGETHER WITH
volatility clustering; this forecast-conditioning variant does NOT reproduce
that result (its returns are strongly autocorrelated, per check_ghw.py). The
file is kept only to document the line of investigation that led to selecting
Franke-Westerhoff as the generator.

Pricing (R = 1 + r), with fundamentalists forecasting 0:

    R * x_t = n_{2,t} * f2_eff_t + eps_t,     eps_t ~ N(0, sigma_eps^2)

so x_t = (n_{2,t} f2_eff_t + eps_t) / R; the pricing shock eps_t therefore
enters scaled by 1/R, consistent with the equation above.

Fractions via multinomial logit on realised-profit fitness (intensity beta),
positions and profits computed from the SAME conditioned forecast.

PARAMETER NOTE: the defaults below are a working regime found by scanning, not a
published calibration. Since the model is shelved, no quantitative claim rests
on these values.

Output: additive return series r_t = x_t - x_{t-1}.
"""

from dataclasses import dataclass, asdict
import numpy as np


@dataclass
class BHGHWParams:
    R: float = 1.01          # gross risk-free rate
    a: float = 1.0           # risk aversion
    sigma2: float = 1.0      # conditional variance of excess returns
    g: float = 1.9           # trend extrapolation (> R: locally destabilising)
    b: float = 0.0           # trend bias
    C1: float = 1.0          # fundamentalist information cost
    beta: float = 2.0        # intensity of choice
    w_mem: float = 0.0       # fitness memory in [0, 1)
    beta_cond: float = 20.0  # GHW conditioning width (smaller = chartists withdraw sooner)
    sigma_eps: float = 0.02  # pricing-noise standard deviation

    def as_dict(self):
        return asdict(self)


def simulate_bh_ghw(params: BHGHWParams, n: int, burn_in: int = 1000, rng=None):
    """Simulate the GHW-conditioned BH model; return n-1 additive returns + flags."""
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
    g, b = params.g, params.b
    C1, beta, w_mem = params.C1, params.beta, params.w_mem
    bcond = params.beta_cond

    blew_up = False
    EXPLODE = 1e8

    def f2_eff(x_lag):
        return np.exp(-(x_lag * x_lag) / bcond) * (g * x_lag + b)

    for t in range(2, T):
        bu1, bu2 = beta * U1, beta * U2
        m = bu1 if bu1 > bu2 else bu2
        e1 = np.exp(bu1 - m)
        e2 = np.exp(bu2 - m)
        n2 = e2 / (e1 + e2)

        fe = f2_eff(x[t - 1])
        eps = rng.normal(0.0, params.sigma_eps)
        x[t] = (n2 * fe + eps) / R

        if not np.isfinite(x[t]) or abs(x[t]) > EXPLODE:
            blew_up = True
            x[t:] = np.nan
            break

        excess = x[t] - R * x[t - 1]
        fe_prev = f2_eff(x[t - 2])
        z2 = (fe_prev - R * x[t - 1]) * inv_var 
        z1 = (0.0 - R * x[t - 1]) * inv_var
        pi2 = excess * z2
        pi1 = excess * z1 - C1
        U1 = pi1 + w_mem * U1
        U2 = pi2 + w_mem * U2

    x = x[burn_in:]
    returns = np.diff(x)
    return {
        "returns": returns,
        "deviation": x,
        "blew_up": bool(blew_up),
        "finite": bool(np.all(np.isfinite(returns))),
    }
