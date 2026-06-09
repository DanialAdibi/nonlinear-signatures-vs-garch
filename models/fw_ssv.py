"""
models/fw_ssv.py -- Franke & Westerhoff structural stochastic volatility model.

Same demand-and-price engine, two interchangeable SWITCHING mechanisms (this is
the Option-2 model pair: shared core, different switching layer):

  DCA  -- Discrete Choice Approach. Fractions are an instantaneous logit of the
          attractiveness index (memoryless).
  TPA  -- Transition Probability Approach. Fractions evolve via per-period
          switching probabilities (inertia / memory).

Both use the same HPM attractiveness index (Herding, Predisposition,
Misalignment). The pair was DESIGNED as a "confusable on moments, different
underneath" pair, but with the faithful Franke-Westerhoff (2012) parameters that
design intent is not realized: DCA's memoryless switching produces strong
volatility clustering and fat tails, while TPA at the paper's nu = 0.05 switches
so sluggishly that the trader mix stays near constant, giving near-Gaussian
returns with little clustering. The two variants are therefore dynamically
distinct and SEPARATE easily on standard moments (a moments classifier reaches
AUC ~0.99; see the nu note below and experiment/discriminate.py); they converge
only as nu -> 1 (the TPA -> DCA limit). The study's matched-moment discrimination
test is consequently carried by a GARCH null (see experiment/), not by this
behavioural pair.

Reference: Franke, R. & Westerhoff, F. (2012). Structural stochastic volatility
in asset pricing dynamics: Estimation and model contest. JEDC 36(8), 1193-1211.

Demands (log price p_t, fundamental p_star = 0; group-specific noise):
    d_t^f = phi * (p_star - p_t)  + eps_t^f,   eps^f ~ N(0, sigma_f^2)
    d_t^c = chi * (p_t - p_{t-1}) + eps_t^c,   eps^c ~ N(0, sigma_c^2)
Price:  p_{t+1} = p_t + mu * (n_t^f d_t^f + n_t^c d_t^c)
Index:  a_t = alpha_0 + alpha_n (n_t^f - n_t^c) + alpha_p (p_t - p_star)^2

DCA fractions:  n_{t+1}^f = 1 / (1 + exp(-a_t))
TPA fractions:  n_{t+1}^f = n_t^f + n_t^c * pi_cf - n_t^f * pi_fc
                pi_cf = min(1, nu * exp(+a_t)),  pi_fc = min(1, nu * exp(-a_t))

PARAMETER NOTE: the structural defaults are the published MSM estimates for the
DCA-HPM model in Franke & Westerhoff (2012), Table 1 (p. 12), verified line by
line against the paper: the seven demand/index constants (phi, chi, sigma_f,
sigma_c, alpha_0, alpha_n, alpha_p) plus mu = 0.010, the paper's common
normalization. Note: the paper also reports an alternative joint-MCR-optimized
DCA-HPM parameter set in Table 7 (p. 27); we deliberately use the standard
Table 1 (quadratic-loss MSM) estimates, not Table 7. The paper's models are
symmetric by construction (p. 4 fn 6; p. 6), which is why the leverage term
below defaults to off.

For the TPA variant the flexibility parameter nu is also fixed by the paper at
nu = 0.050 (p. 12, Table 1 note); it is a scaling normalization (like beta = 1
for DCA), chosen so the unit cap on the transition probabilities rarely binds
(~1% of steps, only during large-misalignment excursions). The discrimination experiment (experiment/discriminate.py) sweeps
nu as a free parameter. NOTE on confusability: with the faithful TPA above, the
DCA and TPA pair separates easily on standard moments wherever the two are
dynamically distinct (a moments classifier reaches AUC ~0.99 across the swept
nu range); they become confusable only as nu -> 1, where TPA degenerates into
the DCA limit. So a "confusable but dynamically distinct" behavioural pair is
not available from these HPM parameters; the study's matched-moment comparison
is instead constructed explicitly against a GARCH null (see experiment/).
Output: additive returns r_t = p_t - p_{t-1}.
"""

from dataclasses import dataclass, asdict
import numpy as np


@dataclass
class FWParams:

    phi: float = 0.12       
    chi: float = 1.50      
    sigma_f: float = 0.758 
    sigma_c: float = 2.087

    alpha_0: float = -0.327  
    alpha_n: float = 1.79    
    alpha_p: float = 18.43  

    alpha_lev: float = 0.0 
    mu: float = 0.01      
    p_star: float = 0.0    
    switching: str = "dca"
    nu: float = 0.05

    def as_dict(self):
        return asdict(self)


def simulate_fw(params: FWParams, n: int, burn_in: int = 1000, rng=None):
    """Simulate FW with the chosen switching mechanism; return n-1 returns + flags."""
    if rng is None:
        rng = np.random.default_rng()

    T = n + burn_in
    p = np.empty(T)
    p[0] = params.p_star
    p[1] = params.p_star

    nf = 0.5
    ps = params.p_star
    phi, chi = params.phi, params.chi
    sf, sc = params.sigma_f, params.sigma_c
    a0, an, ap = params.alpha_0, params.alpha_n, params.alpha_p
    lev = params.alpha_lev
    mu, nu = params.mu, params.nu
    mode = params.switching.lower()
    if mode not in ("dca", "tpa"):
        raise ValueError("switching must be 'dca' or 'tpa'")

    blew_up = False
    EXPLODE = 1e8

    for t in range(1, T - 1):
        nc = 1.0 - nf
        df = phi * (ps - p[t]) + rng.normal(0.0, sf)
        dc = chi * (p[t] - p[t - 1]) + rng.normal(0.0, sc)
        p[t + 1] = p[t] + mu * (nf * df + nc * dc)

        if not np.isfinite(p[t + 1]) or abs(p[t + 1]) > EXPLODE:
            blew_up = True
            p[t + 1:] = np.nan
            break

        a = a0 + an * (nf - nc) + ap * (p[t] - ps) ** 2
        if lev != 0.0:
            a = a + lev * (p[t] - p[t - 1])
        if mode == "dca":
            nf = 1.0 / (1.0 + np.exp(-a))
        else:
            pi_cf = min(1.0, nu * np.exp(a))
            pi_fc = min(1.0, nu * np.exp(-a))
            nf = nf + nc * pi_cf - nf * pi_fc
            nf = min(1.0, max(0.0, nf))

    p = p[burn_in:]
    returns = np.diff(p)
    return {
        "returns": returns,
        "price": p,
        "blew_up": bool(blew_up),
        "finite": bool(np.all(np.isfinite(returns))),
    }