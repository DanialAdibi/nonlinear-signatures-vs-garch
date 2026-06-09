"""
models/chaotic.py -- reference systems with KNOWN dynamics, used to validate the
nonlinear estimators (Referee 2) before they are applied to the FW model pair.

  henon   : discrete map, largest Lyapunov exponent ~ 0.419 per iteration,
            correlation dimension ~ 1.22. The classic Rosenstein test case.
  lorenz  : continuous flow, largest Lyapunov ~ 0.906 per unit time,
            correlation dimension ~ 2.05.
  iid_noise / ar1 : NEGATIVE controls -- no deterministic structure, so the
            largest-Lyapunov estimate should sit at/below the finite-sample floor.

Each returns a 1-D observable (scalar time series), which is what the estimators
consume after delay embedding.
"""

import numpy as np


def henon(n, a=1.4, b=0.3, burn_in=1000, noise=0.0, rng=None):
    """Henon map x-coordinate. Chaotic at (a, b) = (1.4, 0.3).

    The initial condition is drawn from `rng` (within the attractor basin) so
    that distinct seeds produce distinct, decorrelated trajectories after
    burn-in -- required for the positive control to have within-class variance.

    NOTE: the IC bounds below are calibrated for the default chaotic parameters
    (a, b) = (1.4, 0.3); for other (a, b) the basin of attraction differs and
    starts outside it diverge to infinity, so the bounds would need revisiting.
    """
    if rng is None:
        rng = np.random.default_rng()
    T = n + burn_in
    x = np.empty(T)
    y = np.empty(T)
    # random IC inside the basin of attraction (safely bounded for a=1.4,b=0.3)
    x[0], y[0] = rng.uniform(-0.2, 0.2), rng.uniform(-0.05, 0.05)
    for t in range(1, T):
        x[t] = 1.0 - a * x[t - 1] ** 2 + y[t - 1]
        y[t] = b * x[t - 1]
    obs = x[burn_in:]
    if noise > 0:
        obs = obs + rng.normal(0.0, noise * np.std(obs), size=obs.shape)
    return obs


def lorenz(n, dt=0.01, sample_every=10, sigma=10.0, rho=28.0, beta=8.0 / 3.0,
           burn_in=5000, noise=0.0, rng=None):
    """
    Lorenz x-coordinate, RK4-integrated and subsampled. The effective time step
    between returned samples is dt * sample_every (relevant for Lyapunov units).
    """
    if rng is None:
        rng = np.random.default_rng()

    def deriv(s):
        x, y, z = s
        return np.array([sigma * (y - x), x * (rho - z) - y, x * y - beta * z])

    total_steps = (n * sample_every) + burn_in

    s = np.array([1.0, 1.0, 1.0]) + rng.normal(0.0, 1.0, size=3)
    out = np.empty(n)
    k = 0
    for step in range(total_steps):
        k1 = deriv(s)
        k2 = deriv(s + 0.5 * dt * k1)
        k3 = deriv(s + 0.5 * dt * k2)
        k4 = deriv(s + dt * k3)
        s = s + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        if step >= burn_in and (step - burn_in) % sample_every == 0 and k < n:
            out[k] = s[0]
            k += 1
    if noise > 0:
        out = out + rng.normal(0.0, noise * np.std(out), size=out.shape)
    return out


def iid_noise(n, rng=None):
    """Negative control: i.i.d. Gaussian."""
    if rng is None:
        rng = np.random.default_rng()
    return rng.standard_normal(n)


def ar1(n, phi=0.5, burn_in=500, rng=None):
    """Negative control: linear AR(1) (dependent but not chaotic)."""
    if rng is None:
        rng = np.random.default_rng()
    T = n + burn_in
    x = np.empty(T)
    x[0] = 0.0
    e = rng.standard_normal(T)
    for t in range(1, T):
        x[t] = phi * x[t - 1] + e[t]
    return x[burn_in:]