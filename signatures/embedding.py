"""
signatures/embedding.py -- Takens delay embedding and the data-driven rules for
its two parameters: delay tau (first minimum of average mutual information) and
dimension m (false nearest neighbours).

These are fixed a priori per the analysis plan and held constant across models;
the functions here are also used to justify those choices.
"""

import numpy as np
from scipy.spatial import cKDTree


def embed(x, m, tau):
    """Delay embedding: returns array of shape (N - (m-1)*tau, m)."""
    x = np.asarray(x, float)
    N = len(x) - (m - 1) * tau
    if N <= 0:
        raise ValueError("series too short for requested (m, tau)")
    return np.column_stack([x[i * tau: i * tau + N] for i in range(m)])


def average_mutual_information(x, max_lag=30, bins=16):
    """
    AMI(tau) for tau = 0..max_lag, via 2-D histogram binning. The first local
    minimum is the conventional choice of embedding delay.
    """
    x = np.asarray(x, float)
    ami = np.empty(max_lag + 1)
    for lag in range(max_lag + 1):
        a = x[: len(x) - lag] if lag > 0 else x
        b = x[lag:] if lag > 0 else x
        c_xy, _, _ = np.histogram2d(a, b, bins=bins)
        p_xy = c_xy / c_xy.sum()
        p_x = p_xy.sum(axis=1)
        p_y = p_xy.sum(axis=0)
        nz = p_xy > 0
        outer = p_x[:, None] * p_y[None, :]
        ami[lag] = np.sum(p_xy[nz] * np.log(p_xy[nz] / outer[nz]))
    return ami


def ami_first_min(x, max_lag=30, bins=16, floor_frac=0.10):
    """First *genuine* local minimum of AMI; fallback to 1.

    The plain "first strict local minimum" rule is fragile when AMI decays
    monotonically (e.g. fast-decorrelating maps): it picks up a tiny fluctuation
    in the noise floor of the tail and returns a spuriously large delay. We
    therefore accept a local minimum only if it sits meaningfully above the AMI
    noise floor -- specifically more than `floor_frac` of the way from the floor
    (min over lags >= 1) up to AMI(0). A genuine minimum (where AMI dips and then
    rises again, as for a flow) clears this; a tail artifact does not. If no such
    minimum exists, AMI is effectively monotone and the conventional choice is
    tau = 1.
    """
    ami = average_mutual_information(x, max_lag, bins)
    floor = ami[1:].min()
    span = ami[0] - floor
    for i in range(1, len(ami) - 1):
        if ami[i] < ami[i - 1] and ami[i] < ami[i + 1]:
            if span <= 0 or (ami[i] - floor) > floor_frac * span:
                return i
    return 1


def false_nearest_neighbours(x, tau, m_max=10, rtol=15.0, atol=2.0):
    """
    Fraction of false nearest neighbours for m = 1..m_max (Kennel et al.).
    The embedding dimension is the smallest m at which this fraction drops
    near zero.
    """
    x = np.asarray(x, float)
    sigma = np.std(x)
    fracs = []
    for m in range(1, m_max + 1):
        emb = embed(x, m, tau)
        emb_next = embed(x, m + 1, tau)
        n = min(len(emb), len(emb_next))
        emb = emb[:n]
        tree = cKDTree(emb)
        dist, idx = tree.query(emb, k=2)
        nn_dist = dist[:, 1]
        nn_idx = idx[:, 1]
        false_count = 0
        valid = 0
        for i in range(n):
            j = nn_idx[i]
            if nn_dist[i] == 0:
                continue
            d_extra = abs(emb_next[i, m] - emb_next[j, m])
            valid += 1
            if d_extra / nn_dist[i] > rtol or \
               np.sqrt(nn_dist[i] ** 2 + d_extra ** 2) / sigma > atol:
                false_count += 1
        fracs.append(false_count / max(valid, 1))
    return np.array(fracs)


def fnn_dimension(x, tau, m_max=10, threshold=0.05):
    """Smallest m with FNN fraction below threshold; fallback to m_max."""
    fracs = false_nearest_neighbours(x, tau, m_max)
    for m, f in enumerate(fracs, start=1):
        if f < threshold:
            return m
    return m_max