"""
seeds.py -- deterministic, collision-resistant RNG derivation.

Every stochastic step in the project draws from an RNG derived from a single
master seed plus human-readable tags (model name, parameter value, replication
index). This guarantees that any result regenerates exactly from the master
seed alone, and that two different grid cells never share a stream by accident.

    rng = get_rng(MASTER_SEED, "fw", alpha_lev, rep)

Design choices that make it robust:
  * SHA-256 via hashlib (NOT Python's built-in hash(), which is salted per
    process and would break cross-run reproducibility).
  * Type-tagged, length-framed tag encoding (netstring style), so distinct tag
    tuples can never collide -- e.g. ("fw","50"), ("fw",50) and ("fw|50",) are
    all different streams.
  * Full 256-bit hash entropy into the Generator (no 32-bit truncation), so the
    "no accidental shared stream" promise holds even for tens of thousands of
    cells.
  * Float tags rounded to 12 significant figures, so a parameter written as a
    literal (0.05) and the same value reached by arithmetic (0.1+0.1-0.15)
    seed the SAME stream, while genuinely distinct values stay distinct.

CONTRACT: use a consistent type per parameter (don't mix int 50 and float 50.0
for the same knob -- they are deliberately different streams). Non-finite float
tags (nan/inf) are rejected.
"""

import hashlib
import math
import numpy as np

MASTER_SEED = 20240601
_FLOAT_SIG = 12  # significant figures retained when a float is used as a tag


def _round_sig(x, sig=_FLOAT_SIG):
    if x == 0.0:
        return 0.0
    return round(x, -int(math.floor(math.log10(abs(x)))) + (sig - 1))


def _canonical(tag):
    """Type-tagged, length-framed encoding -> unambiguous, collision-free."""
    if isinstance(tag, bool):
        body = "b:" + ("1" if tag else "0")
    elif isinstance(tag, (int, np.integer)):
        body = "i:" + str(int(tag))
    elif isinstance(tag, (float, np.floating)):
        if not math.isfinite(tag):
            raise ValueError("non-finite float tag: %r" % (tag,))
        body = "f:" + repr(_round_sig(float(tag)))
    elif isinstance(tag, str):
        body = "s:" + tag
    else:
        raise TypeError("unsupported tag type %s: %r" % (type(tag).__name__, tag))
    return "%d:%s" % (len(body), body)         # netstring framing


def _payload(master, *tags):
    return _canonical(master) + "".join(_canonical(t) for t in tags)


def _digest_int(master, *tags):
    return int(hashlib.sha256(_payload(master, *tags).encode()).hexdigest(), 16)


def derive_seed(master, *tags, bits=64):
    """Deterministic integer seed in [0, 2**bits). Stable across runs/platforms.

    NOTE: scikit-learn / arch `random_state` must be < 2**32. For those, pass
    bits=32, or better, draw the seed from get_rng(...).integers(2**32) so the
    library seed is itself part of the reproducible stream.
    """
    if not (1 <= bits <= 256):
        raise ValueError("bits must be in 1..256")
    return _digest_int(master, *tags) % (2 ** bits)


def get_rng(master, *tags):
    """NumPy Generator seeded deterministically from master + tags.

    Uses the full 256-bit hash entropy (no truncation), so distinct tag tuples
    never share a stream by accident.
    """
    return np.random.default_rng(np.random.SeedSequence(_digest_int(master, *tags)))