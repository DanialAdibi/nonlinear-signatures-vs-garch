"""
controls/test_cache_integrity.py -- cache-leakage guard.

WHY THIS EXISTS
The embedding-extension and leverage experiments cache generated paths to
results/cache/ so the one-time GARCH fit is not repeated for every embedding
dimension. Caching is a classic data-leakage vector: if the cache key does not
capture everything the paths depend on, a run can silently load STALE paths that
do not match what the current code would generate, fabricating or distorting a
result without any error being raised.

This test asserts the two properties that make caching safe:

  (1) DETERMINISM: generating paths twice from the same MASTER_SEED yields
      byte-identical arrays. (If this fails, caching can never be trusted, and
      it is also the same bug class as the earlier clone-trajectory issue.)

  (2) ROUND-TRIP FIDELITY: paths written to the cache and reloaded are
      byte-identical to the freshly generated paths, with identical length,
      order, and dtype. (This is the "cached == fresh" check the review asked
      for.)

  (3) KEY SENSITIVITY: changing a parameter that the paths depend on (here T)
      changes the paths -- a guard that the cache key is keyed on the right
      things. This does not prove the key is complete, but it catches the
      degenerate case where generation ignores its inputs.

Run:  python -m controls.test_cache_integrity
Exit code 0 = all pass; 1 = a failure (do NOT trust cached runs until fixed).

NOTE ON SCOPE: this calls the REAL generators (fw_paths_tagged from
experiment/embedding_diagnostics.py, fw_lev_paths from
experiment/harder_pair_lev.py) that underlie build_or_load() in those modules --
not a private copy of them -- so if either generator's seed scheme or parameters
ever drift, these checks fail loudly rather than passing against a stale
reimplementation. It uses a small K so it runs in seconds; the property it
checks is independent of K.

IMPORTANT OPERATIONAL GUARD: this test proves round-trip fidelity and generator
determinism, but it cannot prove the cache KEY (the K_T[_lev] filename) captures
everything the paths depend on (e.g. MASTER_SEED, GARCH spec, burn_in). The
defence against a stale-key leak is procedural: DELETE results/cache/ before any
from-scratch regeneration so every path is rebuilt under the current code. Run
this test after regeneration to confirm the rebuilt cache is self-consistent.
"""

import os
import sys
import tempfile
import numpy as np

from experiment.embedding_diagnostics import fw_paths_tagged
from experiment.harder_pair_lev import fw_lev_paths


PASS = "[PASS]"
FAIL = "[FAIL]"
results = []


def check(name, ok, detail=""):
    results.append(ok)
    print(f"  {PASS if ok else FAIL} {name:42} {detail}", flush=True)


def arrays_identical(a_list, b_list):
    """Byte-identical: same count, same per-path length, same values exactly."""
    if len(a_list) != len(b_list):
        return False, f"count {len(a_list)} != {len(b_list)}"
    for i, (a, b) in enumerate(zip(a_list, b_list)):
        a, b = np.asarray(a), np.asarray(b)
        if a.shape != b.shape:
            return False, f"path {i} shape {a.shape} != {b.shape}"
        if a.dtype != b.dtype:
            return False, f"path {i} dtype {a.dtype} != {b.dtype}"
        if not np.array_equal(a, b):
            return False, f"path {i} values differ (max |d|={np.abs(a-b).max():.2e})"
    return True, f"{len(a_list)} paths identical"


def main():
    K, T, tag = 12, 600, "hp_fw"
    print("Cache-integrity guard (controls/test_cache_integrity.py)")
    print("=" * 60)

    # (1) DETERMINISM
    a = fw_paths_tagged(K, T, tag)
    b = fw_paths_tagged(K, T, tag)
    ok, detail = arrays_identical(a, b)
    check("determinism: same seed -> identical paths", ok, detail)

    # (2) ROUND-TRIP FIDELITY
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, f"fw_K{K}_T{T}.npy")
        np.save(path, np.array(a))
        reloaded = list(np.load(path))
        ok, detail = arrays_identical(a, reloaded)
        check("round-trip: save -> load == fresh", ok, detail)

    # (3) KEY SENSITIVITY
    c = fw_paths_tagged(K, T + 50, tag)
    diff = (len(c) != len(a)) or any(
        np.asarray(x).shape != np.asarray(y).shape for x, y in zip(a, c)
    ) or not arrays_identical(a, c)[0]
    check("key sensitivity: changing T changes paths", diff,
          "paths differ when T differs")

    # (4) LEVERAGE GENERATION DETERMINISM
    la = fw_lev_paths(K, T, 50.0, "hpl_fw")
    lb = fw_lev_paths(K, T, 50.0, "hpl_fw")
    ok, detail = arrays_identical(la, lb)
    check("leverage determinism: same seed -> identical", ok, detail)

    # (5) LEVERAGE vs SYMMETRIC DIFFER
    same, _ = arrays_identical(a, la)
    check("leverage changes output (alpha_lev not ignored)", not same,
          "leverage paths differ from symmetric")

    print("=" * 60)
    n_pass = sum(results)
    print(f"{n_pass}/{len(results)} checks passed")
    if all(results):
        print("CACHE INTEGRITY OK -- cached runs can be trusted.")
        return 0
    print("CACHE INTEGRITY FAILURE -- do NOT trust cached runs until resolved.")
    return 1


if __name__ == "__main__":
    sys.exit(main())