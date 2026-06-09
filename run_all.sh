#!/usr/bin/env bash
# =====================================================================
# run_all.sh -- full from-scratch regeneration of the discrimination study.
#
# Runs the complete pipeline in dependency order: clears the cache, runs
# the validation gates, regenerates every experiment, then the new
# robustness checks (EGARCH null; the BDS bootstrap runs inside
# residual_tests). Each stage is timestamped and logged; the script stops
# at the first failure so a half-finished run is never mistaken for a
# complete one.
#
# USAGE
#   bash run_all.sh                 # defaults (leverage sweep at K=500)
#   LEV_SWEEP_K=800 bash run_all.sh # heavier leverage sweep
#   SKIP_GATES=1 bash run_all.sh    # skip Stage 1 gates (not recommended)
#
# REQUIREMENTS
#   - run from the repo root (the directory containing seeds.py)
#   - python3 with numpy/scipy/scikit-learn/arch installed
#   - keep the machine AWAKE: single-core jobs suspend on sleep, which is
#     what corrupted the earlier leverage sweep. Use caffeinate / disable
#     sleep for the duration.
#
# OUTPUT
#   - results/*.txt        : one result file per experiment
#   - results/run_all.log  : full combined log with timings
#   - results/cache/       : regenerated path cache (deleted at start)
# =====================================================================

set -euo pipefail   # stop on any error, undefined var, or failed pipe

# ---- config (override via environment) ----
LEV_SWEEP_K="${LEV_SWEEP_K:-500}"   # higher K for the leverage sweep (Phase 8 fix)
LEV_SWEEP_T="${LEV_SWEEP_T:-2500}"
SKIP_GATES="${SKIP_GATES:-0}"
PY="${PY:-python3}"

LOG="results/run_all.log"
mkdir -p results

# ---- helpers ----
ts()   { date "+%Y-%m-%d %H:%M:%S"; }
say()  { echo "[$(ts)] $*" | tee -a "$LOG"; }
stage(){ echo | tee -a "$LOG"; say "================ $* ================"; }

# run a module, tee its output, and time it; abort the whole script on failure
run_step() {
  local label="$1"; shift
  say ">>> START: $label"
  local t0=$SECONDS
  if "$@" >>"$LOG" 2>&1; then
    say "<<< DONE : $label  ($((SECONDS - t0))s)"
  else
    say "!!! FAILED: $label  -- see $LOG. Aborting."
    exit 1
  fi
}

# ---- sanity: are we in the repo root? ----
if [[ ! -f seeds.py ]]; then
  echo "ERROR: run this from the repo root (the directory with seeds.py)." >&2
  exit 1
fi

say "########## FULL REGENERATION START ##########"
say "leverage sweep K=$LEV_SWEEP_K T=$LEV_SWEEP_T ; python=$($PY --version 2>&1)"

# =====================================================================
stage "STAGE 0  cache reset (no stale paths)"
# Critical: delete the cache so every path is rebuilt under the current,
# parameter-verified code. This is the procedural guard the cache test
# cannot enforce on its own.
rm -rf results/cache
say "deleted results/cache/"

# =====================================================================
if [[ "$SKIP_GATES" != "1" ]]; then
stage "STAGE 1  validation gates (tools + generator)"
run_step "validate_estimators (estimator gate)" $PY -m controls.validate_estimators
run_step "check_fw (generator gate)"            $PY check_fw.py
run_step "check_fw_pair (confusability gate)"   $PY check_fw_pair.py
run_step "check_cont_facts (symmetric)"         $PY check_cont_facts.py
run_step "check_cont_facts (leverage 50)"       $PY check_cont_facts.py 50
else
say "SKIP_GATES=1 set -- skipping Stage 1 (not recommended)."
fi

# =====================================================================
stage "STAGE 2  experiments in dependency order"
# Order matters: harder_pair builds the cache + helpers the rest reuse.
run_step "discriminate (Finding 1: DCA vs TPA)"        $PY -m experiment.discriminate
run_step "robustness (T-sweep + positive control)"     $PY -m experiment.robustness
run_step "harder_pair (Finding 2: FW vs GARCH, K=300)" $PY -m experiment.harder_pair
# cache now exists -> confirm it is self-consistent before the cached runs
run_step "test_cache_integrity (post-build check)"     $PY -m controls.test_cache_integrity
run_step "embedding_diagnostics (m-sweep + nulls)"     $PY -m experiment.embedding_diagnostics
run_step "embedding_extend (m=7,8 via cache)"          $PY -m experiment.embedding_extend
run_step "garch_spec_check (GJR vs plain)"             $PY -m experiment.garch_spec_check
run_step "harder_pair_lev (leverage arm)"              $PY -m experiment.harder_pair_lev
run_step "leverage_sweep (higher K=$LEV_SWEEP_K)"      $PY -m experiment.leverage_sweep "$LEV_SWEEP_K" "$LEV_SWEEP_T"
run_step "residual_tests (symmetric arm + BDS boot)"   $PY -m experiment.residual_tests
run_step "residual_tests (leverage arm + BDS boot)"    $PY -m experiment.residual_tests 50

# =====================================================================
stage "STAGE 3  alternative-null robustness"
run_step "egarch_null (EGARCH leverage-arm check)"     $PY -m experiment.egarch_null

# =====================================================================
stage "COMPLETE"
say "all stages finished. Result files in results/:"
ls -1 results/*.txt 2>/dev/null | sed 's/^/    /' | tee -a "$LOG"
say "########## FULL REGENERATION END ##########"