#!/usr/bin/env bash
# =====================================================================
# run_phase0to6.sh -- fresh run of the reviewed phases (0 through 6).
#
# Extends run_phase0to4.sh with the now-reviewed Phase 5 (Finding 2 hub)
# and Phase 6 (embedding sweep / diagnostics / extension):
#
#   Phase 0  seeds.py                       (imported; no run step)
#   Phase 1  FW generator gate              check_fw.py
#   Phase 2  DCA/TPA separability diag.     check_fw_pair.py
#   Phase 3  estimator validation gate      controls/validate_estimators.py
#   Phase 4  Finding 1 + robustness         discriminate.py, robustness.py
#   Phase 5  Finding 2 (FW vs GARCH) + CI   experiment/harder_pair.py  (HP_BOOTSTRAP=1)
#   Phase 6  embedding sweep/diag/extend    embedding_sweep, embedding_diagnostics, embedding_extend
#
# It does NOT run Phase 7-10 (leverage arm/sweep, residual probes, the
# cache-integrity test, the BDS bootstrap, the EGARCH null, check_cont_facts):
# those files are not reviewed yet. Use run_all.sh once they are.
#
# IMPORTANT -- this is a LONG run (order of several hours on one core):
#   harder_pair K=300 + bootstrap, embedding_sweep K=120, embedding_diagnostics
#   K=300 (the heavy one), and embedding_extend K=300 (rebuilds its path cache).
#   Each of harder_pair / diagnostics / extend refits the GARCH null on 300
#   paths, so GARCH is fitted a few times over. Run it where it can finish
#   uninterrupted (overnight is sensible).
#
# arch IS required here (the GARCH null is fitted). The path cache at
# results/cache/ is CLEARED at the start so embedding_extend rebuilds under the
# current code: that cache is keyed only on (K,T), so a stale cache from an
# earlier code/seed state would otherwise be silently reused.
#
# USAGE
#   PY=python bash run_phase0to6.sh              # Windows (python, not python3)
#   bash run_phase0to6.sh                        # if `python3` is on PATH
#   SKIP_GATES=1 bash run_phase0to6.sh           # skip Stage 1 gates (not recommended)
#   HP_NBOOT=4000 bash run_phase0to6.sh          # more bootstrap resamples for Finding 2
#
# REQUIREMENTS
#   - run from the repo root (the directory containing seeds.py)
#   - the pinned stack (see requirements_frozen.txt): numpy, scipy,
#     scikit-learn, arch, pandas, statsmodels
#   - keep the machine awake for the duration
#
# OUTPUT (in results/)
#   robustness.txt, harder_pair_tight.txt, embedding_sweep.txt,
#   embedding_diagnostics.txt, embedding_extend.txt, run_phase0to6.log
# =====================================================================

set -euo pipefail   # stop on any error, undefined var, or failed pipe

# ---- config (override via environment) ----
SKIP_GATES="${SKIP_GATES:-0}"
PY="${PY:-python3}"

LOG="results/run_phase0to6.log"
mkdir -p results
: > "$LOG"          # reset the log so each run starts clean

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

say "########## PHASE 0-6 RUN START (reviewed phases only) ##########"
say "python=$("$PY" --version 2>&1)"

# capture exact package versions for this run
"$PY" -m pip freeze > results/requirements_frozen.txt 2>/dev/null || true
say "key packages: $(grep -iE '^(numpy|scipy|scikit-learn|arch|pandas|statsmodels)==' results/requirements_frozen.txt | tr '\n' ' ' || true)"

# ---- soft integrity check against the committed pin at the repo root ----
if [[ -f requirements_frozen.txt ]]; then
  for pkg in numpy scipy scikit-learn arch pandas statsmodels; do
    want="$(grep -iE "^${pkg}==" requirements_frozen.txt || true)"
    have="$(grep -iE "^${pkg}==" results/requirements_frozen.txt || true)"
    if [[ -n "$want" && -n "$have" && "$want" != "$have" ]]; then
      say "WARNING: $pkg differs from committed pin (pin=$want, env=$have)"
    fi
  done
fi

# ---- clear the embedding_extend path cache (keyed only on K,T) ----
if [[ -d results/cache ]]; then
  say "clearing results/cache/ so embedding_extend rebuilds paths under current code"
  rm -rf results/cache
fi

# =====================================================================
if [[ "$SKIP_GATES" != "1" ]]; then
stage "STAGE 1  validation gates (Phases 1-3)"
# check_fw_pair is a separability DIAGNOSTIC (documents Finding 1), not pass/fail.
run_step "validate_estimators (Phase 3 estimator gate)"     "$PY" -m controls.validate_estimators
run_step "check_fw (Phase 1 generator gate)"                "$PY" check_fw.py
run_step "check_fw_pair (Phase 2 DCA/TPA separability)"     "$PY" check_fw_pair.py
else
say "SKIP_GATES=1 set -- skipping Stage 1 (not recommended)."
fi

# =====================================================================
stage "STAGE 2  Phase-4 experiments (Finding 1 + robustness)"
run_step "discriminate (Finding 1: DCA vs TPA)"             "$PY" -m experiment.discriminate
run_step "robustness (T-sweep + positive control + richer)" "$PY" -m experiment.robustness

# =====================================================================
stage "STAGE 3  Phase-5 experiment (Finding 2: FW vs GARCH, with bootstrap CI)"
export HP_BOOTSTRAP=1                       # report the paired bootstrap CI on Delta
run_step "harder_pair (Finding 2 + paired bootstrap CI)"    "$PY" -m experiment.harder_pair
unset HP_BOOTSTRAP

# =====================================================================
stage "STAGE 4  Phase-6 experiments (embedding sweep / diagnostics / extension)"
run_step "embedding_sweep (m=3-6 at K=120, exploratory)"      "$PY" -m experiment.embedding_sweep
run_step "embedding_diagnostics (artifact controls, K=300)"   "$PY" -m experiment.embedding_diagnostics
run_step "embedding_extend (m=7,8 + faithfulness, K=300)"     "$PY" -m experiment.embedding_extend

# =====================================================================
stage "COMPLETE (Phases 0-6)"
say "reviewed-phase run finished. Result files in results/:"
find results -maxdepth 1 -name '*.txt' | sort | sed 's/^/    /' | tee -a "$LOG"
say "NOTE: Phases 7-10 (leverage arm/sweep, residual probes, cache-integrity,"
say "      BDS bootstrap, EGARCH null, check_cont_facts) were NOT run -- not yet"
say "      reviewed. Use run_all.sh after those files are reviewed."
say "########## PHASE 0-6 RUN END ##########"
