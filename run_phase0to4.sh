#!/usr/bin/env bash
# =====================================================================
# run_phase0to4.sh -- fresh run of ONLY the reviewed phases (0 through 4).
#
# This is a deliberately trimmed pipeline. It runs the parts of the study
# that have been code-reviewed and validated end to end:
#
#   Phase 0  seeds.py                      (imported; no run step)
#   Phase 1  FW generator gate             check_fw.py
#   Phase 2  DCA/TPA separability diag.    check_fw_pair.py
#   Phase 3  estimator validation gate     controls/validate_estimators.py
#   Phase 4  Finding 1 + robustness        discriminate.py, robustness.py
#
# It intentionally does NOT run the Phase 5-10 work that run_all.sh runs
# (harder_pair, the embedding sweep, the leverage arm/sweep, the residual
# probes, the cache-integrity test, the BDS bootstrap, the EGARCH null),
# because those files have not been reviewed yet. Use run_all.sh once they
# have. check_cont_facts.py is also omitted here: it is unreviewed and its
# leverage variant belongs to Phase 7.
#
# Expect the numbers to differ from the committed (pre-regeneration) ones in
# two reviewed ways, both correct: Finding 1 is now the faithful separable-pair
# result (AUC_R1 ~ 0.99, Referee 2 not exceeding it), and robustness Parts A/C
# (which use TPA) likewise show the separable pair rather than the old shape.
#
# USAGE
#   bash run_phase0to4.sh                 # defaults
#   SKIP_GATES=1 bash run_phase0to4.sh    # skip Stage 1 gates (not recommended)
#
# REQUIREMENTS
#   - run from the repo root (the directory containing seeds.py)
#   - python3 with numpy/scipy/scikit-learn installed (arch NOT needed here:
#     no GARCH is fitted in Phases 0-4)
#   - keep the machine AWAKE for the duration (single-core, ~30-45 min)
#
# OUTPUT
#   - results/robustness.txt        : robustness battery results
#   - results/run_phase0to4.log     : full combined log with timings
# =====================================================================

set -euo pipefail   # stop on any error, undefined var, or failed pipe

# ---- config (override via environment) ----
SKIP_GATES="${SKIP_GATES:-0}"
PY="${PY:-python3}"

LOG="results/run_phase0to4.log"
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

say "########## PHASE 0-4 RUN START (reviewed phases only) ##########"
say "python=$("$PY" --version 2>&1)"
# capture exact package versions for reproducibility
"$PY" -m pip freeze > results/requirements_frozen.txt 2>/dev/null || true
say "key packages: $(grep -iE '^(numpy|scipy|scikit-learn|arch)==' results/requirements_frozen.txt | tr '\n' ' ' || true)"
say "full frozen requirements written to results/requirements_frozen.txt"

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
stage "STAGE 2  Phase-4 experiments"
run_step "discriminate (Finding 1: DCA vs TPA)"             "$PY" -m experiment.discriminate
run_step "robustness (T-sweep + positive control + richer)" "$PY" -m experiment.robustness

# =====================================================================
stage "COMPLETE (Phases 0-4)"
say "reviewed-phase run finished. Result files in results/:"
find results -maxdepth 1 -name '*.txt' | sort | sed 's/^/    /' | tee -a "$LOG"
say "NOTE: Phases 5-10 (harder_pair onward) were NOT run -- use run_all.sh after"
say "      those files are reviewed."
say "########## PHASE 0-4 RUN END ##########"
