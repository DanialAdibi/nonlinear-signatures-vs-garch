# Project Lab Notebook — What Was Done, Step by Step

**Study:** Do nonlinear-dynamical signatures discriminate market models better
than standard return moments? A controlled simulation study with known ground
truth.

> **Status note (current):** Following a round of reviewer comments, the codebase
> has been hardened (parameter verification, a cache-integrity guard, a
> bootstrapped BDS test, and an EGARCH alternative null) and the entire study has
> now been regenerated from scratch in one controlled run under the pinned
> environment (Phase 15). The process and code described in Phases 0-9 are
> unchanged; the result numbers quoted in Phases 4-9 are now the FINAL regenerated
> values from that single verified pass. The regeneration also surfaced one
> genuine surprise that the pre-regeneration runs had not: the leverage-arm
> collapse is specific to the GJR null and does NOT hold against an EGARCH null
> (Phase 15), which prompted a new fit-quality diagnostic. That diagnostic has now
> run at K = 300 and did NOT confirm the clustering explanation: GJR and EGARCH
> reproduce FW's clustering and leverage about equally, yet only the GJR edge
> collapses. So the under-specified-null reading holds for the GJR axis but the
> EGARCH survival is an open puzzle pointing to a higher-order structural difference
> the nonlinear features detect. That is the one substantive item left open;
> everything else is settled.

**How to read this document.** This is the chronological record of what was done,
in the order it happened, why each step followed from the last, and which code
file and result file correspond to each step. It is meant to be read top to
bottom. Two companion documents support it: `codebase_overview.md` (a reference
map of every file) and `project_review.md` (the detailed findings write-up). This
notebook is the narrative that connects them.

A note on method: the path included dead ends and corrections (a shelved model, a
caught bug, a small-sample illusion). These are kept in, in order, because they
are part of how the result was reached and how it was checked.

---

## Phase 0 — Foundations and reproducibility

**What and why.** Before any experiment, the project was built on a single
reproducibility rule: every random draw derives from one master seed plus
descriptive tags, so any number in the study regenerates exactly.

- `seeds.py` — defines `MASTER_SEED` and `get_rng(seed, *tags)`. Every stochastic
  step in every later file draws from here.

---

## Phase 1 — Choosing the data-generating models

**What and why.** A discrimination study needs models that are genuinely
market-like and genuinely hard to tell apart. Several candidates were built and
tested before settling on the final pair.

- `bh98.py` — Brock–Hommes (1998) adaptive belief system. **Outcome: shelved.**
  Its dynamics are explosive, so it is not a usable market generator. Kept for
  provenance.
- `bh_ghw.py` — Brock–Hommes with Gaunersdorfer–Hommes–Wagener conditioning to
  bound the dynamics. **Outcome: shelved.** It produces autocorrelated returns,
  which violates a basic stylized fact (returns should be near-white), so it was
  set aside.
- `fw_ssv.py` — Franke–Westerhoff structural stochastic volatility model. **This
  became the generator used throughout.** It has one shared demand-and-price
  engine with two interchangeable switching mechanisms (DCA, memoryless; TPA, with
  inertia), so the two variants share everything except the switching layer — the
  "confusable on the surface, different underneath" structure the study needs.

  > *Correction (Phase 11): the "confusable on the surface" framing held only under
  > a bug in the TPA switching probabilities, found in code review and since fixed.
  > With the faithful TPA the DCA/TPA pair is NOT confusable on standard moments.
  > The shared-core design stands; the confusable-pair claim does not. See Phase 11.*
- `check_bh98.py`, `check_ghw.py` — the gates used to evaluate and ultimately
  shelve the BH/GHW models.
- `check_fw.py` — generator gate for FW: confirmed finiteness, stability, and
  basic stylized facts. **Outcome: FW passes.**

**Result of Phase 1:** the FW DCA/TPA pair is the model pair; BH and GHW are
shelved with documented reasons.

---

## Phase 2 — Confirming the pair is hard to tell apart

**What and why.** If the two FW variants were trivially different on standard
moments, the discrimination question would be meaningless. So the pair had to be
tuned to be *confusable* on moments.

- `check_fw_pair.py` — the confusability gate. It measures the largest standardized
  mean difference (Cohen's d) of the standard moments between DCA and TPA and tunes
  the TPA flexibility `nu` so they are closely matched. **Outcome: at the operating
  `nu`, max Cohen's d ≈ 0.44 — the two are genuinely hard to separate on moments.**

  > *Correction (Phase 11): the "max Cohen's d ≈ 0.44" result was an artifact of a
  > bug in the TPA transition probabilities. `nu` was applied outside the min(1, .)
  > cap, `nu * min(1, exp(a))`, instead of inside it, `min(1, nu * exp(a))`, as in
  > Franke–Westerhoff (2012, p. 9). With the formula corrected, the DCA/TPA pair
  > separates strongly on moments wherever it is dynamically distinct (max Cohen's
  > d ≈ 5; a moments classifier reaches AUC ≈ 0.99), and is confusable only as
  > `nu` approaches 1, where TPA degenerates into the DCA limit and there is no real
  > difference to detect. There is no `nu` that gives a confusable-but-distinct pair.
  > The confusable-pair construction is therefore dropped; the matched-moment test
  > (where the nonlinear features get a fair chance) is carried by the GARCH null in
  > Phase 5. `check_fw_pair.py` is now a separability diagnostic, not a confusability
  > gate. See Phase 11.*

---

## Phase 3 — Building and validating the measurement tools

**What and why.** The nonlinear estimators (Referee 2) had to be proven correct on
systems with known answers *before* being trusted on the FW data — otherwise a
null result could just mean broken tools.

Signature estimators built:
- `embedding.py` — Takens delay embedding; rules for delay `tau` (average mutual
  information) and dimension `m` (false nearest neighbours).
- `moments.py` — Referee 1, the standard moments.
- `lyapunov.py` — largest Lyapunov exponent (Rosenstein), returning slope **and**
  scaling-fit R².
- `dimension.py` — Grassberger–Procaccia correlation dimension `d2` (with R²).
- `rqa.py` — recurrence determinism (DET) and laminarity (LAM).
- `bds.py` — the BDS independence statistic.
- `entropy.py` — permutation/sample entropy (kept for a later robustness probe).
- `features.py` — assembles the two named feature sets (`REFEREE_1`, `REFEREE_2`)
  on one fixed embedding shared across models.
- `chaotic.py` — the known test systems: Hénon and Lorenz (chaotic, known
  invariants) and i.i.d. noise / AR(1) (negative controls).

Validation:
- `validate_estimators.py` — the gate. **Outcome: passes.** Hénon Lyapunov ≈ 0.42
  with R² > 0.95; Lorenz positive with R² > 0.90; noise/AR(1) show a positive
  Lyapunov *floor* but with low R² (no scaling region). This established the key
  principle used throughout: **slope alone does not separate chaos from noise;
  slope together with R² does** — which is why the R² values are carried as
  features.

---

## Phase 4 — Finding 1: the first discrimination experiment

**What and why.** With validated tools and a confusable pair, the core question
was asked directly: DCA vs TPA — do nonlinear signatures beat moments?

- `discriminate.py` — generates DCA and TPA paths, trains a classifier on each
  feature set, reports AUC. **Outcome: standard moments win at every setting; the
  gap widens with more data.** Clean negative for the hypothesis.
- `robustness.py` (+ `robustness.txt`) — robustness battery:
  - sample-size (T) sweep — confirms the moments' advantage grows with data;
  - IAAFT positive control — chaotic series vs phase-randomized surrogate; under
    observational noise, Referee 2 still separates them (AUC ≈ 1.0) while Referee 1
    fails (≈ 0.8). **This proves the tools do detect nonlinear structure when it
    exists** — so the negative result is real, not insensitivity;
  - richer-feature probe (adds entropy) — moves AUC ~0.003, i.e. nothing.

**Result of Phase 4:** for behavioural-vs-behavioural discrimination, nonlinear
signatures add nothing over standard moments, and the tools are demonstrably not
blind (positive control passes).

> *Correction (Phase 11): with the corrected TPA the premise above ("a confusable
> pair") no longer holds, so Finding 1 is reframed honestly. The faithful DCA/TPA
> pair separates easily on moments wherever it genuinely differs (a moments
> classifier reaches AUC ≈ 0.99 across the swept `nu`), so moments already capture
> the behavioural difference and the nonlinear signatures have no gap to fill. The
> negative for the hypothesis stands and is arguably cleaner, but it is now a
> baseline negative on an easily-separated pair, not a confusable-pair result. The
> load-bearing matched-moment test, where nonlinear features could in principle
> matter, is Finding 2 (FW vs the moment-matched GARCH null). See Phase 11.*

---

## Phase 5 — Finding 2: behavioural vs a statistical null

**What and why.** A harder, more meaningful test: can the tools tell FW apart from
a fitted GARCH imitation of it? This required matching the distribution so the
comparison isolates temporal structure.

- `harder_pair.py` (+ `harder_pair_tight_k_150.txt`, `harder_pair_tight_k_300.txt`)
  — fits a GJR-GARCH per FW path and simulates a matched path, at two levels:
  LOOSE (as fitted) and TIGHT (rank-matched to FW's exact marginal). This is the
  hub module; later scripts reuse its fitting and scoring helpers.
- **A correction made here, kept on the record:** at small sample size (K = 80–150)
  a small nonlinear edge appeared. Re-running at larger K showed it **halving with
  each doubling of data** (+0.097 → +0.046 → +0.002), the signature of *no effect*.
  **Outcome: at K = 300 the two feature sets tie (ΔAUC ≈ +0.002).** The early
  "edge" was a small-sample illusion. K = 300 became the authoritative sample size
  thereafter.

---

## Phase 6 — The embedding turn (a surprise, then a careful check)

**What and why.** The embedding dimension `m = 4` had been hand-chosen (the
standard FNN rule does not converge for stochastic data). To check the result did
not depend on that choice, `m` was swept — and the result *did* depend on it.

- `embedding_sweep.py` (+ `embedding_sweep.txt`) — first sweep (K = 120).
  **Outcome: ΔAUC rises with m.** This was unexpected and raised the question:
  real, or an artifact of the embedding?
- `embedding_diagnostics.py` (+ `embedding_diagnostics.txt`) — the decisive check
  at K = 300. It added two artifact controls: (a) an **A-vs-A null** (two batches
  of the *same* model) swept across m — stayed flat, so the embedding does not
  manufacture separation; (b) a **tau = 2 re-run** — reproduced the rise, so it is
  not a small-delay vector-overlap artifact. **Outcome: the rise is real.**
- `embedding_extend.py` (+ `embedding_extend.txt`) — extended to m = 7,8 using
  cached paths (so GARCH is not refitted), with an m = 6 faithfulness check that
  reproduced the prior number exactly. **Outcome: ΔAUC climbs to ≈ +0.10 by m = 8
  and settles into a ceiling; the A-vs-A null stays flat throughout.**

**Result of Phase 6:** at conventional embeddings the tools tie, but a *real*
nonlinear advantage emerges and grows with embedding dimension, driven by
recurrence laminarity. Confirmed not an artifact.

---

## Phase 7 — What the edge actually was: the leverage investigation

**What and why.** The natural next question: *what* is laminarity detecting that
GARCH misses? The investigation started from the GARCH null's asymmetry term.

- `garch_spec_check.py` (+ `garch_spec_check.txt`) — checks whether the GJR
  asymmetry parameter is doing any work. **Outcome: on symmetric FW the fitted
  gamma ≈ 0 (sign-mixed), so GJR collapses to plain GARCH; the GJR-vs-plain choice
  is immaterial.** The asymmetry machinery sits idle because symmetric FW has no
  leverage to fit. This motivated the next step.
- `fw_ssv.py` (leverage term added) — a signed-return term (`alpha_lev`) was added
  to the attractiveness index, introducing a leverage effect through the volatility
  channel. Defaults to 0, so the symmetric study is unchanged.
- `check_cont_facts.py` (+ implied validation) — validated the extended model
  against the Cont (2001) stylized facts. **Outcome: at `alpha_lev = 50` the model
  produces a realistic leverage effect (L(τ) ≈ −0.15) while keeping return-whiteness,
  heavy tails, and clustering intact; the marginal stays symmetric (leverage is not
  skewness).** A GJR fitted to it now returns gamma ≈ +0.06 (90% positive) — the
  asymmetry term engages.
- `harder_pair_lev.py` (+ `harder_pair_lev.txt`) — the leverage arm: leverage FW vs
  a now-leverage-aware GJR null, swept over m. **Outcome: the nonlinear advantage
  collapses to ≈ 0 at every m** (peak +0.022 vs +0.104 in the symmetric arm). The
  mechanism: laminarity's separation barely changes, but the standard moments rise
  to meet it — the structure laminarity detected was leverage-type asymmetry, which
  a correctly specified GARCH reproduces.

**Result of Phase 7:** the nonlinear edge was leverage-type volatility-regime
structure. Give the statistical null a working leverage term and the edge
disappears.

---

## Phase 8 — Trying to make the claim airtight: the dose-response

**What and why.** The leverage arm rested on two endpoints (symmetric vs strong
leverage). To show the edge is governed by *how mis-specified* the null is, the
leverage strength was swept across intermediate values.

- `leverage_sweep.py` (+ `leverage_sweep.txt`) — sweeps `alpha_lev` = 0,10,20,30,50,
  reporting fitted gamma and ΔAUC at fixed high m. **Outcome: mixed.** The fitted
  gamma rises cleanly and monotonically (the null engages its leverage term
  progressively), but ΔAUC is **non-monotone** at the intermediate points, and the
  A-vs-A null drifts from zero there — so those intermediate points are noisier and
  less reliable. **The clean two endpoints carry the claim; the smooth dose-response
  is not established.** Reported honestly rather than suppressed.

  > *Update (Phase 10): this sweep is being re-run at K = 500, uninterrupted, to
  > resolve whether the non-monotonicity is sample noise or a genuine feature.*

---

## Phase 9 — Connecting to the classic literature: residual probes

**What and why.** The older chaos-in-finance literature does not test raw returns;
it filters through GARCH and tests the residuals. Two probes connect the study to
that tradition.

- `residual_tests.py` (+ `residual_tests.txt`, `residual_tests_lev50.txt`) —
  - **Probe A (discrimination on residuals):** filter both arms through their own
    GJR, classify the residuals, sweep m. **Outcome: the nonlinear edge reverses to
    negative** (≈ −0.08 at m = 8 on both arms) — once the volatility layer is
    removed, the standard moments do *better* than the nonlinear features. This is
    the sharpest confirmation that the edge was entirely the volatility/leverage
    layer.
  - **Probe B (BDS adequacy):** BDS on raw FW returns (≈ 6.0) vs on GJR residuals
    (≈ 0), a ~95% drop. **Outcome: the raw-return "nonlinearity" is volatility
    clustering that GARCH removes.** This is the bridge to the real-data companion
    study: on a known-non-chaotic generator, GARCH filtering cleans the residuals,
    so a *surviving* BDS rejection on real residuals would be harder to dismiss as
    a filtering artifact. (BDS asymptotic standard errors are unreliable under heavy
    tails; a reported version should bootstrap the null.)

  > *Update (Phase 10): the bootstrap null is now implemented
  > (`signatures/bds_bootstrap.py`) and is the reported version for Probe B; the
  > asymptotic statistic above is kept only as a labelled reference.*

---

## Phase 10 — Reviewer-driven hardening and full regeneration

**What and why.** A round of reviewer comments raised three serious items and two
substantive recommendations. Rather than patch the existing results, the decision
was to fix everything at the code level and regenerate the whole study in one
controlled pass, so the final numbers come from a single verified pipeline. Each
item and the action taken:

- **FW parameter verification (was an open item).** Obtained Franke–Westerhoff
  (2012) and verified all seven structural constants line by line against their
  Table 1 (DCA–HPM variant). **Outcome: every value matches the published estimate
  exactly** (phi, chi, the two demand-noise s.d.s, the three switching-index
  coefficients), as do mu = 0.010 and nu = 0.050. The "working recollection"
  hedging was removed from `fw_ssv.py` and the manuscript; Table 1 is now cited
  explicitly, with a note that the standard MSM estimates (Table 1) are used rather
  than the joint-MCR-optimised set (Table 7). The paper's own statement that the
  models are symmetric by construction now supports the leverage-extension framing.

  > *Correction (Phase 11): the seven structural constants, mu = 0.010, and the
  > nu = 0.050 VALUE all match Table 1 as stated here, and that verification stands.
  > However, the TPA transition-probability FUNCTIONAL FORM, which was not part of
  > the seven-constant check, was found in review to be mis-coded against p. 9 of
  > the paper. See Phase 11.*

- **Cache-integrity guard (`controls/test_cache_integrity.py`, new).** Caching the
  GARCH-fitted paths is a data-leakage vector — the same bug class as the earlier
  clone-trajectory issue. A test was written asserting generator determinism (same
  seed → byte-identical paths), round-trip fidelity (saved-then-reloaded equals
  fresh, exact to dtype), and key sensitivity. **It was confirmed to have teeth**
  by deliberately feeding it a corrupted round-trip and stale paths and checking it
  flags both. The regeneration also deletes the cache at the start (so every path
  is rebuilt under verified code) and runs this test mid-pipeline once the cache is
  rebuilt.

- **Bootstrapped BDS (`signatures/bds_bootstrap.py`, new).** The asymptotic BDS
  N(0,1) calibration is unreliable under heavy tails, so a permutation null was
  implemented: it preserves the exact empirical marginal while destroying temporal
  dependence — the appropriate i.i.d. null for this regime. **Validated** on known
  cases: it does not reject genuine i.i.d. data (including heavy-tailed t), and
  rejects strongly on GARCH and AR(1). Probe B (Phase 9) now reports the bootstrap
  result as primary, with the asymptotic statistic kept only as a labelled
  reference.

- **EGARCH alternative null (`experiment/egarch_null.py`, new).** To show the
  leverage-arm collapse is not specific to the GJR functional form, an EGARCH null
  was built — it captures leverage through asymmetry in the log-variance rather
  than GJR's squared-residual indicator. **The fitting was verified to engage
  correctly** (the EGARCH asymmetry parameter takes the equity-leverage sign on
  leverage data). Whether the advantage also collapses against it is for the
  regeneration to confirm.

- **Higher-K leverage sweep (Phase 8 fix).** The drifting A-vs-A null is treated as
  a tool-stability signal, not just noise. The sweep is re-run at K = 500
  (uninterrupted) inside the regeneration; the earlier non-monotonicity was at
  least partly an interrupted-run artifact. Both outcomes will be reported: a
  smooth curve establishes the dose-response, a persistent non-monotonicity with a
  stable null becomes a genuine finding about the estimators.

- **Single run script (`run_all.sh`, new).** The whole pipeline is now one command:
  it clears the cache, runs the validation gates, regenerates every experiment in
  dependency order (with the cache-integrity check mid-run), then the EGARCH null,
  logging each stage with timing and stopping at the first failure. The validation
  gates were re-confirmed to pass with the verified parameters before the full run.

**Result of Phase 10:** all three serious items and both substantive
recommendations are resolved at the code level; the study is regenerating in one
verified pass. Result numbers in Phases 4–9 will be finalised from this run.

---

## Phase 11 — Review-stage correction: the TPA transition-probability bug and the Finding 1 reframe

**What and why.** A line-by-line code review against the Franke–Westerhoff (2012)
paper confirmed every structural constant but turned up a bug in the TPA switching
mechanism that the seven-constant check had not covered, because it is a functional
form rather than a constant. The bug manufactured the confusability that Phase 2
reported and that Finding 1 was built on.

**The bug.** The paper (p. 9) defines the per-period switching probabilities as
`pi_cf = min(1, nu * exp(a))` and `pi_fc = min(1, nu * exp(-a))`, with `nu`
multiplying `exp(a)` inside the cap. The code computed `pi_cf = nu * min(1, exp(a))`,
with `nu` outside the cap. The two agree when the attractiveness index `a <= 0`,
but differ when `a > 0`: the paper's probability grows as `nu * exp(a)` up to the
cap of 1, while the buggy version flatlines at `nu`. Since the misalignment term
`alpha_p * (p - p*)^2` is always non-negative, `a` is positive in roughly 70% of
steps, so the bug was active most of the time. The paper's own remark that with
`nu = 0.05` the cap "practically never becomes binding" is the design intent the
buggy code inverts (its cap binds whenever `a > 0`).

**The evidence the bug mattered.**
- At `nu = 0.10`, the max Cohen's d between DCA and TPA is ≈ 0.49 with the buggy
  TPA but ≈ 4.6 with the faithful TPA. The Phase 2 confusability existed only
  because of the bug.
- Sweeping `nu` with the faithful TPA, the pair is wildly separable on moments for
  `nu` from 0.02 to about 0.6 (max d ≈ 5), and confusable only at `nu` near 1.0
  (max d ≈ 0.2), where TPA tracks the attractiveness so fast it becomes DCA.
- A moments classifier (the Referee 1 side of `discriminate.py`, reproduced with
  the validated `moments.py` and the same pipeline) gives AUC ≈ 0.99 for DCA vs the
  faithful TPA across the whole `nu` grid, against an A-vs-A null of ≈ 0.52. So with
  the faithful TPA, moments separate the pair almost perfectly; Referee 2 has no gap
  to fill.

**The decision (Finding 1 reframe).** Because no `nu` gives a pair that is both
confusable on moments and dynamically distinct, the confusable-pair construction
is dropped. Finding 1 is reported as the honest negative it becomes: the faithful
behavioural pair separates easily on moments wherever it genuinely differs, so the
nonlinear signatures add nothing the moments do not already capture. This is still
a clean negative for the nonlinear-signatures hypothesis, and arguably a cleaner
one because it does not rest on a tuned confusability. The matched-moment test in
which nonlinear features could in principle matter is constructed explicitly
against a GARCH null (Finding 2, Phase 5), not by tuning `nu`.

**The fix, at the code level.**
- `fw_ssv.py`: TPA probabilities corrected to `min(1, nu * exp(+/-a))` in code and
  docstring; the `nu` default reverted to the paper's 0.05; the docstring rewritten
  to state that `nu` is the paper's fixed scaling value, that the faithful pair is
  moment-separable except as `nu` approaches 1, and that the matched-moment test is
  the GARCH null. The DCA path is untouched and its stylized facts are unchanged, so
  Findings 2 onward are unaffected.
- `check_fw_pair.py`: repurposed from a confusability gate into a DCA/TPA
  moment-separability diagnostic that reports per-moment Cohen's d at `nu = 0.05`
  and max d across `nu`, documenting the separability that is the premise of the
  reframed Finding 1. No pass/fail verdict.
- `run_all.sh`: the corresponding step relabelled from "confusability gate" to
  "separability diagnostic," with a comment that it documents Finding 1 rather than
  gating.

**Scope.** The bug touches only the TPA path, so it affects Finding 1 and the
confusability framing. It does not touch `harder_pair.py`, the embedding sweep, the
leverage arm, or the residual probes, all of which use DCA only. The study's
central conclusion (nonlinear signatures appear to win only when the statistical
null is under-specified) rests on that DCA-versus-GARCH machinery and is unaffected.

**Result of Phase 11:** the TPA model is now faithful to FW (2012); the
confusable-pair premise of Finding 1 is retired and replaced with a separable-pair
negative; the code is corrected and consistent. The prose in this notebook's
Phases 1, 2, 4, and 10 is annotated above; `project_review.md` and the manuscript's
Finding 1 still need the same reframe.

---

## Phase 12 — Review-stage verification of Finding 2: a reproduction discrepancy, the bootstrap fix, and the environment pin

**What and why.** With the estimator stack and the Finding-1 machinery verified in
Phase 11, the review moved to the Finding-2 hub. The plan was simple: re-run
`harder_pair.py` at the authoritative K = 300 under the reviewed code and confirm
it reproduces the documented tie (the committed `harder_pair_tight_k_300.txt`:
AUC_R1 = 0.734, AUC_R2 = 0.737, delta = +0.002). It did not reproduce, and chasing
down why produced two concrete improvements: a robust way to report the tie, and a
pinned environment.

- **The discrepancy.** The fresh K = 300 run (same seed, same K, same T) returned
  AUC_R1 = 0.759, AUC_R2 = 0.732, delta = -0.027. The sign of delta flipped relative
  to the committed +0.002, and both AUCs moved by a few points. This was flagged as a
  red item and diagnosed before anything was accepted.

- **What it was not.** The fresh run prints "300 retained after pairing", the string
  added by the Phase-5 `garch_match` fix, so it ran on the corrected `harder_pair`.
  The corrected `harder_pair` was confirmed path-neutral: the simplified rank-remap
  was shown byte-identical to the old double-remap across trials, and the
  None-and-filter change is inert when all fits succeed (they did, 300/300). The
  `fw_ssv` corrections from Phase 11 were also ruled out: `harder_pair` builds
  `FWParams(switching="dca")` with `alpha_lev = 0`, and in the source `nu` is read
  only inside the `tpa` branch, so neither the TPA min-cap fix nor the `nu` revert
  touches the DCA path. The scope claim "TPA affects Finding 1 only" holds.

- **What it was.** The marginal stayed matched exactly in both runs (std, hill_alpha,
  excess_kurtosis all at Cohen's d = 0.00), so the rank-match still works; everything
  that moved was *temporal* structure (acf_abs up, lam and lyap_r2 down). That points
  upstream of `harder_pair`, to the random stream itself: the `seeds.py` hardening
  from the earlier session (type-tagged hashing changes the derived stream) and/or a
  different `arch` version than the one that produced the committed file. Either
  reshuffles the FW draws and the GARCH simulation seeds. **Conclusion: the committed
  +0.002 is a stale, pre-correction number; the -0.027 is the current, deterministic,
  authoritative one.** FW-DCA generation was confirmed deterministic (same seed gives
  a byte-identical path).

- **The scientific reading (the conclusion does not change, and arguably sharpens).**
  Across runs, delta now reads +0.067 at K = 30, +0.002 at the old K = 300, and
  -0.027 at this K = 300. The point estimate wanders around zero and its sign is not
  stable. The honest reading of Finding 2 is a tie: once the GARCH null is matched on
  the marginal, FW and GARCH are barely separable, and the residual separation is
  carried as much by acf_abs (an R1 feature) as by the nonlinear set. Reporting it as
  a signed point estimate (+0.002, "R2 edges R1") was over-precise for what the data
  support.

- **The fix that makes the tie robust (`harder_pair.py`, bootstrap added).** A paired
  bootstrap CI on delta = AUC_R2 - AUC_R1 was added as an optional pass (enabled with
  the `HP_BOOTSTRAP=1` environment variable; the main run is unchanged when it is
  off). Design choices, each made to be defensible at refereeing: the resampling unit
  is the pair (an FW path and the GARCH path fit to it), drawn with replacement, so
  the coupling between the arms is propagated rather than assumed away; predicted
  scores are computed out-of-fold once via `cross_val_predict` and the AUC difference
  is bootstrapped over those fixed scores, which removes the bootstrap-plus-CV
  leakage where a duplicated row lands in both train and test; and it is a paired
  bootstrap on the difference, the correct construction for "is the difference zero".
  Holding the scores fixed makes the interval mildly conservative for a tie, which
  cuts the right way. **Outcome at K = 300:** delta(oof) = -0.021, 95% CI
  [-0.059, +0.019], P(delta > 0) = 0.142. The CI brackets zero, so FW and GJR-GARCH
  are statistically indistinguishable on the nonlinear-versus-moment AUC difference
  once the marginal is matched. The interval is *tight* (width about 0.08 at n = 300),
  so this is the strong form of a tie (the difference is genuinely near zero), not a
  no-power tie. The weight of evidence (about 86 percent of resamples) leans toward
  standard moments being marginally better, never toward the nonlinear features
  winning. Finding 2 is now reported as this CI tie, which is immune to the reseeding
  that moved +0.002 to -0.027.

- **Environment pinned (`requirements.txt`, `requirements_frozen.txt`, new).** The
  reproduction discrepancy is exactly the failure mode an unpinned environment
  produces, so the stack was pinned. `requirements.txt` lists the four direct
  dependencies (numpy, scipy, scikit-learn, arch) with floors for getting the code
  running. `requirements_frozen.txt` pins the exact versions that generated the
  verified run, captured from that machine: Python 3.13.13, numpy 2.3.3, scipy
  1.16.2, scikit-learn 1.7.2, arch 8.0.0, pandas 2.3.3, statsmodels 0.14.5. The
  `arch` version matches the sandbox used for the code review, so the validation
  transfers directly. numpy's `default_rng` (PCG64) is stream-stable across the 2.x
  range by numpy's own guarantee, so the FW paths do not depend on the numpy version;
  the version-sensitive pieces are `arch` (the GARCH simulate stream), `scipy` (the
  `arch` optimizer backend), and `scikit-learn` (AUC at the third decimal), which are
  the ones the frozen file locks. Both files live at the repo root.

- **Repository hygiene (guidance recorded; not yet committed).** Reviewed code should
  go to a `code-review` branch, not `main`, until Phases 5 onward are reviewed and the
  docs are reconciled. Target layout: requirements files and gates at the root, the
  module packages as-is, the three documentation files moved under `docs/`, a
  `.gitignore` excluding `__pycache__/`, `results/cache/`, `*.log`, and the `.venv/`
  built from the frozen file, and a `.gitattributes` with `*.sh text eol=lf` to keep
  Unix line endings on the shell scripts under a Windows checkout. The stale
  pre-correction result files (including `harder_pair_tight_k_*.txt`) are to be
  archived, not committed as current.

**Result of Phase 12:** Finding 2 is verified under reviewed code and reported as a
measured bootstrap-CI tie rather than a wandering point estimate; the discrepancy
that surfaced it is fully diagnosed (stale committed number from an old random
stream, not a code regression); and the environment is pinned so results regenerate
exactly. The harder-pair hub is now review-complete.

---

## Phase 13 — Review-stage verification of Phase 6 (the embedding turn) and a full Phase 0-6 regeneration

**What and why.** Phase 12 closed the Finding-2 hub. The next load-bearing claim is
Phase 6: that a real nonlinear advantage emerges and grows with the embedding
dimension m. The three embedding files were reviewed and run, then the whole
reviewed pipeline (Phases 0 through 6) was regenerated in one pass under the pinned
environment to confirm everything holds together.

- **Code review of the three embedding files.** `embedding_sweep.py` (exploratory,
  K = 120), `embedding_diagnostics.py` (the decisive K = 300 file with the A-vs-A
  null and the tau = 2 overlap test), and `embedding_extend.py` (m = 7,8 via a disk
  path cache) were each read and executed. Findings: the code is sound; the
  independent null batch is genuinely decorrelated from the primary; the built-in
  AUC_R1-constant-across-m guard passes (R1 does not leak the embedding); the
  clustering separation is correctly m- and tau-invariant; and the extend disk cache
  round-trips exactly (saved-then-reloaded equals fresh generation, byte-identical,
  and the m = 6 faithfulness check reproduces the diagnostics m = 6 value). One
  regression introduced during this review (an `evaluate` return-signature change
  that silently turned extend's laminarity column into the clustering column) was
  caught and fixed. All three files were changed in one way: the hardcoded
  laminarity column was replaced with a per-feature Cohen's d table over all Referee
  2 features, so the mechanism is read off the data rather than assumed.

- **The full Phase 0-6 run (`run_phase0to6.sh`, new).** A single trimmed pipeline
  runs the reviewed phases only: the three gates, Finding 1 and robustness,
  `harder_pair` with the bootstrap CI, then the embedding sweep, diagnostics, and
  extension. It clears the extend path cache at the start (the cache is keyed only on
  K and T, so a stale cache from an earlier code state would otherwise be reused),
  checks the running environment against the committed pin, and aborts on the first
  failure. It completed in about three and a half hours with no failures and no pin
  mismatch (the stack matched: numpy 2.3.3, scipy 1.16.2, scikit-learn 1.7.2,
  arch 8.0.0, pandas 2.3.3, statsmodels 0.14.5).

- **Phase 6 result, verified at K = 300.** The qualitative finding holds; its
  mechanism and magnitude are corrected.
  - **Real, not an artifact.** The A-vs-A null Delta stays flat across m = 3 to 8
    (about -0.06 to -0.11, no upward drift) with per-feature nonlinear d's pinned
    below ~0.15 at every m, so the embedding does not manufacture separation. The
    FW-vs-GARCH rise survives at tau = 2 (it is stronger there: Delta +0.070 at m = 6
    versus +0.026 at tau = 1), so it is not a tau = 1 vector-overlap artifact.
  - **It keeps climbing through m = 8; it does not plateau.** The tau = 1
    FW-vs-GARCH Delta runs -0.061, -0.027, -0.007, +0.026, +0.046, +0.073 over
    m = 3 to 8. This corrects the earlier "settles into a ceiling near +0.10 by
    m = 8" claim: under the verified stream there is no ceiling in range and the
    advantage is still growing at m = 8.
  - **Mechanism corrected: laminarity AND correlation dimension, not laminarity
    alone.** BDS is the largest single nonlinear feature at the conventional m = 4
    (0.45) but is flat across every m (it uses its own internal embedding and cannot
    grow with the sweep). The two features that climb are recurrence laminarity
    (0.18 to 0.58) and correlation dimension d2 (0.01 to 0.56), rising almost in
    lockstep; the Lyapunov exponent is roughly flat. The earlier "driven by
    laminarity" was half right.
  - **A null-baseline reporting note (yellow flag, resolved by framing).** The
    A-vs-A null is not marginal-matched to the primary FW batch, so its sample
    marginals differ slightly and R1 gains a small edge (AUC_R1 = 0.529), leaving the
    null Delta offset to about -0.07 rather than zero. The artifact test is
    unaffected (it asks whether the null Delta climbs with m, and it does not), but
    the cleanest statement of the finding is the differential: the FW-vs-GARCH Delta
    relative to the null, whose gap grows from about +0.02 at m = 3 to +0.13 to +0.15
    at m = 6 to 8. The decision is to report the differential (correct, needs no
    re-run) and to treat a rank-matched null (which would centre the null at zero) as
    an optional clean-up folded into the canonical regeneration, not a correction.

- **Everything else reproduced.** Finding 1 is the separable-pair baseline negative
  (AUC_R1 ~ 0.99, Referee 2 not exceeding it, DCA-vs-DCA null 0.492); robustness
  shows the positive control with the system-dependent margin framing; Finding 2 is
  the same TIGHT Delta -0.027 with the bootstrap 95% CI [-0.059, +0.019] bracketing
  zero.

**Result of Phase 13:** Phases 0 through 6 are now reviewed and verified end to end
under the pinned environment. The embedding claim survives with a corrected
mechanism (laminarity plus correlation dimension), corrected magnitude (lower, and
still climbing rather than plateauing), and an honest null-baseline framing. This is
consistent with the throughline: the m-growing advantage is real against the *plain*
GARCH null, and Phase 7 (leverage) is where it is attributed and then collapses.

---

## Phase 14 — Completing the file-by-file review (Phases 7-10) and a tested-and-reverted null dead end

**What and why.** Phase 13 left Phases 7 through 10 still to verify file by file. This
phase ran every remaining file in the sandbox, end to end, and closed the review.
Each file was executed, not just read.

- **Phase 7-8 files.** `garch_spec_check.py` (GJR vs plain, fixed an `except:continue`
  index misalignment by tracking the successful subset), `harder_pair_lev.py` (the
  leverage arm; fixed two broken function signatures left from earlier edits and added
  AUC and per-feature columns), `check_cont_facts.py` (verified correct, no changes;
  validates the alpha_lev = 50 operating point), and `leverage_sweep.py` (fixed the
  same `evaluate` mislabel; the non-monotone interior and the K = 500 support were
  confirmed as the honestly-flagged open item). `fw_ssv.py` was re-verified fresh and
  its top docstring reconciled to the Phase 11 reframe.
- **Phase 9 + Phase 10 files.** `bds_bootstrap.py` was reviewed first because
  `residual_tests.py` imports it; it was validated standalone (iid normal and
  heavy-tailed t both reject at the nominal 4-5%, AR(1) 100%, GARCH 52%, deterministic)
  and needed no changes. `residual_tests.py` was then run against it: fixed a latent
  IndexError (`n_bds` could exceed the truncated residual count), dropped dead imports,
  and gave it per-arm output filenames so the symmetric and leverage runs do not clobber
  each other. `test_cache_integrity.py` passed 5/5 and was strengthened to call the REAL
  generators (`fw_paths_tagged`, `fw_lev_paths`) instead of a private copy, so it now
  fails loudly if the real generation ever drifts (confirmed to have teeth by injecting a
  non-deterministic generator). `egarch_null.py` had nine dead imports trimmed and the
  `lam` mislabel fixed. `run_all.sh` was checked end to end (argument and module paths,
  the diagnostic-not-gate `check_fw_pair`, `bash -n`) and needed no changes; the only bug
  it surfaced was the `residual_tests` filename collision, fixed above.

- **A tested-and-reverted null dead end (kept on record).** During this phase the
  review considered folding a rank-matched A-vs-A null into the canonical run, to centre
  the embedding null's Delta at zero rather than the differential framing of Phase 13.
  The change was implemented across the four null sites and then TESTED before trusting
  it. It backfired: rank-matching a second FW batch onto the primary batch's marginal
  builds each null path from one batch's magnitudes at the other's rank-times, which
  breaks the `acf_abs` (volatility-clustering) match in REFEREE_1 and *manufactures*
  separation. Measured at K = 30, the raw null sat at AUC_R1 = 0.477 (near 0.5) while the
  rank-matched null swung to 0.326. The change made the calibration worse, not better, so
  it was fully reverted. The lesson, and the correct design: rank-matching is the tool for
  matching a DIFFERENT model (GARCH) to FW's marginal; two same-DGP FW batches already
  agree in distribution and must NOT be rank-matched. The small AUC_R1 offset is batch
  sampling noise that shrinks with K, and the differential reading (test Delta minus null
  Delta) was correct all along. The manuscript should state explicitly why the null is
  not rank-matched, since a referee may ask.

**Result of Phase 14:** the file-by-file review is complete (Phases 0 through 10, all
files executed and verified). The codebase is ready for the canonical regeneration.

---

## Phase 15 — The canonical regeneration, the EGARCH surprise, and the null fit-quality diagnostic

**What and why.** With the review complete, the whole study was regenerated in one
uninterrupted pass via `run_all.sh` under the pinned environment (Python 3.13.13,
arch 8.0.0), machine kept awake. The run completed clean in roughly eight hours: every
gate passed, the cache-integrity guard passed 5/5 mid-pipeline, no stage failed, and the
final result set contained both per-arm residual files with no stale `embedding_sweep.txt`
leaking through. The Phase 4-9 numbers are now final. The headline values:

- **Finding 1:** clean negative (AUC_R1 ~ 0.99 wherever the pair differs; DCA-vs-DCA
  null 0.49).
- **Finding 2 (tight):** Delta = -0.027 (AUC_R1 = 0.759, AUC_R2 = 0.732) at m = 4, the
  bootstrap-CI tie.
- **Embedding:** FW-vs-GARCH Delta rises from -0.061 (m = 3) to +0.073 (m = 8) with the
  null flat (-0.06 to -0.11) and no plateau; laminarity (0.18 to 0.58) and correlation
  dimension (0.01 to 0.56) co-drive, BDS flat at 0.45. The rise survives tau = 2.
- **GJR leverage arm:** gamma engaged (mean +0.065, 89% positive); Delta ~ 0 across m
  (+0.016 at m = 8). The collapse holds.
- **Leverage sweep (K = 500):** the A-vs-A null is now stable at ~0 (the K = 500 fix
  worked), and with a stable null the dose-response is a genuine finding rather than
  noise: it is THRESHOLD-like, not a smooth gradient. dAUC(m = 8) holds at +0.108, +0.131,
  +0.128, +0.120 for alpha_lev = 0, 10, 20, 30 (gamma rising 0 to 0.030), then drops to
  +0.016 at alpha_lev = 50 (gamma 0.063). The edge persists until the null's leverage term
  engages strongly enough, then collapses. The two endpoints carry the claim; the smooth
  monotone dose-response is not established, and that is reported as the threshold result.
- **Residual probes:** Part A reverses negative on residuals (-0.08 to -0.10 at m = 8,
  both arms); Part B BDS rejection collapses from 92-93% on raw returns to 3-8% on GJR
  residuals (median p 0.005 to ~0.4-0.5).

**The EGARCH surprise.** The alternative-null robustness check did NOT confirm what its
name expected. Against an EGARCH null with its asymmetry term engaged (gamma -0.046, 88%
negative, the equity sign), the nonlinear edge does not collapse: dAUC RISES with m to
+0.146 at m = 8. That is larger than even the symmetric-GARCH arm (+0.073), while the
A-vs-A null stays flat at ~0. So a null that merely CARRIES a leverage term is not enough
to kill the edge. Lining up the three nulls at m = 8: symmetric GARCH +0.073, GJR
(leverage engaged) +0.016, EGARCH (leverage engaged) +0.146. This falsified the
pre-regeneration expectation that the collapse was robust to the asymmetric-null form, and
it is recorded as such rather than smoothed over.

**The null fit-quality diagnostic (`experiment/null_fit_quality.py`, new).** The thesis
("the nonlinear edge measures how badly the null reproduces FW's dynamics, not whether the
null carries an asymmetry label") makes a falsifiable prediction about the EGARCH surprise:
EGARCH should reproduce FW's dynamics WORSE than GJR. The diagnostic fits both models to the
same leverage FW paths, simulates each model's native (non-rank-matched) null, and compares
both against FW on the two quantities the nonlinear features were thought to key on, using
the same estimators `check_cont_facts` validates: volatility clustering ACF(|r|) and the
leverage curve L(tau). Native simulations are used on purpose, since rank-matching only
matches the marginal and cannot repair a temporal mismatch.

The K = 20 preliminary pointed the predicted way (EGARCH clustering distance 0.028 vs GJR
0.010), but at K = 300 that gap VANISHED and the prediction was NOT confirmed. The two nulls
fit FW essentially identically: clustering distance GJR 0.011 vs EGARCH 0.010, leverage
distance GJR 0.050 vs EGARCH 0.051, both ties within noise (and both reproduce FW's sharp
lag-2 leverage trough poorly, sitting flat near -0.02 to -0.03 against FW's -0.143). The
script's auto-verdict fired "EGARCH leverage worse" on the 0.001 leverage difference, which
is noise; the verdict logic was tightened afterward to require a meaningful margin and now
reports COMPARABLE on these numbers. So the diagnostic landed on the case we had flagged as
the one to confront: comparable summary-statistic fit, opposite discrimination. The
clustering-fidelity explanation is therefore NOT established. Whatever the nonlinear
features (laminarity, correlation dimension) detect as different between the GJR and EGARCH
nulls is not captured by 2-point ACF(|r|) or the leverage curve; it lies in higher-order
structure (the natural candidate is volatility-of-volatility or conditional-variance tails,
which an EGARCH log-variance specification can render very differently from GJR even with
matched clustering). Characterising that is the open question.

**Result of Phase 15:** the study is regenerated and final. The under-specified-null
reading is well evidenced on the GJR axis (symmetric edge, collapse when the engaged GJR
matches FW, residual probes, the K = 500 threshold sweep). The EGARCH result is a genuine
open puzzle: a leverage-aware null that matches FW's clustering and leverage as well as GJR
does, yet does not kill the edge. The clustering-fidelity explanation was tested and did not
hold, so the conclusion is stated conservatively (see the updated throughline below): the
edge reflects null mis-specification, demonstrated cleanly for GJR, with the EGARCH case
showing that "adequate on clustering and leverage" is not the same as "adequate for the
nonlinear features," for reasons not yet characterised.

---

## Summary of the throughline

1. **Behavioural vs behavioural (Finding 1):** nonlinear signatures do not beat
   standard moments. (With the corrected TPA the faithful pair separates easily on
   moments wherever it differs, so this is a baseline negative on an easily-separated
   pair rather than a confusable-pair result; see Phase 11.)
2. **Behavioural vs GARCH (Finding 2):** they tie at conventional embeddings, but a
   real, growing advantage appears at higher embedding dimensions — confirmed not an
   artifact. (At m = 4 the tie is verified under reviewed code and pinned environment
   and reported as a bootstrap CI bracketing zero; see Phase 12. The higher-m
   advantage is now verified too: real (null flat across m, survives tau = 2), driven
   by laminarity and correlation dimension together, still climbing at m = 8 with no
   plateau; BDS leads at m = 4 but is m-flat. See Phase 13.)
3. **The leverage investigation:** that advantage is leverage-type volatility
   structure; against a GJR null that adequately reproduces FW's clustering the
   advantage collapses (leverage arm), and once the volatility layer is filtered out
   entirely, the nonlinear tools do *worse* than moments (residual probe). The
   collapse is NOT, however, a property of any asymmetry-aware null: against an EGARCH
   null, whose leverage term engages, the advantage survives and grows with m (Phase
   15). The natural explanation, that EGARCH reproduces FW's clustering worse, was
   tested at K = 300 and did NOT hold: the two nulls fit FW's clustering and leverage
   about equally. So the EGARCH survival is an open puzzle. It points to a higher-order
   structural difference (the nonlinear features detect something distinguishing the
   two nulls that 2-point clustering and the leverage curve do not capture), and it
   means "adequate on clustering and leverage" is not sufficient for the edge to
   collapse.

**The conclusion:** nonlinear signatures appear to beat standard moments only when
the statistical null is mis-specified for the data's volatility dynamics. The apparent
edge is a symptom of an under-specified null and disappears once the null captures
those dynamics, demonstrated cleanly against the GJR null (which matches FW's clustering
and, when its leverage term engages, its asymmetry) and reinforced by the residual
probes and the threshold sweep. One result complicates the simplest version of this
reading and is reported as an open puzzle: an EGARCH null matches FW's clustering and
leverage as well as GJR does, yet leaves the edge intact and growing. So "mis-specified"
cannot be reduced to "poor on clustering and leverage"; the nonlinear features are
detecting a higher-order difference between the two nulls that those summary statistics
miss, and characterising it is the remaining work. The central claim (the edge reflects
null mis-specification rather than a fixed nonlinear property of the data) stands on the
GJR evidence; the EGARCH case sharpens the open question rather than the answer.

---

## Open items (honestly flagged)

- **FW parameter verification — RESOLVED (Phase 10).** All seven structural
  constants verified exactly against Franke–Westerhoff (2012) Table 1; the code and
  manuscript no longer hedge.
- **BDS asymptotic standard errors — RESOLVED (Phase 10).** A permutation
  (bootstrap) null is now implemented and used for the reported Probe B result; the
  asymptotic statistic is kept only as a labelled reference.
- **Phase 8 leverage-sweep drift — RESOLVED (Phase 15).** Re-run at K = 500 inside the
  canonical regeneration. The A-vs-A null is now stable at ~0, so the non-monotone
  interior is a genuine finding rather than noise: the dose-response is threshold-like
  (dAUC holds ~0.12 through alpha_lev = 30, then drops to ~0.02 at 50 as gamma crosses
  ~0.06), not a smooth gradient. Reported as the threshold result; the two endpoints
  carry the claim.
- **Asymmetric-null robustness — OPEN (Phase 15).** The leverage-arm collapse is NOT
  robust to the asymmetric-null functional form: against EGARCH the edge survives and
  grows with m (+0.146 at m = 8). The natural explanation (EGARCH reproduces FW's
  clustering worse) was tested at K = 300 and did NOT hold: the two nulls fit FW's
  clustering and leverage about equally. So this is an open puzzle, not a clean
  sharpening, and points to a higher-order structural difference the nonlinear features
  detect. The scope statement stands: demonstrated for one leverage mechanism against
  GJR; no claim that every behavioural asymmetry is GARCH-representable.
- **TPA transition-probability bug — RESOLVED in code (Phase 11).** Found in
  review: `nu` was applied outside the min(1, .) cap instead of inside it, contrary
  to Franke–Westerhoff (2012, p. 9). Fixed in `fw_ssv.py`; `check_fw_pair.py`
  repurposed to a separability diagnostic; `run_all.sh` relabelled. Affects Finding
  1 and the confusability framing only; the DCA-based Findings 2 onward are
  unaffected.
- **Finding 1 reframe (prose) — IN PROGRESS (Phase 11).** This notebook's Phases 1,
  2, 4, and 10 are annotated. Still to do: reframe `project_review.md` and the
  manuscript's Finding 1 from the confusable-pair story to the separable-pair
  baseline negative.
- **Finding 2 verification and reporting — RESOLVED (Phase 12).** The K = 300 hub was
  re-run under reviewed code; the committed delta = +0.002 did not reproduce
  (regenerated delta = -0.027). Diagnosed as a stale committed number from an old
  random stream (the `seeds.py` hardening and/or `arch` version), not a code
  regression: the `harder_pair` fix is byte-identical and the DCA path does not use
  the corrected TPA `nu`. Finding 2 is now reported as a paired bootstrap CI on delta
  (95% CI [-0.059, +0.019], brackets zero) rather than a signed point estimate.
- **Environment pin — RESOLVED (Phase 12).** `requirements.txt` (floors) and
  `requirements_frozen.txt` (exact: Python 3.13.13, numpy 2.3.3, scipy 1.16.2,
  scikit-learn 1.7.2, arch 8.0.0, pandas 2.3.3, statsmodels 0.14.5) added at the repo
  root. `arch` matches the review sandbox; numpy `default_rng` is stream-stable, so
  the version-sensitive pieces (`arch`, `scipy`, `scikit-learn`) are the ones locked.
- **Stale result files — RESOLVED (Phase 15).** The pre-correction outputs were
  archived before the canonical regeneration, which then wrote the verified set (both
  per-arm residual files present, no stale `embedding_sweep.txt`).
- **Phase 6 embedding verification — RESOLVED (Phase 13).** The three embedding files
  are reviewed and the full Phase 0-6 pipeline regenerated under the pinned
  environment. The m-growing advantage is confirmed real (null flat, survives
  tau = 2), driven by laminarity and correlation dimension, still climbing at m = 8.
  Docs (notebook Phase 6 prose and `project_review.md` Section 6.4) reconciled to the
  verified numbers.
- **Rank-matched A-vs-A null — RESOLVED, REJECTED (Phase 14).** Implemented across the
  four null sites and tested before trusting it; it made the calibration worse (raw null
  AUC_R1 ~ 0.48 vs rank-matched 0.33 at K = 30) by breaking the volatility-clustering
  match, so it was fully reverted. The raw same-DGP null is the correct control and the
  differential reading was right all along. The manuscript should state why the null is
  not rank-matched.
- **Regeneration — RESOLVED (Phase 15).** The full study was regenerated in one
  verified pass under the pinned environment (clean, all gates pass, cache integrity
  5/5, no failures); the Phase 4-9 numbers in this notebook are now final.
- **EGARCH fit-quality confirmation — RAN, did NOT confirm (Phase 15).** The K = 300
  run of `experiment/null_fit_quality.py` came back COMPARABLE, not the predicted
  EGARCH-worse: clustering distance GJR 0.011 vs EGARCH 0.010, leverage GJR 0.050 vs
  EGARCH 0.051, both ties (the K = 20 clustering gap was small-sample noise). So the
  clustering-fidelity explanation is not established, and the EGARCH survival is the
  confront-case. The script's auto-verdict misfired on the 0.001 leverage difference;
  the verdict logic was tightened to require a meaningful margin and now reports
  COMPARABLE.
- **EGARCH survival — RESOLVED as the additive-vs-log axis (Phase 15).** Why does FW-vs-EGARCH
  separate (and grow with m) while FW-vs-GJR collapses, when the two nulls fit FW's clustering
  and leverage equally? Resolved below: the edge tracks the functional FORM of the
  conditional-variance recursion (additive vs log), confirmed with two nulls on each side. A
  first dig (Step 1, `experiment/null_pathology_check.py`) ruled out the dull
  explanation: at K = 300 the EGARCH null paths are NOT pathological, in fact milder than
  GJR (median excess kurtosis 1.28 vs GJR 1.35 vs FW 1.90; zero non-finite, zero freak
  paths, fewer extreme moves than GJR), and its fitted persistence is comparable to GJR's
  (both nulls fit FW's near-integrated volatility). So the edge is a real higher-order
  effect, not a simulation artifact. The difference the nonlinear features detect must be a
  genuine higher-order property of the conditional-variance dynamics (volatility-of-
  volatility, conditional-variance tails, long-memory of |r|, or the run-length of high-vol
  spells). Until then, the manuscript's leverage-robustness section should report the
  EGARCH result honestly: the edge survives against a clean, well-fitting alternative
  asymmetric null, for a higher-order reason not yet characterised.

  Step 2 (`experiment/localise_egarch_gap.py`, K = 300) localised the carrier. The gap is
  carried by the recurrence/geometry cluster: laminarity (lam, single-feature AUC
  0.64->0.74 across m = 3->8, Cohen d 0.88 at the peak), correlation dimension (d2,
  0.60->0.72, and the largest leave-one-out drop at m = 8 so the most unique signal), and
  determinism (det), all of which grow with embedding dimension and so track the dAUC rise.
  The Lyapunov features (lambda1, lyap_r2) carry NOTHING (flat at chance ~0.52), so this is
  NOT a trajectory-divergence effect (a K = 30 trial had wrongly suggested lambda1 was the
  carrier; the full run reversed it, the usual small-sample lesson). BDS adds a separate,
  embedding-independent contribution (flat single-AUC 0.59 but a large leave-one-out drop),
  a different axis of serial dependence; because it is m-flat, the m-DEPENDENCE of the edge
  is entirely the recurrence/dimension cluster. The moments stay flat throughout. Reading:
  FW and a well-fitted EGARCH differ in the LAMINAR/recurrence structure and effective
  complexity of the volatility dynamics (how long the process dwells in slowly-changing
  states, i.e. volatility-regime persistence), not in how fast trajectories diverge.
  Step 3 (`experiment/vol_structure_probe.py`, K = 300) tried to reduce that to a named
  volatility statistic, using rank-matched nulls (identical marginals, so only temporal
  arrangement differs) and a selectivity test: find the statistic where EGARCH sits FAR
  from FW while GJR stays CLOSE. NONE of six standard summaries (volatility-of-volatility
  and its tail, long-memory of |r| over lags 11-100, ACF(|r|) at lag 50, high- and low-vol
  run-lengths) has that signature; calibration was clean (max |d(FW,FWb)| = 0.14). More
  pointedly, the one axis on which the nulls differ from FW at all, long-memory of |r|, runs
  the WRONG way: GJR is FARTHER from FW (Cohen d -0.64) than EGARCH (-0.50). So the
  feature-space proximity (which rates GJR the closer match to FW) is DISCORDANT with raw
  clustering proximity (which rates GJR the worse match). This rules out long-memory as the
  mechanism and shows something stronger: the recurrence features are not a proxy for any
  standard volatility diagnostic. They detect a higher-order structural difference between
  FW and EGARCH that is orthogonal to (here, anti-correlated with) clustering/long-memory
  fidelity, vol-of-vol, and regime dwell-time, and that GJR happens to reproduce while
  EGARCH does not. Step 3 alone could not reduce it to a named economic/statistical property.

  Follow-up (`experiment/aparch_null.py`, K = 300) NAMED the axis. The GJR-vs-EGARCH contrast
  confounds additive-vs-log variance form with parameterisation, so a third leverage-aware
  null was added: APARCH, which is additive-FAMILY (a power of the conditional std dev,
  power delta free) but parameterised unlike GJR. APARCH TIES: dAUC flat across m
  (-0.037, -0.042, +0.015), landing at +0.015 at m = 8, essentially identical to GJR's
  +0.016, while EGARCH rose to +0.146. The in-run GJR re-score reproduced +0.016 exactly (a
  faithfulness check). Crucially APARCH chose delta ~ 1.7-1.9 (not GJR's 2) with leverage
  engaged (gamma +0.37, 88% positive), so a genuinely different additive parameterisation
  still collapses the edge, it is the additive FAMILY that ties, not GJR's specific form.
  Two additive leverage-aware nulls (GJR power-2, APARCH power-1.7) tie; the one log-variance
  leverage-aware null (EGARCH) separates. So the structural difference the recurrence
  features detect tracks the FORM of the conditional-variance recursion (additive vs log),
  not leverage-awareness, and not any standard volatility summary (Step 3).

  Resolution: this SHARPENS rather than breaks the central thesis. The edge collapses against
  nulls whose variance recursion is additive, the form that matches FW's effective volatility
  dynamics, and survives against the log-variance null, which has the wrong recursion form.
  EGARCH is leverage-aware but mis-specifies the variance FORM, and the recurrence features
  (laminarity, correlation dimension) detect exactly that, where the moments largely cannot
  (the moments catch only the shared clustering deficit, AUC_R1 ~ 0.72 for all three nulls).
  So the nonlinear features earn their keep in a precisely scoped way: they are sensitive to
  conditional-variance functional-form mis-specification that standard moment diagnostics
  miss. CONFIRMED (`experiment/loggarch_null.py`, K = 300): a second, independent log-variance
  null locks the axis. Log-GARCH (asymmetric, custom Student-t QML) uses the SAME log-variance
  recursion as EGARCH but a DIFFERENT news-impact transform (log eps^2 rather than EGARCH's
  |z|), so a shared result isolates the log form. It SEPARATES like EGARCH: dAUC climbs
  +0.014 -> +0.056 -> +0.103 across m = 3,5,8 (EGARCH-shaped), while the in-run GJR re-score
  reproduced +0.016 exactly and the A-vs-A null stayed flat. The clincher is the per-feature
  table at m = 8: laminarity 0.93 and d2 0.81 sit right on the EGARCH values (0.88, 0.82) and
  far above GJR (0.43, 0.38), so the SAME two features carry it to the SAME magnitude, via a
  different news-impact. Fit health clean (300/300 converged, persistence 0.98 near-integrated,
  xi 83% negative = leverage engaged). So: two additive leverage-aware nulls (GJR power-2,
  APARCH power-1.7) tie; two log-variance leverage-aware nulls (EGARCH |z|, log-GARCH log eps^2)
  separate. The additive-vs-log axis is LOCKED and can be stated as a finding, not a
  conjecture. (The scalar dAUC is a little below EGARCH's +0.146, but the carrier features
  match exactly, so the mechanism is identical.) Remaining confirmatory check: floor-sensitivity
  of the log-GARCH inlier guard (re-run at FLOOR_PCT 0.5 and 2.5; persistence and dAUC should
  be stable) before the manuscript states it.
- **Finding 1 and EGARCH prose in `project_review.md` and the manuscript — IN PROGRESS.**
  This notebook is current. Still to do: fold the Finding 1 separable-pair reframe
  (Phase 11) and the EGARCH open-puzzle framing (Phase 15) into the review write-up and
  the manuscript.