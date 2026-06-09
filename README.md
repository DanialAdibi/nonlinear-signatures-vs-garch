# Do Nonlinear-Dynamical Signatures Add Anything to Return Moments?

**Functional-Form Mis-specification of the Volatility Null in a Controlled Discrimination Study**

A controlled simulation study testing whether nonlinear-dynamical signatures of
returns (recurrence quantification, correlation dimension, Lyapunov exponent, BDS)
discriminate market models better than standard return moments. Because the
data-generating process is known and contains no low-dimensional deterministic
structure, the study can say what these signatures actually detect: not chaos, but a
mis-specification in the *functional form* of a statistical null's conditional-variance
recursion (additive versus log).

This repository contains all code needed to regenerate every result in the paper from
scratch.

## Summary of the finding

The nonlinear signatures carry no general advantage over standard return moments. They
acquire one only when the statistical null is mis-specified in the functional form of
its variance recursion: with a leverage term fitted in every null, the nonlinear
advantage collapses against additive-variance nulls (GJR, APARCH) and survives, growing
with embedding dimension, against log-variance nulls (EGARCH, log-GARCH). Recurrence
laminarity and the correlation dimension act as a model-free diagnostic of
conditional-variance functional-form mis-specification, a property the return moments do
not register.

## Repository structure

```
models/        the Franke-Westerhoff generator and its leverage extension, plus the
               known-dynamics reference systems (Henon, Lorenz, noise, AR(1)) used to
               validate the estimators
signatures/    the feature estimators (embedding, moments, Lyapunov, dimension, RQA, BDS)
experiment/    the discrimination experiments and the statistical-null family
controls/      validation gates and the cache-integrity guard
seeds.py       the master seed and tagged RNG (project root)
run_all.sh     single-command regeneration of the entire study
```

## Requirements

- Python 3.13
- `numpy`, `scipy`, `scikit-learn`
- `arch` (pinned; its GJR-GARCH and EGARCH maximum-likelihood fits feed directly into
  the reported AUCs, so the version matters)

`statsmodels` is deliberately not used, to keep volatility fitting within a single
pinned implementation. Exact pinned versions are listed in `requirements.txt`.

## Reproducing the results

```
bash run_all.sh
```

This clears the path cache, runs the estimator-validation gates, regenerates every
experiment in dependency order (with a cache-integrity check once the cache is rebuilt),
and finishes with the alternative-null robustness checks, logging each stage with timing
and stopping at the first failure. Every result file behind the paper's tables is
produced by this single command. All randomness derives from one master seed, so the run
reproduces exactly.

## Paper

[Link to the paper / preprint, once available.]

## Citation

If you use this code, please cite the paper. A `CITATION.cff` is provided.

## License

Released under the [LICENSE](LICENSE).