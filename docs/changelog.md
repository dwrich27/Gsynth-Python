# Changelog

## 0.1.0 (2026-04-05)

Initial release — Python port of R gsynth implementing:

- Three estimators: gsynth, ife (ALS), mc (nuclear-norm)
- All four fixed-effects specifications (none/unit/time/two-way)
- Cross-validation for factor number (LOO on treated pre-treatment data for gsynth; k-fold for ife)
- Cross-validation for lambda (matrix completion)
- Three inference methods: parametric bootstrap (pseudo-treatment), nonparametric block bootstrap, jackknife
- Six plot types: gap, raw, counterfactual, factors, loadings, missing
- effect() for cumulative and subgroup ATT
- GsynthResult with full output fields
- Pure NumPy implementation — no scipy dependency
- Support for staggered adoption
- Time-varying covariate partial-out
- Unit-level analytic/IPW weights
- Cluster bootstrap
- Event-time ATT representation (0 = first treatment period)
- Treatment-cell-weighted att_avg matching R gsynth
