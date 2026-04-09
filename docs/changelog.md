# Changelog

## 0.2.0 (2026-04-09)

### Bug fixes

**Covariate estimation — joint alternating loop (major fix)**

The previous release estimated `beta` sequentially: demean `Y` (assuming `beta=0`), then run OLS on demeaned `Y` using control-unit observations only. This approach is biased when covariates `X` are correlated with the latent factors, because the unit/time fixed effects absorb covariate variation during the demeaning step, leaving little signal for OLS to recover.

The fix implements the same joint alternating loop used by R `fect`:

1. Given the current `beta`, demean `Y − X @ beta` to obtain updated unit and time fixed effects.
2. Run ALS on the demeaned control outcomes to estimate factors `F` and loadings `Λ`.
3. Update `beta` via OLS on all D=0 residuals: `(Y − FE) − Λ @ F'`.
4. Repeat until convergence.

This joint loop ensures that fixed effects, factors, and `beta` are estimated consistently. On the validation dataset (N=20, T=30, N_tr=5, r=2, true ATT=2.0):

| Metric | Before fix | After fix | R gsynth | True |
|--------|-----------|-----------|----------|------|
| `att_avg` | 1.927 | **2.004** | 2.003 | 2.000 |
| `beta_X1` | 0.407 | **0.617** | 0.650 | 0.600 |
| `beta_X2` | −0.061 | **−0.373** | −0.355 | −0.400 |

**Covariate observation set — all D=0 cells (moderate fix)**

For `ife` and `mc` estimators, `partial_out_covariates` previously used only control-unit observations for the OLS. It now uses all D=0 observations (control units for all periods + treated units for pre-treatment periods), matching R fect's behaviour.

**Cross-validation methodology (moderate fix)**

For both `gsynth` and `ife` estimators, CV now uses k-fold hold-out on control-unit cells (matching R fect) rather than leave-one-out on treated pre-treatment data. Both methods selected the same `r*` on the validation dataset; on real data they may disagree.

---

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
