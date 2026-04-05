# `gsynth()` — API Reference

Estimates a Generalized Synthetic Control model (or Interactive Fixed Effects / Matrix Completion) on panel data.

---

## Function Signature

```python
def gsynth(
    formula=None, data=None, Y=None, D=None, X=None, na_rm=False,
    index=None, weight=None, force="unit", cl=None,
    r=0, lam=None, nlambda=10, CV=True, criterion="mspe", k=5,
    EM=False, estimator="gsynth", se=False, nboots=200,
    inference=None, parallel=False, cores=None, tol=1e-3,
    seed=None, min_T0=5, alpha=0.05, normalize=False,
) -> GsynthResult
```

---

## Parameters

### `formula`
**Type**: `str` or `None` | **Default**: `None`

Patsy-style formula string specifying the outcome and treatment. The format is `"Y ~ D"` or `"Y ~ D + X1 + X2"`. The left-hand side is the outcome variable name; `D` on the right-hand side is the treatment indicator; any additional right-hand-side terms are covariates. If `formula` is provided, `Y`, `D`, and `X` are ignored.

---

### `data`
**Type**: `pd.DataFrame` | **Default**: `None`

The panel dataset in long format (one row per unit-time observation). Must contain columns matching the names specified in `formula` and `index`. All column names referenced in `formula` must exist in `data`.

---

### `Y`
**Type**: `str` or `None` | **Default**: `None`

Name of the outcome column in `data`. Used when `formula=None`. Must be a numeric column.

---

### `D`
**Type**: `str` or `None` | **Default**: `None`

Name of the binary treatment indicator column in `data`. Used when `formula=None`. Must contain only 0 and 1; must be absorbing (once 1, always 1 for that unit).

---

### `X`
**Type**: `list of str` or `None` | **Default**: `None`

List of covariate column names in `data`. Used when `formula=None`. All named columns must be numeric. Equivalent to specifying covariates in `formula`.

---

### `na_rm`
**Type**: `bool` | **Default**: `False`

If `True`, rows with `NaN` values in `Y`, `D`, or any covariate are silently dropped before estimation. If `False`, any missing value raises an error. Setting `na_rm=True` can create an unbalanced panel.

---

### `index`
**Type**: `list of str` | **Default**: `None`

A two-element list `[unit_id_column, time_id_column]` identifying the unit and time dimensions of the panel. These two columns must uniquely identify every row. Example: `index=["state", "year"]`.

---

### `weight`
**Type**: `str` or `None` | **Default**: `None`

Name of a column in `data` containing unit-level analytic or inverse-probability weights. Weights are applied during factor and loading estimation. If `None`, all units receive equal weight.

---

### `force`
**Type**: `str` | **Default**: `"unit"`

Controls which additive fixed effects are partialled out before IFE estimation. Must be one of:
- `"none"`: no additive FEs removed; pure IFE
- `"unit"`: unit FEs `alpha_i` removed (recommended default)
- `"time"`: time FEs `xi_t` removed
- `"two-way"`: both unit and time FEs removed

See [Fixed Effects](../user-guide/fixed-effects.md) for details.

---

### `cl`
**Type**: `str` or `None` | **Default**: `None`

Name of a column in `data` identifying clusters for cluster-robust bootstrap inference. Clusters are resampled as units during bootstrap. Requires `se=True`. If `None`, bootstrap resamples at the individual unit level.

---

### `r`
**Type**: `int` or `list of int` | **Default**: `0`

Number of latent factors. When `CV=True`, this is the maximum `r` to evaluate (or a list of specific values to try). When `CV=False`, this is the fixed number of factors used. Capped at `min_T0 - 2` (with unit FE) or `min_T0 - 1` (without). Setting `r=0` with `CV=False` fits only additive fixed effects without any factor structure.

---

### `lam`
**Type**: `float` or `None` | **Default**: `None`

Regularization parameter for the matrix completion (`mc`) estimator. If `None` and `CV=True`, the optimal lambda is selected by cross-validation over a grid of `nlambda` values. Ignored for `gsynth` and `ife` estimators.

---

### `nlambda`
**Type**: `int` | **Default**: `10`

Number of lambda values in the CV grid for the `mc` estimator. Larger values give finer lambda search but increase computation time. Ignored when `lam` is specified or when `estimator != "mc"`.

---

### `CV`
**Type**: `bool` | **Default**: `True`

If `True`, cross-validation is used to select the optimal `r` (and `lambda` for `mc`). If `False`, the value of `r` is used directly without CV. See [Cross-Validation](../user-guide/cross-validation.md).

---

### `criterion`
**Type**: `str` | **Default**: `"mspe"`

CV criterion for factor number selection. `"mspe"` selects `r` minimizing mean squared prediction error. `"pc"` uses a panel information criterion analogous to AIC/BIC for factor models. `"mspe"` is more robust in small samples.

---

### `k`
**Type**: `int` | **Default**: `5`

Number of folds for k-fold cross-validation used with `ife` and `mc` estimators. Ignored for `gsynth` (which uses LOO). Larger `k` reduces bias in CV estimates but increases computation time.

---

### `EM`
**Type**: `bool` | **Default**: `False`

If `True`, uses an Expectation-Maximization algorithm for parameter initialization. Currently experimental.

---

### `estimator`
**Type**: `str` | **Default**: `"gsynth"`

Estimation method. Must be one of:
- `"gsynth"`: Generalized Synthetic Control — factors from control group only (default, recommended)
- `"ife"`: Interactive Fixed Effects via Alternating Least Squares on all pre-treatment data
- `"mc"`: Matrix Completion with nuclear-norm regularization

See [Estimators](../user-guide/estimators.md) for a full comparison.

---

### `se`
**Type**: `bool` | **Default**: `False`

If `True`, computes standard errors and confidence intervals via bootstrap or jackknife. Populates `att_avg_se`, `att_avg_ci_lower/upper`, `att_se`, `att_ci_lower/upper`, and `att_boot` on the result. Requires specifying `nboots` (for bootstrap) or setting `inference="jackknife"`.

---

### `nboots`
**Type**: `int` | **Default**: `200`

Number of bootstrap replicates. Only used when `se=True` and `inference` is `"parametric"` or `"nonparametric"`. For publication-quality results, use 500 or more. Increasing `nboots` reduces Monte Carlo variability in SE estimates.

---

### `inference`
**Type**: `str` or `None` | **Default**: `None`

Inference method. When `None`, the default is chosen by estimator: `"parametric"` for `gsynth`, `"nonparametric"` for `ife` and `mc`. Must be one of:
- `"parametric"`: pseudo-treatment bootstrap using control units as placebo treated
- `"nonparametric"`: block bootstrap resampling entire units with replacement
- `"jackknife"`: leave-one-unit-out jackknife (does not use `nboots`)

See [Inference](../user-guide/inference.md).

---

### `parallel`
**Type**: `bool` | **Default**: `False`

If `True`, bootstrap replicates are run in parallel using Python's multiprocessing. Can significantly reduce wall-clock time for large `nboots`. Requires `cores` to be set.

---

### `cores`
**Type**: `int` or `None` | **Default**: `None`

Number of CPU cores for parallel bootstrap. When `parallel=True` and `cores=None`, uses all available cores minus one. Ignored when `parallel=False`.

---

### `tol`
**Type**: `float` | **Default**: `1e-3`

Convergence tolerance for the ALS algorithm (used by `ife` estimator). Iteration stops when the relative change in the objective falls below `tol`. Smaller values give more precise estimates at the cost of more iterations.

---

### `seed`
**Type**: `int` or `None` | **Default**: `None`

Random seed for reproducibility. Controls random number generation for bootstrap resampling and k-fold CV fold assignment. Set to any integer for reproducible results.

---

### `min_T0`
**Type**: `int` | **Default**: `5`

Minimum required number of pre-treatment periods for a treated unit. Units with fewer pre-treatment periods are dropped with a warning. Also determines the maximum allowable `r` (see `r` parameter). Increase this for more reliable loading estimation.

---

### `alpha`
**Type**: `float` | **Default**: `0.05`

Significance level for confidence intervals. Default `0.05` gives 95% CIs. Used when `se=True`. Set to `0.10` for 90% CIs.

---

### `normalize`
**Type**: `bool` | **Default**: `False`

If `True`, normalizes the outcome and covariates to unit variance before estimation. Can improve numerical stability when variables have very different scales. The reported ATTs are transformed back to the original scale.

---

## Returns

Returns a [`GsynthResult`](result.md) object containing all estimation output.

---

## Notes

### Three-Stage Pipeline

Internally, `gsynth()` runs three stages:

1. **Data preparation**: parse formula, validate panel structure, partial out covariates via OLS on control units, demean by fixed effects per `force`.
2. **Factor estimation**: estimate latent factors `F_t` and loadings `lambda_i` using the selected estimator and `r`.
3. **ATT computation**: impute counterfactuals `Y_ct = alpha_i + lambda_i @ F_t + beta @ X_it`; compute `att_it = Y_tr_it - Y_ct_it` for post-treatment treated cells; aggregate to event-time ATTs and `att_avg`.

### Formula Parsing Rules

- `formula` must be a string of the form `"Y ~ D"` or `"Y ~ D + X1 + X2"`.
- The left-hand side must be a single variable name (the outcome).
- The right-hand side must include exactly one treatment indicator `D`. Additional terms are treated as covariates.
- Interaction terms and transformations (e.g., `"log(X)"`) are not supported. Apply transformations to `data` before calling `gsynth`.

### Normalization

When `normalize=True`, each column (outcome and covariates) is divided by its standard deviation (computed on control unit pre-treatment observations). This affects factor estimation and can improve ALS convergence. The reported `att_avg` and `att` are always in the original units.

### r Constraint Rule

The maximum `r` evaluated in CV (or used when `CV=False`) is capped at:
- `min_T0 - 2` when `force` is `"unit"` or `"two-way"`
- `min_T0 - 1` when `force` is `"none"` or `"time"`

This prevents overfitting in the loading estimation step. If you specify `r` that exceeds this cap, gsynth silently reduces it to the cap.

---

## See Also

- [`GsynthResult`](result.md) — full description of all return fields
- [`plot()`](plot.md) — visualization
- [`effect()`](effect.md) — effect summary and subgroup analysis
- [Estimators guide](../user-guide/estimators.md)
- [Cross-Validation guide](../user-guide/cross-validation.md)
- [Inference guide](../user-guide/inference.md)

---

## Examples

### Basic Estimation

```python
import numpy as np
import pandas as pd
from gsynth import gsynth

rng = np.random.default_rng(42)
N, T, N_tr, T0 = 30, 20, 5, 10
units = [f"unit_{i}" for i in range(N)]
times = list(range(2000, 2020))
alpha = rng.normal(0, 1, N)
F = rng.normal(0, 1, (T, 2))
lam = rng.normal(0, 1, (N, 2))
eps = rng.normal(0, 0.5, (N, T))
Y_base = alpha[:, None] + lam @ F.T + eps

rows = []
for i, u in enumerate(units):
    for t_idx, yr in enumerate(times):
        D = 1 if i < N_tr and t_idx >= T0 else 0
        rows.append({"unit": u, "year": yr, "Y": Y_base[i, t_idx] + D * 1.5, "D": D})
df = pd.DataFrame(rows)

result = gsynth("Y ~ D", data=df, index=["unit", "year"])
result.summary()
```

### With Cross-Validation

```python
# CV selects optimal r from {0, 1, 2, 3}
result = gsynth("Y ~ D", data=df, index=["unit", "year"],
                r=[0, 1, 2, 3], CV=True, criterion="mspe")
print(f"CV-selected r: {result.r}")
print("CV scores:", result.r_cv)
```

### With Inference

```python
result = gsynth("Y ~ D", data=df, index=["unit", "year"],
                r=2, CV=False,
                se=True, nboots=500, inference="parametric", seed=42)
print(f"ATT: {result.att_avg:.3f} +/- {result.att_avg_se:.3f}")
print(f"95% CI: [{result.att_avg_ci_lower:.3f}, {result.att_avg_ci_upper:.3f}]")
```

### With Covariates

```python
# Add time-varying covariate X1
rng2 = np.random.default_rng(99)
df["X1"] = rng2.normal(0, 1, len(df))

result = gsynth("Y ~ D + X1", data=df, index=["unit", "year"],
                r=2, CV=False)
print(f"beta_X1: {result.beta[0]:.3f}")
print(f"ATT:     {result.att_avg:.3f}")
```
