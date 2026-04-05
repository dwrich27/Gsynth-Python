# Covariates

gsynth supports time-varying covariates that are partialled out before factor estimation. This allows you to control for observed confounders that vary both across units and over time.

---

## Formula Syntax

Add covariates to the right-hand side of the formula, after `D`:

```python
# One covariate
result = gsynth("Y ~ D + X1", data=df, index=["unit", "year"])

# Multiple covariates
result = gsynth("Y ~ D + X1 + X2 + X3", data=df, index=["unit", "year"])
```

The column names `X1`, `X2`, etc. must exist in `data` and be numeric.

---

## How Covariates Are Used: Partialling Out

gsynth removes the linear covariate contribution from the outcome before estimating the interactive fixed effects model. The procedure is:

1. **OLS on control units** (pre-treatment period): regress `Y` on `X` (and the FE structure implied by `force`) using only control-unit observations.
2. **Estimate `beta`**: the OLS coefficient vector `beta` is stored in `result.beta`.
3. **Partial out**: the adjusted outcome `Y_tilde = Y - beta * X` is used for all subsequent IFE estimation (factors, loadings, counterfactuals).
4. **Counterfactual**: the imputed Y(0) adds back the covariate contribution: `Y_ct = alpha_i + lambda_i @ F_t + beta @ X_it`.

This approach avoids conflating covariate effects with treatment effects in the factor estimation step.

---

## Covariate Coefficient: `result.beta`

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
X = rng.normal(0, 1, (N, T))     # one covariate
beta_true = 0.8
Y_base = alpha[:, None] + lam @ F.T + beta_true * X + eps
TAU = 1.5

rows = []
for i, u in enumerate(units):
    is_treated = i < N_tr
    for t_idx, yr in enumerate(times):
        D = 1 if is_treated and t_idx >= T0 else 0
        Y = Y_base[i, t_idx] + D * TAU
        rows.append({"unit": u, "year": yr, "Y": Y, "D": D, "X1": X[i, t_idx]})

df = pd.DataFrame(rows)

result = gsynth("Y ~ D + X1", data=df, index=["unit", "year"], r=2, CV=False)
print(f"Estimated beta: {result.beta}")      # should be close to 0.8
print(f"True beta:      {beta_true}")
print(f"ATT:            {result.att_avg:.3f}")  # should be close to 1.5
```

---

## Multiple Covariates

With multiple covariates, `result.beta` is a 1-D array of length `K` (number of covariates), in the same order as they appear in the formula:

```python
result = gsynth("Y ~ D + log_income + turnout", data=df, index=["unit", "year"])
print("Covariate names:", result.X_names)
print("Coefficients:   ", result.beta)
# X_names: ['log_income', 'turnout']
# beta[0] = coefficient on log_income
# beta[1] = coefficient on turnout
```

---

## Interpreting `result.X_names`

`result.X_names` stores the list of covariate names in the order they appear in the formula. This lets you match `beta` values to variable names:

```python
for name, coef in zip(result.X_names, result.beta):
    print(f"  {name}: {coef:.4f}")
```

---

## Limitations

### Time-invariant covariates

Do **not** include time-invariant covariates (e.g., a dummy for region) in the formula. Such variables are collinear with the unit fixed effects (`force="unit"` or `"two-way"`) and will cause a singular matrix error. Time-invariant covariates are automatically absorbed by unit FEs.

```python
# BAD: 'region' is time-invariant -> collinear with unit FEs
result = gsynth("Y ~ D + region", data=df, index=["unit", "year"])

# GOOD: only include time-varying covariates
result = gsynth("Y ~ D + log_gdp + inflation", data=df, index=["unit", "year"])
```

### Treatment-induced covariates

Do not include covariates that are themselves affected by the treatment (mediators or "bad controls"). If `X_it` responds to `D_it`, including it will bias the counterfactual by absorbing part of the treatment effect. Only include pre-determined or exogenous covariates.

### Missing values in covariates

If covariate columns contain `NaN`, gsynth will raise an error unless `na_rm=True`. Missing covariate values are treated the same as missing outcome values.

!!! tip "Covariate centering"
    Covariates are not automatically centered. If your covariates have very different scales, consider standardizing them before estimation to improve numerical stability, especially with many covariates.
