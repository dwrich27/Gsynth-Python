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

## How Covariates Are Used: Joint Alternating Estimation

The IFE model with covariates is:

```
Y_it = α_i + ξ_t + λ_i · F_t + β · X_it + ε_it
```

Estimating `β` and the factor structure jointly is important: when `X` is correlated with the latent factors (which is common — both vary by unit and time), a naive sequential approach absorbs covariate variation into the fixed effects during demeaning, biasing `β` toward zero.

gsynth uses the same joint alternating loop as R `fect`:

1. **Initialize** `β = 0`.
2. **Demean** `Y − X@β` to remove unit and time fixed effects, yielding `Ỹ`.
3. **ALS** on `Ỹ` for control units to estimate latent factors `F` and loadings `Λ`.
4. **Update `β`** via OLS using all D=0 residuals:
   `(Ỹ + X@β) − Λ@F'` regressed on `X` (D=0 cells).
5. **Repeat** steps 2–4 until `‖Δβ‖ < tol`.

The **D=0 mask** in step 4 covers control units for all periods and treated units for their pre-treatment periods. Using the full D=0 set (not just control units) avoids bias when covariate variation in the control group does not span the full covariate space.

After convergence, `result.beta` holds the final `β`, and the counterfactual is:

```
Ŷ(0)_it  =  α̂_i  +  ξ̂_t  +  λ̂_i · F̂_t  +  β̂ · X_it
```

!!! note "Estimator differences"
    The joint loop applies to the `gsynth` estimator. For `ife` and `mc`, `β` is estimated via a single sequential OLS on all D=0 observations (no iteration), which is the standard approach for those estimators.

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
