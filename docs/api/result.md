# `GsynthResult` — API Reference

The `GsynthResult` class holds all output from a [`gsynth()`](gsynth.md) estimation. All fields are accessible as attributes.

---

## Class Description

`GsynthResult` is the return type of `gsynth()`. It bundles point estimates, counterfactuals, model components, inference output, and metadata into a single object. All array shapes are documented below; scalar fields are annotated with their Python type.

---

## Fields

### Treatment Effects

| Field | Type | Shape | Description |
|---|---|---|---|
| `att_avg` | `float` | scalar | Treatment-cell-weighted average ATT across all treated units and post-treatment periods. The primary estimand. |
| `att` | `ndarray` | `(T_all,)` | Event-time ATT. `att[k]` corresponds to `att_time[k]`. Negative event times are pre-treatment placebo estimates. |
| `att_time` | `ndarray` | `(T_all,)` | Event-time indices. 0 = treatment onset, negative = pre-treatment, positive = post-treatment. |

**Event-time convention**: `att_time = 0` is the first period in which any treated unit receives treatment (or each unit's own onset in staggered settings). Negative values are pre-treatment lags; positive values are post-treatment leads.

```python
# Access treatment effects
print(f"ATT average: {result.att_avg:.3f}")

# Event-time array
import numpy as np
for t, a in zip(result.att_time, result.att):
    print(f"  t={t:+d}  ATT={a:.3f}")

# Separate pre and post
pre_att  = result.att[result.att_time < 0]
post_att = result.att[result.att_time >= 0]
print(f"Mean pre-treatment ATT:  {pre_att.mean():.3f}  (should be ~0)")
print(f"Mean post-treatment ATT: {post_att.mean():.3f}")
```

---

### Counterfactuals

| Field | Type | Shape | Description |
|---|---|---|---|
| `Y_tr` | `ndarray` | `(N_tr, T)` | Observed outcomes for treated units, all time periods. |
| `Y_ct` | `ndarray` | `(N_tr, T)` | Imputed counterfactual Y(0) for treated units, all time periods. |

The ATT for treated unit `i` at post-treatment period `t` is `Y_tr[i, t] - Y_ct[i, t]`.

```python
# Counterfactuals
print("Y_tr shape:", result.Y_tr.shape)   # (N_tr, T)
print("Y_ct shape:", result.Y_ct.shape)   # (N_tr, T)

# Unit-level post-treatment ATT matrix
import numpy as np
post_mask = result.att_time >= 0
# Note: att_time indexing is in event time; Y_tr/Y_ct are in calendar time.
# Use result.T0 to align if needed.
```

---

### Model Components

| Field | Type | Shape | Description |
|---|---|---|---|
| `factors` | `ndarray` | `(T, r)` | Estimated latent common factors `F_t`. Each column is one factor. |
| `loadings` | `ndarray` | `(N, r)` | Estimated factor loadings `lambda_i` for all units (treated + control). Each row is one unit. |
| `alpha` | `ndarray` | `(N,)` | Estimated unit fixed effects. Zero for all units when `force="none"` or `"time"`. |
| `xi` | `ndarray` | `(T,)` | Estimated time fixed effects. Zero for all periods when `force="none"` or `"unit"`. |
| `beta` | `ndarray` | `(K,)` | Estimated covariate coefficients. Empty array `[]` when no covariates. |
| `r` | `int` | scalar | Number of factors used in the final model (CV-selected or user-specified). |
| `lambda_opt` | `float` | scalar | Optimal regularization lambda for `mc` estimator. `nan` for `gsynth` and `ife`. |
| `mse` | `float` | scalar | In-sample mean squared error on pre-treatment control observations. |
| `sigma2` | `float` | scalar | Estimated error variance from the IFE fit. |

```python
# Model components
print("Factors shape:  ", result.factors.shape)   # (T, r)
print("Loadings shape: ", result.loadings.shape)  # (N, r)
print("Unit FEs:       ", result.alpha[:5])
print("Time FEs:       ", result.xi)
print("Covariate beta: ", result.beta)
print("r selected:     ", result.r)

# Factor for treated unit 0 (row in loadings)
treated_unit_idx = 0
loading_i = result.loadings[treated_unit_idx, :]
print(f"Loading for treated unit 0: {loading_i}")
```

---

### Inference

These fields are only populated when `se=True` was passed to `gsynth()`. Otherwise they are `None` or `nan`.

| Field | Type | Shape | Description |
|---|---|---|---|
| `att_avg_se` | `float` | scalar | Bootstrap/jackknife standard error of `att_avg`. |
| `att_avg_ci_lower` | `float` | scalar | Lower bound of `(1-alpha)` confidence interval for `att_avg`. |
| `att_avg_ci_upper` | `float` | scalar | Upper bound of `(1-alpha)` confidence interval for `att_avg`. |
| `att_se` | `ndarray` | `(T_all,)` | Standard error for each event-time ATT. |
| `att_ci_lower` | `ndarray` | `(T_all,)` | Lower CI bound for each event-time ATT. |
| `att_ci_upper` | `ndarray` | `(T_all,)` | Upper CI bound for each event-time ATT. |
| `att_boot` | `ndarray` | `(nboots, T_post)` | Bootstrap draws of post-treatment ATTs. Columns correspond to post-treatment event times only. |

```python
if result.att_avg_se is not None:
    print(f"ATT: {result.att_avg:.3f} (SE={result.att_avg_se:.3f})")
    print(f"95% CI: [{result.att_avg_ci_lower:.3f}, {result.att_avg_ci_upper:.3f}]")

    # Inspect bootstrap distribution
    import numpy as np
    print(f"Bootstrap std:   {result.att_boot.mean(axis=1).std():.3f}")
    print(f"Bootstrap shape: {result.att_boot.shape}")
```

---

### Diagnostics

| Field | Type | Shape | Description |
|---|---|---|---|
| `r_cv` | `ndarray` | `(r_max+1,)` | CV MSPE scores indexed by candidate `r`. `r_cv[k]` is the CV score for `r=k`. |
| `lambda_cv` | `ndarray` or `None` | `(nlambda,)` | CV scores for each lambda value (`mc` estimator only). `None` for other estimators. |

```python
# Inspect CV scores
import numpy as np
print("CV scores by r:", result.r_cv)
print("Selected r:    ", result.r)
if result.lambda_cv is not None:
    print("Lambda CV scores:", result.lambda_cv)
    print("Optimal lambda:  ", result.lambda_opt)
```

---

### Metadata

| Field | Type | Description |
|---|---|---|
| `N` | `int` | Total number of units (treated + control). |
| `T` | `int` | Number of time periods. |
| `N_tr` | `int` | Number of treated units. |
| `N_co` | `int` | Number of control units (`N - N_tr`). |
| `T0` | `dict` | Dictionary mapping each treated unit ID to its number of pre-treatment periods. |
| `treat_units` | `list` | List of treated unit IDs, in the order they appear in `Y_tr` and `Y_ct`. |
| `units` | `list` | List of all unit IDs (treated + control). |
| `times` | `list` | List of all time period values in calendar order. |
| `index` | `list` | The `index` parameter passed to `gsynth()`. |
| `Y_name` | `str` | Name of the outcome variable. |
| `D_name` | `str` | Name of the treatment variable. |
| `X_names` | `list` | Names of covariates in the order they appear in `beta`. Empty list if no covariates. |

```python
# Metadata
print(f"N={result.N}, N_tr={result.N_tr}, N_co={result.N_co}, T={result.T}")
print(f"Outcome: {result.Y_name}, Treatment: {result.D_name}")
print(f"Covariates: {result.X_names}")
print(f"Treated units: {result.treat_units}")
print(f"Pre-treatment periods: {result.T0}")
```

---

## Methods

### `summary()`

Prints a formatted text summary of the estimation results to stdout:

```python
result.summary()
```

Output includes: estimator, sample sizes, selected `r`, `att_avg` with SE/CI (if available), and a table of event-time ATTs.

### `__repr__()`

Returns a concise string representation of the result:

```python
print(result)
# <GsynthResult: estimator=gsynth, N=30, N_tr=5, T=20, r=2, att_avg=1.487>
```

---

## Example: Accessing All Major Field Groups

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

result = gsynth("Y ~ D", data=df, index=["unit", "year"],
                r=2, CV=False, se=True, nboots=200, seed=42)

# --- Treatment effects ---
print("=== Treatment Effects ===")
print(f"att_avg:  {result.att_avg:.3f}")
print(f"att_se:   {result.att_avg_se:.3f}")
print(f"att_time: {result.att_time}")
print(f"att:      {result.att.round(3)}")

# --- Counterfactuals ---
print("\n=== Counterfactuals ===")
print(f"Y_tr shape: {result.Y_tr.shape}")
print(f"Y_ct shape: {result.Y_ct.shape}")
gap_post = (result.Y_tr - result.Y_ct)[:, T0:]
print(f"Mean post-treatment gap (per unit): {gap_post.mean(axis=1).round(3)}")

# --- Model components ---
print("\n=== Model Components ===")
print(f"r:        {result.r}")
print(f"factors:  {result.factors.shape}")
print(f"loadings: {result.loadings.shape}")
print(f"alpha:    {result.alpha[:5].round(3)}")
print(f"xi:       {result.xi[:5].round(3)}")

# --- Metadata ---
print("\n=== Metadata ===")
print(f"N={result.N}, N_tr={result.N_tr}, N_co={result.N_co}, T={result.T}")
print(f"T0 (pre-treatment periods): {result.T0}")
print(f"Treated units: {result.treat_units}")
```

---

## See Also

- [`gsynth()`](gsynth.md) — produces this object
- [`plot()`](plot.md) — visualization using this object
- [`effect()`](effect.md) — effect summaries using this object
