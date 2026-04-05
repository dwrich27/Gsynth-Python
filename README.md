# gsynth — Generalized Synthetic Control Method (Python)

A pure-NumPy Python port of the R [`gsynth`](https://yiqingxu.org/packages/gsynth/) package by Yiqing Xu (2017).

Estimates counterfactual outcomes and average treatment effects on the treated (ATT) for panel data using **Interactive Fixed Effects (IFE)** and **Matrix Completion (MC)** models.

---

## Features

| Feature | Details |
|---------|---------|
| **Estimators** | `gsynth` (default), `ife` (EM algorithm), `mc` (nuclear-norm matrix completion) |
| **Fixed effects** | `none`, `unit`, `time`, `two-way` |
| **Factor selection** | Cross-validation over r ∈ {0…5} or user-specified grid |
| **Lambda selection** | CV over log-spaced sequence (MC estimator) |
| **Inference** | Parametric bootstrap, nonparametric block bootstrap, jackknife |
| **Treatment** | Binary (0/1), simultaneous or staggered adoption |
| **Covariates** | Time-varying covariates (partially out before IFE) |
| **Weights** | Unit-level analytic / IPW weights |
| **Clustering** | Block bootstrap clustered by any unit-level variable |
| **Plot types** | `gap`, `raw`, `counterfactual`, `factors`, `loadings`, `missing` |
| **Effect summary** | Cumulative and subgroup ATT via `effect()` |
| **Dependencies** | NumPy ≥ 1.24, pandas ≥ 1.5, matplotlib ≥ 3.6 (for plots) |

---

## Installation

```bash
pip install gsynth          # once published on PyPI
# or directly from source:
pip install .
```

---

## Quick Start

```python
import pandas as pd
from gsynth import gsynth, plot, effect

# Long-format panel data
df = pd.read_csv("panel_data.csv")

# --- Basic estimation ---
result = gsynth(
    formula="Y ~ D",
    data=df,
    index=["unit_id", "year"],   # [unit column, time column]
    force="unit",                 # unit fixed effects
    r=0,                          # r=0 + CV=True → auto-select
    CV=True,
    se=True,
    nboots=200,
    seed=42,
)

print(result.summary())

# --- Plots ---
plot(result, type="gap")                   # period-by-period ATT
plot(result, type="counterfactual")        # actual vs Y(0)
plot(result, type="factors", nfactors=2)  # latent factors
plot(result, type="loadings")             # factor loadings
plot(result, type="raw")                  # raw outcome trajectories
plot(result, type="missing")              # data / treatment status

# --- Cumulative effect ---
eff = effect(result, cumu=True, period=(2010, 2015))
print(f"Cumulative ATT 2010–2015: {eff['att_cumulative']:.3f}")

# --- Subgroup effect ---
eff_sub = effect(result, id=["unit_1", "unit_2"])
```

---

## All Parameters

### `gsynth()`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `formula` | str | None | R-style formula `"Y ~ D + X1 + X2"` |
| `data` | DataFrame | **required** | Long-format panel |
| `Y` | str | None | Outcome column |
| `D` | str | None | Binary treatment column (0/1) |
| `X` | str or list | None | Time-varying covariate column(s) |
| `index` | list[str, str] | **required** | `[unit_col, time_col]` |
| `na_rm` | bool | False | Drop rows with any missing value |
| `weight` | str | None | Unit weight column |
| `force` | str | `"unit"` | Fixed effects: `"none"`, `"unit"`, `"time"`, `"two-way"` |
| `cl` | str | None | Cluster variable for block bootstrap |
| `r` | int or list | 0 | Number of factors (CV searches 0–5 when `CV=True, r=0`) |
| `lam` | float or list | None | Regularisation lambda (MC only) |
| `nlambda` | int | 10 | Lambda grid length for CV |
| `CV` | bool | True | Enable cross-validation |
| `criterion` | str | `"mspe"` | CV criterion: `"mspe"` or `"pc"` |
| `k` | int | 5 | Number of CV folds |
| `EM` | bool | False | Use EM (equivalent to `estimator="ife"`) |
| `estimator` | str | `"gsynth"` | `"gsynth"`, `"ife"`, or `"mc"` |
| `se` | bool | False | Compute bootstrap standard errors |
| `nboots` | int | 200 | Bootstrap replications |
| `inference` | str | None | `"parametric"`, `"nonparametric"`, or `"jackknife"` |
| `tol` | float | 1e-3 | ALS/MC convergence tolerance |
| `seed` | int | None | Random seed |
| `min_T0` | int | 5 | Minimum pre-treatment periods |
| `alpha` | float | 0.05 | Significance level for CIs |
| `normalize` | bool | False | Standardise Y and X before estimation |

### `plot(result, type=...)`

| `type` | Description |
|--------|-------------|
| `"gap"` | Period-by-period ATT with CI band |
| `"raw"` | Raw outcome trajectories (treated units) |
| `"counterfactual"` / `"ct"` | Actual vs imputed Y(0) |
| `"factors"` | Estimated latent factors over time |
| `"loadings"` | Factor loadings across units |
| `"missing"` | Treatment status / data availability heatmap |

Common `plot()` options: `xlim`, `ylim`, `xlab`, `ylab`, `main`, `legendOff`, `theme_bw`, `shade_post`, `id`, `nfactors`, `raw`, `figsize`, `show`.

### `effect(result, ...)`

| Parameter | Default | Description |
|-----------|---------|-------------|
| `cumu` | True | Compute cumulative sum of ATT |
| `period` | None | `(start, end)` time range |
| `id` | None | Restrict to unit subgroup |
| `plot` | False | Return bar-plot figure |

---

## Output Object (`GsynthResult`)

```python
result.att_avg          # float — overall ATT (average across post periods)
result.att              # ndarray (T_post,) — period-by-period ATT
result.att_time         # ndarray (T_post,) — time indices for ATT
result.Y_tr             # ndarray (N_tr, T) — observed treated outcomes
result.Y_ct             # ndarray (N_tr, T) — counterfactual outcomes Y(0)
result.factors          # ndarray (T, r) — estimated latent factors
result.loadings         # ndarray (N, r) — estimated factor loadings
result.alpha            # ndarray (N,) — unit fixed effects
result.xi               # ndarray (T,) — time fixed effects
result.beta             # ndarray (K,) — covariate coefficients
result.r                # int — number of factors used
result.lambda_opt       # float — optimal lambda (MC only)
result.mse              # float — in-sample MSE
result.sigma2           # float — estimated error variance
result.att_avg_se       # float — SE of overall ATT (if se=True)
result.att_avg_ci_lower # float — CI lower bound
result.att_avg_ci_upper # float — CI upper bound
result.att_se           # ndarray (T_post,) — per-period SE
result.att_ci_lower     # ndarray (T_post,) — per-period CI lower
result.att_ci_upper     # ndarray (T_post,) — per-period CI upper
result.att_boot         # ndarray (nboots, T_post) — bootstrap draws
result.r_cv             # ndarray — CV scores per r candidate
result.lambda_cv        # ndarray — CV scores per lambda candidate
result.N, result.T      # int — panel dimensions
result.N_tr, result.N_co# int — treated / control unit counts
result.T0               # dict — {unit_id: T0} pre-treatment periods
result.treat_units      # ndarray — treated unit identifiers
```

---

## Model

The `gsynth` estimator fits an Interactive Fixed Effects (IFE) model:

```
Y_it = α_i + λ_i · F_t + D_it · τ_it + β · X_it + ε_it
```

where **α_i** are unit fixed effects, **λ_i** are unit-specific factor loadings, **F_t** are latent time factors, and **τ_it** is the heterogeneous treatment effect.

**Estimation steps:**
1. Partial out fixed effects (demean by unit, time, or both)
2. Partial out time-varying covariates X via OLS on control units
3. Estimate latent factors F from control units via ALS
4. Project treated units' pre-treatment data onto the factor space to recover λ_i
5. Impute counterfactuals: Ŷ(0)_it = λ̂_i · F̂_t
6. Compute ATT: τ̂_it = Y_it − Ŷ(0)_it  (post-treatment)

---

## Reference

Xu, Y. (2017). Generalized Synthetic Control Method: Causal Inference with
Interactive Fixed Effects Models. *Political Analysis*, 25(1), 57–76.
[doi:10.1017/pan.2016.2](https://doi.org/10.1017/pan.2016.2)
