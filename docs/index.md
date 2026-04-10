# gsynth — Generalized Synthetic Control for Python

!!! note "Credit where it's due"
    This package is a Python port of the [`gsynth`](https://yiqingxu.org/packages/gsynth/) and [`fect`](https://yiqingxu.org/packages/fect/) R packages by **[Yiqing Xu](https://yiqingxu.org/)** (Stanford) and collaborators. The methodology, estimation algorithm, and validation benchmarks are entirely his work. If this package is useful to you, please cite the original papers below and consider using his R packages directly — they are more mature, better tested, and actively maintained. This Python port exists to make the method accessible to Python-first workflows.

**gsynth** estimates causal effects in panel data settings using Generalized Synthetic Control methods — combining the intuition of synthetic control with the flexibility of interactive fixed effects (IFE) models.

---

## Installation

```python
pip install git+https://github.com/dwrich27/Gsynth-Python.git
```

---

## Five-Minute Example

```python
import numpy as np
import pandas as pd
from gsynth import gsynth, plot, effect

# --- generate a small panel dataset ---
rng = np.random.default_rng(42)
N, T, N_tr, T0 = 30, 20, 5, 10
units = [f"unit_{i}" for i in range(N)]
times = list(range(2000, 2020))

alpha = rng.normal(0, 1, N)
F = rng.normal(0, 1, (T, 2))
lam = rng.normal(0, 1, (N, 2))
eps = rng.normal(0, 0.5, (N, T))
Y_base = alpha[:, None] + lam @ F.T + eps
TAU = 1.5

rows = []
for i, u in enumerate(units):
    is_treated = i < N_tr
    for t_idx, yr in enumerate(times):
        D = 1 if is_treated and t_idx >= T0 else 0
        Y = Y_base[i, t_idx] + D * TAU
        rows.append({"unit": u, "year": yr, "Y": Y, "D": D})

df = pd.DataFrame(rows)

# --- estimate ---
result = gsynth("Y ~ D", data=df, index=["unit", "year"], r=2, CV=False)
result.summary()

# --- visualize and summarize ---
plot(result, type="gap")
eff = effect(result)
print(f"Average treatment effect: {eff['att_avg']:.3f}")
```

---

## Model

gsynth fits the interactive fixed effects model:

```
Y_it = alpha_i + lambda_i * F_t + D_it * tau_it + beta * X_it + eps_it
```

where:

| Symbol | Meaning |
|---|---|
| `alpha_i` | Unit fixed effects |
| `lambda_i` | Factor loadings (N x r) |
| `F_t` | Latent common factors (T x r) |
| `D_it` | Treatment indicator (binary, absorbing) |
| `tau_it` | Unit-time treatment effect |
| `beta` | Covariate coefficients |
| `X_it` | Time-varying covariates |
| `eps_it` | Idiosyncratic error |

The counterfactual Y(0) for treated units is imputed as `alpha_i + lambda_i_hat * F_t_hat + beta_hat * X_it`, with factors and loadings estimated from control-group pre-treatment data.

---

## Features

| Feature | gsynth Python |
|---|---|
| **Estimators** | `gsynth` (factors from control), `ife` (ALS on all pre-treatment), `mc` (matrix completion) |
| **Fixed effects** | `none`, `unit`, `time`, `two-way` |
| **Cross-validation** | LOO (gsynth), k-fold (ife/mc), criterion: MSPE or PC |
| **Inference** | Parametric bootstrap, nonparametric block bootstrap, jackknife |
| **Staggered adoption** | Yes — each unit has its own T0; event-time ATT alignment |
| **Covariates** | Time-varying covariates partialled out via OLS on controls |
| **Plot types** | `gap`, `raw`, `counterfactual`, `factors`, `loadings`, `missing` |
| **Weights** | Unit-level analytic / IPW weights |
| **Cluster bootstrap** | Yes, via `cl` parameter |
| **Pure NumPy** | No scipy dependency |

---

## Estimators at a Glance

- **`gsynth`** (default): Estimates factors from the control group only, then projects treated units onto the factor space. Closest in spirit to the original synthetic control method. Best choice when treatment effect heterogeneity is a concern.
- **`ife`**: Interactive Fixed Effects via Alternating Least Squares. Uses all pre-treatment data (treated and control) to estimate factors. More efficient but assumes no treatment anticipation in the pre-period.
- **`mc`**: Matrix completion with nuclear-norm regularization. Non-parametric approach, does not assume a low-rank factor structure explicitly. Works well when the number of factors is unclear.

---

## User Guide

- [Data Format](user-guide/data-format.md) — required columns, treatment indicator rules, missing data
- [Basic Estimation](user-guide/basic-estimation.md) — formula interface, interpreting results
- [Estimators](user-guide/estimators.md) — gsynth vs ife vs mc
- [Fixed Effects](user-guide/fixed-effects.md) — the `force` parameter
- [Covariates](user-guide/covariates.md) — time-varying covariates in the formula
- [Cross-Validation](user-guide/cross-validation.md) — selecting r and lambda
- [Inference](user-guide/inference.md) — bootstrap and jackknife standard errors
- [Staggered Adoption](user-guide/staggered-adoption.md) — multiple treatment onset times
- [Visualization](user-guide/visualization.md) — six plot types
- [Effect Summary](user-guide/effect-summary.md) — the `effect()` function

---

## Credits and Citation

This Python package implements the methods developed by **Yiqing Xu** (Stanford University) and collaborators. The methodology, algorithms, and the R reference implementations ([`gsynth`](https://yiqingxu.org/packages/gsynth/), [`fect`](https://yiqingxu.org/packages/fect/)) are entirely their work.

If you use this package in published research, **please cite the original papers**:

**Core GSC methodology:**
> Xu, Y. (2017). Generalized Synthetic Control Method: Causal Inference with Interactive Fixed Effects Models. *Political Analysis*, 25(1), 57–76. doi:[10.1017/pan.2016.2](https://doi.org/10.1017/pan.2016.2)

**Panel estimation with staggered treatment and covariates (fect):**
> Liu, L., Wang, Y., & Xu, Y. (2024). A Practical Guide to Counterfactual Estimators for Causal Inference with Time-Series Cross-Sectional Data. *American Journal of Political Science*, 68(1), 160–176. doi:[10.1111/ajps.12723](https://doi.org/10.1111/ajps.12723)

Yiqing Xu's work has made rigorous causal inference with panel data dramatically more accessible. His R packages remain the authoritative implementations — use them if you can.

---

## Tutorials

- [Quick Start](tutorials/quickstart.md) — get running in 5 minutes
- [Political Economy Example](tutorials/election.md) — campaign finance reform
- [Staggered Adoption](tutorials/staggered.md) — minimum wage with staggered rollout
