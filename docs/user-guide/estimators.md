# Estimators

gsynth implements three estimators, each making different assumptions about how to identify the factor model. The estimator is selected via the `estimator` parameter.

---

## Overview

| Estimator | `estimator=` | Factor estimation data | Best for |
|---|---|---|---|
| Generalized Synthetic Control | `"gsynth"` | Control group only | Most applications; conservative |
| Interactive Fixed Effects | `"ife"` | All pre-treatment obs | Efficiency when anticipation is absent |
| Matrix Completion | `"mc"` | All control obs | When factor structure is unclear |

---

## `gsynth` — Generalized Synthetic Control (Default)

**Mathematical intuition**: Factors `F_t` are estimated exclusively from the control group using PCA on de-meaned outcomes. Treated units' loadings `lambda_i` are then estimated by regressing each treated unit's pre-treatment outcomes on the estimated factors. The counterfactual is `alpha_i_hat + lambda_i_hat @ F_t_hat`.

Because the factors come only from control units, the treated units' post-treatment outcomes play no role in factor estimation. This is the key identification advantage: treatment effects cannot "leak" into factor estimates.

**When to use**: This is the default and the safest choice for most applied settings. It is most similar to the original synthetic control in spirit — treated units are compared to a weighted combination of control units — but allows for time-varying weights (through the factor structure).

```python
result_gsynth = gsynth("Y ~ D", data=df, index=["unit", "year"],
                        estimator="gsynth", r=2, CV=False)
print(f"gsynth ATT: {result_gsynth.att_avg:.3f}")
```

---

## `ife` — Interactive Fixed Effects (ALS)

**Mathematical intuition**: Factors and loadings are estimated jointly from all pre-treatment observations (both treated and control units) using Alternating Least Squares (ALS). This is the standard IFE estimator from Bai (2009). The algorithm alternates between:
1. Fixing `F`, regressing out `alpha_i` and `lambda_i` via OLS.
2. Fixing `lambda`, estimating `F` via OLS.

Convergence is achieved when parameter updates fall below tolerance `tol`.

**When to use**: When you have many treated units and believe there is no treatment anticipation (no pre-treatment effect), `ife` uses more data for factor estimation and can be more efficient. In datasets with few control units relative to treated units, `ife` may outperform `gsynth`.

!!! warning "Efficiency vs robustness trade-off"
    `ife` uses treated pre-treatment data in factor estimation. If there is treatment anticipation (outcomes changing before the official treatment date), this can bias the factors and hence the counterfactuals. Inspect the pre-treatment ATT (t < 0 in the gap plot) carefully.

```python
result_ife = gsynth("Y ~ D", data=df, index=["unit", "year"],
                     estimator="ife", r=2, CV=False)
print(f"ife ATT: {result_ife.att_avg:.3f}")
```

---

## `mc` — Matrix Completion

**Mathematical intuition**: The panel matrix of outcomes is treated as a partially observed low-rank matrix. Missing entries (post-treatment treated cells, which represent the missing counterfactual) are imputed by minimizing a nuclear-norm-penalized loss:

```
min_{M}  ||W * (Y - M)||_F^2  +  lambda * ||M||_*
```

where `W` is the mask matrix (0 for treated post-treatment cells, 1 otherwise), `M` is the completed matrix, and `||.||_*` is the nuclear norm (sum of singular values). The regularization parameter `lambda` is selected by cross-validation.

**When to use**: When you are unsure about the number of factors, or when the panel is large and sparse. The nuclear norm penalty automatically induces low-rank structure without requiring you to specify `r` explicitly. Also useful as a robustness check against the factor-model-based estimators.

```python
result_mc = gsynth("Y ~ D", data=df, index=["unit", "year"],
                    estimator="mc")
print(f"mc ATT: {result_mc.att_avg:.3f}")
print(f"Optimal lambda: {result_mc.lambda_opt:.4f}")
```

---

## Comparing All Three

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
TAU = 1.5

rows = []
for i, u in enumerate(units):
    is_treated = i < N_tr
    for t_idx, yr in enumerate(times):
        D = 1 if is_treated and t_idx >= T0 else 0
        Y = Y_base[i, t_idx] + D * TAU
        rows.append({"unit": u, "year": yr, "Y": Y, "D": D})
df = pd.DataFrame(rows)

for est in ["gsynth", "ife", "mc"]:
    kwargs = {"r": 2, "CV": False} if est != "mc" else {}
    res = gsynth("Y ~ D", data=df, index=["unit", "year"], estimator=est, **kwargs)
    print(f"{est:8s}  ATT = {res.att_avg:.3f}")
```

---

## Practical Guidance

**Start with `gsynth`**. It is the most conservative and the most interpretable. If you have a theoretical reason to believe `ife` is more appropriate (many treated units, no anticipation), try it as a robustness check. Use `mc` as an additional robustness check, particularly when you want to avoid specifying `r`.

In most applied settings, all three estimators should produce qualitatively similar results if the data has a true low-rank factor structure. Large discrepancies are a signal to investigate data quality or model specification.

!!! tip "Default inference"
    The default `inference` mode differs by estimator: `"parametric"` for `gsynth`, `"nonparametric"` for `ife` and `mc`. See [Inference](inference.md) for details.
