# Basic Estimation

## Formula Interface vs Keyword Interface

gsynth supports two equivalent ways to specify the outcome and treatment:

**Formula interface** (recommended):

```python
result = gsynth("Y ~ D", data=df, index=["unit", "year"])
```

The formula `"Y ~ D"` tells gsynth that `Y` is the outcome and `D` is the treatment indicator. Additional covariates go on the right-hand side: `"Y ~ D + X1 + X2"`.

**Keyword interface**:

```python
result = gsynth(data=df, Y="Y", D="D", index=["unit", "year"])
```

Both forms are equivalent. The formula interface is more concise and is used throughout this documentation.

---

## Minimal Example

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

# Two lines to estimate and inspect:
result = gsynth("Y ~ D", data=df, index=["unit", "year"], r=2, CV=False)
result.summary()
```

---

## Interpreting GsynthResult

### `att_avg`: The Summary Statistic

`result.att_avg` is the treatment-cell-weighted average ATT — a single number summarizing the overall average treatment effect across all treated units and post-treatment periods:

```python
print(f"Average Treatment Effect: {result.att_avg:.3f}")
```

This is the primary estimand. It is a weighted average of `tau_it` over all (i, t) cells where `D_it = 1`.

### `att`: Period-by-Period Effects

`result.att` is a 1-D array of average treatment effects indexed by **event time**. Event time 0 is the first treatment period; negative event times are pre-treatment periods.

```python
import numpy as np

for t, a in zip(result.att_time, result.att):
    label = "pre" if t < 0 else "post"
    print(f"  t={t:+d}  ({label})  ATT = {a:.3f}")
```

### `att_time`: Event-Time Index

`result.att_time` is the array of event-time values corresponding to entries in `result.att`. It always contains 0 (the onset period), negative integers for pre-treatment lags, and positive integers for post-treatment leads.

---

## ATT Sign Convention

A **positive** `att_avg` means the treatment **increased** the outcome `Y`. A negative value means the treatment decreased `Y`. There is no direction assumption built in.

---

## Event-Time Convention

| Event time | Meaning |
|---|---|
| `t < 0` | Pre-treatment period (should be near zero under parallel trends) |
| `t = 0` | First period of treatment (treatment onset) |
| `t > 0` | Post-treatment period |

Under the null of no treatment effect, pre-treatment ATTs (t < 0) should be close to zero. Systematic deviation from zero in the pre-period indicates failure of the parallel trends / factor model assumption. This is the **placebo check** built into the gap plot.

In staggered adoption settings, all treated units are aligned to their own t=0, so the event-time ATT averages across heterogeneous adoption cohorts.

---

## result.summary() Walkthrough

Calling `result.summary()` prints a formatted table:

```
=== gsynth Results ===

Estimator   : gsynth
N (total)   : 30
N_tr        : 5
N_co        : 25
T           : 20
r           : 2

Average Treatment Effect (ATT):
  att_avg   : 1.487  (se=nan, 95% CI: [nan, nan])

Pre-treatment ATT (placebo, should be ~0):
  t=-9 : -0.023
  t=-8 :  0.041
  ...

Post-treatment ATT:
  t= 0 :  1.502
  t= 1 :  1.491
  ...
```

When `se=True`, standard errors and confidence intervals are populated. Without `se=True`, those fields show `nan`.

---

## Accessing Result Fields Programmatically

All result fields are accessible as attributes:

```python
# Sample sizes
print(result.N, result.N_tr, result.N_co, result.T)

# Event-time ATT array
print(result.att)          # shape: (T_all,)
print(result.att_time)     # event-time indices

# Scalar summary
print(result.att_avg)

# Model components
print(result.factors.shape)    # (T, r)
print(result.loadings.shape)   # (N, r)
print(result.alpha)            # unit FEs, shape (N,)
print(result.xi)               # time FEs, shape (T,)

# Counterfactuals for treated units
print(result.Y_tr.shape)   # (N_tr, T) — observed
print(result.Y_ct.shape)   # (N_tr, T) — imputed Y(0)

# ATT = Y_tr - Y_ct (post-treatment columns)
import numpy as np
att_matrix = result.Y_tr - result.Y_ct
print(att_matrix[:, result.att_time >= 0])  # post-treatment cells

# Pre-treatment periods per unit (staggered adoption)
print(result.T0)  # dict: {unit_id: n_pre_periods}

# Metadata
print(result.Y_name, result.D_name)
print(result.units[:5])
print(result.times)
```

---

## Cross-Validation Output

When `CV=True` (the default), gsynth selects the number of factors `r` automatically. You can inspect the CV scores to understand how the selection was made:

```python
# r_cv[k] = MSPE for r=k
print(result.r_cv)

# Optimal r selected
print(result.r)
```

See [Cross-Validation](cross-validation.md) for a full explanation.
