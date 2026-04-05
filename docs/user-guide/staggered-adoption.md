# Staggered Adoption

Staggered adoption (or staggered rollout) occurs when different treated units adopt the treatment at different calendar time periods. gsynth handles staggered adoption natively.

---

## What Staggered Adoption Means

In a non-staggered design, all treated units switch from `D=0` to `D=1` at the same calendar time. In staggered adoption, each treated unit has its own treatment onset date. For example:

| Unit | Treatment onset |
|---|---|
| State A | 2002 |
| State B | 2005 |
| State C | 2005 |
| State D | 2010 |
| Control states | Never treated |

Each treated unit's pre-treatment period is defined relative to its own onset date, not a common calendar date.

---

## How gsynth Handles Staggered Adoption

gsynth handles staggered adoption with no special configuration required. The key steps are:

1. **Factor estimation**: Latent factors `F_t` are estimated from all control-unit observations (all calendar periods). This pools information across the full time series of control units.
2. **Loading estimation**: For each treated unit `i`, loading `lambda_i` is estimated from unit `i`'s own pre-treatment observations (however many that unit has, determined by its own onset date).
3. **Counterfactual imputation**: The counterfactual for treated unit `i` in each post-treatment period is `alpha_i + lambda_i @ F_t + beta @ X_it`, using the jointly estimated factors.
4. **Event-time alignment**: All unit-level ATTs are aligned to event time 0 = treatment onset, regardless of the calendar date of onset.

---

## Event-Time Representation

The `att` array and `att_time` array represent event-time averages:

- `att_time = -k` for `k` periods before treatment onset
- `att_time = 0` for the treatment onset period
- `att_time = k` for `k` periods after treatment onset

In staggered adoption, the event-time ATT at `t=k` averages across all treated units that have at least `k` post-treatment periods. Units with fewer post-treatment periods drop out at later event times, so the effective sample size can decrease at long horizons.

---

## `att_avg` Interpretation

`result.att_avg` is the **treatment-cell-weighted** average ATT: it averages `tau_it` over all cells `(i, t)` where `D_it = 1`. This weighting scheme accounts for:

- Different numbers of post-treatment periods per unit (units with longer post-treatment exposure get more weight)
- Heterogeneous adoption cohorts

This is consistent with the estimand in Xu (2017) and is comparable to the ATT estimand in heterogeneous treatment effect contexts.

---

## The `T0` Dictionary

`result.T0` is a dictionary mapping each treated unit ID to its number of pre-treatment periods:

```python
print(result.T0)
# {'state_A': 21, 'state_B': 18, 'state_C': 18, 'state_D': 9}
```

Units with fewer pre-treatment periods (small values in `T0`) contribute less reliable loading estimates. Make sure `min_T0` is set to exclude units with very short pre-treatment windows:

```python
result = gsynth("Y ~ D", data=df, index=["unit", "year"],
                min_T0=5)  # drop treated units with < 5 pre-treatment periods
```

---

## `min_T0` in Staggered Settings

In staggered adoption, late adopters naturally have fewer pre-treatment periods. The `min_T0` parameter drops treated units that fall below the threshold:

```python
result = gsynth("Y ~ D", data=df, index=["unit", "year"],
                min_T0=8)
if result.N_tr < original_n_tr:
    print("Some treated units were dropped due to min_T0 constraint")
```

Dropped units appear in the console output as a warning. This is important to check: if your latest adopters are systematically different from early adopters, dropping them could introduce selection bias.

---

## Code Example: Staggered Dataset

```python
import numpy as np
import pandas as pd
from gsynth import gsynth, plot

rng = np.random.default_rng(42)
N_co, T = 30, 20
units_co = [f"ctrl_{i}" for i in range(N_co)]
times = list(range(2000, 2000 + T))

# Three treated cohorts: adopt at t_idx = 5, 8, 11
cohorts = [
    ("treat_early_0", 5),
    ("treat_early_1", 5),
    ("treat_mid_0",   8),
    ("treat_mid_1",   8),
    ("treat_late_0", 11),
    ("treat_late_1", 11),
]

alpha = rng.normal(0, 1, N_co + len(cohorts))
F = rng.normal(0, 1, (T, 2))
lam = rng.normal(0, 1, (N_co + len(cohorts), 2))
eps = rng.normal(0, 0.5, (N_co + len(cohorts), T))
Y_base = alpha[:, None] + lam @ F.T + eps
TAU = 1.5

rows = []
all_units = units_co + [c[0] for c in cohorts]

# Control units
for i, u in enumerate(units_co):
    for t_idx, yr in enumerate(times):
        rows.append({"unit": u, "year": yr, "Y": Y_base[i, t_idx], "D": 0})

# Treated cohorts
for j, (u, onset) in enumerate(cohorts):
    i = N_co + j
    for t_idx, yr in enumerate(times):
        D = 1 if t_idx >= onset else 0
        Y = Y_base[i, t_idx] + D * TAU
        rows.append({"unit": u, "year": yr, "Y": Y, "D": D})

df = pd.DataFrame(rows)

# Estimate
result = gsynth("Y ~ D", data=df, index=["unit", "year"], r=2, CV=False)

# Check pre-treatment periods per treated unit
print("Pre-treatment periods per treated unit:")
for u, t0 in result.T0.items():
    print(f"  {u}: {t0} periods")

# Event-time ATT
print("\nEvent-time ATT:")
for t, a in zip(result.att_time, result.att):
    marker = "<-- onset" if t == 0 else ""
    print(f"  t={t:+3d}  ATT={a:.3f}  {marker}")

# Plot
plot(result, type="gap")
```

---

## Practical Advice for Staggered Settings

1. **Inspect `result.T0`** before trusting results. Units with very few pre-treatment periods produce unreliable loadings.
2. **Set `min_T0` conservatively**. A minimum of 5–8 pre-treatment periods is a reasonable floor; more is better.
3. **Check the pre-treatment placebo** (negative event times in the gap plot). With staggered adoption, the pre-treatment ATT represents an average across units at different lags from their treatment onset; systematic deviation indicates model misfit.
4. **Event-time sample sizes shrink at long horizons**. Very late post-treatment event times may be based on only a few units; interpret them with caution.

!!! note "Never-treated vs not-yet-treated controls"
    gsynth uses all units with `D=0` at a given calendar time as controls. Units that are not yet treated (but will be treated later) are included as controls for earlier adopters during their pre-treatment periods. This is different from some DiD estimators that restrict to never-treated controls. If this is a concern, subset your data to exclude not-yet-treated units from the control pool.
