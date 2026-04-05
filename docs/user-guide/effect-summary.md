# Effect Summary

The `effect()` function provides a flexible way to summarize treatment effects over specific time windows and/or unit subgroups.

---

## Purpose

While `result.att_avg` gives the overall treatment-cell-weighted average, `effect()` lets you:

- Restrict to a specific post-treatment time window (e.g., only the first 3 years)
- Restrict to a subgroup of treated units
- Compute a cumulative (summed) rather than average effect
- Generate a bar chart of the effects

---

## Basic Usage

```python
from gsynth import effect

eff = effect(result)
print(eff)
```

By default, `effect()` averages over all post-treatment periods (event times where `att_time >= 0`).

The return value is a dictionary:

| Key | Type | Description |
|---|---|---|
| `"att"` | `ndarray` | ATT at each event time in the window |
| `"att_cumulative"` | `float` | Cumulative (summed) ATT over the window |
| `"att_avg"` | `float` | Average ATT over the window |
| `"times"` | `ndarray` | Event-time indices for the window |
| `"n_units"` | `int` | Number of treated units included |
| `"fig"` | `Figure` or `None` | Bar chart figure (if `plot=True`) |

---

## Time Window: `period`

Use `period` to restrict to a specific event-time range. It accepts a tuple `(start, end)` where both endpoints are inclusive:

```python
# Only the first 3 post-treatment periods (t=0, 1, 2)
eff_early = effect(result, period=(0, 2))
print(f"Average ATT (t=0 to t=2): {eff_early['att_avg']:.3f}")

# A specific window (t=3 to t=6)
eff_mid = effect(result, period=(3, 6))
print(f"Average ATT (t=3 to t=6): {eff_mid['att_avg']:.3f}")
```

When `period=None` (default), all post-treatment event times are included.

---

## Unit Subgroup: `id`

Use `id` to restrict to one or more specific treated units:

```python
# Single unit
eff_unit0 = effect(result, id="unit_0")
print(f"ATT for unit_0: {eff_unit0['att_avg']:.3f}")

# Subgroup of units
eff_group = effect(result, id=["unit_0", "unit_1", "unit_2"])
print(f"ATT for group: {eff_group['att_avg']:.3f}")
print(f"N units in group: {eff_group['n_units']}")
```

This is useful for checking effect heterogeneity across subgroups of treated units.

---

## Cumulative Effect: `cumu`

By default (`cumu=False`), `effect()` returns time-averaged ATT. Set `cumu=True` to return the cumulative sum instead:

```python
# Cumulative effect summed over all post-treatment periods
eff_cumu = effect(result, cumu=True)
print(f"Cumulative ATT (all post-periods): {eff_cumu['att_cumulative']:.3f}")

# Cumulative over first 5 periods
eff_cumu5 = effect(result, period=(0, 4), cumu=True)
print(f"Cumulative ATT (t=0 to t=4): {eff_cumu5['att_cumulative']:.3f}")
```

The `"att_cumulative"` key holds the sum; `"att_avg"` still holds the mean.

---

## Bar Chart: `plot`

Set `plot=True` to generate a bar chart of the period-by-period ATTs in the window:

```python
eff = effect(result, plot=True)
fig = eff["fig"]
fig.savefig("effect_bar.png", dpi=150, bbox_inches="tight")
```

The bar chart shows one bar per event time, with bars colored by sign (positive = blue, negative = red, following a common convention). If CI data is available (from `se=True`), error bars are drawn.

---

## Combining Period and Subgroup

You can combine `period` and `id` to compute subgroup effects over a specific time window:

```python
eff = effect(result,
             period=(0, 5),
             id=["unit_0", "unit_1"],
             cumu=False,
             plot=True)
print(f"Group ATT (t=0–5): {eff['att_avg']:.3f}")
```

---

## Full Example

```python
import numpy as np
import pandas as pd
from gsynth import gsynth, effect

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
result = gsynth("Y ~ D", data=df, index=["unit", "year"], r=2, CV=False)

# 1. Overall summary
eff_all = effect(result)
print(f"Overall ATT: {eff_all['att_avg']:.3f}")
print(f"N treated units: {eff_all['n_units']}")

# 2. Early post-treatment window
eff_early = effect(result, period=(0, 2))
print(f"Early ATT (t=0-2): {eff_early['att_avg']:.3f}")

# 3. Cumulative effect
eff_cumu = effect(result, cumu=True)
print(f"Cumulative ATT: {eff_cumu['att_cumulative']:.3f}")

# 4. Single unit subgroup
eff_u0 = effect(result, id="unit_0")
print(f"unit_0 ATT: {eff_u0['att_avg']:.3f}")

# 5. With bar chart
eff_plot = effect(result, plot=True)
eff_plot["fig"].savefig("att_by_period.png", dpi=150, bbox_inches="tight")
```

---

## Notes

!!! note "Relation to `result.att_avg`"
    `effect(result)["att_avg"]` is equivalent to `result.att_avg` when no `period` or `id` filter is applied. Use `effect()` when you need filtered or cumulative summaries.

!!! note "Staggered adoption and `period`"
    In staggered adoption settings, `period` refers to event-time (relative to each unit's own treatment onset), not calendar time. This is consistent with how `result.att_time` is defined.
