# `effect()` — API Reference

Summarizes treatment effects over specific time windows and/or unit subgroups.

---

## Function Signature

```python
def effect(
    result,
    period=None,
    id=None,
    cumu=False,
    plot=False,
) -> dict
```

---

## Parameters

### `result`
**Type**: `GsynthResult`

A fitted result object returned by [`gsynth()`](gsynth.md). Must not be `None`.

---

### `period`
**Type**: `tuple of (int, int)` or `None` | **Default**: `None`

Event-time window over which to summarize effects. A tuple `(start, end)` with both endpoints **inclusive**. For example, `period=(0, 4)` computes effects for event times 0, 1, 2, 3, 4.

When `None` (default), all post-treatment event times are included (all `att_time >= 0`).

Note: `period` refers to event time (relative to each unit's treatment onset), not calendar time. This is consistent with `result.att_time`.

---

### `id`
**Type**: `str`, `int`, or `list` or `None` | **Default**: `None`

Unit identifier(s) to restrict the summary to a subgroup of treated units. Pass a single unit ID (as string or int, matching the unit identifier used in `data`) or a list of unit IDs.

When `None` (default), all treated units are included.

---

### `cumu`
**Type**: `bool` | **Default**: `False`

If `False` (default), the `"att_avg"` key returns the time-averaged ATT over the specified window.
If `True`, the `"att_cumulative"` key is the primary statistic and holds the cumulative sum of ATTs over the window.

Both `"att_avg"` and `"att_cumulative"` are always populated in the return dict regardless of this setting.

---

### `plot`
**Type**: `bool` | **Default**: `False`

If `True`, generates a bar chart of the period-by-period ATTs in the window and stores it in the `"fig"` key of the return dict. Each bar corresponds to one event time in the window. If `se=True` was used during estimation, error bars are drawn.

---

## Returns

A dictionary with the following keys:

| Key | Type | Description |
|---|---|---|
| `"att"` | `ndarray` | ATT at each event time within the specified window |
| `"att_cumulative"` | `float` | Cumulative sum of ATTs over the window |
| `"att_avg"` | `float` | Time-average of ATTs over the window |
| `"times"` | `ndarray` | Event-time indices for the window |
| `"n_units"` | `int` | Number of treated units included (after `id` filter) |
| `"fig"` | `Figure` or `None` | Bar chart figure if `plot=True`, else `None` |

---

## Examples

### Basic Usage

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

rows = []
for i, u in enumerate(units):
    for t_idx, yr in enumerate(times):
        D = 1 if i < N_tr and t_idx >= T0 else 0
        rows.append({"unit": u, "year": yr, "Y": Y_base[i, t_idx] + D * 1.5, "D": D})
df = pd.DataFrame(rows)

result = gsynth("Y ~ D", data=df, index=["unit", "year"], r=2, CV=False)

# Default: average over all post-treatment periods
eff = effect(result)
print(f"ATT (all post-treatment): {eff['att_avg']:.3f}")
print(f"N treated units:          {eff['n_units']}")
print(f"Event times in window:    {eff['times']}")
print(f"Cumulative ATT:           {eff['att_cumulative']:.3f}")
```

---

### Period Filter

```python
# First 3 post-treatment periods (event times 0, 1, 2)
eff_early = effect(result, period=(0, 2))
print(f"Early ATT (t=0 to t=2): {eff_early['att_avg']:.3f}")
print(f"ATT by period: {eff_early['att']}")

# A specific window (t=3 to t=7)
eff_late = effect(result, period=(3, 7))
print(f"Later ATT (t=3 to t=7): {eff_late['att_avg']:.3f}")
```

---

### Subgroup Analysis

```python
# Single treated unit
eff_u0 = effect(result, id="unit_0")
print(f"ATT for unit_0: {eff_u0['att_avg']:.3f}")

# Subgroup of treated units
eff_group = effect(result, id=["unit_0", "unit_1", "unit_2"])
print(f"ATT for group (unit_0/1/2): {eff_group['att_avg']:.3f}")
print(f"N units in group:           {eff_group['n_units']}")
```

---

### Cumulative Effect with Bar Chart

```python
# Cumulative sum over first 5 post-treatment periods, with bar chart
eff_cumu = effect(result,
                  period=(0, 4),
                  cumu=True,
                  plot=True)
print(f"Cumulative ATT (t=0 to t=4): {eff_cumu['att_cumulative']:.3f}")

# Save the bar chart
fig = eff_cumu["fig"]
fig.savefig("cumulative_effect.png", dpi=150, bbox_inches="tight")
```

---

## See Also

- [`gsynth()`](gsynth.md) — estimation
- [`plot()`](plot.md) — gap and counterfactual plots
- [`GsynthResult`](result.md) — all result fields
- [Effect Summary guide](../user-guide/effect-summary.md)
