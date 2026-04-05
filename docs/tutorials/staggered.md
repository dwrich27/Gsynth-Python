# Tutorial: Effect of Minimum Wage Increase on Employment Rate — Staggered State Adoption

This tutorial demonstrates how gsynth handles staggered adoption, where different units adopt treatment at different calendar times.

!!! warning "Synthetic demonstration data"
    All data in this tutorial are **synthetically generated** for illustration. The scenario and results do not reflect any real minimum wage study.

---

## Scenario

- **Units**: 40 US states
- **Time**: 16 years, 2005–2020
- **Treatment**: State minimum wage increase (above federal level)
- **Cohorts**: 15 states adopt in three cohorts:
  - Early adopters (5 states): adopt at year index 5 (2009)
  - Mid adopters (5 states): adopt at year index 8 (2012)
  - Late adopters (5 states): adopt at year index 11 (2015)
  - Never treated: 25 states
- **Outcome**: Log employment rate
- **True ATT**: -0.02 (log points) — modest negative employment effect

---

## Step 1: Understanding Staggered Data

In staggered adoption, the treatment indicator `D` looks like this:

| State | 2005-2008 | 2009 | 2010-2011 | 2012 | 2013-2014 | 2015 | 2016-2020 |
|---|---|---|---|---|---|---|---|
| Early adopter | 0 | 1 | 1 | 1 | 1 | 1 | 1 |
| Mid adopter | 0 | 0 | 0 | 1 | 1 | 1 | 1 |
| Late adopter | 0 | 0 | 0 | 0 | 0 | 1 | 1 |
| Never treated | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

Each treated unit has its own treatment onset; once treated, it stays treated (absorbing treatment).

---

## Step 2: Generate the Dataset

```python
import numpy as np
import pandas as pd
from gsynth import gsynth, plot, effect

rng = np.random.default_rng(2025)

N_co  = 25    # never-treated control states
T     = 16    # years 2005-2020
TAU   = -0.02  # true ATT: log employment rate

years = list(range(2005, 2005 + T))

# Three treatment cohorts: (name_prefix, n_units, onset_index)
cohorts = [
    ("early", 5,  5),   # adopt at t_idx=5 (year 2009)
    ("mid",   5,  8),   # adopt at t_idx=8 (year 2012)
    ("late",  5, 11),   # adopt at t_idx=11 (year 2015)
]

N_tr = sum(c[1] for c in cohorts)  # 15 treated states
N    = N_co + N_tr                  # 40 total

# Unit names
ctrl_names = [f"ctrl_{i:02d}" for i in range(N_co)]
treat_names = []
for prefix, n, _ in cohorts:
    treat_names += [f"{prefix}_{i:02d}" for i in range(n)]
all_names = ctrl_names + treat_names

# Latent factor structure (r=2)
alpha = rng.normal(-0.3, 0.1, N)        # log employment rate base
F     = rng.normal(0, 0.5, (T, 2))      # common factors (business cycle)
lam   = rng.normal(0, 0.3, (N, 2))      # state-specific factor loadings
eps   = rng.normal(0, 0.01, (N, T))     # small idiosyncratic noise

Y_base = alpha[:, None] + lam @ F.T + eps

# Build treatment onset index for each unit
onset_by_unit = {}
for j, name in enumerate(ctrl_names):
    onset_by_unit[name] = None  # never treated

unit_idx = N_co
for prefix, n, onset in cohorts:
    for i in range(n):
        name = f"{prefix}_{i:02d}"
        onset_by_unit[name] = onset
        unit_idx += 1

# Build long-format DataFrame
rows = []
for i, name in enumerate(all_names):
    onset = onset_by_unit[name]
    for t_idx, yr in enumerate(years):
        D = 1 if (onset is not None and t_idx >= onset) else 0
        Y = Y_base[i, t_idx] + D * TAU
        rows.append({"state": name, "year": yr, "log_emp": round(Y, 5), "min_wage": D})

df = pd.DataFrame(rows)

print(df.head(10))
print(f"\nPanel: {df['state'].nunique()} states x {df['year'].nunique()} years")
print(f"Treated: {df.groupby('state')['min_wage'].max().sum():.0f} states")
print(f"Treatment events by year:\n{df.groupby('year')['min_wage'].sum()}")
```

---

## Step 3: Estimate with gsynth

```python
result = gsynth(
    "log_emp ~ min_wage",
    data=df,
    index=["state", "year"],
    r=2,
    CV=False,
    seed=42
)
result.summary()
```

gsynth handles the staggered design automatically. Factors are estimated from the 25 never-treated states. Each treated state's loadings are estimated from its own pre-treatment observations.

---

## Step 4: Inspect the T0 Dictionary

`result.T0` tells you how many pre-treatment periods each treated unit has:

```python
print("Pre-treatment periods per treated state:")
print("-" * 40)
for state, t0 in sorted(result.T0.items()):
    cohort = state.split("_")[0]
    print(f"  {state:<12s}  ({cohort:5s} adopter)  T0 = {t0}")
```

Expected output:
```
  early_00      (early adopter)  T0 = 5
  early_01      (early adopter)  T0 = 5
  ...
  mid_00        (mid   adopter)  T0 = 8
  ...
  late_00       (late  adopter)  T0 = 11
  ...
```

Early adopters have only 5 pre-treatment periods; late adopters have 11. The `min_T0=5` default is just sufficient for early adopters (capping `r` at 3 with unit FEs).

---

## Step 5: Event-Time Alignment Across Cohorts

The key insight of the gsynth approach in staggered settings is that all cohorts are aligned to their own event time 0:

```python
print("\n--- Event-time ATTs (averaged across cohorts) ---")
print(f"{'t':>5}  {'ATT':>8}  {'note':}")
for t, a in zip(result.att_time, result.att):
    note = ""
    if t == 0:
        note = "<-- avg treatment onset"
    elif t == -5:
        note = "<-- earliest pre-period (early adopters)"
    print(f"  {t:+3d}  {a:+.4f}  {note}")
```

At any event time `t`, the ATT averages over all treated units that have data at that event time. Early adopters contribute to more event times than late adopters.

---

## Step 6: Gap Plot

```python
fig = plot(result, type="gap",
           main="Effect of Minimum Wage on Log Employment (Staggered)",
           xlab="Years relative to minimum wage adoption",
           ylab="ATT (log points)",
           ylim=(-0.08, 0.04))
fig.savefig("minwage_gap.png", dpi=150, bbox_inches="tight")
```

Observations:
- Pre-treatment ATTs (negative event times) should cluster near zero
- Post-treatment ATTs should be near -0.02 (the true effect)
- The number of units contributing to each event time declines at long horizons

---

## Step 7: Interpreting `att_avg`

```python
print(f"\nSummary:")
print(f"  att_avg (treatment-cell-weighted): {result.att_avg:.4f}")
print(f"  True ATT:                          {TAU:.4f}")
print(f"  N treated states:                  {result.N_tr}")
print(f"  N control states:                  {result.N_co}")
```

The treatment-cell-weighted average weights each (state, post-treatment period) cell equally. This implicitly gives more weight to units with longer post-treatment exposure (late adopters have fewer post-treatment periods in this dataset, so they get less weight).

---

## Step 8: Comparison with Naive Pre-Post Estimate

To appreciate what gsynth adds, compare it to a simple pre-post difference (which does not account for common trends):

```python
# Naive pre-post estimator for each treated state
naive_atts = []
for state in result.treat_units:
    state_df = df[df["state"] == state].sort_values("year")
    onset_yr = state_df[state_df["min_wage"] == 1]["year"].min()
    pre_mean  = state_df[state_df["year"] < onset_yr]["log_emp"].mean()
    post_mean = state_df[state_df["year"] >= onset_yr]["log_emp"].mean()
    naive_atts.append(post_mean - pre_mean)

naive_avg = np.mean(naive_atts)
print(f"\nNaive pre-post ATT:    {naive_avg:.4f}")
print(f"gsynth ATT:            {result.att_avg:.4f}")
print(f"True ATT:              {TAU:.4f}")
print()
print("The naive estimator conflates the treatment effect with common")
print("time trends. gsynth removes the counterfactual time trend.")
```

In most datasets with factor structure, the naive pre-post estimate will be biased. gsynth accounts for the time-varying counterfactual trend by constructing a synthetic Y(0).

---

## Step 9: Matrix Completion as Robustness Check

Use the `mc` estimator as a non-parametric alternative:

```python
result_mc = gsynth(
    "log_emp ~ min_wage",
    data=df,
    index=["state", "year"],
    estimator="mc",
    seed=42
)

print(f"\ngsynth ATT (IFE): {result.att_avg:.4f}")
print(f"mc ATT:           {result_mc.att_avg:.4f}")
print(f"True ATT:         {TAU:.4f}")
print(f"mc lambda_opt:    {result_mc.lambda_opt:.4f}")
```

Close agreement between `gsynth` and `mc` estimates suggests the results are robust to different modeling assumptions.

---

## Step 10: Plot the Missing/Treatment Pattern

The `"missing"` plot is especially useful for staggered designs — it gives an immediate visual of which states are treated when:

```python
fig = plot(result, type="missing",
           main="Treatment Status by State and Year (Staggered Adoption)")
fig.savefig("minwage_missing.png", dpi=150, bbox_inches="tight")
```

---

## Key Takeaways from This Tutorial

1. **gsynth handles staggered adoption natively** — no special configuration needed. Just ensure `D` is an absorbing 0/1 indicator.
2. **`result.T0`** shows how many pre-treatment periods each treated unit has. Check this to make sure all treated units have adequate pre-treatment data.
3. **Event-time alignment** (`att_time`) puts all cohorts on a common scale regardless of calendar adoption date.
4. **att_avg** is treatment-cell-weighted and accounts for heterogeneous post-treatment exposure lengths across cohorts.
5. **The naive pre-post estimator is biased** when common trends exist. gsynth removes the counterfactual trend, recovering a much more accurate estimate.
6. **Matrix completion (`mc`)** provides a useful robustness check that does not assume a specific factor structure.
