# Quick Start Tutorial

This tutorial walks you through a complete gsynth workflow in about 5 minutes using a synthetic panel dataset — no external data files required.

---

## Step 1: Generate a Synthetic Panel Dataset

We create a 30-unit, 20-period panel with 5 treated units. Units are named `unit_0` through `unit_29`; time periods are years 2000–2019. Treatment begins at period index 10 (year 2009) for treated units. The true treatment effect is `TAU = 1.5`.

The data-generating process follows the interactive fixed effects model:

```
Y_it = alpha_i + lambda_i @ F_t + D_it * TAU + eps_it
```

```python
import numpy as np
import pandas as pd
from gsynth import gsynth, plot, effect

# Reproducible random number generator
rng = np.random.default_rng(42)

# Dimensions
N     = 30    # total units
T     = 20    # time periods
N_tr  = 5     # treated units
T0    = 10    # pre-treatment periods (treatment starts at index T0)
TAU   = 1.5   # true treatment effect

units = [f"unit_{i}" for i in range(N)]
times = list(range(2000, 2000 + T))  # years 2000-2019

# IFE data-generating process
# alpha_i: unit-level intercepts (unit fixed effects)
alpha = rng.normal(0, 1, N)

# F: T x 2 matrix of latent common factors
F = rng.normal(0, 1, (T, 2))

# lam: N x 2 matrix of factor loadings (unit-specific factor sensitivities)
lam = rng.normal(0, 1, (N, 2))

# eps: idiosyncratic noise
eps = rng.normal(0, 0.5, (N, T))

# Baseline outcome (no treatment)
Y_base = alpha[:, None] + lam @ F.T + eps

# Build long-format DataFrame
rows = []
for i, u in enumerate(units):
    is_treated = i < N_tr     # first N_tr units are treated
    for t_idx, yr in enumerate(times):
        # Treatment kicks in at period T0 for treated units
        D = 1 if is_treated and t_idx >= T0 else 0
        Y = Y_base[i, t_idx] + D * TAU   # add treatment effect when D=1
        rows.append({"unit": u, "year": yr, "Y": Y, "D": D})

df = pd.DataFrame(rows)
print(df.head(10))
print(f"\nPanel dimensions: {df['unit'].nunique()} units x {df['year'].nunique()} periods")
print(f"Treated units: {df.groupby('unit')['D'].max().sum():.0f}")
```

Expected output:
```
      unit  year         Y  D
0   unit_0  2000  0.234567  0
1   unit_0  2001  1.432890  0
...
Panel dimensions: 30 units x 20 periods
Treated units: 5
```

!!! note "Why synthetic data?"
    Using synthetic data with a known true effect lets you verify that the estimator recovers `TAU = 1.5`. In this tutorial, `att_avg` should be close to 1.5.

---

## Step 2: Estimate with gsynth

Run gsynth with the formula interface. We fix `r=2` and turn off CV to match the true number of factors in the DGP:

```python
# Estimate the generalized synthetic control model
result = gsynth(
    "Y ~ D",            # formula: Y is outcome, D is treatment
    data=df,
    index=["unit", "year"],  # unit and time identifiers
    r=2,                # number of latent factors (we know the truth is 2)
    CV=False,           # skip cross-validation (we know r)
    seed=42             # reproducibility
)
```

The formula `"Y ~ D"` specifies:
- `Y`: outcome variable
- `D`: treatment indicator

`index=["unit", "year"]` tells gsynth which columns identify units and time periods.

---

## Step 3: Print Summary

```python
# Print formatted summary of results
result.summary()
```

This prints the model specification, sample sizes, the estimated `att_avg`, and a table of event-time ATTs. You should see `att_avg` near 1.5.

You can also access fields directly:

```python
# Overall average treatment effect
print(f"Estimated ATT:  {result.att_avg:.3f}")
print(f"True ATT:       {TAU}")
print(f"Difference:     {abs(result.att_avg - TAU):.3f}")

# Sample sizes
print(f"\nN (total):   {result.N}")
print(f"N_tr:        {result.N_tr}")
print(f"N_co:        {result.N_co}")
print(f"T:           {result.T}")
print(f"r selected:  {result.r}")
```

---

## Step 4: Inspect Event-Time ATTs

The `att` array contains ATT estimates for every event time — both pre-treatment (placebo) and post-treatment:

```python
print("\n--- Event-time ATTs ---")
for t, a in zip(result.att_time, result.att):
    marker = " <-- onset" if t == 0 else ""
    status = "pre " if t < 0 else "post"
    print(f"  t={t:+3d}  [{status}]  ATT = {a:+.3f}{marker}")
```

Pre-treatment ATTs (t < 0) should be near zero — they are placebo estimates under the null of no treatment anticipation. Post-treatment ATTs (t >= 0) should be near 1.5.

---

## Step 5: Plot the Gap (ATT over Event Time)

The gap plot shows ATT estimates over event time, with a dashed line at t=0 and shaded post-treatment region:

```python
# Gap plot: period-by-period ATT
fig = plot(result, type="gap",
           main="Treatment Effect: Estimated ATT by Event Time",
           xlab="Years relative to treatment onset",
           ylab="Average Treatment Effect (ATT)")
```

The pre-treatment period (left of the dashed line) should fluctuate near zero. The post-treatment period should show estimates near 1.5.

---

## Step 6: Add Inference

Re-estimate with bootstrap standard errors:

```python
result_se = gsynth(
    "Y ~ D",
    data=df,
    index=["unit", "year"],
    r=2,
    CV=False,
    se=True,         # compute standard errors
    nboots=200,      # 200 bootstrap replicates
    inference="parametric",  # pseudo-treatment bootstrap (default for gsynth)
    seed=42
)

print(f"ATT:    {result_se.att_avg:.3f}")
print(f"SE:     {result_se.att_avg_se:.3f}")
print(f"95% CI: [{result_se.att_avg_ci_lower:.3f}, {result_se.att_avg_ci_upper:.3f}]")
```

Now plot the gap with CI band:

```python
fig = plot(result_se, type="gap",
           main="ATT with 95% Confidence Interval")
```

---

## Step 7: Plot the Counterfactual

The counterfactual plot shows the observed trajectory alongside the imputed Y(0) for treated units:

```python
# Average counterfactual across all treated units
fig = plot(result, type="counterfactual",
           main="Observed vs Counterfactual (all treated units)")

# Counterfactual for a specific treated unit
fig = plot(result, type="counterfactual",
           id="unit_0",
           main="unit_0: Observed vs Counterfactual")
```

In the pre-treatment period, the observed and counterfactual lines should overlap closely (good model fit). After treatment onset, the observed line rises above the counterfactual — the gap is the estimated treatment effect.

---

## Step 8: Compute Cumulative Effect

Use `effect()` to compute the cumulative treatment effect:

```python
from gsynth import effect

# Average ATT over all post-treatment periods
eff = effect(result)
print(f"Average ATT (post-treatment): {eff['att_avg']:.3f}")
print(f"N treated units:              {eff['n_units']}")

# Cumulative ATT (sum over post-treatment periods)
eff_cumu = effect(result, cumu=True)
print(f"Cumulative ATT:               {eff_cumu['att_cumulative']:.3f}")
T_post = T - T0
print(f"Expected cumulative (TAU x T_post): {TAU * T_post:.1f}")
```

---

## Summary

In this tutorial you:

1. Generated a synthetic IFE panel dataset with a known treatment effect of 1.5
2. Estimated the generalized synthetic control model with `gsynth()`
3. Inspected the summary output and confirmed the ATT is near 1.5
4. Verified the pre-treatment placebo (event-time ATTs near zero before t=0)
5. Added bootstrap inference and plotted the gap with a CI band
6. Plotted the counterfactual trajectory for treated units
7. Computed the cumulative treatment effect with `effect()`

From here, explore the [User Guide](../user-guide/basic-estimation.md) for more detailed documentation on each step, or the [API Reference](../api/gsynth.md) for the full parameter list.
