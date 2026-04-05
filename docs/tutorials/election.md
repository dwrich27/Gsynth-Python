# Tutorial: Effect of Campaign Finance Reform on Electoral Competitiveness

This tutorial demonstrates the full gsynth workflow using a realistic synthetic dataset. We examine whether campaign finance reform laws — which cap individual campaign contributions — affect electoral competitiveness measured by vote margin.

!!! warning "Synthetic demonstration data"
    All data in this tutorial are **synthetically generated** for illustration purposes. The scenario, states, and results do not reflect any real policy evaluation.

---

## Scenario

- **Units**: 50 US states
- **Time**: 20 biennial election cycles, 1980–2018 (years 1980, 1982, ..., 2018)
- **Treatment**: Campaign finance reform law, adopted by 10 states beginning at election cycle 11 (year 2000)
- **Outcome**: Vote margin (percentage points; lower values indicate more competitive races)
- **Covariates**: `log_income` (log median household income), `turnout` (voter turnout rate)

The hypothesized effect: campaign finance reform reduces the incumbency advantage and lowers vote margins (increases competitiveness). We expect `att_avg < 0`.

---

## Step 1: Data Generation

```python
import numpy as np
import pandas as pd
from gsynth import gsynth, plot, effect

rng = np.random.default_rng(2024)

# Dimensions
N_states  = 50
N_treated = 10
T         = 20         # 1980, 1982, ..., 2018 (biennial)
T0        = 10         # treatment starts at cycle 11 (year 2000)
TAU       = -3.0       # true ATT: reform reduces margin by 3 pp
years     = list(range(1980, 1980 + 2 * T, 2))  # biennial

states    = [f"state_{i:02d}" for i in range(N_states)]

# Latent factor structure (r=2)
alpha  = rng.normal(15, 5, N_states)       # state-level baseline margin (%)
F      = rng.normal(0, 1, (T, 2))           # common factors (e.g., national mood)
lam    = rng.normal(0, 1, (N_states, 2))    # state factor loadings
eps    = rng.normal(0, 2, (N_states, T))    # idiosyncratic noise

# Time-varying covariates
# log_income: rising over time with state-level heterogeneity
log_income_base = rng.uniform(10.5, 11.5, N_states)
log_income = (log_income_base[:, None]
              + np.linspace(0, 0.3, T)[None, :]
              + rng.normal(0, 0.05, (N_states, T)))

# turnout: correlated with competitiveness (lower margin -> higher turnout)
turnout_base = rng.uniform(0.45, 0.70, N_states)
turnout = (turnout_base[:, None]
           + rng.normal(0, 0.02, (N_states, T)))

# Covariate effects (higher income -> higher margin; higher turnout -> lower margin)
beta_income  =  2.5
beta_turnout = -8.0

# Baseline outcome with covariate effects
Y_base = (alpha[:, None]
          + lam @ F.T
          + beta_income * log_income
          + beta_turnout * turnout
          + eps)

# Build long-format DataFrame
rows = []
for i, state in enumerate(states):
    is_treated = i < N_treated
    for t_idx, yr in enumerate(years):
        D = 1 if is_treated and t_idx >= T0 else 0
        Y = Y_base[i, t_idx] + D * TAU
        rows.append({
            "state":      state,
            "year":       yr,
            "margin":     round(Y, 3),
            "reform":     D,
            "log_income": round(log_income[i, t_idx], 4),
            "turnout":    round(turnout[i, t_idx], 4),
        })

df = pd.DataFrame(rows)
print(df.head(12))
print(f"\nUnits: {df['state'].nunique()}, Periods: {df['year'].nunique()}")
print(f"Treated states: {df.groupby('state')['reform'].max().sum():.0f}")
```

---

## Step 2: Treatment Summary

Before estimation, examine the treatment pattern:

```python
# Treatment onset by state
treated_states = df[df["reform"] == 1]["state"].unique()
print(f"Treated states ({len(treated_states)}):")
for s in treated_states:
    onset = df[(df["state"] == s) & (df["reform"] == 1)]["year"].min()
    print(f"  {s}: reform adopted in {onset}")

# Cross-tab: treatment status by year
print("\nNumber of treated states per year:")
print(df.groupby("year")["reform"].sum().to_string())
```

All 10 treated states adopt simultaneously in 2000 (this is a non-staggered design; see the [Staggered Adoption tutorial](staggered.md) for the staggered case).

---

## Step 3: Basic Estimation (No Standard Errors)

Start with a quick estimate without inference to check model fit:

```python
result0 = gsynth(
    "margin ~ reform",
    data=df,
    index=["state", "year"],
    r=2,
    CV=False,
    seed=42
)

print(f"ATT (no covariates): {result0.att_avg:.3f}")
print(f"True ATT:            {TAU}")
```

---

## Step 4: Gap Plot — First Look

```python
fig = plot(result0, type="gap",
           main="Effect of Campaign Finance Reform on Vote Margin",
           xlab="Election cycles relative to reform (2000)",
           ylab="Average Treatment Effect (pp)",
           ylim=(-8, 4))
```

Examine the pre-treatment ATTs (negative event times). They should fluctuate near zero, indicating that treated and control states had similar trends before reform.

---

## Step 5: Adding Inference

Re-estimate with parametric bootstrap standard errors:

```python
result = gsynth(
    "margin ~ reform",
    data=df,
    index=["state", "year"],
    r=2,
    CV=False,
    se=True,
    nboots=200,
    inference="parametric",
    seed=42
)

print("\n=== Main Results ===")
print(f"ATT:    {result.att_avg:.3f}")
print(f"SE:     {result.att_avg_se:.3f}")
print(f"95% CI: [{result.att_avg_ci_lower:.3f}, {result.att_avg_ci_upper:.3f}]")
print(f"t-stat: {result.att_avg / result.att_avg_se:.2f}")
```

---

## Step 6: Gap Plot with Confidence Intervals

```python
fig = plot(result, type="gap",
           main="Effect of Campaign Finance Reform (95% CI)",
           xlab="Election cycles relative to reform",
           ylab="Vote margin change (pp)",
           ylim=(-10, 5),
           shade_post=True)
fig.savefig("reform_gap.png", dpi=150, bbox_inches="tight")
```

If the CI excludes zero in the post-treatment period and the pre-treatment CI consistently includes zero, you have a credible estimate.

---

## Step 7: Counterfactual Plot for a Specific State

Visualize one state's observed trajectory versus the synthetic control:

```python
fig = plot(result, type="counterfactual",
           id="state_00",
           raw="band",   # show band of control state outcomes
           main="state_00: Observed vs Counterfactual Margin")
fig.savefig("reform_ct_state00.png", dpi=150, bbox_inches="tight")
```

The pre-treatment overlap between observed and counterfactual validates the model. Post-treatment, the gap between lines is the estimated treatment effect for this state.

---

## Step 8: Covariate Version

Add income and turnout as controls to absorb observable confounding:

```python
result_cov = gsynth(
    "margin ~ reform + log_income + turnout",
    data=df,
    index=["state", "year"],
    r=2,
    CV=False,
    se=True,
    nboots=200,
    seed=42
)

print("\n=== With Covariates ===")
print(f"ATT:    {result_cov.att_avg:.3f}")
print(f"SE:     {result_cov.att_avg_se:.3f}")
print(f"95% CI: [{result_cov.att_avg_ci_lower:.3f}, {result_cov.att_avg_ci_upper:.3f}]")

# Covariate coefficients
for name, coef in zip(result_cov.X_names, result_cov.beta):
    print(f"  beta_{name}: {coef:.3f}  (true: {{'log_income': 2.5, 'turnout': -8.0}[name]})")
```

The coefficients should be close to the true values (2.5 for log_income, -8.0 for turnout), and the ATT should be close to -3.0.

---

## Step 9: Pre-Treatment Placebo Check

The pre-treatment ATTs are a built-in specification test. Extract and tabulate them:

```python
import numpy as np

pre_mask = result.att_time < 0
pre_atts = result.att[pre_mask]
pre_times = result.att_time[pre_mask]

print("\n=== Pre-treatment placebo check ===")
print(f"Mean pre-treatment ATT:   {pre_atts.mean():.4f}  (should be ~0)")
print(f"Max abs pre-treatment ATT: {np.abs(pre_atts).max():.4f}")

if result.att_se is not None:
    pre_ses = result.att_se[pre_mask]
    pre_tstats = pre_atts / (pre_ses + 1e-10)
    n_sig = (np.abs(pre_tstats) > 1.96).sum()
    print(f"Pre-treatment t-stats > 1.96: {n_sig} of {len(pre_atts)}")
    print("  (expect ~ 5% by chance under the null)")
```

A well-specified model should show pre-treatment ATTs that are both statistically insignificant and economically small relative to the post-treatment ATTs.

---

## Step 10: Reporting Results

Summarize the final results for a publication table:

```python
print("\n" + "=" * 55)
print("  Table: Effect of Campaign Finance Reform on Vote Margin")
print("=" * 55)
print(f"  Estimator:         gsynth (generalized synthetic control)")
print(f"  N states:          {result_cov.N} ({result_cov.N_tr} treated, {result_cov.N_co} control)")
print(f"  T periods:         {result_cov.T} biennial cycles (1980-2018)")
print(f"  Factors (r):       {result_cov.r}")
print(f"  Covariates:        log_income, turnout")
print(f"  ATT:               {result_cov.att_avg:.2f} pp")
print(f"  SE (parametric):   {result_cov.att_avg_se:.2f}")
print(f"  95% CI:            [{result_cov.att_avg_ci_lower:.2f}, {result_cov.att_avg_ci_upper:.2f}]")
print("=" * 55)
print("  True ATT (DGP):   -3.00 pp")
print("=" * 55)
print()
print("Note: Negative ATT indicates reform reduced vote margins")
print("      (increased electoral competitiveness), as hypothesized.")
```

---

## Key Takeaways from This Tutorial

1. **gsynth recovers the true effect** (-3.0 pp) reliably when the factor model is correctly specified.
2. **Pre-treatment placebo check** (event times < 0 in the gap plot) is essential for validating the parallel counterfactual assumption.
3. **Covariates improve precision** when they explain substantial variation in the outcome.
4. **Counterfactual plots** for individual units help build intuition and identify potential outliers.
5. **Parametric bootstrap** (default for `gsynth`) is appropriate for single-onset, balanced treatment designs.

For extensions to staggered adoption (different states adopting at different times), see the [Staggered Adoption tutorial](staggered.md).
