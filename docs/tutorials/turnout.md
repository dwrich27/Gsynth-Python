# Voter Turnout: Replicating Xu (2017)

This tutorial reproduces the canonical empirical example from Xu (2017) using the bundled `turnout` dataset — the same example used in the original R `gsynth` package. Results closely match the R output.

---

## Background

Xu (2017) studies the effect of **Election Day Registration (EDR)** laws on voter turnout across US states. Nine states adopted EDR at various points between 1976 and 2012:

| Year | States |
|------|--------|
| 1976 | ME, MN, WI |
| 1996 | ID, NH, WY |
| 2008 | IA, MT |
| 2012 | CT |

This is a **staggered adoption** design: treated units switch on at different times. The dataset covers 47 states from 1920 to 2012 (every 4 years, 24 time periods).

---

## Load and Inspect the Data

```python
from gsynth.data import load_turnout

df = load_turnout()
print(df.shape)        # (1128, 6)
print(df.head())
print(df.dtypes)
```

```
(1128, 6)
   abb  year  turnout  policy_edr  policy_mail_in  policy_motor
0   AL  1920    31.49           0               0             0
1   AL  1924    17.19           0               0             0
2   AL  1928    19.37           0               0             0
...

abb               object
year               int64
turnout          float64
policy_edr         int64
policy_mail_in     int64
policy_motor       int64
```

**Columns:**

| Column | Description |
|--------|-------------|
| `abb` | Two-letter state abbreviation (unit ID) |
| `year` | Election year, 1920–2012 every 4 years (time ID) |
| `turnout` | Voter turnout (%) — outcome Y |
| `policy_edr` | 1 if Election Day Registration in effect (treatment D) |
| `policy_mail_in` | 1 if no-excuse absentee / mail-in voting in effect |
| `policy_motor` | 1 if motor-voter registration in effect |

---

## Estimate the Treatment Effect

We replicate the main result from Xu (2017). The model includes two time-varying covariates (`policy_mail_in`, `policy_motor`) and two-way fixed effects:

```python
from gsynth import gsynth, plot, effect

result = gsynth(
    "turnout ~ policy_edr + policy_mail_in + policy_motor",
    data=df,
    index=["abb", "year"],
    force="two-way",    # unit + time fixed effects
    CV=True,            # auto-select number of factors
    r=0,                # r=0 + CV=True → search over r in {0,1,2,3,4,5}
    se=True,
    nboots=200,
    seed=42,
)

print(result.summary())
```

```
===== Gsynth Estimation Results =====
Estimator : gsynth
Fixed effects : two-way
Factors (r) : 2
Obs (N×T) : 47 × 24 = 1128
Treated units : 9
Control units : 38

Average Treatment Effect on Treated (ATT):
  att_avg    = 4.93  (SE = 1.22,  95% CI: [2.54, 7.32])

Covariate estimates (beta):
  policy_mail_in =  0.34
  policy_motor   = -0.81
```

**Comparison to R `gsynth`:**

| Quantity | Python gsynth | R gsynth |
|----------|:-------------:|:--------:|
| r (selected) | 2 | 2 |
| ATT (average) | 4.93 | 4.90 |
| Difference | +0.03 pp | — |

The Python and R implementations agree to within 0.03 percentage points on the ATT estimate. Both select r = 2 factors via cross-validation.

---

## Visualize the Results

### Gap plot (period-by-period ATT)

```python
plot(result, type="gap", main="EDR Effect on Voter Turnout")
```

The gap plot shows the estimated treatment effect (ATT) for each event-time period, with 95% bootstrap confidence intervals. Effects are near zero in pre-treatment periods (placebo test) and positive post-treatment.

### Counterfactual plot

```python
plot(result, type="counterfactual", main="Actual vs. Counterfactual Turnout")
```

This shows actual turnout for treated states (solid) against the estimated counterfactual Y(0) (dashed) — what turnout would have been without EDR. The post-treatment gap is the ATT.

### Factor plot

```python
plot(result, type="factors", nfactors=2)
```

The two estimated latent factors capture shared time trends across states. The model uses these to construct the synthetic counterfactual for each treated state.

---

## Subgroup and Cumulative Effects

```python
eff = effect(result, cumu=True)
print(f"Cumulative ATT (all post periods): {eff['att_cumulative']:.3f}")

# Early adopters (1976)
early = ["ME", "MN", "WI"]
eff_early = effect(result, id=early)
print(f"ATT for early adopters (ME/MN/WI): {eff_early['att_avg']:.3f}")

# Late adopters (2008–2012)
late = ["IA", "MT", "CT"]
eff_late = effect(result, id=late)
print(f"ATT for late adopters (IA/MT/CT): {eff_late['att_avg']:.3f}")
```

---

## Full Working Script

```python
from gsynth import gsynth, plot, effect
from gsynth.data import load_turnout

df = load_turnout()

result = gsynth(
    "turnout ~ policy_edr + policy_mail_in + policy_motor",
    data=df,
    index=["abb", "year"],
    force="two-way",
    CV=True,
    r=0,
    se=True,
    nboots=200,
    seed=42,
)

print(result.summary())

plot(result, type="gap",             main="EDR Effect on Voter Turnout")
plot(result, type="counterfactual",  main="Actual vs. Counterfactual Turnout")
plot(result, type="factors",         nfactors=2)
plot(result, type="missing")         # treatment status heatmap

eff = effect(result, cumu=True)
print(f"Cumulative ATT: {eff['att_cumulative']:.3f}")
```

---

## Reference

Xu, Y. (2017). Generalized Synthetic Control Method: Causal Inference with Interactive Fixed Effects Models. *Political Analysis*, 25(1), 57–76. [doi:10.1017/pan.2016.2](https://doi.org/10.1017/pan.2016.2)
