# Data Format

gsynth requires panel data in **long format**: one row per unit-time observation.

---

## Required Structure

Every dataset must contain:

| Column | Type | Description |
|---|---|---|
| Unit ID | str or int | Identifies the cross-sectional unit (e.g., state, firm, country) |
| Time ID | int or float | Identifies the time period (e.g., year, quarter) |
| Outcome `Y` | float | The outcome variable of interest |
| Treatment `D` | int (0/1) | Binary treatment indicator |

Tell gsynth which columns are the unit and time IDs via the `index` parameter:

```python
result = gsynth("Y ~ D", data=df, index=["unit_id", "time_id"])
```

The first element of `index` is the unit identifier; the second is the time identifier.

---

## Minimal Example

```python
import numpy as np
import pandas as pd
from gsynth import gsynth

# Minimal panel: 3 units, 6 time periods, unit 0 treated from t=3 onward
rows = [
    {"unit": "A", "year": 1, "Y": 2.1, "D": 0},
    {"unit": "A", "year": 2, "Y": 2.4, "D": 0},
    {"unit": "A", "year": 3, "Y": 2.6, "D": 1},
    {"unit": "A", "year": 4, "Y": 4.0, "D": 1},
    {"unit": "B", "year": 1, "Y": 1.8, "D": 0},
    {"unit": "B", "year": 2, "Y": 1.9, "D": 0},
    {"unit": "B", "year": 3, "Y": 2.0, "D": 0},
    {"unit": "B", "year": 4, "Y": 2.1, "D": 0},
    {"unit": "C", "year": 1, "Y": 3.0, "D": 0},
    {"unit": "C", "year": 2, "Y": 3.1, "D": 0},
    {"unit": "C", "year": 3, "Y": 3.2, "D": 0},
    {"unit": "C", "year": 4, "Y": 3.3, "D": 0},
]
df = pd.DataFrame(rows)
result = gsynth("Y ~ D", data=df, index=["unit", "year"], r=1, CV=False)
```

---

## Treatment Indicator Rules

The treatment indicator `D` must satisfy two conditions:

1. **Binary**: values must be 0 (untreated) or 1 (treated) at every observation.
2. **Absorbing**: once a unit switches to `D=1`, it must remain treated for all subsequent periods. The pattern `0 → 1` is valid; `1 → 0` is not.

!!! warning "Non-absorbing treatment"
    If any unit has `D` switching from 1 back to 0, gsynth will raise an error. Recode such units as always-control if the reversal is a data artifact, or exclude them if treatment truly reversed.

**Staggered adoption** is fully supported: different units can switch to `D=1` at different calendar times, as long as each unit's own sequence is absorbing. See [Staggered Adoption](staggered-adoption.md) for details.

---

## Optional Columns

### Covariates

Time-varying covariates can be added to the formula:

```python
result = gsynth("Y ~ D + X1 + X2", data=df, index=["unit", "year"])
```

Covariate columns must be numeric. See [Covariates](covariates.md) for how they are used in estimation.

### Weights

A column of unit-level weights (analytic or inverse-probability weights) can be passed via the `weight` parameter:

```python
result = gsynth("Y ~ D", data=df, index=["unit", "year"], weight="w")
```

### Cluster Variable

For cluster-robust bootstrap inference, specify a column identifying clusters:

```python
result = gsynth("Y ~ D", data=df, index=["unit", "year"], se=True, cl="state_group")
```

---

## The `index` Parameter

`index` is a list of two column names: `[unit_id_column, time_id_column]`. These columns should uniquely identify every row (no duplicate unit-time pairs).

```python
# Using default column names
result = gsynth("Y ~ D", data=df, index=["unit", "year"])

# With custom column names
result = gsynth("Y ~ D", data=df, index=["firm_id", "quarter"])
```

---

## Missing Values and `na_rm`

By default, gsynth will raise an error if the dataset contains `NaN` values in `Y`, `D`, or any covariate column. To silently drop rows with missing values before estimation, set `na_rm=True`:

```python
result = gsynth("Y ~ D", data=df, index=["unit", "year"], na_rm=True)
```

!!! note "Unbalanced panels"
    Dropping missing rows can create an unbalanced panel (different numbers of observations per unit). gsynth handles unbalanced panels internally. The `missing` plot type is useful for visualizing which unit-time cells have data. See [Visualization](visualization.md).

---

## The `min_T0` Parameter

`min_T0` sets the minimum number of pre-treatment periods required for a treated unit to be included in estimation. Units with fewer pre-treatment periods are dropped with a warning.

```python
result = gsynth("Y ~ D", data=df, index=["unit", "year"], min_T0=5)
```

**Why this matters**: factor loadings for treated units are estimated by projecting their pre-treatment outcomes onto the factor space. With too few pre-treatment periods, this projection is unreliable. The default `min_T0=5` is a conservative lower bound; increase it if you have a long panel.

The number of factors `r` is also capped by `min_T0`: with unit fixed effects, `r <= min_T0 - 2`; without unit FEs, `r <= min_T0 - 1`. This prevents overfitting in the pre-treatment projection.

---

## Constructing Panel Data from Wide Format

If your data is in wide format (one column per time period), convert it using `pd.melt`:

```python
# Wide format: columns are years
df_wide = pd.read_csv("outcomes_wide.csv")  # columns: unit, Y_2000, Y_2001, ..., D_2000, ...

# Convert outcome to long
df_y = df_wide.melt(id_vars="unit", value_vars=[c for c in df_wide if c.startswith("Y_")],
                    var_name="year", value_name="Y")
df_y["year"] = df_y["year"].str.replace("Y_", "").astype(int)

# Convert treatment to long and merge
df_d = df_wide.melt(id_vars="unit", value_vars=[c for c in df_wide if c.startswith("D_")],
                    var_name="year", value_name="D")
df_d["year"] = df_d["year"].str.replace("D_", "").astype(int)

df_long = df_y.merge(df_d, on=["unit", "year"])
```

---

## Common Pitfalls

### Duplicate unit-time rows
If the same unit-time combination appears more than once, gsynth will raise an error. Use `df.duplicated(["unit", "year"])` to check.

### Wrong column types
The outcome `Y` and covariates must be numeric. Categorical columns will cause errors. Encode categoricals before passing to gsynth.

### Time IDs are not evenly spaced
gsynth uses the lexicographic order of time IDs. If time periods are not contiguous integers, make sure the ordering is correct. Non-integer time IDs (e.g., year-quarter strings) are supported as long as they sort correctly.

### Treatment variable is not 0/1
Ensure `D` contains only 0 and 1. A continuous treatment variable or an indicator coded as True/False should be cast: `df["D"] = df["D"].astype(int)`.

### Too few control units
gsynth needs enough control units to estimate factors reliably. As a rule of thumb, the number of control units should be at least twice the number of factors `r`. Very small control groups will produce poor counterfactuals.
