# Fixed Effects

The `force` parameter controls which fixed effects are partialled out of the outcome before interactive fixed effects (IFE) estimation.

---

## What `force` Does

Before estimating the latent factor structure, gsynth can absorb additive fixed effects from the outcome matrix. This is done by within-demeaning:

- **Unit FEs** (`alpha_i`): subtract each unit's time-series mean (computed on pre-treatment control observations)
- **Time FEs** (`xi_t`): subtract each period's cross-sectional mean (computed on control units)

Demeaning happens within the control group during the pre-treatment period; the estimated fixed effects are then applied to treated units.

---

## Four Options

| `force=` | Fixed effects removed | Model |
|---|---|---|
| `"none"` | None | Pure IFE, no additive FEs |
| `"unit"` (default) | Unit FEs `alpha_i` | Most common; recommended |
| `"time"` | Time FEs `xi_t` | Rarely used alone |
| `"two-way"` | Both `alpha_i` and `xi_t` | Macro-shock settings |

---

## Unit Fixed Effects (`"unit"`, default)

Unit FEs absorb time-invariant heterogeneity — differences in average outcomes across units that persist over time. This is the most common specification and is the default.

```python
result = gsynth("Y ~ D", data=df, index=["unit", "year"],
                force="unit")  # default
print(result.alpha)  # estimated unit FEs, shape (N,)
```

The estimated unit FEs are stored in `result.alpha`. After demeaning, the IFE model fits on the residual outcome.

---

## Time Fixed Effects (`"time"`)

Time FEs absorb aggregate time trends — shocks that affect all units equally in the same period. Use this if your data has a strong common trend but you do not believe unit-level heterogeneity is important.

```python
result = gsynth("Y ~ D", data=df, index=["unit", "year"],
                force="time")
print(result.xi)  # estimated time FEs, shape (T,)
```

---

## Two-Way Fixed Effects (`"two-way"`)

Two-way FEs partial out both unit-level and time-level additive effects before estimating factors. This is appropriate when:

- Units have persistently different baseline outcome levels (unit FEs)
- All units are affected by common aggregate shocks (time FEs)

For example, in macroeconomic panel data where both country-specific level differences and common business cycle shocks are present.

```python
result = gsynth("Y ~ D", data=df, index=["unit", "year"],
                force="two-way")
print(result.alpha)  # unit FEs
print(result.xi)     # time FEs
```

---

## No Fixed Effects (`"none"`)

Using `force="none"` fits the IFE model directly without any additive demeaning. This is rarely appropriate in practice because it forces the factor model to absorb all heterogeneity, requiring more factors and making CV-selected `r` larger. Use only if you have a specific theoretical reason.

```python
result = gsynth("Y ~ D", data=df, index=["unit", "year"],
                force="none")
```

---

## Effect on Factor Count

Unit FEs augment the factor matrix before pre-treatment loading estimation. When `force="unit"`, one implicit "factor" per treated unit is used to accommodate the unit mean, which costs one degree of freedom. The practical consequence is the r constraint:

- With unit FEs: `r <= min_T0 - 2`
- Without unit FEs: `r <= min_T0 - 1`

where `min_T0` is the minimum number of pre-treatment periods across treated units. This cap prevents the factor model from perfectly fitting the pre-treatment data (overfitting the pre-treatment projection).

```python
# With min_T0=10 and force="unit":
# r can be at most 10 - 2 = 8
result = gsynth("Y ~ D", data=df, index=["unit", "year"],
                min_T0=10, r=5, CV=False)  # r=5 is fine
```

---

## Recommendation

For most applied panel data settings, `force="unit"` (the default) is the right choice. It removes cross-sectional heterogeneity while allowing time-varying common shocks to be captured by the latent factors.

Use `force="two-way"` when:

- You have strong prior evidence of aggregate shocks affecting all units equally (e.g., macroeconomic crises, national policy changes)
- The raw outcomes show parallel aggregate trends across all units that are not well-captured by factors alone

!!! note "Unit FEs vs factors"
    Unit FEs capture *time-invariant* heterogeneity. Latent factors capture *time-varying* heterogeneity in a low-rank way. The two work together: unit FEs remove the level differences, then factors capture the remaining time-varying co-movements.
