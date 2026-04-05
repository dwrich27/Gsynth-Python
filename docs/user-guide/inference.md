# Inference

Standard errors and confidence intervals are computed via bootstrap or jackknife resampling. Enable inference with `se=True`.

---

## When to Use `se=True`

By default, gsynth returns only point estimates. Add `se=True` to also compute uncertainty measures:

```python
result = gsynth("Y ~ D", data=df, index=["unit", "year"],
                se=True, nboots=200, seed=42)
```

With `se=True`, the following additional fields are populated on `GsynthResult`:

| Field | Description |
|---|---|
| `att_avg_se` | Standard error of `att_avg` |
| `att_avg_ci_lower` | Lower bound of `(1-alpha)` CI for `att_avg` |
| `att_avg_ci_upper` | Upper bound of `(1-alpha)` CI for `att_avg` |
| `att_se` | SE for each event-time ATT, shape `(T_all,)` |
| `att_ci_lower` | Lower CI for event-time ATT, shape `(T_all,)` |
| `att_ci_upper` | Upper CI for event-time ATT, shape `(T_all,)` |
| `att_boot` | Bootstrap draws of post-treatment ATT, shape `(nboots, T_post)` |

The confidence level is controlled by the `alpha` parameter (default 0.05 for 95% CI).

---

## Three Inference Methods

### Parametric Bootstrap (default for `gsynth`)

The parametric bootstrap generates pseudo-treatment studies using control units:

1. Randomly select a control unit as the "pseudo-treated" unit.
2. Resample the remaining control units (with replacement) as the "pseudo-control" group.
3. Run gsynth on this pseudo-dataset. Because the pseudo-treated unit was never actually treated, the true ATT is zero by construction.
4. Record the estimated `att_avg` as one bootstrap draw.
5. Repeat `nboots` times.
6. SE = standard deviation of the bootstrap distribution of `att_avg`.

This method directly estimates the sampling variability of the gsynth estimator under the null of no treatment effect.

```python
result = gsynth("Y ~ D", data=df, index=["unit", "year"],
                se=True, inference="parametric", nboots=200, seed=42)
print(f"ATT: {result.att_avg:.3f} (SE={result.att_avg_se:.3f})")
print(f"95% CI: [{result.att_avg_ci_lower:.3f}, {result.att_avg_ci_upper:.3f}]")
```

!!! note "Default for gsynth"
    When `estimator="gsynth"` and `inference=None`, gsynth automatically uses `"parametric"` bootstrap.

### Nonparametric Block Bootstrap (default for `ife` and `mc`)

The nonparametric bootstrap resamples entire units (with replacement) to preserve the time-series structure:

1. Resample treated units independently with replacement.
2. Resample control units independently with replacement.
3. Run gsynth on the resampled dataset.
4. Record `att_avg`.
5. Repeat `nboots` times.

This is a standard block bootstrap at the unit level. It is asymptotically valid under mild conditions and is the default for `ife` and `mc` where parametric resampling is less natural.

```python
result = gsynth("Y ~ D", data=df, index=["unit", "year"],
                estimator="ife", se=True, inference="nonparametric",
                nboots=200, seed=42)
```

!!! warning "Small N"
    With few treated units (N_tr < 5), the nonparametric bootstrap can be unreliable because the bootstrap distribution is coarse. Consider jackknife in this case.

### Jackknife

Leave-one-unit-out jackknife:

1. Drop one treated unit at a time.
2. Re-estimate on the remaining dataset.
3. Use jackknife variance formula to compute SE.

This method does not require `nboots` and is faster than bootstrap. However, it can be conservative (overestimate SE) in small samples.

```python
result = gsynth("Y ~ D", data=df, index=["unit", "year"],
                se=True, inference="jackknife")
print(f"Jackknife SE: {result.att_avg_se:.3f}")
```

---

## Number of Bootstrap Draws: `nboots`

`nboots=200` (the default) is adequate for exploratory analysis. For published results, use `nboots=500` or more:

```python
result = gsynth("Y ~ D", data=df, index=["unit", "year"],
                se=True, nboots=500, seed=42)
```

With more bootstrap draws, the SE and CI estimates are more stable (less Monte Carlo variability), but computation time increases proportionally.

---

## Reproducibility: `seed`

Set `seed` to make bootstrap results reproducible:

```python
result = gsynth("Y ~ D", data=df, index=["unit", "year"],
                se=True, nboots=200, seed=12345)
```

---

## Cluster Bootstrap: `cl`

If units within clusters are correlated (e.g., states within regions), use cluster-robust bootstrap by specifying the cluster column:

```python
result = gsynth("Y ~ D", data=df, index=["unit", "year"],
                se=True, cl="region", nboots=200)
```

Entire clusters are resampled together, preserving within-cluster correlation. The cluster variable must be a column in `data`.

---

## Confidence Level: `alpha`

The `alpha` parameter sets the significance level for CIs (default 0.05 for 95% CI):

```python
result = gsynth("Y ~ D", data=df, index=["unit", "year"],
                se=True, nboots=200, alpha=0.1)  # 90% CI
print(f"90% CI: [{result.att_avg_ci_lower:.3f}, {result.att_avg_ci_upper:.3f}]")
```

---

## Bootstrap Draws: `result.att_boot`

The raw bootstrap draws are stored in `result.att_boot`, shape `(nboots, T_post)`, where `T_post` is the number of post-treatment periods. You can use these for custom inference:

```python
import numpy as np

# Manual 95% CI via quantiles (percentile method)
ci_lower = np.percentile(result.att_boot, 2.5, axis=0)
ci_upper = np.percentile(result.att_boot, 97.5, axis=0)
print("Percentile CI (first 3 post-treatment periods):")
for i in range(3):
    print(f"  t={i}: [{ci_lower[i]:.3f}, {ci_upper[i]:.3f}]")
```

---

## Parallelization

Bootstrap is computationally intensive. Enable parallel execution with `parallel=True` and specify the number of cores:

```python
result = gsynth("Y ~ D", data=df, index=["unit", "year"],
                se=True, nboots=500, parallel=True, cores=4, seed=42)
```

!!! note "Reproducibility with parallel"
    Parallel bootstrap may not produce exactly the same results as serial execution even with the same `seed`, due to non-deterministic scheduling of parallel tasks.
