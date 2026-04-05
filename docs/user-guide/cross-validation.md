# Cross-Validation

Cross-validation (CV) is used to select the number of factors `r` (for `gsynth` and `ife`) and the regularization parameter `lambda` (for `mc`) without overfitting the pre-treatment data.

---

## Why CV?

The factor model's in-sample fit always improves as `r` increases. Without CV, the model would select the maximum allowed `r`, producing a pre-treatment fit that is artificially good and post-treatment counterfactuals that are noisy. CV uses held-out pre-treatment observations to select the `r` that generalizes best.

---

## CV for `gsynth`: LOO on Treated Pre-Treatment Data

For the default `gsynth` estimator, CV uses **leave-one-out (LOO)** on the treated units' pre-treatment observations:

1. Fix a candidate value of `r`.
2. For each pre-treatment period `t` of each treated unit `i`:
   a. Estimate factors from control units (all periods).
   b. Estimate loading `lambda_i` from treated unit `i`'s pre-treatment data, **excluding period `t`**.
   c. Predict the held-out period: `Y_hat_it = alpha_i + lambda_i @ F_t`.
   d. Record the squared prediction error `(Y_it - Y_hat_it)^2`.
3. Average squared prediction error = MSPE for that `r`.
4. Select `r` minimizing MSPE.

This LOO procedure directly tests how well the factor model predicts held-out pre-treatment periods for treated units — exactly the out-of-sample prediction task that determines counterfactual quality.

---

## CV for `ife`: k-Fold on Control Unit Cells

For the `ife` estimator, CV uses **k-fold** cross-validation on control unit observations:

1. Randomly partition control unit-time cells into `k` folds (default `k=5`).
2. For each candidate `r`, for each fold:
   a. Estimate the IFE model on all cells except the held-out fold.
   b. Predict held-out cells.
   c. Record MSPE.
3. Average MSPE across folds; select `r` minimizing average MSPE.

---

## CV for `mc`: k-Fold on Control Cells for Lambda

For the `mc` estimator, CV selects `lambda` (the nuclear norm penalty) using k-fold on control unit-time cells, following the same procedure as `ife` but over a grid of `lambda` values.

```python
# mc CV: 10 lambda values by default; inspecting results
result_mc = gsynth("Y ~ D", data=df, index=["unit", "year"],
                    estimator="mc", nlambda=10)
print(f"Optimal lambda: {result_mc.lambda_opt:.4f}")
print("Lambda CV scores:", result_mc.lambda_cv)
```

The `nlambda` parameter controls how many lambda values to search over. The lambda grid is constructed automatically based on the data scale.

---

## Reading CV Scores

```python
result = gsynth("Y ~ D", data=df, index=["unit", "year"])

# r_cv[k] = MSPE for r=k (for k=0, 1, 2, ...)
print("CV scores by r:", result.r_cv)

# Selected r
print("Selected r:", result.r)

# For mc: lambda CV scores
# print("Lambda CV scores:", result.lambda_cv)
```

A well-behaved CV profile has a clear minimum. If the scores are flat or noisy, the data may not have strong factor structure and `r=0` (simple additive FE) might be adequate.

---

## r Constraint

The maximum `r` evaluated during CV is capped by the minimum number of pre-treatment periods to prevent overfitting:

- With unit FEs (`force="unit"` or `"two-way"`): `r_max = min_T0 - 2`
- Without unit FEs (`force="none"` or `"time"`): `r_max = min_T0 - 1`

For example, if the treated unit with the fewest pre-treatment periods has `min_T0=8` and `force="unit"`, CV evaluates `r` in `{0, 1, 2, 3, 4, 5, 6}` (up to `r_max=6`).

```python
result = gsynth("Y ~ D", data=df, index=["unit", "year"],
                min_T0=8, force="unit")
print(f"r_max evaluated: {8 - 2}")  # = 6
print(f"Selected r: {result.r}")
```

---

## Disabling CV

To fix `r` at a specific value and skip CV, set `CV=False` and specify `r`:

```python
result = gsynth("Y ~ D", data=df, index=["unit", "year"],
                CV=False, r=2)
```

This is faster and useful when you have a strong prior on the number of factors, or for debugging.

---

## Specifying a Custom r Grid

Pass a list to `r` to evaluate only specific values:

```python
# CV over r in {0, 1, 2, 3} only
result = gsynth("Y ~ D", data=df, index=["unit", "year"],
                r=[0, 1, 2, 3])
```

This is useful when you want to restrict the search space, for example if `min_T0` is small and you want to avoid the maximum cap.

---

## The `criterion` Parameter

Two CV criteria are available:

| `criterion=` | Description |
|---|---|
| `"mspe"` (default) | Mean squared prediction error — selects `r` minimizing average prediction error |
| `"pc"` | Panel criterion — a penalized criterion analogous to information criteria for factors |

`"mspe"` is more robust in small samples. `"pc"` can favor larger `r` in datasets with many units and long time series.

```python
result = gsynth("Y ~ D", data=df, index=["unit", "year"],
                criterion="mspe")  # default

result_pc = gsynth("Y ~ D", data=df, index=["unit", "year"],
                   criterion="pc")
```

---

## The `k` Parameter

`k` sets the number of folds for k-fold CV (used for `ife` and `mc`):

```python
result = gsynth("Y ~ D", data=df, index=["unit", "year"],
                estimator="ife", k=10)  # 10-fold instead of default 5
```

Larger `k` gives more stable CV estimates but increases computation time.

---

## Practical Tips

!!! tip "When CV selects r=0"
    CV selecting `r=0` means the best-predicting model has no factors — only additive FEs. This often happens when the panel is short (few pre-treatment periods) or when unit FEs alone explain most of the variation. Consider whether your panel has enough time variation to support factor estimation.

!!! tip "Slow CV with many r values"
    If `min_T0` is large, CV evaluates many `r` values and can be slow. You can speed up with a custom grid: `r=[0, 1, 2]`.

!!! note "CV and randomness"
    For `ife` and `mc`, k-fold CV involves random assignment of cells to folds. Set `seed` for reproducibility:
    ```python
    result = gsynth("Y ~ D", data=df, index=["unit", "year"],
                    estimator="ife", seed=42)
    ```
