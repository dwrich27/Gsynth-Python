# `plot()` — API Reference

Generates diagnostic and results plots from a fitted [`GsynthResult`](result.md).

---

## Function Signature

```python
def plot(
    result,
    type="gap",
    xlim=None,
    ylim=None,
    xlab=None,
    ylab=None,
    main=None,
    legendOff=False,
    theme_bw=False,
    shade_post=True,
    axis_adjust=False,
    id=None,
    nfactors=None,
    raw="none",
    alpha=0.2,
    figsize=None,
    show=True,
) -> matplotlib.figure.Figure
```

---

## Parameters

### `result`
**Type**: `GsynthResult`

A fitted result object returned by [`gsynth()`](gsynth.md). Must not be `None`.

---

### `type`
**Type**: `str` | **Default**: `"gap"`

The plot type to generate. Must be one of:
- `"gap"`: event-time ATT with optional CI band
- `"raw"`: raw outcome trajectories over calendar time
- `"counterfactual"` (alias `"ct"`): observed vs imputed Y(0) for treated units
- `"factors"`: time series of estimated latent factors
- `"loadings"`: bar chart of factor loadings by unit
- `"missing"`: heatmap of treatment status and data availability

---

### `xlim`
**Type**: `tuple of (float, float)` or `None` | **Default**: `None`

Limits for the x-axis. For `"gap"`, `xlim` is in event-time units (e.g., `(-5, 10)`). For `"raw"` and `"counterfactual"`, it is in calendar time. If `None`, the axis range is set automatically.

---

### `ylim`
**Type**: `tuple of (float, float)` or `None` | **Default**: `None`

Limits for the y-axis. If `None`, set automatically from the data range plus a small margin.

---

### `xlab`
**Type**: `str` or `None` | **Default**: `None`

Label for the x-axis. If `None`, a sensible default is used (e.g., `"Event time"` for gap plots, `"Year"` for raw/counterfactual plots).

---

### `ylab`
**Type**: `str` or `None` | **Default**: `None`

Label for the y-axis. If `None`, defaults to the outcome variable name from the result.

---

### `main`
**Type**: `str` or `None` | **Default**: `None`

Plot title. If `None`, a default title describing the plot type and estimator is used.

---

### `legendOff`
**Type**: `bool` | **Default**: `False`

If `True`, suppresses the legend. Useful when the legend overlaps data or for publication figures where a separate caption is provided.

---

### `theme_bw`
**Type**: `bool` | **Default**: `False`

If `True`, applies a black-and-white styling: white background, grey gridlines, no colored backgrounds. Suitable for print publications that do not support color.

---

### `shade_post`
**Type**: `bool` | **Default**: `True`

For `"gap"` and `"counterfactual"` plots: if `True`, shades the post-treatment region (event time >= 0) with a light fill to visually distinguish it from the pre-treatment placebo period.

---

### `axis_adjust`
**Type**: `bool` | **Default**: `False`

If `True`, adjusts tick label formatting for better readability (e.g., rotating x-axis labels). Useful for plots with many time periods.

---

### `id`
**Type**: `str`, `int`, or `list` or `None` | **Default**: `None`

For `"counterfactual"` and `"loadings"` plots: restricts the plot to the specified unit(s). Pass a single unit ID (matching the unit identifier column) or a list of unit IDs. If `None`, all treated units are included (or averaged for `"counterfactual"`).

---

### `nfactors`
**Type**: `int` or `None` | **Default**: `None`

For `"factors"` plots: the number of factors to display. If `None`, all `r` factors are shown. Use `nfactors=2` to limit display to the first two factors when `r` is large.

---

### `raw`
**Type**: `str` | **Default**: `"none"`

For `"counterfactual"` plots only: controls overlay of control unit outcomes.
- `"none"`: no control overlay
- `"band"`: shaded band showing the range (or IQR) of control outcomes
- `"all"`: plot individual trajectories of all control units

---

### `alpha`
**Type**: `float` | **Default**: `0.2`

Transparency (alpha) value for shaded regions (CI band, post-treatment shade, control band). Range 0 (fully transparent) to 1 (fully opaque).

---

### `figsize`
**Type**: `tuple of (float, float)` or `None` | **Default**: `None`

Figure dimensions in inches as `(width, height)`. If `None`, a default size appropriate to the plot type is used. Example: `figsize=(10, 5)`.

---

### `show`
**Type**: `bool` | **Default**: `True`

If `True`, calls `plt.show()` to display the figure interactively. Set `show=False` in scripts or batch processing to suppress display and use the returned `Figure` object directly.

---

## Returns

`matplotlib.figure.Figure` — the figure object. You can further modify it or save it:

```python
fig = plot(result, type="gap", show=False)
ax = fig.axes[0]
ax.axhline(0, color="black", linewidth=0.8, linestyle="--")  # custom zero line
fig.savefig("custom_gap.png", dpi=150, bbox_inches="tight")
```

---

## Plot Types

### `"gap"` — Event-Time ATT

Shows the estimated ATT at each event time. The vertical dashed line marks `t=0` (treatment onset). Pre-treatment periods (t < 0) serve as a placebo check and should be near zero. When `se=True` on the result, a shaded confidence interval band is drawn.

**Relevant parameters**: `xlim`, `ylim`, `shade_post`, `xlab`, `ylab`, `main`, `alpha` (CI band transparency), `legendOff`, `theme_bw`

```python
fig = plot(result, type="gap",
           main="ATT by Event Time",
           xlab="Years relative to treatment",
           ylab="ATT",
           xlim=(-8, 10),
           ylim=(-2, 4),
           shade_post=True,
           theme_bw=True)
```

---

### `"raw"` — Raw Trajectories

Shows observed `Y` over calendar time for all treated units (colored) and all control units (grey). No counterfactual is shown. Used for visual inspection of common trends.

**Relevant parameters**: `xlim`, `ylim`, `xlab`, `ylab`, `main`, `legendOff`, `theme_bw`

```python
fig = plot(result, type="raw",
           main="Raw Observed Outcomes",
           legendOff=False)
```

---

### `"counterfactual"` / `"ct"` — Observed vs Y(0)

Shows the observed trajectory (solid line) and imputed counterfactual Y(0) (dashed line) for treated units. The gap between lines is the estimated treatment effect. Pre-treatment overlap validates model fit.

**Relevant parameters**: `id`, `raw`, `xlim`, `ylim`, `shade_post`, `xlab`, `ylab`, `main`, `alpha`, `theme_bw`

```python
# Average across all treated units
fig = plot(result, type="counterfactual")

# Single unit with control band
fig = plot(result, type="ct",
           id="unit_0",
           raw="band",
           main="unit_0: Observed vs Counterfactual")

# Single unit with all control trajectories
fig = plot(result, type="counterfactual",
           id="unit_0",
           raw="all")
```

---

### `"factors"` — Latent Factor Time Series

Shows the time series of estimated latent factors `F_t` as one line per factor over calendar time.

**Relevant parameters**: `nfactors`, `xlim`, `ylim`, `xlab`, `ylab`, `main`, `theme_bw`, `figsize`

```python
fig = plot(result, type="factors",
           nfactors=2,
           main="Estimated Latent Factors",
           xlab="Year")
```

---

### `"loadings"` — Factor Loading Bar Chart

Shows a bar chart of factor loadings `lambda_i` for all units. Treated units are highlighted in a distinct color (red by default); control units are shown in grey. When `r > 1`, multiple subplots show loadings for each factor.

**Relevant parameters**: `id`, `main`, `theme_bw`, `figsize`, `axis_adjust`, `legendOff`

```python
fig = plot(result, type="loadings",
           main="Factor Loadings: Treated (red) vs Control (grey)")
```

---

### `"missing"` — Treatment Status Heatmap

Shows a unit x time heatmap indicating:
- Dark blue: treated unit, post-treatment period
- Light blue: treated unit, pre-treatment period
- Grey: control unit
- White: missing observation

**Relevant parameters**: `main`, `figsize`, `axis_adjust`

```python
fig = plot(result, type="missing",
           main="Panel Data Structure",
           figsize=(12, 6))
```

---

## Saving Figures

```python
# Save to PNG
fig = plot(result, type="gap", show=False)
fig.savefig("gap.png", dpi=150, bbox_inches="tight")

# Save to PDF
fig.savefig("gap.pdf", bbox_inches="tight")

# Save to SVG (vector, for editing)
fig.savefig("gap.svg", bbox_inches="tight")
```

---

## See Also

- [`gsynth()`](gsynth.md) — estimation
- [`effect()`](effect.md) — effect summary
- [`GsynthResult`](result.md) — result fields
- [Visualization guide](../user-guide/visualization.md)
