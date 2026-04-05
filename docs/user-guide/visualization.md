# Visualization

gsynth provides six built-in plot types via the `plot()` function. All plots return a matplotlib `Figure` object that can be further customized or saved.

```python
from gsynth import plot

fig = plot(result, type="gap")
fig.savefig("gap_plot.png", dpi=150, bbox_inches="tight")
```

---

## `"gap"` — Period-by-Period ATT

**What it shows**: The estimated ATT at each event time, with the pre-treatment periods displayed as a placebo check. A vertical dashed line marks `t=0` (treatment onset). The post-treatment region is shaded. When `se=True`, a confidence interval band is drawn around the point estimates.

**When to use**: This is the primary output plot. Always start here. Pre-treatment ATTs (negative event times) should be near zero — systematic deviation indicates that the parallel trends assumption underlying the factor model may not hold.

```python
fig = plot(result, type="gap",
           main="ATT: Effect of Treatment",
           xlab="Years relative to treatment",
           ylab="Estimated ATT",
           ylim=(-2, 4),
           shade_post=True)
```

**Key parameters**:
- `ylim`: y-axis limits, e.g., `(-1, 3)`
- `xlim`: x-axis limits (event-time range)
- `shade_post`: shade post-treatment region (default `True`)
- `main`: plot title
- `xlab`, `ylab`: axis labels
- `legendOff`: suppress legend
- `theme_bw`: use black-and-white theme

When `se=True` on the result, the CI band is drawn automatically.

---

## `"raw"` — Raw Outcome Trajectories

**What it shows**: Raw observed outcome `Y` over calendar time for all treated units (colored lines) and, optionally, all control units (grey lines). No counterfactual or treatment effect shown — this is purely descriptive.

**When to use**: Before estimation, to visually assess whether treated and control units move together in the pre-treatment period (common trends). Also useful for spotting outlier units or obvious data problems.

```python
fig = plot(result, type="raw",
           main="Raw Outcomes: Treated vs Control",
           ylab="Outcome Y",
           legendOff=False)
```

**Key parameters**:
- `legendOff`: suppress the treated/control legend
- `ylim`, `xlim`: axis limits (in calendar time)
- `main`, `xlab`, `ylab`: labels

---

## `"counterfactual"` (alias `"ct"`) — Actual vs Imputed Y(0)

**What it shows**: For treated unit(s), plots the observed trajectory (solid line) alongside the imputed counterfactual Y(0) (dashed line). The gap between the two lines is the estimated treatment effect at each period. A vertical dashed line marks treatment onset.

**When to use**: To visualize the quality of the counterfactual fit for specific units. The pre-treatment portion (both lines should overlap closely) serves as a visual model validation. In the post-treatment period, the gap between lines shows the treatment effect over time.

```python
# All treated units (average counterfactual)
fig = plot(result, type="counterfactual",
           main="Observed vs Counterfactual")

# Single treated unit
fig = plot(result, type="counterfactual",
           id="unit_0",
           main="unit_0: Observed vs Counterfactual")

# With control band overlay
fig = plot(result, type="counterfactual",
           raw="band",   # "none" / "band" / "all"
           id="unit_0")
```

**Key parameters**:
- `id`: restrict plot to a single treated unit (unit ID as string/int) or list of units
- `raw`: overlay control outcomes. `"none"` (default): no overlay; `"band"`: shaded band of control outcomes; `"all"`: all individual control trajectories
- `ylim`, `xlim`, `main`, `xlab`, `ylab`: standard customization

---

## `"factors"` — Latent Factors Over Time

**What it shows**: Time series of the estimated latent factors `F_t` (one line per factor), plotted over calendar time. A vertical dashed line marks the average treatment onset (or first treatment onset in staggered settings).

**When to use**: To interpret the factor structure. If factors show sharp breaks or unusual patterns around treatment time, this could indicate that factor estimation is being influenced by treatment timing rather than reflecting pre-existing dynamics.

```python
fig = plot(result, type="factors",
           nfactors=2,  # plot only first 2 factors
           main="Estimated Latent Factors")
```

**Key parameters**:
- `nfactors`: number of factors to display (default: all `r` factors)
- `main`, `xlab`, `ylab`, `ylim`, `xlim`: standard customization
- `figsize`: figure size as `(width, height)` tuple

---

## `"loadings"` — Factor Loadings Bar Chart

**What it shows**: Bar chart of the factor loadings `lambda_i` for each unit. Treated units are shown in red (or a distinct color); control units in grey. When `r > 1`, multiple panels or facets show one factor's loadings per panel.

**When to use**: To inspect the factor loading distribution and identify how treated units relate to control units in factor space. Treated units with extreme loadings (far from the control unit loading mass) may have poor counterfactual estimates because the factor space is extrapolating.

```python
fig = plot(result, type="loadings",
           main="Factor Loadings: Treated (red) vs Control (grey)")
```

**Key parameters**:
- `main`, `figsize`: standard customization
- `theme_bw`: black-and-white styling
- `axis_adjust`: adjust axis label formatting

---

## `"missing"` — Treatment Status and Data Availability Heatmap

**What it shows**: A heatmap over units (rows) x time periods (columns). Cell colors indicate:

| Color | Meaning |
|---|---|
| Dark blue | Treated unit, post-treatment period |
| Light blue | Treated unit, pre-treatment period |
| Grey | Control unit (all periods) |
| White | Missing / unavailable observation |

**When to use**: Essential for unbalanced panels or staggered adoption designs. Gives an immediate visual overview of which units are treated when, and whether there are gaps in the data.

```python
fig = plot(result, type="missing",
           main="Panel Structure: Treatment Status and Data Availability")
```

**Key parameters**:
- `main`, `figsize`: standard customization

---

## Common Plot Customization

### Axis Labels and Title

```python
fig = plot(result, type="gap",
           main="Effect of Treatment on Outcome",
           xlab="Event time (years from treatment)",
           ylab="Average Treatment Effect")
```

### Axis Limits

```python
fig = plot(result, type="gap",
           xlim=(-5, 10),   # event times from -5 to 10
           ylim=(-0.5, 3))  # ATT from -0.5 to 3
```

### Figure Size

```python
fig = plot(result, type="gap",
           figsize=(10, 5))  # width=10 inches, height=5 inches
```

### Black-and-White Theme

```python
fig = plot(result, type="gap", theme_bw=True)
```

### Suppress Display (headless / scripting)

```python
fig = plot(result, type="gap", show=False)
fig.savefig("gap.png", dpi=150, bbox_inches="tight")
```

### Saving to File

```python
fig = plot(result, type="gap")
fig.savefig("gap_plot.png", dpi=150, bbox_inches="tight")
fig.savefig("gap_plot.pdf", bbox_inches="tight")
```

---

## Plotting All Six Types in One Script

```python
from gsynth import plot

plot_types = ["gap", "raw", "counterfactual", "factors", "loadings", "missing"]
for ptype in plot_types:
    fig = plot(result, type=ptype, show=False)
    fig.savefig(f"{ptype}.png", dpi=120, bbox_inches="tight")
    print(f"Saved {ptype}.png")
```
