"""
Plotting capabilities for GsynthResult — replicating all R gsynth plot types.

Plot types:
  "gap"           — period-by-period ATT with optional CI band
  "raw"           — raw outcome trajectories for treated and control
  "counterfactual"— actual vs counterfactual outcome for treated units
  "factors"       — estimated latent factors over time
  "loadings"      — distribution of factor loadings across units
  "missing"       — data availability / treatment status heatmap
"""
from __future__ import annotations

from typing import Any, Optional, Union

import numpy as np

try:
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False

# Default colour palette (colour-blind friendly)
_COL_TREAT = "#E41A1C"
_COL_CT    = "#377EB8"
_COL_CTRL  = "#AAAAAA"
_COL_BAND  = "#BBCCEE"
_COL_ZERO  = "#555555"


def _require_mpl():
    if not _HAS_MPL:
        raise ImportError("matplotlib is required for plotting: pip install matplotlib")


def _apply_theme(ax: "Axes", theme_bw: bool):
    if theme_bw:
        ax.set_facecolor("white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#CCCCCC")
        ax.tick_params(colors="#333333")
        ax.xaxis.label.set_color("#333333")
        ax.yaxis.label.set_color("#333333")
        ax.title.set_color("#333333")


def plot(
    result,
    type: str = "gap",
    *,
    xlim: Optional[tuple] = None,
    ylim: Optional[tuple] = None,
    xlab: Optional[str] = None,
    ylab: Optional[str] = None,
    main: Optional[str] = None,
    legendOff: bool = False,
    theme_bw: bool = True,
    shade_post: bool = True,
    axis_adjust: bool = False,
    id: Optional[Union[Any, list]] = None,
    nfactors: Optional[int] = None,
    raw: str = "none",
    alpha: float = 0.3,
    figsize: tuple = (9, 5),
    show: bool = True,
) -> "Figure":
    """
    Plot a GsynthResult object.

    Parameters
    ----------
    result     : GsynthResult from gsynth()
    type       : one of "gap", "raw", "counterfactual"/"ct", "factors",
                 "loadings", "missing"
    xlim       : (xmin, xmax) limits for the x-axis
    ylim       : (ymin, ymax) limits for the y-axis
    xlab       : x-axis label (auto-generated if None)
    ylab       : y-axis label (auto-generated if None)
    main       : plot title (auto-generated if None)
    legendOff  : if True, suppress legend
    theme_bw   : use minimalist black-and-white theme
    shade_post : shade the post-treatment region (gap plot only)
    axis_adjust: rotate/adjust x-axis tick labels for string times
    id         : unit id(s) to highlight (counterfactual / raw plots)
    nfactors   : max number of factors to show (factors plot)
    raw        : "none", "band", or "all" — raw data overlay in ct plot
    alpha      : transparency of CI/shading
    figsize    : figure size in inches
    show       : call plt.show() automatically

    Returns
    -------
    matplotlib Figure
    """
    _require_mpl()
    type_lower = type.lower()
    if type_lower in ("counterfactual", "ct"):
        fig = _plot_counterfactual(
            result, id=id, raw=raw, xlim=xlim, ylim=ylim,
            xlab=xlab, ylab=ylab, main=main, legendOff=legendOff,
            theme_bw=theme_bw, axis_adjust=axis_adjust, alpha=alpha,
            figsize=figsize,
        )
    elif type_lower == "gap":
        fig = _plot_gap(
            result, xlim=xlim, ylim=ylim, xlab=xlab, ylab=ylab,
            main=main, legendOff=legendOff, theme_bw=theme_bw,
            shade_post=shade_post, axis_adjust=axis_adjust, alpha=alpha,
            figsize=figsize,
        )
    elif type_lower == "raw":
        fig = _plot_raw(
            result, id=id, xlim=xlim, ylim=ylim, xlab=xlab, ylab=ylab,
            main=main, legendOff=legendOff, theme_bw=theme_bw,
            axis_adjust=axis_adjust, figsize=figsize,
        )
    elif type_lower == "factors":
        fig = _plot_factors(
            result, nfactors=nfactors, xlim=xlim, ylim=ylim,
            xlab=xlab, ylab=ylab, main=main, theme_bw=theme_bw,
            axis_adjust=axis_adjust, figsize=figsize,
        )
    elif type_lower == "loadings":
        fig = _plot_loadings(
            result, xlim=xlim, ylim=ylim, xlab=xlab, ylab=ylab,
            main=main, theme_bw=theme_bw, figsize=figsize,
        )
    elif type_lower == "missing":
        fig = _plot_missing(
            result, id=id, xlab=xlab, ylab=ylab, main=main,
            theme_bw=theme_bw, figsize=figsize,
        )
    else:
        raise ValueError(
            f"Unknown plot type '{type}'. Choose from: "
            "'gap', 'raw', 'counterfactual', 'factors', 'loadings', 'missing'."
        )

    if show:
        plt.tight_layout()
        plt.show()
    return fig


# ---------------------------------------------------------------------------
# gap plot
# ---------------------------------------------------------------------------

def _plot_gap(result, **kw) -> "Figure":
    fig, ax = plt.subplots(figsize=kw["figsize"])
    _apply_theme(ax, kw["theme_bw"])

    times = result.att_time   # event-time; negative = pre-treatment
    att   = result.att
    has_se = result.att_se is not None

    # Shade post-treatment region (event-time >= 0)
    if kw["shade_post"] and len(times) > 0:
        post_min = 0.0
        post_max = float(times.max()) + 0.5
        if post_max > post_min:
            ax.axvspan(post_min - 0.5, post_max, color="#EEEEEE", zorder=0)

    # Vertical line at treatment onset
    ax.axvline(0, color="#888888", linewidth=0.8, linestyle=":", zorder=1)

    # Zero line
    ax.axhline(0, color=_COL_ZERO, linewidth=0.8, linestyle="--", zorder=1)

    # CI band
    if has_se:
        ax.fill_between(
            times, result.att_ci_lower, result.att_ci_upper,
            color=_COL_BAND, alpha=kw["alpha"], zorder=2, label="95% CI"
        )

    # ATT line (covers all event-time periods including pre-treatment)
    ax.plot(times, att, color=_COL_TREAT, linewidth=2, marker="o", ms=4,
            zorder=3, label="ATT")

    ax.set_xlabel(kw["xlab"] or "Event Time (periods since treatment)")
    ax.set_ylabel(kw["ylab"] or "Average Treatment Effect")
    ax.set_title(kw["main"] or "Period-by-Period ATT")
    if kw["xlim"]:
        ax.set_xlim(kw["xlim"])
    if kw["ylim"]:
        ax.set_ylim(kw["ylim"])
    if not kw["legendOff"] and has_se:
        ax.legend(frameon=False)
    if kw["axis_adjust"]:
        plt.xticks(rotation=45, ha="right")
    return fig


# ---------------------------------------------------------------------------
# raw outcomes plot
# ---------------------------------------------------------------------------

def _plot_raw(result, id=None, **kw) -> "Figure":
    fig, ax = plt.subplots(figsize=kw["figsize"])
    _apply_theme(ax, kw["theme_bw"])

    times = result.times
    Y_tr  = result.Y_tr          # (N_tr, T)
    treat_units = result.treat_units

    # Control units
    # We don't store Y_co directly but can reconstruct from Y_tr + Y_ct
    # (Y_co is not part of the result object; plot control band from Y_ct proxy)

    # Plot treated unit trajectories
    highlighted = set()
    if id is not None:
        highlighted = {id} if not isinstance(id, (list, tuple, np.ndarray)) else set(id)

    for k, u in enumerate(treat_units):
        col = _COL_TREAT if not highlighted or u in highlighted else "#FFAAAA"
        lw  = 2.0 if not highlighted or u in highlighted else 0.5
        y_k = Y_tr[k]
        ax.plot(times, y_k, color=col, linewidth=lw, alpha=0.8)

    ax.set_xlabel(kw["xlab"] or str(result.index[1] if result.index else "Time"))
    ax.set_ylabel(kw["ylab"] or result.Y_name)
    ax.set_title(kw["main"] or "Raw Outcome Trajectories (Treated Units)")
    if kw["xlim"]:
        ax.set_xlim(kw["xlim"])
    if kw["ylim"]:
        ax.set_ylim(kw["ylim"])
    if kw["axis_adjust"]:
        plt.xticks(rotation=45, ha="right")
    return fig


# ---------------------------------------------------------------------------
# counterfactual plot
# ---------------------------------------------------------------------------

def _plot_counterfactual(result, id=None, raw="none", **kw) -> "Figure":
    """
    Plot actual vs imputed counterfactual for treated units.

    Parameters
    ----------
    id  : specific unit id; if None plots the average across treated units
    raw : "none" | "band" (IQR of individual units) | "all" (all unit lines)
    """
    fig, ax = plt.subplots(figsize=kw["figsize"])
    _apply_theme(ax, kw["theme_bw"])

    times = result.times
    Y_tr  = result.Y_tr    # (N_tr, T)
    Y_ct  = result.Y_ct    # (N_tr, T)
    treat_units = result.treat_units

    if id is not None:
        # Single unit plot
        matches = np.where(treat_units == id)[0]
        if not len(matches):
            raise ValueError(f"Unit '{id}' not found among treated units.")
        k = matches[0]
        y_obs = Y_tr[k]
        y_ct  = Y_ct[k]
        label_obs = f"Observed (unit {id})"
        label_ct  = f"Counterfactual (unit {id})"
    else:
        # Average across treated units
        y_obs = np.nanmean(Y_tr, axis=0)
        y_ct  = np.nanmean(Y_ct, axis=0)
        label_obs = "Observed (average)"
        label_ct  = "Counterfactual (average)"

    # Raw data overlay
    if raw == "all":
        for k in range(len(treat_units)):
            ax.plot(times, Y_tr[k], color=_COL_TREAT, alpha=0.15, linewidth=0.8)
            ax.plot(times, Y_ct[k], color=_COL_CT,    alpha=0.15, linewidth=0.8)
    elif raw == "band":
        obs_q25 = np.nanquantile(Y_tr, 0.25, axis=0)
        obs_q75 = np.nanquantile(Y_tr, 0.75, axis=0)
        ct_q25  = np.nanquantile(Y_ct, 0.25, axis=0)
        ct_q75  = np.nanquantile(Y_ct, 0.75, axis=0)
        ax.fill_between(times, obs_q25, obs_q75, color=_COL_TREAT, alpha=0.15)
        ax.fill_between(times, ct_q25,  ct_q75,  color=_COL_CT,    alpha=0.15)

    # CI from bootstrap if available
    if result.att_se is not None and id is None:
        att_times = result.att_time
        lo = y_ct[np.isin(times, att_times)] + result.att_ci_lower
        hi = y_ct[np.isin(times, att_times)] + result.att_ci_upper
        ax.fill_between(att_times, lo, hi, color=_COL_BAND, alpha=kw["alpha"])

    # Main lines
    ax.plot(times, y_obs, color=_COL_TREAT, linewidth=2.0, label=label_obs)
    ax.plot(times, y_ct,  color=_COL_CT,    linewidth=2.0, linestyle="--", label=label_ct)

    # Treatment onset line(s)
    if len(result.T0) > 0:
        t0_vals = np.unique(list(result.T0.values()))
        for t0 in t0_vals:
            if t0 < len(times):
                ax.axvline(times[t0], color="#555555", linewidth=1.0, linestyle=":")

    ax.set_xlabel(kw["xlab"] or str(result.index[1] if result.index else "Time"))
    ax.set_ylabel(kw["ylab"] or result.Y_name)
    ax.set_title(kw["main"] or "Actual vs Counterfactual Outcome")
    if kw["xlim"]:
        ax.set_xlim(kw["xlim"])
    if kw["ylim"]:
        ax.set_ylim(kw["ylim"])
    if not kw["legendOff"]:
        ax.legend(frameon=False)
    if kw["axis_adjust"]:
        plt.xticks(rotation=45, ha="right")
    return fig


# ---------------------------------------------------------------------------
# factors plot
# ---------------------------------------------------------------------------

def _plot_factors(result, nfactors=None, **kw) -> "Figure":
    if result.factors is None or result.r == 0:
        raise ValueError("No estimated factors available (r=0 or MC estimator).")

    F = result.factors      # (T, r)
    times = result.times
    r = result.r
    n_show = min(nfactors or r, r, 4)

    fig, axes = plt.subplots(n_show, 1, figsize=(kw["figsize"][0], 3 * n_show),
                             sharex=True, squeeze=False)
    colours = plt.cm.tab10(np.linspace(0, 0.9, n_show))

    for j in range(n_show):
        ax = axes[j, 0]
        _apply_theme(ax, kw["theme_bw"])
        ax.plot(times, F[:, j], color=colours[j], linewidth=1.8)
        ax.axhline(0, color=_COL_ZERO, linewidth=0.6, linestyle="--")
        ax.set_ylabel(kw["ylab"] or f"Factor {j + 1}")
        ax.set_title(kw["main"] or f"Estimated Latent Factor {j + 1}" if j == 0 else "")
        if kw["xlim"]:
            ax.set_xlim(kw["xlim"])

    axes[-1, 0].set_xlabel(kw["xlab"] or str(result.index[1] if result.index else "Time"))
    if kw["axis_adjust"]:
        plt.xticks(rotation=45, ha="right")
    return fig


# ---------------------------------------------------------------------------
# loadings plot
# ---------------------------------------------------------------------------

def _plot_loadings(result, **kw) -> "Figure":
    if result.loadings is None or result.r == 0:
        raise ValueError("No estimated loadings available (r=0 or MC estimator).")

    Lambda = result.loadings   # (N, r)
    units  = result.units
    r      = result.r
    n_show = min(r, 4)

    fig, axes = plt.subplots(1, n_show, figsize=(kw["figsize"][0], kw["figsize"][1]),
                             squeeze=False)
    treat_set = set(result.treat_units.tolist())

    for j in range(n_show):
        ax = axes[0, j]
        _apply_theme(ax, kw["theme_bw"])
        loadings_j = Lambda[:, j]
        colors_bar = [_COL_TREAT if u in treat_set else _COL_CTRL for u in units]
        ax.bar(range(len(units)), loadings_j, color=colors_bar, edgecolor="none")
        ax.axhline(0, color=_COL_ZERO, linewidth=0.6, linestyle="--")
        ax.set_title(kw["main"] or f"Factor Loading {j + 1}" if j == 0 else f"Loading {j + 1}")
        ax.set_xlabel(kw["xlab"] or "Unit")
        ax.set_ylabel(kw["ylab"] or "Loading" if j == 0 else "")
        if kw["xlim"]:
            ax.set_xlim(kw["xlim"])
        if kw["ylim"]:
            ax.set_ylim(kw["ylim"])

    # Legend
    treat_patch = mpatches.Patch(color=_COL_TREAT, label="Treated")
    ctrl_patch  = mpatches.Patch(color=_COL_CTRL, label="Control")
    axes[0, -1].legend(handles=[treat_patch, ctrl_patch], frameon=False)
    return fig


# ---------------------------------------------------------------------------
# missing / status heatmap
# ---------------------------------------------------------------------------

def _plot_missing(result, id=None, **kw) -> "Figure":
    """
    Heatmap showing treatment status and data availability.

    Colour codes:
      dark blue  = treated (D==1)
      light blue = pre-treatment (D==0, treated unit)
      grey       = control unit
      white      = missing observation
    """
    from matplotlib.colors import ListedColormap

    units = result.units
    times = result.times
    treat_set = set(result.treat_units.tolist())

    # Build status matrix
    # 0=missing, 1=control, 2=treated-pre, 3=treated-post
    status = np.ones((len(units), len(times)), dtype=int)  # default: control

    for i, u in enumerate(units):
        if u in treat_set:
            k_tr = np.where(result.treat_units == u)[0]
            if len(k_tr):
                t0 = result.T0.get(u, 0)
                status[i, :t0] = 2   # pre-treatment treated
                status[i, t0:] = 3   # post-treatment treated

    # Subset units if id given
    if id is not None:
        id_list = [id] if not isinstance(id, (list, tuple, np.ndarray)) else list(id)
        keep = np.isin(units, id_list)
        units = units[keep]
        status = status[keep]

    cmap = ListedColormap(["white", "#AAAAAA", "#AAC4E8", "#2166AC"])

    fig, ax = plt.subplots(figsize=(max(9, len(times) * 0.3), max(5, len(units) * 0.3)))
    _apply_theme(ax, kw["theme_bw"])

    im = ax.imshow(status, aspect="auto", cmap=cmap, vmin=0, vmax=3,
                   interpolation="nearest")

    ax.set_xlabel(kw["xlab"] or str(result.index[1] if result.index else "Time"))
    ax.set_ylabel(kw["ylab"] or str(result.index[0] if result.index else "Unit"))
    ax.set_title(kw["main"] or "Treatment Status and Data Availability")

    # Tick labels (subsample if many)
    step_t = max(1, len(times) // 10)
    step_u = max(1, len(units) // 20)
    ax.set_xticks(range(0, len(times), step_t))
    ax.set_xticklabels([str(t) for t in times[::step_t]], rotation=45, ha="right")
    ax.set_yticks(range(0, len(units), step_u))
    ax.set_yticklabels([str(u) for u in units[::step_u]])

    # Colorbar legend
    cbar = fig.colorbar(im, ax=ax, ticks=[0.375, 1.125, 1.875, 2.625])
    cbar.ax.set_yticklabels(["Missing", "Control", "Pre-treat", "Post-treat"])

    return fig
