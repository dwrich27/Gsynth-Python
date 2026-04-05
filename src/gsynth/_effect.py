"""
effect() — cumulative and subgroup average treatment effects.
"""
from __future__ import annotations

from typing import Any, Optional, Union

import numpy as np


def effect(
    result,
    cumu: bool = True,
    period: Optional[tuple] = None,
    id: Optional[Union[Any, list]] = None,
    plot: bool = False,
) -> dict:
    """
    Calculate cumulative or subgroup average treatment effects.

    Parameters
    ----------
    result : GsynthResult
    cumu   : if True, compute cumulative ATT over ``period``; otherwise
             return period-by-period ATT for the selected period/units
    period : (start, end) time range to accumulate; None uses all post-treatment
    id     : unit id or list of ids to restrict computation to a subgroup;
             None uses all treated units
    plot   : if True, return a matplotlib Figure alongside the dict

    Returns
    -------
    dict with keys:
        att          : ndarray of period-by-period ATTs in the window
        att_cumulative : float, cumulative sum of ATT (if cumu=True)
        att_avg       : float, time-average ATT in the window
        times         : ndarray of time periods
        n_units       : number of treated units included
        fig           : Figure (only if plot=True)
    """
    att_time = result.att_time
    att      = result.att
    treat_units = result.treat_units

    # --- filter by unit subgroup ---
    if id is not None:
        id_list = np.atleast_1d(id)
        # Subset _att_unit if available
        if result._att_unit is not None:
            keep_k = np.isin(treat_units, id_list)
            n_units = int(keep_k.sum())
            att_unit_sub = np.array(result._att_unit, dtype=object)[keep_k]
            # Rebuild period-by-period ATT for the subgroup
            att = np.zeros(len(att_time))
            counts = np.zeros(len(att_time))
            for k in range(n_units):
                unit_gaps = att_unit_sub[k]
                for jj in range(min(len(unit_gaps), len(att_time))):
                    if not np.isnan(unit_gaps[jj]):
                        att[jj] += unit_gaps[jj]
                        counts[jj] += 1
            nz = counts > 0
            att[nz] /= counts[nz]
        else:
            n_units = int(np.isin(treat_units, id_list).sum())
    else:
        n_units = result.N_tr

    # --- filter by time period ---
    # Default: post-treatment periods only (att_time >= 0 in event-time)
    if period is not None:
        p_start, p_end = period[0], period[1]
        mask = (att_time >= p_start) & (att_time <= p_end)
    else:
        mask = att_time >= 0

    att_sel   = att[mask]
    times_sel = att_time[mask]

    att_avg        = float(np.nanmean(att_sel)) if len(att_sel) else 0.0
    att_cumulative = float(np.nansum(att_sel))  if cumu else None

    out = {
        "att": att_sel,
        "att_cumulative": att_cumulative,
        "att_avg": att_avg,
        "times": times_sel,
        "n_units": n_units,
    }

    if plot:
        try:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.bar(times_sel, att_sel, color="#2166AC", edgecolor="white", linewidth=0.5)
            ax.axhline(0, color="#555555", linewidth=0.8, linestyle="--")
            ax.set_xlabel("Time")
            ax.set_ylabel("ATT")
            title = "Cumulative ATT" if cumu else "Period ATT"
            if id is not None:
                title += f" (subgroup: {id})"
            ax.set_title(title)
            plt.tight_layout()
            out["fig"] = fig
        except ImportError:
            pass

    return out
