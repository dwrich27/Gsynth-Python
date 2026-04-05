"""Data preparation utilities: parse panel, build matrices, handle missing."""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


def parse_panel(
    data: pd.DataFrame,
    Y: str,
    D: str,
    X: Optional[list[str]],
    index: list[str],
    weight: Optional[str],
    cl: Optional[str],
    na_rm: bool,
    min_T0: int,
) -> dict:
    """
    Validate and reshape a long-format panel DataFrame into NumPy arrays.

    Returns
    -------
    dict with keys:
        Y_mat   : (N, T) outcome matrix (NaN where missing)
        D_mat   : (N, T) treatment indicator matrix
        X_mat   : (N, T, K) covariate tensor, or None
        W_vec   : (N,) unit weights, or None
        cl_vec  : (N,) cluster labels, or None
        units   : (N,) array of unit ids
        times   : (T,) array of time ids
        is_balanced : bool
        ctrl_idx    : array of control unit positions
        treat_idx   : array of treated unit positions
        T0_vec      : (N_tr,) number of pre-treatment periods per treated unit
    """
    uid, tid = index[0], index[1]

    if na_rm:
        cols = [Y, D] + (X or [])
        data = data.dropna(subset=cols).copy()

    # Validate treatment is binary
    d_vals = data[D].dropna().unique()
    if not set(d_vals).issubset({0, 1, 0.0, 1.0}):
        raise ValueError(f"Treatment variable '{D}' must be dichotomous (0/1). Found: {d_vals}")

    units = np.array(sorted(data[uid].unique()))
    times = np.array(sorted(data[tid].unique()))
    N, T = len(units), len(times)

    unit_map = {u: i for i, u in enumerate(units)}
    time_map = {t: j for j, t in enumerate(times)}

    Y_mat = np.full((N, T), np.nan)
    D_mat = np.zeros((N, T), dtype=float)
    X_mat = np.full((N, T, len(X or [])), np.nan) if X else None

    for _, row in data.iterrows():
        i = unit_map[row[uid]]
        j = time_map[row[tid]]
        Y_mat[i, j] = row[Y]
        D_mat[i, j] = row[D]
        if X:
            for k, xname in enumerate(X):
                X_mat[i, j, k] = row[xname]

    # Unit weights
    W_vec = None
    if weight:
        w_lookup = data.groupby(uid)[weight].first()
        W_vec = np.array([w_lookup.get(u, 1.0) for u in units], dtype=float)
        W_vec = W_vec / W_vec.mean()

    # Cluster labels
    cl_vec = None
    if cl:
        cl_lookup = data.groupby(uid)[cl].first()
        cl_vec = np.array([cl_lookup[u] for u in units])

    # Identify treated units: any D==1
    is_treated = D_mat.any(axis=1)
    treat_idx = np.where(is_treated)[0]
    ctrl_idx = np.where(~is_treated)[0]

    if len(treat_idx) == 0:
        raise ValueError("No treated units found (no unit has D == 1).")

    # Compute T0 per treated unit (number of pre-treatment periods)
    T0_vec = []
    drop_treat = []
    for pos in treat_idx:
        first_treat = int(np.argmax(D_mat[pos] == 1))
        t0 = first_treat  # periods 0..first_treat-1 are pre-treatment
        if t0 < min_T0:
            drop_treat.append(pos)
        else:
            T0_vec.append(t0)

    if drop_treat:
        import warnings
        warnings.warn(
            f"{len(drop_treat)} treated unit(s) dropped: fewer than min_T0={min_T0} "
            f"pre-treatment periods. Unit positions: {drop_treat}",
            stacklevel=3,
        )
        treat_idx = np.array([p for p in treat_idx if p not in drop_treat])

    T0_vec = np.array(T0_vec, dtype=int)

    is_balanced = not np.any(np.isnan(Y_mat))

    return {
        "Y_mat": Y_mat,
        "D_mat": D_mat,
        "X_mat": X_mat,
        "W_vec": W_vec,
        "cl_vec": cl_vec,
        "units": units,
        "times": times,
        "N": N,
        "T": T,
        "ctrl_idx": ctrl_idx,
        "treat_idx": treat_idx,
        "T0_vec": T0_vec,
        "is_balanced": is_balanced,
    }


def demean_panel(Y_mat: np.ndarray, D_mat: np.ndarray, force: str, ctrl_idx: np.ndarray):
    """
    Partial out unit and/or time fixed effects from the outcome matrix.

    Only the control observations (D==0 in the pre-treatment window) are used
    to estimate time effects to avoid contamination by treatment.

    Returns
    -------
    Y_dem   : demeaned outcome (same shape as Y_mat)
    alpha   : unit fixed effects (N,) or None
    xi      : time fixed effects (T,) or None
    """
    N, T = Y_mat.shape
    alpha = np.zeros(N)
    xi = np.zeros(T)

    Y_dem = Y_mat.copy()

    if force == "none":
        return Y_dem, None, None

    if force in ("unit", "two-way"):
        # unit means computed over control observations
        for i in range(N):
            mask = ~np.isnan(Y_mat[i]) & (D_mat[i] == 0)
            if mask.any():
                alpha[i] = Y_mat[i, mask].mean()
        Y_dem = Y_dem - alpha[:, None]

    if force in ("time", "two-way"):
        # time means on already unit-demeaned values, control units only
        ctrl_Y = Y_dem[ctrl_idx]  # (N_co, T)
        for j in range(T):
            col = ctrl_Y[:, j]
            col = col[~np.isnan(col)]
            if len(col):
                xi[j] = col.mean()
        Y_dem = Y_dem - xi[None, :]

    alpha_out = alpha if force in ("unit", "two-way") else None
    xi_out = xi if force in ("time", "two-way") else None

    return Y_dem, alpha_out, xi_out


def partial_out_covariates(
    Y_dem: np.ndarray,
    X_mat: np.ndarray,
    D_mat: np.ndarray,
    ctrl_idx: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Partial out time-varying covariates from Y using OLS on control units.

    Parameters
    ----------
    Y_dem : (N, T) demeaned outcome
    X_mat : (N, T, K) covariate tensor
    D_mat : (N, T) treatment indicator

    Returns
    -------
    Y_res : (N, T) residuals after removing covariate effect
    beta  : (K,) OLS coefficient estimates
    """
    N, T, K = X_mat.shape

    # Stack control observations
    y_stack = []
    x_stack = []
    for i in ctrl_idx:
        for j in range(T):
            if not np.isnan(Y_dem[i, j]) and not np.any(np.isnan(X_mat[i, j])):
                y_stack.append(Y_dem[i, j])
                x_stack.append(X_mat[i, j])

    Y_vec = np.array(y_stack)
    X_arr = np.array(x_stack)

    # OLS: beta = (X'X)^-1 X'Y
    XtX = X_arr.T @ X_arr
    XtY = X_arr.T @ Y_vec
    beta = np.linalg.lstsq(XtX, XtY, rcond=None)[0]

    # Partial out
    Y_res = Y_dem.copy()
    for i in range(N):
        for j in range(T):
            if not np.any(np.isnan(X_mat[i, j])):
                Y_res[i, j] -= X_mat[i, j] @ beta

    return Y_res, beta
