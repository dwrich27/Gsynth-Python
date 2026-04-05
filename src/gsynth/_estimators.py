"""
Core estimators for the Generalized Synthetic Control Method.

Three algorithms are implemented:
  - ``ife``  : Interactive Fixed Effects via alternating least squares (ALS).
  - ``gsynth``: Factors estimated on control group only; treated units projected.
  - ``mc``   : Matrix Completion via nuclear-norm regularisation (soft-SVD).
"""
from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _soft_threshold(sigma: np.ndarray, lam: float) -> np.ndarray:
    """Soft-threshold singular values for nuclear-norm proximal operator."""
    return np.maximum(sigma - lam, 0.0)


def _svd_reconstruct(U, s, Vt) -> np.ndarray:
    return (U * s) @ Vt


def _nan_obs_mask(Y: np.ndarray) -> np.ndarray:
    """Return boolean mask True where Y is observed (not NaN)."""
    return ~np.isnan(Y)


# ---------------------------------------------------------------------------
# Interactive Fixed Effects (ALS) — used by both "ife" and "gsynth" modes
# ---------------------------------------------------------------------------

def _ife_als(
    Y: np.ndarray,
    obs_mask: np.ndarray,
    r: int,
    tol: float = 1e-4,
    max_iter: int = 1000,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Alternating Least Squares for the Interactive Fixed Effects model.

        Y_it ≈ F_t @ Lambda_i   (r-factor approximation)

    Parameters
    ----------
    Y        : (N, T) outcome matrix (NaN = unobserved)
    obs_mask : (N, T) bool mask — True where Y is observed
    r        : number of factors
    tol      : convergence tolerance on objective
    max_iter : maximum ALS iterations

    Returns
    -------
    F       : (T, r) factor matrix
    Lambda  : (N, r) loading matrix
    residuals : (N, T) residuals Y - Lambda @ F.T  (NaN where unobserved)
    mse     : in-sample MSE
    """
    N, T = Y.shape
    if r == 0:
        resid = np.where(obs_mask, Y, np.nan)
        mse = float(np.nanmean(resid ** 2))
        return np.zeros((T, 0)), np.zeros((N, 0)), resid, mse

    # Initialise F via truncated SVD on mean-imputed Y
    Y_imp = Y.copy()
    col_means = np.nanmean(Y, axis=0)
    inds = np.where(np.isnan(Y))
    Y_imp[inds] = col_means[inds[1]]

    try:
        U, s, Vt = np.linalg.svd(Y_imp, full_matrices=False)
        F = Vt[:r].T * s[:r]          # (T, r)
        Lambda = U[:, :r]              # (N, r)
    except np.linalg.LinAlgError:
        rng = np.random.default_rng(42)
        F = rng.standard_normal((T, r))
        Lambda = rng.standard_normal((N, r))

    prev_obj = np.inf
    for _ in range(max_iter):
        # Update Lambda: for each unit i, OLS on observed time points
        for i in range(N):
            mask_i = obs_mask[i]
            if mask_i.sum() < r:
                continue
            F_obs = F[mask_i]          # (T_obs, r)
            Y_obs = Y[i, mask_i]       # (T_obs,)
            Lambda[i] = np.linalg.lstsq(F_obs, Y_obs, rcond=None)[0]

        # Update F: for each time t, OLS on observed units
        for j in range(T):
            mask_j = obs_mask[:, j]
            if mask_j.sum() < r:
                continue
            L_obs = Lambda[mask_j]     # (N_obs, r)
            Y_obs = Y[mask_j, j]       # (N_obs,)
            F[j] = np.linalg.lstsq(L_obs, Y_obs, rcond=None)[0]

        # Objective: MSE on observed cells
        fitted = Lambda @ F.T          # (N, T)
        resid_sq = np.where(obs_mask, (Y - fitted) ** 2, 0.0)
        obj = resid_sq.sum() / obs_mask.sum()

        if abs(prev_obj - obj) < tol:
            break
        prev_obj = obj

    fitted = Lambda @ F.T
    residuals = np.where(obs_mask, Y - fitted, np.nan)
    mse = float(np.nanmean(residuals ** 2))
    return F, Lambda, residuals, mse


# ---------------------------------------------------------------------------
# gsynth estimator
# ---------------------------------------------------------------------------

def estimate_gsynth(
    Y: np.ndarray,
    D: np.ndarray,
    ctrl_idx: np.ndarray,
    treat_idx: np.ndarray,
    T0_vec: np.ndarray,
    r: int,
    tol: float = 1e-4,
) -> dict:
    """
    Generalized Synthetic Control estimator.

    Factors are estimated **only** from control units.  Treated units'
    factor loadings are recovered by projecting their pre-treatment
    outcomes onto the estimated factor space.

    Parameters
    ----------
    Y        : (N, T) demeaned/residualized outcome
    D        : (N, T) treatment indicator
    ctrl_idx : indices of control units in Y
    treat_idx: indices of treated units in Y
    T0_vec   : pre-treatment period count per treated unit
    r        : number of factors

    Returns
    -------
    dict with keys: F, Lambda, Y_ct, att, att_avg, residuals, mse
    """
    N, T = Y.shape
    Y_ctrl = Y[ctrl_idx]                         # (N_co, T)
    obs_ctrl = _nan_obs_mask(Y_ctrl)

    F, Lambda_ctrl, resid_ctrl, mse = _ife_als(Y_ctrl, obs_ctrl, r, tol)

    # Estimate factor loadings for treated units using pre-treatment window
    Lambda_treat = np.zeros((len(treat_idx), max(r, 1)))
    if r > 0:
        for k, i in enumerate(treat_idx):
            t0 = int(T0_vec[k])
            if t0 < r:
                # Fallback: use all available pre-treatment
                t0 = T
            F_pre = F[:t0]                        # (T0, r)
            Y_pre = Y[i, :t0]                     # (T0,)
            obs_pre = ~np.isnan(Y_pre)
            if obs_pre.sum() >= r:
                Lambda_treat[k] = np.linalg.lstsq(
                    F_pre[obs_pre], Y_pre[obs_pre], rcond=None
                )[0]

    # Counterfactuals: Y_ct_i = Lambda_i @ F.T
    Y_ct = np.zeros((len(treat_idx), T))
    if r > 0:
        Y_ct = Lambda_treat[:, :r] @ F.T          # (N_tr, T)

    # Treatment effect: gap between observed and counterfactual
    # Use only post-treatment periods (D==1)
    att_unit = []

    for k, i in enumerate(treat_idx):
        t0 = int(T0_vec[k])
        obs_i = ~np.isnan(Y[i])
        gaps = Y[i] - Y_ct[k]
        post_gaps = []
        for j in range(T):
            if D[i, j] == 1 and obs_i[j]:
                post_gaps.append((j, gaps[j]))
        att_unit.append([g for _, g in post_gaps])

    # Average ATT across treated units, by calendar time
    post_times = sorted(
        {j for k, i in enumerate(treat_idx) for j in range(T) if D[i, j] == 1}
    )
    att = np.zeros(len(post_times))
    att_counts = np.zeros(len(post_times))
    for k, i in enumerate(treat_idx):
        for jj, j in enumerate(post_times):
            if D[i, j] == 1 and not np.isnan(Y[i, j]):
                att[jj] += Y[i, j] - Y_ct[k, j]
                att_counts[jj] += 1
    nz = att_counts > 0
    att[nz] /= att_counts[nz]

    att_avg = float(att[nz].mean()) if nz.any() else 0.0

    # Build full Lambda (all units)
    Lambda = np.zeros((N, max(r, 1)))
    if r > 0:
        Lambda[ctrl_idx] = Lambda_ctrl
        Lambda[treat_idx] = Lambda_treat[:, :r]

    return {
        "F": F,
        "Lambda": Lambda,
        "Lambda_ctrl": Lambda_ctrl,
        "Lambda_treat": Lambda_treat[:, :r] if r > 0 else Lambda_treat,
        "Y_ct": Y_ct,
        "att": att,
        "att_time": np.array(post_times),
        "att_avg": att_avg,
        "att_unit": att_unit,
        "residuals": resid_ctrl,
        "mse": mse,
    }


# ---------------------------------------------------------------------------
# IFE estimator (EM algorithm — treats pre-treatment treated obs too)
# ---------------------------------------------------------------------------

def estimate_ife(
    Y: np.ndarray,
    D: np.ndarray,
    ctrl_idx: np.ndarray,
    treat_idx: np.ndarray,
    T0_vec: np.ndarray,
    r: int,
    tol: float = 1e-4,
) -> dict:
    """
    Interactive Fixed Effects via EM / ALS on all pre-treatment observations.

    Treated units' pre-treatment data is used to estimate factors jointly.
    Post-treatment counterfactuals are imputed from the estimated model.
    """
    N, T = Y.shape

    # Pre-treatment mask: observed and not yet treated
    pre_mask = np.zeros((N, T), dtype=bool)
    for k, i in enumerate(treat_idx):
        t0 = int(T0_vec[k])
        pre_mask[i, :t0] = ~np.isnan(Y[i, :t0])
    for i in ctrl_idx:
        pre_mask[i] = ~np.isnan(Y[i])

    F, Lambda, resid, mse = _ife_als(Y, pre_mask, r, tol)

    # Counterfactuals: Y_ct = Lambda @ F.T
    Y_ct = Lambda[treat_idx] @ F.T            # (N_tr, T)

    # ATT
    post_times = sorted(
        {j for i in treat_idx for j in range(T) if D[i, j] == 1}
    )
    att = np.zeros(len(post_times))
    att_counts = np.zeros(len(post_times))
    for k, i in enumerate(treat_idx):
        for jj, j in enumerate(post_times):
            if D[i, j] == 1 and not np.isnan(Y[i, j]):
                att[jj] += Y[i, j] - Y_ct[k, j]
                att_counts[jj] += 1
    nz = att_counts > 0
    att[nz] /= att_counts[nz]
    att_avg = float(att[nz].mean()) if nz.any() else 0.0

    return {
        "F": F,
        "Lambda": Lambda,
        "Lambda_ctrl": Lambda[ctrl_idx],
        "Lambda_treat": Lambda[treat_idx],
        "Y_ct": Y_ct,
        "att": att,
        "att_time": np.array(post_times),
        "att_avg": att_avg,
        "residuals": resid,
        "mse": mse,
    }


# ---------------------------------------------------------------------------
# Matrix Completion estimator
# ---------------------------------------------------------------------------

def _nuclear_norm_minimise(
    Y: np.ndarray,
    obs_mask: np.ndarray,
    lam: float,
    tol: float = 1e-4,
    max_iter: int = 500,
) -> np.ndarray:
    """
    Minimise 0.5 * ||P_Ω(Y - M)||_F^2 + λ * ||M||_*

    using an iterative soft-thresholded SVD (proximal gradient).
    """
    N, T = Y.shape
    M = np.where(obs_mask, Y, 0.0)
    prev_obj = np.inf

    for _ in range(max_iter):
        # Gradient step (step size = 1 for this problem)
        grad = np.where(obs_mask, M - Y, 0.0)
        Z = M - grad

        # Proximal step: soft-threshold singular values
        try:
            U, s, Vt = np.linalg.svd(Z, full_matrices=False)
        except np.linalg.LinAlgError:
            break
        s_thresh = _soft_threshold(s, lam)
        M_new = _svd_reconstruct(U, s_thresh, Vt)

        obj = (
            0.5 * np.where(obs_mask, (Y - M_new) ** 2, 0.0).sum()
            + lam * s_thresh.sum()
        )
        if abs(prev_obj - obj) < tol:
            M = M_new
            break
        M = M_new
        prev_obj = obj

    return M


def estimate_mc(
    Y: np.ndarray,
    D: np.ndarray,
    ctrl_idx: np.ndarray,
    treat_idx: np.ndarray,
    T0_vec: np.ndarray,
    lam: float,
    tol: float = 1e-4,
) -> dict:
    """
    Matrix Completion estimator.

    Treats post-treatment outcomes for treated units as *missing*, then
    recovers them via nuclear-norm regularised matrix completion.

    Parameters
    ----------
    Y   : (N, T) demeaned outcome
    lam : regularization parameter (nuclear norm penalty)
    """
    N, T = Y.shape

    # Observation mask: exclude post-treatment treated cells
    obs_mask = np.zeros((N, T), dtype=bool)
    for i in range(N):
        obs_mask[i] = ~np.isnan(Y[i])
    for k, i in enumerate(treat_idx):
        obs_mask[i] &= (D[i] == 0)   # zero out post-treatment

    M_complete = _nuclear_norm_minimise(Y, obs_mask, lam, tol)

    # Counterfactuals for treated units
    Y_ct = M_complete[treat_idx]       # (N_tr, T)

    # ATT
    post_times = sorted(
        {j for i in treat_idx for j in range(T) if D[i, j] == 1}
    )
    att = np.zeros(len(post_times))
    att_counts = np.zeros(len(post_times))
    for k, i in enumerate(treat_idx):
        for jj, j in enumerate(post_times):
            if D[i, j] == 1 and not np.isnan(Y[i, j]):
                att[jj] += Y[i, j] - Y_ct[k, j]
                att_counts[jj] += 1
    nz = att_counts > 0
    att[nz] /= att_counts[nz]
    att_avg = float(att[nz].mean()) if nz.any() else 0.0

    # MSE on observed control cells
    resid = np.where(obs_mask, Y - M_complete, np.nan)
    mse = float(np.nanmean(resid[ctrl_idx] ** 2))

    # Rank of completed matrix
    try:
        s = np.linalg.svd(M_complete, compute_uv=False)
        r = int(np.sum(s > 1e-8))
    except np.linalg.LinAlgError:
        r = 0

    return {
        "M": M_complete,
        "F": None,
        "Lambda": None,
        "Y_ct": Y_ct,
        "att": att,
        "att_time": np.array(post_times),
        "att_avg": att_avg,
        "residuals": resid,
        "mse": mse,
        "r": r,
    }
