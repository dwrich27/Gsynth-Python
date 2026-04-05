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


def _compute_att_by_event_time(
    Y: np.ndarray,
    Y_ct: np.ndarray,
    D: np.ndarray,
    treat_idx: np.ndarray,
    T0_vec: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Compute period-by-period ATT using **relative event-time** (matching R gsynth).

    Event-time 0 = first treatment period, -1 = one period before treatment, etc.
    All observed periods for treated units are included (pre and post).

    att_avg is the treatment-cell weighted average:
        att_avg = sum(gap[i,j] for (i,j) where D[i,j]=1) / sum(D[i,j]=1)
    matching R: ``att.avg = sum(eff * D) / sum(D)``.

    Returns
    -------
    att       : (n_event_times,) per-event-time average treatment effect
    att_time  : (n_event_times,) corresponding event-time values (integers)
    att_avg   : float, treatment-weighted overall ATT
    """
    T = Y.shape[1]
    N_tr = len(treat_idx)

    # Build dict: event_time -> (sum_gaps, count)
    et_sums: dict[int, float] = {}
    et_counts: dict[int, int] = {}

    total_eff = 0.0
    total_D = 0

    for k in range(N_tr):
        i = treat_idx[k]
        t0 = int(T0_vec[k])
        for j in range(T):
            if np.isnan(Y[i, j]):
                continue
            et = j - t0   # event-time: 0 = first treatment period
            gap = float(Y[i, j] - Y_ct[k, j])
            et_sums[et] = et_sums.get(et, 0.0) + gap
            et_counts[et] = et_counts.get(et, 0) + 1
            # att_avg only over treated cells
            if D[i, j] == 1:
                total_eff += gap
                total_D += 1

    event_times_sorted = sorted(et_sums.keys())
    att = np.array([et_sums[et] / et_counts[et] for et in event_times_sorted])
    att_time = np.array(event_times_sorted, dtype=int)
    att_avg = total_eff / total_D if total_D > 0 else 0.0

    return att, att_time, att_avg


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
    force: str = "unit",
) -> dict:
    """
    Generalized Synthetic Control estimator.

    Factors are estimated **only** from control units.  Treated units'
    factor loadings are recovered by projecting their pre-treatment
    outcomes onto the estimated factor space.

    Bug fixes vs original:
    - ``att_avg`` is now treatment-cell weighted (matches R ``sum(eff*D)/sum(D)``).
    - ``att`` / ``att_time`` use relative **event-time** (0 = first treatment
      period), not calendar time, and include pre-treatment periods.
    - When ``force`` includes unit fixed effects the factor matrix is augmented
      with a constant column so treated-unit FEs are estimated jointly with
      loadings from pre-treatment data (matching R ``fect_nevertreated``).

    Parameters
    ----------
    Y        : (N, T) demeaned/residualized outcome
    D        : (N, T) treatment indicator
    ctrl_idx : indices of control units in Y
    treat_idx: indices of treated units in Y
    T0_vec   : pre-treatment period count per treated unit
    r        : number of factors
    force    : fixed-effects specification (passed in to determine augmentation)

    Returns
    -------
    dict with keys: F, Lambda, Y_ct, att, att_time, att_avg, residuals, mse
    """
    N, T = Y.shape
    Y_ctrl = Y[ctrl_idx]                         # (N_co, T)
    obs_ctrl = _nan_obs_mask(Y_ctrl)

    F, Lambda_ctrl, resid_ctrl, mse = _ife_als(Y_ctrl, obs_ctrl, r, tol)

    # ---------------------------------------------------------------
    # Estimate factor loadings for treated units from pre-treatment data.
    #
    # FIX (Bug 3): When force includes unit FE, augment F with a constant
    # column so that the treated unit's unit FE is estimated jointly with
    # factor loadings from pre-treatment data — matching R's:
    #   F.hat <- cbind(F.hat, rep(1, TT))   # when force %in% c(1, 3)
    #   lambda.tr <- solve(F.hat.pre' @ F.hat.pre) @ F.hat.pre' @ U.tr.pre
    # ---------------------------------------------------------------
    has_unit_fe = force in ("unit", "two-way")

    Lambda_treat = np.zeros((len(treat_idx), r))
    alpha_treat  = np.zeros(len(treat_idx))    # treated unit FEs (if applicable)

    if r > 0 or has_unit_fe:
        for k, i in enumerate(treat_idx):
            t0 = int(T0_vec[k])
            # Build augmented factor matrix [F | 1] if unit FE present
            if has_unit_fe and r > 0:
                ones_col = np.ones((T, 1))
                F_aug = np.hstack([F, ones_col])     # (T, r+1)
            elif has_unit_fe:
                F_aug = np.ones((T, 1))              # (T, 1) — only unit FE
            else:
                F_aug = F                            # (T, r)

            r_aug = F_aug.shape[1]

            F_pre = F_aug[:t0]                       # (T0, r_aug)
            Y_pre = Y[i, :t0]                        # (T0,)
            obs_pre = ~np.isnan(Y_pre)
            if obs_pre.sum() < r_aug:
                continue

            coef = np.linalg.lstsq(F_pre[obs_pre], Y_pre[obs_pre], rcond=None)[0]

            if r > 0:
                Lambda_treat[k] = coef[:r]
            if has_unit_fe:
                alpha_treat[k] = float(coef[-1])

    # ---------------------------------------------------------------
    # Counterfactuals: Y_ct_i = Lambda_i @ F.T  (+ alpha_i if unit FE)
    # ---------------------------------------------------------------
    Y_ct = np.zeros((len(treat_idx), T))
    if r > 0:
        Y_ct = Lambda_treat @ F.T                    # (N_tr, T)
    if has_unit_fe:
        Y_ct += alpha_treat[:, None]                 # broadcast unit FEs

    # ---------------------------------------------------------------
    # FIX (Bugs 1 & 2): ATT by event-time; att_avg treatment-cell weighted
    # ---------------------------------------------------------------
    att, att_time, att_avg = _compute_att_by_event_time(
        Y, Y_ct, D, treat_idx, T0_vec
    )

    # Build full Lambda (all units)
    Lambda = np.zeros((N, max(r, 1)))
    if r > 0:
        Lambda[ctrl_idx] = Lambda_ctrl
        Lambda[treat_idx] = Lambda_treat

    return {
        "F": F,
        "Lambda": Lambda,
        "Lambda_ctrl": Lambda_ctrl,
        "Lambda_treat": Lambda_treat,
        "Y_ct": Y_ct,
        "att": att,
        "att_time": att_time,
        "att_avg": att_avg,
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

    FIX: att_avg is now treatment-cell weighted; att uses event-time.
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

    # FIX (Bugs 1 & 2): event-time ATT, treatment-cell weighted avg
    att, att_time, att_avg = _compute_att_by_event_time(
        Y, Y_ct, D, treat_idx, T0_vec
    )

    return {
        "F": F,
        "Lambda": Lambda,
        "Lambda_ctrl": Lambda[ctrl_idx],
        "Lambda_treat": Lambda[treat_idx],
        "Y_ct": Y_ct,
        "att": att,
        "att_time": att_time,
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

    FIX: att_avg is now treatment-cell weighted; att uses event-time.

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

    # FIX (Bugs 1 & 2): event-time ATT, treatment-cell weighted avg
    att, att_time, att_avg = _compute_att_by_event_time(
        Y, Y_ct, D, treat_idx, T0_vec
    )

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
        "att_time": att_time,
        "att_avg": att_avg,
        "residuals": resid,
        "mse": mse,
        "r": r,
    }
