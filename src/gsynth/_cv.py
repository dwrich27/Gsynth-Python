"""
Cross-validation utilities for selecting the number of factors (r) or
regularisation parameter (lambda) in gsynth / mc estimators.
"""
from __future__ import annotations

import numpy as np

from ._estimators import _ife_als, _nan_obs_mask, _nuclear_norm_minimise

# ---------------------------------------------------------------------------
# CV for factor number (r) — gsynth: LOO on treated pre-treatment data
# ---------------------------------------------------------------------------

def cv_factor_number_gsynth(
    Y: np.ndarray,
    D: np.ndarray,
    ctrl_idx: np.ndarray,
    treat_idx: np.ndarray,
    T0_vec: np.ndarray,
    r_candidates: list[int],
    force: str = "unit",
    tol: float = 1e-4,
    rng: np.random.Generator | None = None,
) -> tuple[int, np.ndarray]:
    """
    Leave-one-out CV on treated units' pre-treatment data for gsynth.

    FIX (Bug 9): R's gsynth (fect_nevertreated) selects r by holding out
    individual pre-treatment periods from treated units (LOO), not by
    masking control-unit cells.

    Algorithm
    ---------
    For each candidate r:
      For each unique pre-treatment time t (across treated units):
        1. Estimate factors on control group with full data.
        2. Project treated unit i onto factors using pre-treatment periods
           EXCLUDING time t.
        3. Compute prediction error at time t.
      MSPE = mean squared prediction error over all LOO held-out cells.
    Select r minimising MSPE.

    Parameters
    ----------
    Y         : (N, T) demeaned outcome
    D         : (N, T) treatment indicator
    ctrl_idx  : control unit indices
    treat_idx : treated unit indices
    T0_vec    : (N_tr,) pre-treatment period count per treated unit
    r_candidates : list of r values to evaluate
    force     : fixed-effects spec (affects whether constant col is appended)
    tol       : ALS convergence tolerance

    Returns
    -------
    r_opt     : selected number of factors
    cv_scores : (len(r_candidates),) mean MSPE per r
    """
    if rng is None:
        rng = np.random.default_rng(0)

    N, T = Y.shape
    Y_ctrl = Y[ctrl_idx]
    obs_ctrl = _nan_obs_mask(Y_ctrl)
    has_unit_fe = force in ("unit", "two-way")

    cv_scores = np.zeros(len(r_candidates))

    for ri, r in enumerate(r_candidates):
        # Estimate factors on control group
        F, _, _, _ = _ife_als(Y_ctrl, obs_ctrl, r, tol)

        if r == 0 and not has_unit_fe:
            # No factors and no unit FE: all predictions = 0, error = var(Y_pre)
            err_sq = []
            for k, i in enumerate(treat_idx):
                t0 = int(T0_vec[k])
                y_pre = Y[i, :t0]
                obs_pre = ~np.isnan(y_pre)
                if obs_pre.sum() > 0:
                    err_sq.extend((y_pre[obs_pre] ** 2).tolist())
            cv_scores[ri] = float(np.mean(err_sq)) if err_sq else np.inf
            continue

        # Build augmented factor matrix
        if has_unit_fe and r > 0:
            ones_col = np.ones((T, 1))
            F_aug = np.hstack([F, ones_col])   # (T, r+1)
        elif has_unit_fe:
            F_aug = np.ones((T, 1))            # (T, 1)
        else:
            F_aug = F                          # (T, r)

        r_aug = F_aug.shape[1]

        # LOO over each pre-treatment time point
        # Collect all pre-treatment (unit, time) cells from treated units
        pre_cells: list[tuple[int, int]] = []   # (treat_unit_k, time_j)
        for k, i in enumerate(treat_idx):
            t0 = int(T0_vec[k])
            for j in range(t0):
                if not np.isnan(Y[i, j]):
                    pre_cells.append((k, j))

        if len(pre_cells) == 0:
            cv_scores[ri] = np.inf
            continue

        # Unique pre-treatment time indices across all treated units
        unique_pre_times = sorted({j for _, j in pre_cells})

        sq_errors = []
        for t_held in unique_pre_times:
            for k, i in enumerate(treat_idx):
                t0 = int(T0_vec[k])
                if t_held >= t0 or np.isnan(Y[i, t_held]):
                    continue

                # Train on pre-treatment periods excluding t_held
                train_times = [j for j in range(t0) if j != t_held and not np.isnan(Y[i, j])]
                if len(train_times) < r_aug:
                    continue

                F_train = F_aug[train_times]            # (T_train, r_aug)
                Y_train = Y[i, train_times]             # (T_train,)
                coef = np.linalg.lstsq(F_train, Y_train, rcond=None)[0]

                pred = float(F_aug[t_held] @ coef)
                sq_errors.append((Y[i, t_held] - pred) ** 2)

        cv_scores[ri] = float(np.mean(sq_errors)) if sq_errors else np.inf

    r_opt = r_candidates[int(np.argmin(cv_scores))]
    return r_opt, cv_scores


# ---------------------------------------------------------------------------
# CV for factor number (r) — ife: K-fold on control units (original)
# ---------------------------------------------------------------------------

def cv_factor_number(
    Y_ctrl: np.ndarray,
    r_candidates: list[int],
    k: int = 5,
    criterion: str = "mspe",
    tol: float = 1e-4,
    rng: np.random.Generator | None = None,
) -> tuple[int, np.ndarray]:
    """
    K-fold cross-validation over control units to select the number of factors.

    Used by the ``ife`` estimator.  For ``gsynth`` use
    ``cv_factor_number_gsynth`` instead (LOO on treated pre-treatment data).
    """
    if rng is None:
        rng = np.random.default_rng(0)

    obs_mask = _nan_obs_mask(Y_ctrl)
    obs_idx = np.argwhere(obs_mask)          # (n_obs, 2)
    n_obs = len(obs_idx)

    fold_ids = np.tile(np.arange(k), int(np.ceil(n_obs / k)))[:n_obs]
    rng.shuffle(fold_ids)

    cv_scores = np.zeros(len(r_candidates))

    for ri, r in enumerate(r_candidates):
        fold_mse = []
        for fold in range(k):
            test_pos = obs_idx[fold_ids == fold]
            train_mask = obs_mask.copy()
            train_mask[test_pos[:, 0], test_pos[:, 1]] = False

            F, Lambda, _, _ = _ife_als(Y_ctrl, train_mask, r, tol)
            if r == 0:
                fitted = np.zeros_like(Y_ctrl)
            else:
                fitted = Lambda @ F.T

            test_vals = Y_ctrl[test_pos[:, 0], test_pos[:, 1]]
            pred_vals = fitted[test_pos[:, 0], test_pos[:, 1]]
            fold_mse.append(np.mean((test_vals - pred_vals) ** 2))

        if criterion == "pc":
            N_co, T = Y_ctrl.shape
            sigma2 = np.mean(fold_mse)
            penalty = r * sigma2 * (N_co + T - r) / (N_co * T) * np.log(
                (N_co * T) / (N_co + T)
            )
            cv_scores[ri] = sigma2 + penalty
        else:
            cv_scores[ri] = float(np.mean(fold_mse))

    r_opt = r_candidates[int(np.argmin(cv_scores))]
    return r_opt, cv_scores


# ---------------------------------------------------------------------------
# CV for lambda — used by mc estimator
# ---------------------------------------------------------------------------

def cv_lambda(
    Y: np.ndarray,
    D: np.ndarray,
    ctrl_idx: np.ndarray,
    lambda_seq: np.ndarray,
    k: int = 5,
    tol: float = 1e-4,
    rng: np.random.Generator | None = None,
) -> tuple[float, np.ndarray]:
    """
    K-fold cross-validation over control-unit observations to select lambda.
    """
    if rng is None:
        rng = np.random.default_rng(0)

    Y_ctrl = Y[ctrl_idx]
    obs_mask = _nan_obs_mask(Y_ctrl)
    obs_idx = np.argwhere(obs_mask)
    n_obs = len(obs_idx)

    fold_ids = np.tile(np.arange(k), int(np.ceil(n_obs / k)))[:n_obs]
    rng.shuffle(fold_ids)

    cv_scores = np.zeros(len(lambda_seq))

    for li, lam in enumerate(lambda_seq):
        fold_mse = []
        for fold in range(k):
            test_pos = obs_idx[fold_ids == fold]
            train_mask = obs_mask.copy()
            train_mask[test_pos[:, 0], test_pos[:, 1]] = False

            M_hat = _nuclear_norm_minimise(Y_ctrl, train_mask, lam, tol)

            test_vals = Y_ctrl[test_pos[:, 0], test_pos[:, 1]]
            pred_vals = M_hat[test_pos[:, 0], test_pos[:, 1]]
            fold_mse.append(float(np.mean((test_vals - pred_vals) ** 2)))

        cv_scores[li] = float(np.mean(fold_mse))

    lam_opt = float(lambda_seq[int(np.argmin(cv_scores))])
    return lam_opt, cv_scores


def build_lambda_seq(Y_ctrl: np.ndarray, nlambda: int = 10) -> np.ndarray:
    """
    Build a log-spaced lambda sequence from lambda_max down to lambda_max/100.
    """
    try:
        s = np.linalg.svd(np.nan_to_num(Y_ctrl), compute_uv=False)
        lam_max = float(s[0])
    except np.linalg.LinAlgError:
        lam_max = 1.0

    lam_min = lam_max / 100.0
    return np.logspace(np.log10(lam_max), np.log10(max(lam_min, 1e-6)), nlambda)
