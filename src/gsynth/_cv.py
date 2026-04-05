"""
Cross-validation utilities for selecting the number of factors (r) or
regularisation parameter (lambda) in gsynth / mc estimators.
"""
from __future__ import annotations

import numpy as np

from ._estimators import _ife_als, _nan_obs_mask, _nuclear_norm_minimise

# ---------------------------------------------------------------------------
# CV for factor number (r) — used by gsynth / ife
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

    A random subset of observed (unit, time) cells in the control panel is held
    out; the model is fit on the remaining cells and MSPE is computed on the
    holdout cells.  This is repeated k times.

    Parameters
    ----------
    Y_ctrl       : (N_co, T) control outcome matrix
    r_candidates : list of r values to evaluate
    k            : number of folds
    criterion    : "mspe" (mean squared prediction error) or "pc"

    Returns
    -------
    r_opt    : selected number of factors
    cv_scores : (len(r_candidates),) mean CV criterion per r
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
            # Bai & Ng (2002) information criterion
            # PC_p1: sigma^2 * (N_co + T - r) / (N_co * T) * ln(N_co * T / (N_co + T))
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

    Parameters
    ----------
    Y          : (N, T) full outcome matrix
    D          : (N, T) treatment indicator
    ctrl_idx   : control unit indices
    lambda_seq : sequence of lambda values to evaluate (decreasing order)
    k          : number of CV folds

    Returns
    -------
    lam_opt   : selected lambda
    cv_scores : (len(lambda_seq),) mean CV MSPE per lambda
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

    lambda_max is the largest singular value of Y_ctrl (divided by
    sqrt(N*T)), which is the smallest lambda that produces the zero solution.
    """
    try:
        s = np.linalg.svd(np.nan_to_num(Y_ctrl), compute_uv=False)
        lam_max = float(s[0])
    except np.linalg.LinAlgError:
        lam_max = 1.0

    lam_min = lam_max / 100.0
    return np.logspace(np.log10(lam_max), np.log10(max(lam_min, 1e-6)), nlambda)
