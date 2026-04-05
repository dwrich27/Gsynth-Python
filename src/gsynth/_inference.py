"""
Bootstrap and jackknife inference for the Generalized Synthetic Control Method.

Two main routines:
  - ``parametric_bootstrap``: resample from estimated error distribution
  - ``nonparametric_bootstrap``: block-resample units with replacement

Both return a (nboots, T_post) array of replicated ATT estimates plus the
overall ATT scalar for each replication.
"""
from __future__ import annotations

from typing import Optional

import numpy as np


def _run_one_boot(
    Y_boot: np.ndarray,
    D_boot: np.ndarray,
    ctrl_idx: np.ndarray,
    treat_idx: np.ndarray,
    T0_boot: np.ndarray,
    r: int,
    lam: Optional[float],
    estimator: str,
    tol: float,
    post_times: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Fit the model on one bootstrap replicate and return ATT array + scalar."""
    from ._estimators import estimate_gsynth, estimate_ife, estimate_mc

    if estimator == "mc":
        res = estimate_mc(Y_boot, D_boot, ctrl_idx, treat_idx, T0_boot, lam or 1.0, tol)
    elif estimator == "ife":
        res = estimate_ife(Y_boot, D_boot, ctrl_idx, treat_idx, T0_boot, r, tol)
    else:
        res = estimate_gsynth(Y_boot, D_boot, ctrl_idx, treat_idx, T0_boot, r, tol)

    # Align ATT to the global post_times grid
    att_full = np.full(len(post_times), np.nan)
    if len(res["att_time"]) > 0:
        for jj, t in enumerate(post_times):
            match = np.where(res["att_time"] == t)[0]
            if len(match):
                att_full[jj] = res["att"][match[0]]

    return att_full, res["att_avg"]


def parametric_bootstrap(
    Y: np.ndarray,
    D: np.ndarray,
    alpha_hat: Optional[np.ndarray],
    xi_hat: Optional[np.ndarray],
    beta_hat: Optional[np.ndarray],
    X_mat: Optional[np.ndarray],
    F_hat: Optional[np.ndarray],
    Lambda_hat: Optional[np.ndarray],
    M_hat: Optional[np.ndarray],
    sigma2: float,
    ctrl_idx: np.ndarray,
    treat_idx: np.ndarray,
    T0_vec: np.ndarray,
    r: int,
    lam: Optional[float],
    estimator: str,
    nboots: int,
    alpha_level: float,
    post_times: np.ndarray,
    force: str,
    tol: float,
    seed: Optional[int],
) -> dict:
    """
    Two-stage parametric bootstrap.

    1.  Draw error matrix  ε* ~ N(0, σ²)
    2.  Reconstruct  Y* = fitted + ε*
    3.  Re-estimate the model on Y*
    4.  Store ATT replications
    """
    rng = np.random.default_rng(seed)
    N, T = Y.shape

    # Build fitted values from estimated components
    if estimator == "mc" and M_hat is not None:
        Y_fitted = M_hat.copy()
    else:
        Y_fitted = np.zeros((N, T))
        if alpha_hat is not None:
            Y_fitted += alpha_hat[:, None]
        if xi_hat is not None:
            Y_fitted += xi_hat[None, :]
        if F_hat is not None and Lambda_hat is not None and r > 0:
            Y_fitted += Lambda_hat @ F_hat.T
        if beta_hat is not None and X_mat is not None:
            for k_x in range(len(beta_hat)):
                Y_fitted += beta_hat[k_x] * X_mat[:, :, k_x]

    att_boots = np.full((nboots, len(post_times)), np.nan)
    att_avg_boots = np.full(nboots, np.nan)

    std = float(np.sqrt(max(sigma2, 1e-12)))

    for b in range(nboots):
        eps = rng.normal(0.0, std, size=(N, T))
        Y_boot = np.where(~np.isnan(Y), Y_fitted + eps, np.nan)

        try:
            att_full, att_avg = _run_one_boot(
                Y_boot, D, ctrl_idx, treat_idx, T0_vec, r, lam, estimator, tol, post_times
            )
        except Exception:
            continue

        att_boots[b] = att_full
        att_avg_boots[b] = att_avg

    return _summarise_boots(att_boots, att_avg_boots, alpha_level)


def nonparametric_bootstrap(
    Y: np.ndarray,
    D: np.ndarray,
    ctrl_idx: np.ndarray,
    treat_idx: np.ndarray,
    T0_vec: np.ndarray,
    r: int,
    lam: Optional[float],
    estimator: str,
    nboots: int,
    alpha_level: float,
    post_times: np.ndarray,
    force: str,
    tol: float,
    seed: Optional[int],
    cl_vec: Optional[np.ndarray],
) -> dict:
    """
    Block bootstrap at unit level (or cluster level when cl_vec is given).

    Units are resampled with replacement, preserving the full time series for
    each unit.  The treatment / control split is preserved by resampling
    treated and control units independently.
    """
    rng = np.random.default_rng(seed)
    N, T = Y.shape

    att_boots = np.full((nboots, len(post_times)), np.nan)
    att_avg_boots = np.full(nboots, np.nan)

    # Define sampling blocks
    if cl_vec is not None:
        ctrl_blocks = np.unique(cl_vec[ctrl_idx])
        treat_blocks = np.unique(cl_vec[treat_idx])
    else:
        ctrl_blocks = ctrl_idx
        treat_blocks = treat_idx

    for b in range(nboots):
        # Resample control blocks
        if cl_vec is not None:
            ctrl_samp_blocks = rng.choice(ctrl_blocks, size=len(ctrl_blocks), replace=True)
            ctrl_samp = np.concatenate([
                np.where(cl_vec == blk)[0] for blk in ctrl_samp_blocks
            ])
            treat_samp_blocks = rng.choice(treat_blocks, size=len(treat_blocks), replace=True)
            treat_samp = np.concatenate([
                np.where(cl_vec == blk)[0] for blk in treat_samp_blocks
            ])
        else:
            ctrl_samp = rng.choice(ctrl_idx, size=len(ctrl_idx), replace=True)
            treat_samp = rng.choice(treat_idx, size=len(treat_idx), replace=True)

        # Reconstruct boot panel
        n_ctrl_b = len(ctrl_samp)
        n_treat_b = len(treat_samp)
        n_boot = n_ctrl_b + n_treat_b

        Y_boot = np.vstack([Y[ctrl_samp], Y[treat_samp]])
        D_boot = np.vstack([D[ctrl_samp], D[treat_samp]])

        ctrl_boot = np.arange(n_ctrl_b)
        treat_boot = np.arange(n_ctrl_b, n_boot)

        # T0_vec for resampled treat units
        T0_boot = np.array([
            int(np.argmax(D[i] == 1)) if D[i].any() else T
            for i in treat_samp
        ], dtype=int)
        T0_boot = np.maximum(T0_boot, 1)

        try:
            att_full, att_avg = _run_one_boot(
                Y_boot, D_boot, ctrl_boot, treat_boot, T0_boot,
                r, lam, estimator, tol, post_times
            )
        except Exception:
            continue

        att_boots[b] = att_full
        att_avg_boots[b] = att_avg

    return _summarise_boots(att_boots, att_avg_boots, alpha_level)


def jackknife_inference(
    Y: np.ndarray,
    D: np.ndarray,
    ctrl_idx: np.ndarray,
    treat_idx: np.ndarray,
    T0_vec: np.ndarray,
    r: int,
    lam: Optional[float],
    estimator: str,
    alpha_level: float,
    post_times: np.ndarray,
    force: str,
    tol: float,
    att_full: np.ndarray,
    att_avg_full: float,
) -> dict:
    """
    Leave-one-unit-out jackknife inference.
    """
    all_idx = np.concatenate([ctrl_idx, treat_idx])
    n_jack = len(all_idx)

    att_jacks = np.full((n_jack, len(post_times)), np.nan)
    att_avg_jacks = np.full(n_jack, np.nan)

    for b, drop in enumerate(all_idx):
        keep = all_idx[all_idx != drop]
        ctrl_j = np.array([i for i in ctrl_idx if i != drop])
        treat_j = np.array([i for i in treat_idx if i != drop])
        if len(treat_j) == 0 or len(ctrl_j) == 0:
            continue

        T0_j = T0_vec[[i for i, ii in enumerate(treat_idx) if ii != drop]]

        Y_j = Y[keep]
        D_j = D[keep]
        ctrl_jj = np.where(np.isin(keep, ctrl_j))[0]
        treat_jj = np.where(np.isin(keep, treat_j))[0]

        try:
            att_full_j, att_avg_j = _run_one_boot(
                Y_j, D_j, ctrl_jj, treat_jj, T0_j, r, lam, estimator, tol, post_times
            )
        except Exception:
            continue

        att_jacks[b] = att_full_j
        att_avg_jacks[b] = att_avg_j

    # Jackknife SE = sqrt((n-1)/n * sum((x_i - x_bar)^2))
    n_valid = np.sum(~np.isnan(att_avg_jacks))
    jk_se_avg = float(np.sqrt(
        (n_valid - 1) / n_valid * np.nanvar(att_avg_jacks)
    )) if n_valid > 1 else 0.0

    jk_se = np.array([
        float(np.sqrt((n_valid - 1) / n_valid * np.nanvar(att_jacks[:, j])))
        if np.sum(~np.isnan(att_jacks[:, j])) > 1 else 0.0
        for j in range(len(post_times))
    ])

    z = float(np.abs(np.quantile(np.random.default_rng(0).normal(size=10000), 1 - alpha_level / 2)))

    return {
        "att_se": jk_se,
        "att_ci_lower": att_full - z * jk_se,
        "att_ci_upper": att_full + z * jk_se,
        "att_avg_se": jk_se_avg,
        "att_avg_ci_lower": att_avg_full - z * jk_se_avg,
        "att_avg_ci_upper": att_avg_full + z * jk_se_avg,
        "att_boot": att_jacks,
    }


def _summarise_boots(
    att_boots: np.ndarray,
    att_avg_boots: np.ndarray,
    alpha_level: float,
) -> dict:
    """Compute SE and CI from bootstrap replications."""
    hi = 1.0 - alpha_level / 2
    z = float(np.quantile(np.abs(np.random.default_rng(99).standard_normal(50000)), hi))

    # Per-period
    att_se = np.nanstd(att_boots, axis=0, ddof=1)
    # Use normal-approximation CI (matching R gsynth behaviour)
    att_mean = np.nanmean(att_boots, axis=0)  # use boot mean for CI center
    att_ci_lo = att_mean - z * att_se
    att_ci_hi = att_mean + z * att_se

    # Overall
    se_avg = float(np.nanstd(att_avg_boots, ddof=1))
    avg_mean = float(np.nanmean(att_avg_boots))
    ci_lo_avg = avg_mean - z * se_avg
    ci_hi_avg = avg_mean + z * se_avg

    return {
        "att_se": att_se,
        "att_ci_lower": att_ci_lo,
        "att_ci_upper": att_ci_hi,
        "att_avg_se": se_avg,
        "att_avg_ci_lower": ci_lo_avg,
        "att_avg_ci_upper": ci_hi_avg,
        "att_boot": att_boots,
        "att_avg_boot": att_avg_boots,
    }
