"""
Bootstrap and jackknife inference for the Generalized Synthetic Control Method.

Three routines:
  - ``parametric_bootstrap``: R-matching pseudo-treatment bootstrap.
      Picks one control unit as a "fake treated" unit, resamples remaining
      controls, runs gsynth on the pseudo-dataset.  SEs come from the
      distribution of att_avg (which has ground truth = 0).
  - ``nonparametric_bootstrap``: block-resample units with replacement.
  - ``jackknife_inference``: leave-one-unit-out.
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
    force: str = "unit",
) -> tuple[np.ndarray, float]:
    """Fit the model on one bootstrap replicate and return ATT array + scalar."""
    from ._estimators import estimate_gsynth, estimate_ife, estimate_mc

    if estimator == "mc":
        res = estimate_mc(Y_boot, D_boot, ctrl_idx, treat_idx, T0_boot, lam or 1.0, tol)
    elif estimator == "ife":
        res = estimate_ife(Y_boot, D_boot, ctrl_idx, treat_idx, T0_boot, r, tol)
    else:
        res = estimate_gsynth(Y_boot, D_boot, ctrl_idx, treat_idx, T0_boot, r, tol, force)

    # Align ATT to the global post_times grid (event-time)
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
    att_full: np.ndarray,
    att_avg_full: float,
) -> dict:
    """
    Pseudo-treatment parametric bootstrap (matching R gsynth).

    FIX (Bug 6 / Bug 5): R's parametric bootstrap for gsynth works as follows:
      1.  Select one control unit at random as the "fake treated" unit.
      2.  Resample the remaining control units with replacement.
      3.  Assign treatment indicator to the fake-treated unit using the
          same adoption timing as the treated group (T0 = same as real T0).
      4.  Run gsynth on this pseudo-dataset.
      5.  Collect att_avg (ground truth = 0 for the fake unit).
      6.  Repeat nboots times; SE = std(att_avg_boots).

    Parameters
    ----------
    Y, D     : full panel matrices (N, T)
    att_full : (len(post_times),) point estimates — used to center CIs
    att_avg_full : float point estimate
    """
    rng = np.random.default_rng(seed)
    N, T = Y.shape

    # Identify valid control units:
    # need enough pre-treatment obs to estimate loadings
    min_t0 = int(np.min(T0_vec)) if len(T0_vec) else T // 2
    has_unit_fe = force in ("unit", "two-way")
    min_obs_needed = r + (1 if has_unit_fe else 0)

    # Valid controls: those with at least (r+1) non-missing obs in pre-period
    valid_ctrl = []
    for i in ctrl_idx:
        pre_obs = int(np.sum(~np.isnan(Y[i, :min_t0])))
        if pre_obs >= max(min_obs_needed, 1):
            valid_ctrl.append(i)

    if len(valid_ctrl) == 0:
        valid_ctrl = list(ctrl_idx)

    att_boots = np.full((nboots, len(post_times)), np.nan)
    att_avg_boots = np.full(nboots, np.nan)

    for b in range(nboots):
        # 1. Pick one valid control as fake-treated
        fake_tr_idx = int(rng.choice(valid_ctrl))

        # 2. Resample remaining controls with replacement
        ctrl_rest = [i for i in ctrl_idx if i != fake_tr_idx]
        if len(ctrl_rest) == 0:
            continue
        ctrl_boot = rng.choice(ctrl_rest, size=len(ctrl_idx), replace=True)

        # 3. Build pseudo-panel: fake_treated + resampled controls
        n_ctrl_b = len(ctrl_boot)
        n_boot = 1 + n_ctrl_b   # 1 fake treated + resampled controls

        Y_boot = np.vstack([Y[[fake_tr_idx]], Y[ctrl_boot]])
        D_boot_mat = np.zeros((n_boot, T))

        # Assign same treatment timing as real treated units (use T0 of first treated unit)
        t0_fake = int(T0_vec[0]) if len(T0_vec) > 0 else T // 2
        D_boot_mat[0, t0_fake:] = 1.0   # fake treated unit

        ctrl_boot_idx = np.arange(1, n_boot)
        treat_boot_idx = np.array([0])
        T0_boot = np.array([t0_fake])

        try:
            att_b, att_avg_b = _run_one_boot(
                Y_boot, D_boot_mat, ctrl_boot_idx, treat_boot_idx,
                T0_boot, r, lam, estimator, tol, post_times, force
            )
        except Exception:
            continue

        att_boots[b] = att_b
        att_avg_boots[b] = att_avg_b

    return _summarise_boots(att_boots, att_avg_boots, att_full, att_avg_full, alpha_level)


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
    att_full: np.ndarray,
    att_avg_full: float,
) -> dict:
    """
    Block bootstrap at unit level (or cluster level when cl_vec is given).

    Units are resampled with replacement, preserving the full time series.
    Treated and control units are resampled independently.
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

        n_ctrl_b = len(ctrl_samp)
        n_treat_b = len(treat_samp)
        n_boot = n_ctrl_b + n_treat_b

        Y_boot = np.vstack([Y[ctrl_samp], Y[treat_samp]])
        D_boot = np.vstack([D[ctrl_samp], D[treat_samp]])

        ctrl_boot = np.arange(n_ctrl_b)
        treat_boot = np.arange(n_ctrl_b, n_boot)

        T0_boot = np.array([
            int(np.argmax(D[i] == 1)) if D[i].any() else T
            for i in treat_samp
        ], dtype=int)
        T0_boot = np.maximum(T0_boot, 1)

        try:
            att_b, att_avg_b = _run_one_boot(
                Y_boot, D_boot, ctrl_boot, treat_boot, T0_boot,
                r, lam, estimator, tol, post_times, force
            )
        except Exception:
            continue

        att_boots[b] = att_b
        att_avg_boots[b] = att_avg_b

    return _summarise_boots(att_boots, att_avg_boots, att_full, att_avg_full, alpha_level)


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
        ctrl_j = np.array([i for i in ctrl_idx if i != drop])
        treat_j = np.array([i for i in treat_idx if i != drop])
        if len(treat_j) == 0 or len(ctrl_j) == 0:
            continue

        T0_j = T0_vec[[kk for kk, ii in enumerate(treat_idx) if ii != drop]]

        keep = np.concatenate([ctrl_j, treat_j])
        # Re-index into compressed panel
        orig_to_new = {orig: new for new, orig in enumerate(sorted(keep))}
        ctrl_jj = np.array([orig_to_new[i] for i in ctrl_j])
        treat_jj = np.array([orig_to_new[i] for i in treat_j])

        Y_j = Y[sorted(keep)]
        D_j = D[sorted(keep)]

        try:
            att_b, att_avg_b = _run_one_boot(
                Y_j, D_j, ctrl_jj, treat_jj, T0_j,
                r, lam, estimator, tol, post_times, force
            )
        except Exception:
            continue

        att_jacks[b] = att_b
        att_avg_jacks[b] = att_avg_b

    # Jackknife SE = sqrt((n-1)/n * sum((x_i - x_bar)^2))
    n_valid = int(np.sum(~np.isnan(att_avg_jacks)))
    jk_se_avg = float(np.sqrt(
        (n_valid - 1) / n_valid * np.nanvar(att_avg_jacks)
    )) if n_valid > 1 else 0.0

    jk_se = np.array([
        float(np.sqrt((n_valid - 1) / n_valid * np.nanvar(att_jacks[:, j])))
        if np.sum(~np.isnan(att_jacks[:, j])) > 1 else 0.0
        for j in range(len(post_times))
    ])

    z = _z_score(alpha_level)

    return {
        "att_se": jk_se,
        "att_ci_lower": att_full - z * jk_se,
        "att_ci_upper": att_full + z * jk_se,
        "att_avg_se": jk_se_avg,
        "att_avg_ci_lower": att_avg_full - z * jk_se_avg,
        "att_avg_ci_upper": att_avg_full + z * jk_se_avg,
        "att_boot": att_jacks,
    }


def _z_score(alpha_level: float) -> float:
    """Normal quantile for (1 - alpha/2) CI via bisection (no scipy)."""
    hi = 1.0 - alpha_level / 2
    # Use large sample from standard normal to get empirical quantile
    rng = np.random.default_rng(42)
    return float(np.quantile(np.abs(rng.standard_normal(100_000)), hi))


def _summarise_boots(
    att_boots: np.ndarray,
    att_avg_boots: np.ndarray,
    att_full: np.ndarray,
    att_avg_full: float,
    alpha_level: float,
) -> dict:
    """
    Compute SE and CI from bootstrap replications.

    CIs use normal approximation centred on point estimates (matching R).
    """
    z = _z_score(alpha_level)

    # Per-period SE
    att_se = np.nanstd(att_boots, axis=0, ddof=1)
    att_ci_lo = att_full - z * att_se
    att_ci_hi = att_full + z * att_se

    # Overall SE
    se_avg = float(np.nanstd(att_avg_boots, ddof=1))
    ci_lo_avg = att_avg_full - z * se_avg
    ci_hi_avg = att_avg_full + z * se_avg

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
