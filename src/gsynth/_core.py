"""
gsynth() — main entry point for the Generalized Synthetic Control estimator.

Implements the three-stage workflow:
  1. Data preparation (parse, demean, partial-out covariates)
  2. Estimation (gsynth / ife / mc) with optional cross-validation
  3. Inference (parametric / nonparametric bootstrap or jackknife)
"""
from __future__ import annotations

import warnings
from typing import Optional, Union

import numpy as np
import pandas as pd

from ._cv import build_lambda_seq, cv_factor_number, cv_lambda
from ._data import demean_panel, parse_panel, partial_out_covariates
from ._estimators import estimate_gsynth, estimate_ife, estimate_mc
from ._inference import (
    jackknife_inference,
    nonparametric_bootstrap,
    parametric_bootstrap,
)
from ._result import GsynthResult


def gsynth(
    formula: Optional[str] = None,
    data: Optional[pd.DataFrame] = None,
    Y: Optional[str] = None,
    D: Optional[str] = None,
    X: Optional[Union[str, list[str]]] = None,
    na_rm: bool = False,
    index: Optional[list[str]] = None,
    weight: Optional[str] = None,
    force: str = "unit",
    cl: Optional[str] = None,
    r: Union[int, list[int]] = 0,
    lam: Optional[Union[float, list[float]]] = None,
    nlambda: int = 10,
    CV: bool = True,
    criterion: str = "mspe",
    k: int = 5,
    EM: bool = False,
    estimator: str = "gsynth",
    se: bool = False,
    nboots: int = 200,
    inference: Optional[str] = None,
    parallel: bool = False,
    cores: Optional[int] = None,
    tol: float = 1e-3,
    seed: Optional[int] = None,
    min_T0: int = 5,
    alpha: float = 0.05,
    normalize: bool = False,
) -> GsynthResult:
    """
    Generalized Synthetic Control Method.

    Estimates counterfactual outcomes and average treatment effects on the
    treated (ATT) for panel data with a binary treatment using Interactive
    Fixed Effects (IFE) or Matrix Completion (MC).

    Parameters
    ----------
    formula : str, optional
        R-style formula string, e.g. ``"Y ~ D + X1 + X2"``.  When provided,
        ``Y``, ``D``, and ``X`` are parsed from the formula; explicit keyword
        arguments take precedence if also supplied.
    data : pd.DataFrame
        Long-format panel data frame.
    Y : str
        Name of the outcome variable column.
    D : str
        Name of the binary treatment variable column (0/1).
    X : str or list of str, optional
        Name(s) of time-varying covariate column(s).
    na_rm : bool
        If True, drop rows with any missing value in Y, D, or X before
        estimation.  Default False (listwise deletion for covariates only).
    index : list of str
        Two-element list ``[unit_col, time_col]`` identifying the panel
        dimensions.
    weight : str, optional
        Column name for unit-level inverse-probability or analytic weights.
    force : str
        Fixed-effects specification: ``"none"``, ``"unit"``, ``"time"``, or
        ``"two-way"``.  Default ``"unit"``.
    cl : str, optional
        Column name for cluster variable used in block bootstrap.  Defaults
        to unit-level blocking when None.
    r : int or list of int
        Number of latent factors.  When ``CV=True`` and ``r=0`` (default),
        factors are selected by cross-validation over ``{0, 1, 2, 3, 4, 5}``.
        Pass a list to restrict the search grid.
    lam : float or list of float, optional
        Regularisation parameter(s) for matrix completion.  When None and
        ``CV=True``, a log-spaced sequence of length ``nlambda`` is searched.
    nlambda : int
        Number of lambda values in the auto-generated search grid.
    CV : bool
        If True, use cross-validation to select r (IFE) or lambda (MC).
    criterion : str
        CV selection criterion: ``"mspe"`` (default) or ``"pc"`` (Bai-Ng
        information criterion).
    k : int
        Number of CV folds.  Default 5.
    EM : bool
        If True, use the EM / IFE algorithm that also conditions on treated
        units' pre-treatment outcomes (``estimator="ife"`` shorthand).
        Ignored when ``estimator`` is set explicitly.
    estimator : str
        Estimation method: ``"gsynth"`` (default), ``"ife"``, or ``"mc"``.
    se : bool
        If True, compute standard errors and confidence intervals via
        bootstrap.  Default False.
    nboots : int
        Number of bootstrap replications.  Default 200.
    inference : str, optional
        Bootstrap type: ``"parametric"`` or ``"nonparametric"``.  When None,
        defaults to ``"parametric"`` for small treatment groups (N_tr < 40)
        and ``"nonparametric"`` otherwise.  ``"jackknife"`` is also accepted.
    parallel : bool
        Reserved for future parallel bootstrap implementation.  Default False.
    cores : int, optional
        Reserved for future use.
    tol : float
        Convergence tolerance for ALS / MC iterations.  Default 1e-3.
    seed : int, optional
        Random seed for bootstrap reproducibility.
    min_T0 : int
        Minimum number of pre-treatment periods required for a treated unit.
        Units with fewer periods are dropped with a warning.  Default 5.
    alpha : float
        Significance level for confidence intervals.  Default 0.05 (95% CIs).
    normalize : bool
        If True, standardise Y and X before estimation (can speed up
        convergence; results are back-transformed).  Default False.

    Returns
    -------
    GsynthResult
        Fitted result object.  See :class:`~gsynth.GsynthResult` for a full
        description of all attributes.

    Examples
    --------
    >>> import pandas as pd
    >>> from gsynth import gsynth
    >>> result = gsynth(
    ...     formula="Y ~ D",
    ...     data=df,
    ...     index=["unit", "year"],
    ...     se=True,
    ...     nboots=200,
    ... )
    >>> print(result)
    >>> from gsynth import plot
    >>> plot(result, type="gap")
    """
    # ------------------------------------------------------------------
    # 0. Parse formula
    # ------------------------------------------------------------------
    if formula is not None:
        Y, D, X = _parse_formula(formula, Y, D, X)

    if data is None:
        raise ValueError("'data' is required.")
    if Y is None or D is None:
        raise ValueError("Both 'Y' and 'D' must be specified (or via formula).")
    if index is None or len(index) != 2:
        raise ValueError("'index' must be a two-element list: [unit_col, time_col].")

    if isinstance(X, str):
        X = [X]
    if EM and estimator == "gsynth":
        estimator = "ife"

    estimator = estimator.lower()
    if estimator not in ("gsynth", "ife", "mc"):
        raise ValueError(f"Unknown estimator '{estimator}'. Choose: 'gsynth', 'ife', 'mc'.")
    if force not in ("none", "unit", "time", "two-way"):
        raise ValueError(f"Unknown force '{force}'. Choose: 'none','unit','time','two-way'.")

    # ------------------------------------------------------------------
    # 1. Parse panel data
    # ------------------------------------------------------------------
    panel = parse_panel(data, Y, D, X or [], index, weight, cl, na_rm, min_T0)
    Y_mat   = panel["Y_mat"]    # (N, T)
    D_mat   = panel["D_mat"]
    X_mat   = panel["X_mat"]    # (N, T, K) or None
    cl_vec  = panel["cl_vec"]
    units   = panel["units"]
    times   = panel["times"]
    N, T    = panel["N"], panel["T"]
    ctrl_idx  = panel["ctrl_idx"]
    treat_idx = panel["treat_idx"]
    T0_vec    = panel["T0_vec"]

    # ------------------------------------------------------------------
    # 2. Normalise (optional)
    # ------------------------------------------------------------------
    Y_scale, Y_center = 1.0, 0.0
    X_scale = None
    if normalize:
        obs_vals = Y_mat[~np.isnan(Y_mat)]
        Y_center = float(obs_vals.mean())
        Y_scale  = float(obs_vals.std()) or 1.0
        Y_mat    = (Y_mat - Y_center) / Y_scale
        if X_mat is not None:
            K = X_mat.shape[2]
            X_scale = np.zeros(K)
            for kk in range(K):
                obs_x = X_mat[:, :, kk][~np.isnan(X_mat[:, :, kk])]
                X_scale[kk] = float(obs_x.std()) or 1.0
                X_mat[:, :, kk] /= X_scale[kk]

    # ------------------------------------------------------------------
    # 3. Demean / partial out fixed effects
    # ------------------------------------------------------------------
    Y_dem, alpha_hat, xi_hat = demean_panel(Y_mat, D_mat, force, ctrl_idx)

    # ------------------------------------------------------------------
    # 4. Partial out time-varying covariates
    # ------------------------------------------------------------------
    beta_hat = None
    if X_mat is not None and X_mat.shape[2] > 0:
        Y_dem, beta_hat = partial_out_covariates(Y_dem, X_mat, D_mat, ctrl_idx)

    # ------------------------------------------------------------------
    # 5. Select r or lambda via CV
    # ------------------------------------------------------------------
    rng = np.random.default_rng(seed)

    r_candidates: list[int]
    if isinstance(r, (list, tuple, np.ndarray)):
        r_candidates = list(r)
    else:
        r_candidates = list(range(0, 6)) if (CV and r == 0) else [int(r)]

    r_cv_scores  = None
    lam_cv_scores = None
    lam_opt      = None
    r_opt        = r_candidates[0]

    if estimator in ("gsynth", "ife"):
        if CV and len(r_candidates) > 1:
            Y_ctrl = Y_dem[ctrl_idx]
            r_opt, r_cv_scores = cv_factor_number(
                Y_ctrl, r_candidates, k=k, criterion=criterion, tol=tol, rng=rng
            )
        else:
            r_opt = r_candidates[0]

    elif estimator == "mc":
        if lam is None:
            lambda_seq = build_lambda_seq(Y_dem[ctrl_idx], nlambda)
        elif isinstance(lam, (list, tuple, np.ndarray)):
            lambda_seq = np.sort(np.asarray(lam, dtype=float))[::-1]
        else:
            lambda_seq = np.array([float(lam)])

        if CV and len(lambda_seq) > 1:
            lam_opt, lam_cv_scores = cv_lambda(
                Y_dem, D_mat, ctrl_idx, lambda_seq, k=k, tol=tol, rng=rng
            )
        else:
            lam_opt = float(lambda_seq[0])

    # ------------------------------------------------------------------
    # 6. Final estimation
    # ------------------------------------------------------------------
    if estimator == "gsynth":
        est = estimate_gsynth(Y_dem, D_mat, ctrl_idx, treat_idx, T0_vec, r_opt, tol)
    elif estimator == "ife":
        est = estimate_ife(Y_dem, D_mat, ctrl_idx, treat_idx, T0_vec, r_opt, tol)
    else:  # mc
        est = estimate_mc(Y_dem, D_mat, ctrl_idx, treat_idx, T0_vec, lam_opt, tol)
        r_opt = est.get("r", 0)

    # ------------------------------------------------------------------
    # 7. Back-transform counterfactuals if normalized
    # ------------------------------------------------------------------
    Y_ct_raw = est["Y_ct"] * Y_scale + Y_center
    Y_tr_raw = Y_mat[treat_idx] * Y_scale + Y_center
    att_raw   = est["att"] * Y_scale
    att_avg_raw = est["att_avg"] * Y_scale

    # ------------------------------------------------------------------
    # 8. Sigma^2 estimate
    # ------------------------------------------------------------------
    resid_ctrl = est.get("residuals")
    if resid_ctrl is not None:
        sigma2 = float(np.nanmean(resid_ctrl ** 2))
    else:
        sigma2 = est.get("mse", 0.0)

    # ------------------------------------------------------------------
    # 9. Inference
    # ------------------------------------------------------------------
    boot_out = None
    if se:
        if inference is None:
            inference = "parametric" if len(treat_idx) < 40 else "nonparametric"
        inference_lc = inference.lower()

        if inference_lc == "jackknife":
            boot_out = jackknife_inference(
                Y_dem, D_mat, ctrl_idx, treat_idx, T0_vec,
                r_opt, lam_opt, estimator, alpha,
                est["att_time"], force, tol,
                att_raw, att_avg_raw,
            )
        elif inference_lc == "parametric":
            if estimator == "mc":
                warnings.warn(
                    "Parametric bootstrap is not recommended for the MC estimator. "
                    "Using nonparametric bootstrap instead.",
                    stacklevel=2,
                )
                inference_lc = "nonparametric"

        if inference_lc == "parametric":
            boot_out = parametric_bootstrap(
                Y_dem, D_mat,
                alpha_hat, xi_hat, beta_hat, X_mat,
                est.get("F"), est.get("Lambda"), est.get("M"),
                sigma2,
                ctrl_idx, treat_idx, T0_vec,
                r_opt, lam_opt, estimator,
                nboots, alpha, est["att_time"], force, tol, seed,
            )
        elif inference_lc == "nonparametric":
            boot_out = nonparametric_bootstrap(
                Y_dem, D_mat,
                ctrl_idx, treat_idx, T0_vec,
                r_opt, lam_opt, estimator,
                nboots, alpha, est["att_time"], force, tol, seed,
                cl_vec,
            )

        if boot_out and normalize:
            for key in ("att_se", "att_ci_lower", "att_ci_upper",
                        "att_avg_se", "att_avg_ci_lower", "att_avg_ci_upper"):
                if boot_out.get(key) is not None:
                    boot_out[key] = np.asarray(boot_out[key]) * Y_scale

    # ------------------------------------------------------------------
    # 10. Build T0 dict (unit -> T0)
    # ------------------------------------------------------------------
    T0_dict = {}
    for kk, i in enumerate(treat_idx):
        T0_dict[units[i]] = int(T0_vec[kk])

    # ------------------------------------------------------------------
    # 11. Assemble result
    # ------------------------------------------------------------------
    res = GsynthResult(
        estimator = estimator,
        force     = force,
        r         = r_opt,
        lambda_opt = lam_opt,

        att_avg   = att_avg_raw,
        att       = att_raw,
        att_time  = est["att_time"],

        Y_tr = Y_tr_raw,
        Y_ct = Y_ct_raw,

        factors  = est.get("F"),
        loadings = est.get("Lambda"),
        alpha    = alpha_hat * Y_scale if alpha_hat is not None else None,
        xi       = xi_hat * Y_scale    if xi_hat   is not None else None,
        beta     = beta_hat,
        residuals = resid_ctrl,

        r_cv    = r_cv_scores,
        lambda_cv = lam_cv_scores,
        mse     = est["mse"] * (Y_scale ** 2),
        sigma2  = sigma2 * (Y_scale ** 2),

        N  = N,
        T  = T,
        N_tr = len(treat_idx),
        N_co = len(ctrl_idx),
        T0   = T0_dict,
        units = units,
        times = times,
        treat_units = units[treat_idx],
        index   = list(index),
        Y_name  = Y,
        D_name  = D,
        X_names = X or [],

        _Y_ct_unit = Y_ct_raw,
    )

    if boot_out:
        res.att_se          = boot_out.get("att_se")
        res.att_ci_lower    = boot_out.get("att_ci_lower")
        res.att_ci_upper    = boot_out.get("att_ci_upper")
        res.att_avg_se      = boot_out.get("att_avg_se")
        res.att_avg_ci_lower = boot_out.get("att_avg_ci_lower")
        res.att_avg_ci_upper = boot_out.get("att_avg_ci_upper")
        res.att_boot        = boot_out.get("att_boot")
        if beta_hat is not None and boot_out.get("beta_se") is not None:
            res.beta_se       = boot_out["beta_se"]
            res.beta_ci_lower = boot_out.get("beta_ci_lower")
            res.beta_ci_upper = boot_out.get("beta_ci_upper")

    return res


# ---------------------------------------------------------------------------
# Formula parser
# ---------------------------------------------------------------------------

def _parse_formula(
    formula: str,
    Y_override: Optional[str],
    D_override: Optional[str],
    X_override: Optional[Union[str, list[str]]],
) -> tuple[str, str, Optional[list[str]]]:
    """
    Parse a formula string of the form ``"Y ~ D + X1 + X2"``.

    Explicit keyword arguments override parsed values.
    """
    formula = formula.replace(" ", "")
    if "~" not in formula:
        raise ValueError(f"Formula must contain '~': got '{formula}'.")
    lhs, rhs = formula.split("~", 1)
    Y_parsed = lhs.strip()
    rhs_terms = [t.strip() for t in rhs.split("+") if t.strip()]
    if not rhs_terms:
        raise ValueError("Formula RHS must contain at least the treatment variable D.")
    D_parsed  = rhs_terms[0]
    X_parsed  = rhs_terms[1:] if len(rhs_terms) > 1 else None

    Y_out = Y_override if Y_override is not None else Y_parsed
    D_out = D_override if D_override is not None else D_parsed
    X_out = X_override if X_override is not None else X_parsed
    return Y_out, D_out, X_out
