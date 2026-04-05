"""GsynthResult — container for all gsynth estimation outputs."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


def _norm_cdf(z: float) -> float:
    """Standard-normal CDF via error function (no scipy required)."""
    return float(0.5 * (1.0 + np.sign(z) * np.real(np.erf(abs(z) / np.sqrt(2)))))


@dataclass
class GsynthResult:
    """
    Container for all outputs from a gsynth estimation.

    Attributes
    ----------
    estimator : str
        Estimation method used ("gsynth", "ife", or "mc").
    force : str
        Fixed effects specification ("none", "unit", "time", "two-way").
    r : int
        Number of latent factors used.
    lambda_opt : float or None
        Optimal regularization lambda (matrix completion only).

    # --- treatment-effect estimates ---
    att_avg : float
        Average treatment effect on the treated (all post-treatment periods).
    att : ndarray, shape (T_post,)
        Period-by-period average treatment effect on treated units.
    att_time : ndarray, shape (T_post,)
        Time indices corresponding to att.

    # --- counterfactuals ---
    Y_tr : ndarray, shape (N_tr, T)
        Actual outcomes for treated units (full panel).
    Y_ct : ndarray, shape (N_tr, T)
        Imputed counterfactual Y(0) for treated units.

    # --- model components ---
    factors : ndarray, shape (T, r) or None
        Estimated latent factors.
    loadings : ndarray, shape (N, r) or None
        Estimated factor loadings for all units.
    alpha : ndarray, shape (N,) or None
        Unit fixed effects.
    xi : ndarray, shape (T,) or None
        Time fixed effects.
    beta : ndarray, shape (K,) or None
        Covariate coefficients.
    residuals : ndarray, shape (N_ctrl, T) or None
        Residuals from the IFE fit on control units.

    # --- inference (populated when se=True) ---
    att_avg_se : float or None
    att_avg_ci_lower : float or None
    att_avg_ci_upper : float or None
    att_se : ndarray or None
    att_ci_lower : ndarray or None
    att_ci_upper : ndarray or None
    beta_se : ndarray or None
    beta_ci_lower : ndarray or None
    beta_ci_upper : ndarray or None
    alpha_se : ndarray or None
    att_boot : ndarray or None, shape (nboots, T_post)

    # --- diagnostics ---
    r_cv : ndarray or None
        Cross-validation criterion values for each r tried.
    lambda_cv : ndarray or None
        Cross-validation criterion values for each lambda tried.
    mse : float
        In-sample MSE on control units.
    sigma2 : float
        Estimated error variance.

    # --- metadata ---
    N : int
    T : int
    N_tr : int
    N_co : int
    T0 : dict[str, int]
        Per-treated-unit number of pre-treatment periods.
    units : ndarray
        All unit identifiers.
    times : ndarray
        All time identifiers.
    treat_units : ndarray
        Treated unit identifiers.
    index : list[str, str]
        Column names used as [unit, time] index.
    Y_name : str
    D_name : str
    X_names : list[str]
    """

    # --- estimator metadata ---
    estimator: str = "gsynth"
    force: str = "unit"
    r: int = 0
    lambda_opt: Optional[float] = None

    # --- treatment effects ---
    att_avg: float = 0.0
    att: np.ndarray = field(default_factory=lambda: np.array([]))
    att_time: np.ndarray = field(default_factory=lambda: np.array([]))

    # --- counterfactuals ---
    Y_tr: np.ndarray = field(default_factory=lambda: np.array([]))
    Y_ct: np.ndarray = field(default_factory=lambda: np.array([]))

    # --- model components ---
    factors: Optional[np.ndarray] = None
    loadings: Optional[np.ndarray] = None
    alpha: Optional[np.ndarray] = None
    xi: Optional[np.ndarray] = None
    beta: Optional[np.ndarray] = None
    residuals: Optional[np.ndarray] = None

    # --- inference ---
    att_avg_se: Optional[float] = None
    att_avg_ci_lower: Optional[float] = None
    att_avg_ci_upper: Optional[float] = None
    att_se: Optional[np.ndarray] = None
    att_ci_lower: Optional[np.ndarray] = None
    att_ci_upper: Optional[float] = None
    beta_se: Optional[np.ndarray] = None
    beta_ci_lower: Optional[np.ndarray] = None
    beta_ci_upper: Optional[np.ndarray] = None
    alpha_se: Optional[np.ndarray] = None
    att_boot: Optional[np.ndarray] = None

    # --- diagnostics ---
    r_cv: Optional[np.ndarray] = None
    lambda_cv: Optional[np.ndarray] = None
    mse: float = 0.0
    sigma2: float = 0.0

    # --- metadata ---
    N: int = 0
    T: int = 0
    N_tr: int = 0
    N_co: int = 0
    T0: dict = field(default_factory=dict)
    units: np.ndarray = field(default_factory=lambda: np.array([]))
    times: np.ndarray = field(default_factory=lambda: np.array([]))
    treat_units: np.ndarray = field(default_factory=lambda: np.array([]))
    index: list = field(default_factory=list)
    Y_name: str = "Y"
    D_name: str = "D"
    X_names: list = field(default_factory=list)

    # internal storage for per-unit counterfactuals (N_tr × T)
    _Y_ct_unit: Optional[np.ndarray] = None
    # per-unit ATT (N_tr × T_post)
    _att_unit: Optional[np.ndarray] = None

    def __repr__(self) -> str:
        lines = [
            "GsynthResult",
            f"  Estimator : {self.estimator}",
            f"  Force     : {self.force}",
            f"  Factors   : {self.r}",
            f"  N / T     : {self.N} / {self.T}",
            f"  Treated   : {self.N_tr}",
            f"  ATT (avg) : {self.att_avg:.4f}",
        ]
        if self.att_avg_se is not None:
            ci_lo = self.att_avg_ci_lower
            ci_hi = self.att_avg_ci_upper
            lines.append(f"  95% CI    : [{ci_lo:.4f}, {ci_hi:.4f}]")
        return "\n".join(lines)

    def summary(self) -> str:
        """Return a detailed text summary matching R gsynth print output."""
        lines = [
            "=" * 56,
            " Generalized Synthetic Control Method",
            "=" * 56,
            f"  Estimator      : {self.estimator}",
            f"  Force          : {self.force}",
            f"  # Factors (r)  : {self.r}",
            f"  N units        : {self.N}  ({self.N_tr} treated, {self.N_co} control)",
            f"  T periods      : {self.T}",
            f"  In-sample MSE  : {self.mse:.6f}",
            "",
            "Average Treatment Effect on the Treated (ATT):",
            f"  {'':8s} {'Estimate':>10s}",
        ]
        if self.att_avg_se is not None:
            hdr = (
                f"  {'':8s} {'Estimate':>10s} {'S.E.':>8s}"
                f" {'CI.lower':>10s} {'CI.upper':>10s} {'p-value':>8s}"
            )
            lines[-1] = hdr
            se = self.att_avg_se
            z = abs(self.att_avg / se) if se else float("nan")
            p = 2 * (1 - _norm_cdf(z))
            lines.append(
                f"  {'ATT.avg':8s} {self.att_avg:>10.4f} {se:>8.4f}"
                f" {self.att_avg_ci_lower:>10.4f} {self.att_avg_ci_upper:>10.4f} {p:>8.4f}"
            )
        else:
            lines.append(f"  {'ATT.avg':8s} {self.att_avg:>10.4f}")

        lines += ["", "Period-by-period ATT:"]
        hdr = f"  {'Period':>8s} {'Estimate':>10s}"
        if self.att_se is not None:
            hdr += f" {'S.E.':>8s} {'CI.lower':>10s} {'CI.upper':>10s}"
        lines.append(hdr)
        for i, (t, est) in enumerate(zip(self.att_time, self.att)):
            row = f"  {str(t):>8s} {est:>10.4f}"
            if self.att_se is not None:
                se_i = self.att_se[i]
                lo_i = self.att_ci_lower[i]
                hi_i = self.att_ci_upper[i]
                row += f" {se_i:>8.4f} {lo_i:>10.4f} {hi_i:>10.4f}"
            lines.append(row)

        if self.beta is not None and len(self.beta) > 0:
            lines += ["", "Covariate Coefficients:"]
            hdr2 = f"  {'Variable':>12s} {'Estimate':>10s}"
            if self.beta_se is not None:
                hdr2 += f" {'S.E.':>8s} {'CI.lower':>10s} {'CI.upper':>10s}"
            lines.append(hdr2)
            for j, name in enumerate(self.X_names):
                row = f"  {name:>12s} {self.beta[j]:>10.4f}"
                if self.beta_se is not None:
                    row += (
                        f" {self.beta_se[j]:>8.4f}"
                        f" {self.beta_ci_lower[j]:>10.4f}"
                        f" {self.beta_ci_upper[j]:>10.4f}"
                    )
                lines.append(row)

        lines.append("=" * 56)
        return "\n".join(lines)
