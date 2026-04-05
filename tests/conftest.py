"""Shared pytest fixtures for the gsynth test suite."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def make_panel(
    N: int = 20,
    T: int = 20,
    N_tr: int = 5,
    T0: int = 10,
    r: int = 2,
    tau: float = 1.0,
    seed: int = 0,
) -> pd.DataFrame:
    """
    Generate a synthetic panel data set from the IFE model.

        Y_it = alpha_i + lambda_i @ f_t + D_it * tau + eps_it

    Parameters
    ----------
    N    : total number of units
    T    : number of time periods
    N_tr : number of treated units (first N_tr units are treated)
    T0   : treatment adoption period (same for all treated)
    r    : number of latent factors
    tau  : homogeneous treatment effect
    seed : random seed
    """
    rng = np.random.default_rng(seed)

    # Latent factors: (T, r)
    F = rng.standard_normal((T, r))
    # Factor loadings: (N, r)
    Lambda = rng.standard_normal((N, r))
    # Unit fixed effects
    alpha = rng.uniform(-2, 2, size=N)
    # Idiosyncratic errors
    eps = rng.standard_normal((N, T)) * 0.5

    # Potential outcome Y(0)
    Y0 = alpha[:, None] + Lambda @ F.T + eps

    # Treatment indicator
    D = np.zeros((N, T))
    D[:N_tr, T0:] = 1.0

    # Observed outcome
    Y_obs = Y0 + D * tau

    rows = []
    for i in range(N):
        for j in range(T):
            rows.append({
                "unit": i,
                "time": j,
                "Y": Y_obs[i, j],
                "D": D[i, j],
            })
    return pd.DataFrame(rows)


@pytest.fixture
def simple_panel():
    """Simple balanced panel: 20 units, 20 periods, 5 treated, T0=10."""
    return make_panel(N=20, T=20, N_tr=5, T0=10, r=2, tau=1.0, seed=42)


@pytest.fixture
def small_panel():
    """Tiny panel for fast tests: 10 units, 12 periods, 3 treated, T0=6."""
    return make_panel(N=10, T=12, N_tr=3, T0=6, r=1, tau=0.5, seed=7)


@pytest.fixture
def staggered_panel():
    """Panel with staggered treatment adoption."""
    rng = np.random.default_rng(99)
    N, T = 30, 24
    r = 2
    F = rng.standard_normal((T, r))
    Lambda = rng.standard_normal((N, r))
    alpha = rng.uniform(-1, 1, N)
    eps = rng.standard_normal((N, T)) * 0.3
    Y0 = alpha[:, None] + Lambda @ F.T + eps

    # Staggered: 5 units treated at T=8, 5 units at T=12, 5 units at T=16
    D = np.zeros((N, T))
    D[0:5, 8:] = 1
    D[5:10, 12:] = 1
    D[10:15, 16:] = 1
    Y_obs = Y0 + D * 1.5

    rows = []
    for i in range(N):
        for j in range(T):
            rows.append({"unit": i, "time": j, "Y": Y_obs[i, j], "D": D[i, j]})
    return pd.DataFrame(rows)


@pytest.fixture
def covariate_panel():
    """Panel with two time-varying covariates."""
    rng = np.random.default_rng(11)
    N, T = 15, 16
    r = 1
    F = rng.standard_normal((T, r))
    Lambda = rng.standard_normal((N, r))
    alpha = rng.uniform(-1, 1, N)
    X1 = rng.standard_normal((N, T))
    X2 = rng.standard_normal((N, T))
    eps = rng.standard_normal((N, T)) * 0.3
    beta = np.array([0.5, -0.3])
    Y0 = alpha[:, None] + Lambda @ F.T + beta[0] * X1 + beta[1] * X2 + eps

    D = np.zeros((N, T))
    D[:4, 8:] = 1
    Y_obs = Y0 + D * 0.8

    rows = []
    for i in range(N):
        for j in range(T):
            rows.append({
                "unit": i, "time": j,
                "Y": Y_obs[i, j], "D": D[i, j],
                "X1": X1[i, j], "X2": X2[i, j],
            })
    return pd.DataFrame(rows)
