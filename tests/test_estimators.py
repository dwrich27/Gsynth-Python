"""Unit tests for the internal estimator functions."""
from __future__ import annotations

import numpy as np

from gsynth._cv import build_lambda_seq, cv_factor_number, cv_lambda
from gsynth._data import demean_panel
from gsynth._estimators import (
    _ife_als,
    _nuclear_norm_minimise,
    estimate_gsynth,
    estimate_mc,
)


class TestIfeAls:
    def test_zero_factors(self):
        rng = np.random.default_rng(0)
        Y = rng.standard_normal((10, 8))
        mask = np.ones_like(Y, dtype=bool)
        F, Lambda, resid, mse = _ife_als(Y, mask, r=0)
        assert F.shape == (8, 0)
        assert Lambda.shape == (10, 0)
        np.testing.assert_array_equal(resid, Y)

    def test_one_factor(self):
        rng = np.random.default_rng(1)
        F_true = rng.standard_normal((15, 1))
        L_true = rng.standard_normal((8, 1))
        Y = L_true @ F_true.T + rng.standard_normal((8, 15)) * 0.1
        mask = np.ones_like(Y, dtype=bool)
        F, Lambda, resid, mse = _ife_als(Y, mask, r=1, tol=1e-6)
        assert F.shape == (15, 1)
        assert Lambda.shape == (8, 1)
        assert mse < 0.5   # should fit closely

    def test_missing_data(self):
        rng = np.random.default_rng(2)
        Y = rng.standard_normal((10, 12))
        mask = rng.random((10, 12)) > 0.2   # ~20% missing
        Y_masked = np.where(mask, Y, np.nan)
        F, Lambda, resid, mse = _ife_als(Y_masked, mask, r=2)
        assert not np.any(np.isnan(Lambda))
        assert not np.any(np.isnan(F))

    def test_residuals_nan_where_missing(self):
        rng = np.random.default_rng(3)
        Y = rng.standard_normal((8, 10))
        mask = np.ones_like(Y, dtype=bool)
        mask[0, 0] = False
        Y[0, 0] = np.nan
        _, _, resid, _ = _ife_als(Y, mask, r=1)
        assert np.isnan(resid[0, 0])


class TestEstimateGsynth:
    def setup_method(self):
        rng = np.random.default_rng(42)
        N, T, r, N_tr = 15, 16, 2, 4
        F = rng.standard_normal((T, r))
        Lambda = rng.standard_normal((N, r))
        eps = rng.standard_normal((N, T)) * 0.3
        self.Y0 = Lambda @ F.T + eps
        self.D = np.zeros((N, T))
        self.D[:N_tr, 8:] = 1.0
        self.ctrl_idx = np.arange(N_tr, N)
        self.treat_idx = np.arange(N_tr)
        self.T0_vec = np.full(N_tr, 8)
        self.tau = 1.0
        self.Y = self.Y0 + self.D * self.tau

    def test_output_keys(self):
        est = estimate_gsynth(
            self.Y, self.D, self.ctrl_idx, self.treat_idx, self.T0_vec, r=2
        )
        for key in ("F", "Lambda", "Y_ct", "att", "att_time", "att_avg", "mse"):
            assert key in est, f"Missing key: {key}"

    def test_att_positive(self):
        est = estimate_gsynth(
            self.Y, self.D, self.ctrl_idx, self.treat_idx, self.T0_vec, r=2
        )
        assert est["att_avg"] > 0

    def test_y_ct_shape(self):
        est = estimate_gsynth(
            self.Y, self.D, self.ctrl_idx, self.treat_idx, self.T0_vec, r=2
        )
        assert est["Y_ct"].shape == (len(self.treat_idx), self.Y.shape[1])


class TestEstimateMC:
    def test_output_keys(self):
        rng = np.random.default_rng(7)
        N, T = 12, 14
        Y = rng.standard_normal((N, T))
        D = np.zeros((N, T))
        D[:3, 8:] = 1.0
        ctrl = np.arange(3, N)
        treat = np.arange(3)
        T0 = np.full(3, 8)
        est = estimate_mc(Y, D, ctrl, treat, T0, lam=0.5)
        for key in ("Y_ct", "att", "att_avg", "mse"):
            assert key in est

    def test_nuclear_norm_convergence(self):
        rng = np.random.default_rng(8)
        Y = rng.standard_normal((8, 10))
        mask = np.ones_like(Y, dtype=bool)
        mask[2:4, 5:] = False  # missing block
        M = _nuclear_norm_minimise(Y, mask, lam=0.5)
        assert M.shape == Y.shape
        assert not np.any(np.isnan(M))


class TestCV:
    def test_cv_factor_number(self):
        rng = np.random.default_rng(0)
        N, T = 10, 12
        Y = rng.standard_normal((N, T))
        r_opt, scores = cv_factor_number(Y, [0, 1, 2], k=3)
        assert r_opt in [0, 1, 2]
        assert len(scores) == 3

    def test_build_lambda_seq(self):
        rng = np.random.default_rng(0)
        Y = rng.standard_normal((8, 10))
        seq = build_lambda_seq(Y, nlambda=7)
        assert len(seq) == 7
        assert seq[0] >= seq[-1]   # decreasing

    def test_cv_lambda(self):
        rng = np.random.default_rng(3)
        N, T = 12, 14
        Y = rng.standard_normal((N, T))
        D = np.zeros((N, T))
        D[:3, 7:] = 1.0
        ctrl = np.arange(3, N)
        seq = build_lambda_seq(Y[ctrl], nlambda=4)
        lam_opt, scores = cv_lambda(Y, D, ctrl, seq, k=3)
        assert lam_opt > 0
        assert len(scores) == 4


class TestDemean:
    def test_none_force(self):
        rng = np.random.default_rng(0)
        Y = rng.standard_normal((8, 10))
        D = np.zeros_like(Y)
        ctrl = np.arange(8)
        Y_d, alpha, xi = demean_panel(Y, D, "none", ctrl)
        assert alpha is None
        assert xi is None
        np.testing.assert_array_equal(Y_d, Y)

    def test_unit_force(self):
        rng = np.random.default_rng(1)
        Y = rng.standard_normal((8, 10))
        D = np.zeros_like(Y)
        ctrl = np.arange(8)
        Y_d, alpha, xi = demean_panel(Y, D, "unit", ctrl)
        assert alpha is not None
        assert xi is None
        # Unit means of demeaned Y should be ~0
        unit_means = np.nanmean(Y_d, axis=1)
        np.testing.assert_allclose(unit_means, 0.0, atol=1e-8)

    def test_two_way_force(self):
        rng = np.random.default_rng(2)
        Y = rng.standard_normal((8, 10))
        D = np.zeros_like(Y)
        ctrl = np.arange(8)
        Y_d, alpha, xi = demean_panel(Y, D, "two-way", ctrl)
        assert alpha is not None
        assert xi is not None
