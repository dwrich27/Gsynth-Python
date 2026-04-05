"""Tests for the main gsynth() function."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gsynth import GsynthResult, gsynth


class TestBasicEstimation:
    def test_returns_result_object(self, simple_panel):
        res = gsynth(formula="Y~D", data=simple_panel, index=["unit","time"],
                     CV=False, r=2, se=False)
        assert isinstance(res, GsynthResult)

    def test_att_sign_positive(self, simple_panel):
        """True tau=1.0 — estimated ATT should be positive."""
        res = gsynth(formula="Y~D", data=simple_panel, index=["unit","time"],
                     CV=False, r=2, se=False)
        assert res.att_avg > 0

    def test_att_reasonable_magnitude(self, simple_panel):
        """ATT estimate should be within ±1 of true tau=1.0."""
        res = gsynth(formula="Y~D", data=simple_panel, index=["unit","time"],
                     CV=False, r=2, se=False)
        assert abs(res.att_avg - 1.0) < 1.0

    def test_counterfactual_shape(self, simple_panel):
        res = gsynth(formula="Y~D", data=simple_panel, index=["unit","time"],
                     CV=False, r=2, se=False)
        N_tr = res.N_tr
        T    = res.T
        assert res.Y_ct.shape == (N_tr, T)
        assert res.Y_tr.shape == (N_tr, T)

    def test_att_time_length(self, simple_panel):
        res = gsynth(formula="Y~D", data=simple_panel, index=["unit","time"],
                     CV=False, r=2, se=False)
        assert len(res.att) == len(res.att_time)
        assert len(res.att) > 0

    def test_metadata(self, simple_panel):
        res = gsynth(formula="Y~D", data=simple_panel, index=["unit","time"],
                     CV=False, r=2, se=False)
        assert res.N == 20
        assert res.T == 20
        assert res.N_tr == 5
        assert res.N_co == 15

    def test_formula_parsing(self, simple_panel):
        res1 = gsynth(formula="Y ~ D", data=simple_panel, index=["unit","time"],
                      CV=False, r=1, se=False)
        res2 = gsynth(Y="Y", D="D", data=simple_panel, index=["unit","time"],
                      CV=False, r=1, se=False)
        np.testing.assert_allclose(res1.att_avg, res2.att_avg, rtol=1e-6)


class TestEstimators:
    def test_gsynth_estimator(self, small_panel):
        res = gsynth(formula="Y~D", data=small_panel, index=["unit","time"],
                     estimator="gsynth", CV=False, r=1, se=False)
        assert res.estimator == "gsynth"
        assert res.factors is not None
        assert res.factors.shape == (res.T, 1)

    def test_ife_estimator(self, small_panel):
        res = gsynth(formula="Y~D", data=small_panel, index=["unit","time"],
                     estimator="ife", CV=False, r=1, se=False)
        assert res.estimator == "ife"

    def test_mc_estimator(self, small_panel):
        res = gsynth(formula="Y~D", data=small_panel, index=["unit","time"],
                     estimator="mc", CV=False, lam=1.0, se=False)
        assert res.estimator == "mc"
        assert res.lambda_opt == pytest.approx(1.0)

    def test_em_shorthand(self, small_panel):
        res = gsynth(formula="Y~D", data=small_panel, index=["unit","time"],
                     EM=True, CV=False, r=1, se=False)
        assert res.estimator == "ife"

    def test_zero_factors(self, small_panel):
        res = gsynth(formula="Y~D", data=small_panel, index=["unit","time"],
                     CV=False, r=0, se=False)
        assert res.r == 0


class TestFixedEffects:
    @pytest.mark.parametrize("force", ["none", "unit", "time", "two-way"])
    def test_force_options(self, small_panel, force):
        res = gsynth(formula="Y~D", data=small_panel, index=["unit","time"],
                     CV=False, r=1, force=force, se=False)
        assert res.force == force

    def test_unit_effects_stored(self, small_panel):
        res = gsynth(formula="Y~D", data=small_panel, index=["unit","time"],
                     CV=False, r=1, force="unit", se=False)
        assert res.alpha is not None
        assert len(res.alpha) == res.N

    def test_time_effects_stored(self, small_panel):
        res = gsynth(formula="Y~D", data=small_panel, index=["unit","time"],
                     CV=False, r=1, force="time", se=False)
        assert res.xi is not None
        assert len(res.xi) == res.T

    def test_two_way_effects(self, small_panel):
        res = gsynth(formula="Y~D", data=small_panel, index=["unit","time"],
                     CV=False, r=1, force="two-way", se=False)
        assert res.alpha is not None
        assert res.xi is not None


class TestCrossValidation:
    def test_cv_selects_r(self, simple_panel):
        res = gsynth(formula="Y~D", data=simple_panel, index=["unit","time"],
                     CV=True, criterion="mspe", k=3, se=False)
        assert res.r >= 0
        assert res.r_cv is not None

    def test_cv_mc_selects_lambda(self, simple_panel):
        res = gsynth(formula="Y~D", data=simple_panel, index=["unit","time"],
                     estimator="mc", CV=True, nlambda=5, k=3, se=False)
        assert res.lambda_opt is not None
        assert res.lambda_cv is not None
        assert len(res.lambda_cv) == 5

    def test_pc_criterion(self, small_panel):
        res = gsynth(formula="Y~D", data=small_panel, index=["unit","time"],
                     CV=True, criterion="pc", k=3, se=False)
        assert res.r >= 0

    def test_r_list(self, small_panel):
        res = gsynth(formula="Y~D", data=small_panel, index=["unit","time"],
                     CV=True, r=[1, 2, 3], k=3, se=False)
        assert res.r in [1, 2, 3]


class TestCovariates:
    def test_covariate_coefficients(self, covariate_panel):
        res = gsynth(formula="Y~D+X1+X2", data=covariate_panel,
                     index=["unit","time"], CV=False, r=1, se=False)
        assert res.beta is not None
        assert len(res.beta) == 2

    def test_covariate_names_stored(self, covariate_panel):
        res = gsynth(formula="Y~D+X1+X2", data=covariate_panel,
                     index=["unit","time"], CV=False, r=1, se=False)
        assert res.X_names == ["X1", "X2"]


class TestInference:
    def test_parametric_bootstrap(self, small_panel):
        res = gsynth(formula="Y~D", data=small_panel, index=["unit","time"],
                     CV=False, r=1, se=True, nboots=20,
                     inference="parametric", seed=0)
        assert res.att_avg_se is not None
        assert res.att_avg_se >= 0
        assert res.att_ci_lower is not None
        assert res.att_ci_upper is not None
        assert res.att_avg_ci_lower <= res.att_avg_ci_upper

    def test_nonparametric_bootstrap(self, simple_panel):
        res = gsynth(formula="Y~D", data=simple_panel, index=["unit","time"],
                     CV=False, r=2, se=True, nboots=20,
                     inference="nonparametric", seed=1)
        assert res.att_avg_se is not None
        assert res.att_boot is not None
        assert res.att_boot.shape[0] <= 20

    def test_ci_brackets_att(self, small_panel):
        res = gsynth(formula="Y~D", data=small_panel, index=["unit","time"],
                     CV=False, r=1, se=True, nboots=30,
                     inference="parametric", seed=5)
        if res.att_se is not None:
            assert np.all(res.att_ci_lower <= res.att_ci_upper)

    def test_per_period_se_length(self, small_panel):
        res = gsynth(formula="Y~D", data=small_panel, index=["unit","time"],
                     CV=False, r=1, se=True, nboots=20,
                     inference="parametric", seed=3)
        if res.att_se is not None:
            assert len(res.att_se) == len(res.att)

    def test_seed_reproducibility(self, small_panel):
        res1 = gsynth(formula="Y~D", data=small_panel, index=["unit","time"],
                      CV=False, r=1, se=True, nboots=20,
                      inference="parametric", seed=42)
        res2 = gsynth(formula="Y~D", data=small_panel, index=["unit","time"],
                      CV=False, r=1, se=True, nboots=20,
                      inference="parametric", seed=42)
        assert res1.att_avg_se == pytest.approx(res2.att_avg_se)

    def test_jackknife(self, small_panel):
        res = gsynth(formula="Y~D", data=small_panel, index=["unit","time"],
                     CV=False, r=1, se=True, nboots=0,
                     inference="jackknife", seed=0)
        assert res.att_avg_se is not None


class TestStaggered:
    def test_staggered_adoption(self, staggered_panel):
        res = gsynth(formula="Y~D", data=staggered_panel,
                     index=["unit","time"], CV=False, r=2, se=False)
        assert res.N_tr == 15
        assert res.att_avg > 0

    def test_staggered_t0_dict(self, staggered_panel):
        res = gsynth(formula="Y~D", data=staggered_panel,
                     index=["unit","time"], CV=False, r=2, se=False)
        assert len(res.T0) == 15
        t0_vals = np.unique(list(res.T0.values()))
        assert len(t0_vals) == 3   # three distinct adoption times


class TestValidation:
    def test_missing_data_error(self):
        with pytest.raises(ValueError, match="data"):
            gsynth(Y="Y", D="D", index=["unit","time"])

    def test_missing_index_error(self, simple_panel):
        with pytest.raises(ValueError, match="index"):
            gsynth(formula="Y~D", data=simple_panel)

    def test_bad_estimator(self, simple_panel):
        with pytest.raises(ValueError, match="estimator"):
            gsynth(formula="Y~D", data=simple_panel, index=["unit","time"],
                   estimator="bad_algo")

    def test_bad_force(self, simple_panel):
        with pytest.raises(ValueError, match="force"):
            gsynth(formula="Y~D", data=simple_panel, index=["unit","time"],
                   force="three-way")

    def test_non_binary_treatment(self):
        df = pd.DataFrame({
            "unit": [0, 0, 1, 1],
            "time": [0, 1, 0, 1],
            "Y": [1.0, 2.0, 1.5, 2.5],
            "D": [0, 2, 0, 1],   # invalid: D=2
        })
        with pytest.raises(ValueError, match="dichotomous"):
            gsynth(formula="Y~D", data=df, index=["unit","time"])

    def test_no_treated_units(self):
        df = pd.DataFrame({
            "unit": [0, 0, 1, 1],
            "time": [0, 1, 0, 1],
            "Y": [1.0, 2.0, 1.5, 2.5],
            "D": [0, 0, 0, 0],
        })
        with pytest.raises(ValueError, match="No treated"):
            gsynth(formula="Y~D", data=df, index=["unit","time"])


class TestNormalize:
    def test_normalize_flag(self, simple_panel):
        res = gsynth(formula="Y~D", data=simple_panel, index=["unit","time"],
                     CV=False, r=2, normalize=True, se=False)
        assert isinstance(res.att_avg, float)
        assert abs(res.att_avg - 1.0) < 1.5


class TestSummary:
    def test_repr(self, simple_panel):
        res = gsynth(formula="Y~D", data=simple_panel, index=["unit","time"],
                     CV=False, r=2, se=False)
        s = repr(res)
        assert "GsynthResult" in s
        assert "ATT" in s

    def test_summary_string(self, simple_panel):
        res = gsynth(formula="Y~D", data=simple_panel, index=["unit","time"],
                     CV=False, r=2, se=False)
        s = res.summary()
        assert "Generalized Synthetic" in s
        assert "ATT" in s
