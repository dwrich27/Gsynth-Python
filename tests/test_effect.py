"""Tests for the effect() function."""
from __future__ import annotations

import numpy as np
import pytest

from gsynth import effect, gsynth


@pytest.fixture
def fitted(simple_panel):
    return gsynth(formula="Y~D", data=simple_panel, index=["unit","time"],
                  CV=False, r=2, se=False)


class TestEffect:
    def test_returns_dict(self, fitted):
        out = effect(fitted)
        assert isinstance(out, dict)

    def test_cumulative(self, fitted):
        out = effect(fitted, cumu=True)
        assert "att_cumulative" in out
        assert out["att_cumulative"] is not None

    def test_time_average(self, fitted):
        out = effect(fitted, cumu=False)
        assert "att_avg" in out
        assert isinstance(out["att_avg"], float)

    def test_period_filter(self, fitted):
        times = fitted.att_time
        if len(times) >= 3:
            p_start, p_end = times[0], times[1]
            out = effect(fitted, period=(p_start, p_end))
            assert len(out["times"]) <= 2

    def test_att_avg_close_to_result(self, fitted):
        out = effect(fitted)
        np.testing.assert_allclose(out["att_avg"], fitted.att_avg, rtol=1e-6)

    def test_cumulative_equals_sum(self, fitted):
        out = effect(fitted, cumu=True)
        expected_sum = float(np.nansum(out["att"]))
        assert out["att_cumulative"] == pytest.approx(expected_sum)

    def test_subgroup_id(self, fitted):
        unit_id = fitted.treat_units[0]
        out = effect(fitted, id=unit_id)
        assert out["n_units"] == 1

    def test_plot_flag(self, fitted):
        pytest.importorskip("matplotlib")
        import matplotlib.pyplot as plt
        out = effect(fitted, plot=True)
        assert "fig" in out
        plt.close("all")

    def test_n_units(self, fitted):
        out = effect(fitted)
        assert out["n_units"] == fitted.N_tr
