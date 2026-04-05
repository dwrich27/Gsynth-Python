"""Tests for the plot() function — ensure all plot types run without error."""
from __future__ import annotations

import pytest

pytest.importorskip("matplotlib")

from gsynth import gsynth, plot


@pytest.fixture
def fitted(small_panel):
    return gsynth(formula="Y~D", data=small_panel, index=["unit","time"],
                  CV=False, r=1, se=True, nboots=20,
                  inference="parametric", seed=0)


@pytest.fixture
def fitted_no_se(small_panel):
    return gsynth(formula="Y~D", data=small_panel, index=["unit","time"],
                  CV=False, r=1, se=False)


class TestPlotGap:
    def test_gap_returns_figure(self, fitted):
        import matplotlib.pyplot as plt
        fig = plot(fitted, type="gap", show=False)
        assert fig is not None
        plt.close("all")

    def test_gap_no_se(self, fitted_no_se):
        import matplotlib.pyplot as plt
        fig = plot(fitted_no_se, type="gap", show=False)
        assert fig is not None
        plt.close("all")

    def test_gap_shade_post(self, fitted):
        import matplotlib.pyplot as plt
        fig = plot(fitted, type="gap", shade_post=True, show=False)
        assert fig is not None
        plt.close("all")

    def test_gap_custom_labels(self, fitted):
        import matplotlib.pyplot as plt
        fig = plot(fitted, type="gap", xlab="Year", ylab="Effect",
                   main="My Gap Plot", show=False)
        ax = fig.axes[0]
        assert ax.get_xlabel() == "Year"
        assert ax.get_ylabel() == "Effect"
        assert ax.get_title() == "My Gap Plot"
        plt.close("all")

    def test_gap_legend_off(self, fitted):
        import matplotlib.pyplot as plt
        fig = plot(fitted, type="gap", legendOff=True, show=False)
        assert fig is not None
        plt.close("all")


class TestPlotRaw:
    def test_raw_returns_figure(self, fitted_no_se):
        import matplotlib.pyplot as plt
        fig = plot(fitted_no_se, type="raw", show=False)
        assert fig is not None
        plt.close("all")


class TestPlotCounterfactual:
    def test_ct_average(self, fitted_no_se):
        import matplotlib.pyplot as plt
        fig = plot(fitted_no_se, type="ct", show=False)
        assert fig is not None
        plt.close("all")

    def test_ct_specific_unit(self, fitted_no_se):
        import matplotlib.pyplot as plt
        unit_id = fitted_no_se.treat_units[0]
        fig = plot(fitted_no_se, type="counterfactual", id=unit_id, show=False)
        assert fig is not None
        plt.close("all")

    def test_ct_raw_band(self, fitted_no_se):
        import matplotlib.pyplot as plt
        fig = plot(fitted_no_se, type="ct", raw="band", show=False)
        assert fig is not None
        plt.close("all")

    def test_ct_raw_all(self, fitted_no_se):
        import matplotlib.pyplot as plt
        fig = plot(fitted_no_se, type="ct", raw="all", show=False)
        assert fig is not None
        plt.close("all")

    def test_ct_invalid_unit(self, fitted_no_se):
        with pytest.raises(ValueError, match="not found"):
            plot(fitted_no_se, type="ct", id=9999, show=False)


class TestPlotFactors:
    def test_factors_returns_figure(self, fitted_no_se):
        import matplotlib.pyplot as plt
        fig = plot(fitted_no_se, type="factors", show=False)
        assert fig is not None
        plt.close("all")

    def test_factors_nfactors_arg(self, fitted_no_se):
        import matplotlib.pyplot as plt
        fig = plot(fitted_no_se, type="factors", nfactors=1, show=False)
        assert fig is not None
        plt.close("all")

    def test_factors_zero_r_raises(self, small_panel):
        import matplotlib.pyplot as plt
        res = gsynth(formula="Y~D", data=small_panel, index=["unit","time"],
                     CV=False, r=0, se=False)
        with pytest.raises(ValueError, match="r=0"):
            plot(res, type="factors", show=False)
        plt.close("all")


class TestPlotLoadings:
    def test_loadings_returns_figure(self, fitted_no_se):
        import matplotlib.pyplot as plt
        fig = plot(fitted_no_se, type="loadings", show=False)
        assert fig is not None
        plt.close("all")


class TestPlotMissing:
    def test_missing_returns_figure(self, fitted_no_se):
        import matplotlib.pyplot as plt
        fig = plot(fitted_no_se, type="missing", show=False)
        assert fig is not None
        plt.close("all")

    def test_missing_subset_units(self, fitted_no_se):
        import matplotlib.pyplot as plt
        ids = fitted_no_se.units[:5]
        fig = plot(fitted_no_se, type="missing", id=ids, show=False)
        assert fig is not None
        plt.close("all")


class TestPlotInvalidType:
    def test_bad_type_raises(self, fitted_no_se):
        with pytest.raises(ValueError, match="Unknown plot type"):
            plot(fitted_no_se, type="banana", show=False)
