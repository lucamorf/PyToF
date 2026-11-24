"""Smoke tests of the debug plots (PlotDebugToF) and of the lazy import."""

import subprocess
import sys

import numpy as np
import pytest

from PyToF import ClassToF
from PyToF.messages import PressureError
from tests.cases import R_JUPITER, jupiter_polytrope

pytest.importorskip("matplotlib")


def _jump(P, param=None):
    return np.sqrt(P) * (1 + 1.0 * (P > 0.3 * np.max(P)))


def test_matplotlib_not_imported_without_debug_plot():
    code = (
        "import sys; import PyToF; from tests.cases import jupiter_polytrope;"
        "X = jupiter_polytrope(N=32); X.relax_to_barotrope();"
        "assert 'matplotlib' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


@pytest.mark.parametrize("debug_plot", [True, "save"])
@pytest.mark.filterwarnings("ignore::PyToF.messages.ConvergenceWarning")
def test_debug_plot_saves_one_large_figure_per_iteration(
    tmp_path, monkeypatch, debug_plot
):
    monkeypatch.chdir(tmp_path)
    # the helper relaxes once:
    jupiter_polytrope(N=32, debug_plot=debug_plot, max_iter_relax_barotrope=2)
    names = {p.name for p in tmp_path.glob("*.png")}
    assert "debug_relax_barotrope_0.png" in names
    assert "debug_AlgoToF_0.png" in names
    # one per iteration: two shape calls of two steps each, one relax figure
    assert sum(n.startswith("debug_relax_") for n in names) == 1
    assert sum(n.startswith("debug_AlgoToF_") for n in names) >= 4


@pytest.mark.filterwarnings("ignore::PyToF.messages.ConvergenceWarning")
def test_debug_plot_show_displays_and_writes_nothing(tmp_path, monkeypatch):
    import matplotlib.pyplot as plt

    shown = []
    monkeypatch.setattr(plt, "show", lambda *a, **k: shown.append(1))
    monkeypatch.chdir(tmp_path)
    jupiter_polytrope(N=32, debug_plot="show", max_iter_relax_barotrope=2)
    assert len(shown) >= 5
    assert not list(tmp_path.glob("*.png"))


def test_debug_plot_option_is_validated():
    with pytest.raises(ValueError, match="debug_plot"):
        ClassToF.ToF(debug_plot="screen")


def test_simpson_check_off_keeps_simpson_and_plots_the_failure(
    tmp_path, monkeypatch
):
    """Without the check a density jump ends in a PressureError, after
    a figure was drawn."""
    monkeypatch.chdir(tmp_path)
    X = ClassToF.ToF(
        N=128,
        R_ref=R_JUPITER,
        simpson_check=False,
        debug_plot="save",
        max_iter_relax_barotrope=3,
    )
    X.set_barotrope(_jump)
    with pytest.raises(PressureError, match="simpson_check"):
        X.relax_to_barotrope()
    assert X.simpson_in_use
    assert list(tmp_path.glob("debug_relax_barotrope_*.png"))


def test_simpson_check_on_is_the_default_and_falls_back():
    X = ClassToF.ToF(N=128, R_ref=R_JUPITER)
    assert X.opts["simpson_check"] is True
    X.set_barotrope(_jump)
    with pytest.warns(Warning, match="trapezoid"):
        X.relax_to_barotrope()
    assert not X.simpson_in_use


@pytest.mark.parametrize("primed", [False, True])
@pytest.mark.filterwarnings("ignore::PyToF.messages.ConvergenceWarning")
def test_debug_primed_selects_the_volume_integrals(
    tmp_path, monkeypatch, primed
):
    monkeypatch.chdir(tmp_path)
    jupiter_polytrope(
        N=32,
        debug_plot="save",
        debug_primed=primed,
        max_iter_relax_barotrope=2,
    )
    assert (tmp_path / "debug_AlgoToF_0.png").exists()
