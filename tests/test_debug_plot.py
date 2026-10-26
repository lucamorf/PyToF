"""Smoke tests of the debug plots (PlotDebugToF) and of the lazy import."""

import subprocess
import sys

import pytest

from tests.cases import jupiter_polytrope

pytest.importorskip("matplotlib")


def test_matplotlib_not_imported_without_debug_plot():
    code = (
        "import sys; import PyToF; from tests.cases import jupiter_polytrope;"
        "X = jupiter_polytrope(N=32); X.relax_to_barotrope();"
        "assert 'matplotlib' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_debug_plot_writes_figures(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    X = jupiter_polytrope(N=32, debug_plot=True, max_iter_relax_barotrope=2)
    X.relax_to_barotrope()
    names = {p.name for p in tmp_path.glob("*.png")}
    assert any(n.startswith("debug_FunctionsToF_rho_") for n in names)
    assert any(n.startswith("debug_FunctionsToF_P_") for n in names)
    assert "debug_AlgoToF_integral_0.png" in names
    assert "debug_AlgoToF_integrand_p_0.png" in names
