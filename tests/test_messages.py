"""Console output: verbosity levels, colours, and warnings."""

import warnings

import numpy as np
import pytest

from PyToF import ClassToF
from PyToF.messages import (
    AccuracyWarning,
    ConvergenceWarning,
    PyToFWarning,
    format_message,
    say,
    warn,
)
from tests.cases import jupiter_polytrope, polytrope_n1, uniform_sphere

ESC = "\033["


# ---------------------------------------------------------------- formatting


def test_format_plain():
    assert format_message("x = {x:.1e}!", x=1234.5) == "x = 1.2e+03!"


def test_format_colour_highlights_only_the_fields():
    out = format_message("a {x} b", colour=True, kind="warn", x=7)
    assert out.startswith("\033[91m")  # warn colour
    assert "\033[93m7\033[91m" in out  # count highlighted, base restored
    assert out.endswith("\033[0m")


def test_format_colour_distinguishes_counts_from_results():
    out = format_message("{n} steps, J = {j:.1e}", colour=True, n=7, j=0.5)
    assert "\033[93m7" in out  # integer
    assert "\033[96m5.0e-01" in out  # float


def test_info_text_stays_plain_and_only_numbers_are_coloured():
    out = format_message("see {name}: {n}", colour=True, name="shape", n=3)
    assert out.startswith("see shape: ")  # no colour before the number
    assert "\033[93m3\033[0m" in out


def test_format_colour_keeps_format_spec_and_literal_braces():
    out = format_message("{{{x:.2f}}}", colour=True, x=1.0)
    assert "{" in out and "1.00" in out and "}" in out


# ------------------------------------------------------------- say / levels


@pytest.mark.parametrize(
    "verbosity, shown", [(0, False), (1, True), (2, True)]
)
def test_say_respects_level(capsys, verbosity, shown):
    say(verbosity, 1, "hello {n}", n=3)
    assert ("hello 3" in capsys.readouterr().out) is shown


def test_no_colour_codes_when_output_is_not_a_terminal(capsys):
    say(5, 1, "value {v}", v=1)
    assert ESC not in capsys.readouterr().out


def test_force_colour(capsys, monkeypatch):
    monkeypatch.setenv("FORCE_COLOR", "1")
    say(5, 1, "value {v}", v=1)
    assert ESC in capsys.readouterr().out


def test_no_colour_wins(capsys, monkeypatch):
    monkeypatch.setenv("FORCE_COLOR", "1")
    monkeypatch.setenv("NO_COLOR", "1")
    say(5, 1, "value {v}", v=1)
    assert ESC not in capsys.readouterr().out


# ------------------------------------------------------------------ warnings


def _library_function():
    warn("n={n}", n=3)


def test_warn_uses_category_and_points_at_the_user():
    # Like a library function: the warning is attributed to whoever called it.
    with pytest.warns(PyToFWarning, match="n=3") as record:
        _library_function()
    assert record[0].filename == __file__


@pytest.mark.parametrize(
    "verbosity, expected",
    [
        (0, []),
        (1, ["Relaxing to barotrope finished after"]),
        (2, ["Relaxing to barotrope finished after", "barotrope done in"]),
        (
            3,
            [
                "Relaxing to barotrope finished after",
                "barotrope done in",
                "Iteration 1: drot",
                "Renormalized the radii",
            ],
        ),
    ],
)
def test_messages_by_verbosity(capsys, verbosity, expected):
    jupiter_polytrope(4, 64, verbosity=verbosity)
    out = capsys.readouterr().out
    for text in expected:
        assert text in out
    if verbosity < 2:
        assert "done in" not in out
    if verbosity < 3:
        assert "Iteration" not in out and "Renormalized" not in out


def test_relax_to_shape_summary(capsys):
    X = uniform_sphere(N=128, period=24 * 3600, verbosity=0)
    X.opts["verbosity"] = 2
    X.relax_to_shape()
    out = capsys.readouterr().out
    assert "Relaxing to shape finished after" in out
    assert "Relaxing to shape done in" in out


def test_simpson_fallback_warns_even_when_quiet():
    X = ClassToF.ToF(
        N=128, M_phys=1e24, R_phys=[1e6, "mean"], Period=np.inf, order=4
    )
    X.li = 1e6 * np.linspace(1, 1e-3, 128) ** 1.5  # non-uniform grid
    X.rhoi = 1000.0 * np.ones(128)
    X.m_rot_calc = 0.0
    with pytest.warns(AccuracyWarning, match="radius grid is non-uniform"):
        X.relax_to_shape()
    assert not X.simpson_in_use


def test_non_equidistant_grid_warns_in_get_NMoI():
    X = jupiter_polytrope(4, 64)
    X.li = X.li ** 1.1
    with pytest.warns(PyToFWarning, match="not equidistant"):
        X.get_NMoI()


def test_atmosphere_transition_is_reported(capsys):
    X = ClassToF.ToF(
        N=128,
        R_ref=71492000,
        verbosity=2,
        use_atmosphere=True,
        atmosphere_until=1e5,
        atmosphere=lambda level, P: 0.1 * np.sqrt(P) / np.sqrt(1e5) + 1.0,
    )
    X.set_barotrope(polytrope_n1)
    try:
        X.relax_to_barotrope()
    except ValueError:  # the toy atmosphere may create a density inversion
        pass
    assert "The atmosphere model now extends to level surface" in (
        capsys.readouterr().out
    )


def test_normal_run_issues_no_warnings():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        jupiter_polytrope(7, 128)


def test_density_jump_warns_about_the_trapezoid_fallback():
    # A uniform sphere has a density jump at its surface, which makes S_0
    # non-monotonic.
    with pytest.warns(AccuracyWarning, match="trapezoid rule"):
        uniform_sphere(N=128, period=24 * 3600)


def test_shape_not_converging_warns():
    with pytest.warns(ConvergenceWarning, match="dJ_tol"):
        uniform_sphere(N=128, period=24 * 3600, max_iter_relax_shape=1)


def test_barotrope_not_converging_warns_and_names_the_limit():
    X = ClassToF.ToF(N=64, order=4, R_ref=71492000, max_iter_relax_barotrope=3)
    X.set_barotrope(polytrope_n1)
    with pytest.warns(
        ConvergenceWarning, match="max_iter_relax_barotrope = 3"
    ) as record:
        X.relax_to_barotrope()
    assert record[0].filename == __file__  # points at the user's line


def test_inconsistent_radius_warns():
    # The sphere helper fixes the *mean* radius; declaring it equatorial
    # contradicts the flattening of a rotating body.
    with pytest.warns(PyToFWarning, match="equatorial radius"):
        uniform_sphere(N=128, period=24 * 3600, R_phys=[1e6, "equatorial"])


def test_error_estimate_unavailable_warns():
    X = jupiter_polytrope(4, 64, alphas=np.r_[1.0, np.zeros(11)])
    with pytest.warns(PyToFWarning, match="differential rotation"):
        X.get_Js_errors()
