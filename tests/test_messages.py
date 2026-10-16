"""Console output: verbosity levels, colours, and warnings."""

import warnings

import numpy as np
import pytest

from PyToF import ClassToF
from PyToF.messages import (
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
    assert "\033[96m7\033[91m" in out  # number highlighted, base restored
    assert out.endswith("\033[0m")


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
        (1, []),
        (2, ["Relaxing to barotrope done"]),
        (3, ["Relaxing to barotrope done", "Relaxing to shape done"]),
    ],
)
def test_timing_messages_by_verbosity(capsys, verbosity, expected):
    jupiter_polytrope(4, 64, verbosity=verbosity)
    out = capsys.readouterr().out
    for text in expected:
        assert text in out
    if verbosity < 3:
        assert "Relaxing to shape done" not in out


def test_normal_run_issues_no_warnings():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        jupiter_polytrope(7, 128)
        uniform_sphere(N=128, period=24 * 3600)


def test_shape_not_converging_warns():
    with pytest.warns(ConvergenceWarning, match="dJ_tol"):
        uniform_sphere(N=128, period=24 * 3600, max_iter_shape=1)


def test_barotrope_not_converging_warns_and_names_the_limit():
    X = ClassToF.ToF(N=64, order=4, R_ref=71492000, max_iter_barotrope=3)
    X.set_barotrope(polytrope_n1)
    with pytest.warns(ConvergenceWarning, match="max_iter_barotrope = 3"):
        X.relax_to_barotrope()


def test_inconsistent_radius_warns():
    # The sphere helper fixes the *mean* radius; declaring it equatorial
    # contradicts the flattening of a rotating body.
    with pytest.warns(PyToFWarning, match="equatorial radius"):
        uniform_sphere(N=128, period=24 * 3600, R_phys=[1e6, "equatorial"])


def test_error_estimate_unavailable_warns():
    X = jupiter_polytrope(4, 64, alphas=np.r_[1.0, np.zeros(11)])
    with pytest.warns(PyToFWarning, match="differential rotation"):
        X.get_Js_errors()
