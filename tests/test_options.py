"""The Options object: defaults, dict interface, validation, state."""

import pickle

import numpy as np
import pytest

from PyToF import ClassToF
from PyToF.options import Options
from tests.cases import jupiter_polytrope

# -------------------------------------------------------------- defaults


def test_defaults_are_the_wisdom_hubbard_jupiter():
    opts = Options()
    assert opts["N"] == 1024
    assert opts["order"] == 4
    assert opts["R_phys"] == [71492000, "equatorial"]
    assert opts["M_phys"] == 126686536.1 * (1000) ** 3 / 6.6738480e-11
    assert opts["Target_Js"][0] == 1.398851089834702e-2
    assert opts["alphas"].shape == (12,)


def test_array_defaults_are_not_shared_between_instances():
    a, b = Options(), Options()
    a["Target_Js"][0] = 0.0
    a["R_phys"][0] = 1.0
    assert b["Target_Js"][0] != 0.0
    assert b["R_phys"][0] == 71492000


# -------------------------------------------------------- dict interface


def test_dict_interface():
    opts = Options.from_kwargs(N=64, order=7)
    assert opts["N"] == 64 and opts.get("order") == 7
    assert "verbosity" in opts and "nonsense" not in opts
    assert opts.get("nonsense", "fallback") == "fallback"
    assert dict(opts.items())["N"] == 64
    assert set(opts.keys()) == set(Options.names())

    opts["N"] = 128
    assert opts["N"] == 128


def test_copy_is_independent():
    a = Options()
    b = a.copy()
    b["N"] = 8
    assert a["N"] == 1024


def test_options_can_be_pickled():
    # Needed for multiprocessing in the MCMC.
    opts = Options.from_kwargs(N=64, atmosphere=np.sqrt)
    assert pickle.loads(pickle.dumps(opts))["N"] == 64


# ------------------------------------------------------------ validation


def test_unknown_option_suggests_the_right_name():
    with pytest.raises(TypeError, match="Did you mean 'verbosity'"):
        ClassToF.ToF(Verbosity=2)


def test_unknown_name_cannot_be_read_or_written():
    opts = Options()
    with pytest.raises(TypeError, match="not a PyToF option"):
        opts["ordr"]
    with pytest.raises(TypeError, match="Did you mean 'order'"):
        opts["ordr"] = 7


@pytest.mark.parametrize(
    "kwargs, error",
    [
        ({"order": 5}, ValueError),
        ({"R_phys": [7e7, "diameter"]}, ValueError),
        ({"R_phys": 7e7}, TypeError),
        ({"R_phys": [7e7]}, TypeError),
        ({"verbosity": -1}, ValueError),
        ({"verbosity": "loud"}, ValueError),
    ],
)
def test_invalid_values_are_rejected(kwargs, error):
    with pytest.raises(error):
        ClassToF.ToF(**kwargs)


def test_validation_also_applies_after_construction():
    X = ClassToF.ToF()
    with pytest.raises(ValueError):
        X.opts["order"] = 5
    assert X.opts["order"] == 4


# ----------------------------------------------------------------- state


def test_reference_radius_defaults_to_the_physical_radius():
    X = ClassToF.ToF(R_phys=[1e6, "mean"])
    assert X.opts["R_ref"] == 1e6


def test_relaxing_does_not_change_options():
    X = ClassToF.ToF(N=64, order=4)
    before = {k: v for k, v in X.opts.items() if np.isscalar(v)}
    X.set_barotrope(lambda P, p: np.sqrt(P))
    X.relax_to_barotrope()
    X.get_Js_errors()
    after = {k: v for k, v in X.opts.items() if np.isscalar(v)}
    assert after == before  # in particular n_bin stays -1


def test_simpson_fallback_is_state_not_an_option():
    N = 128
    X = ClassToF.ToF(
        N=N, M_phys=1e24, R_phys=[1e6, "mean"], Period=np.inf, order=4
    )
    assert X.simpson_in_use
    # A non-uniform grid forces the trapezoid rule:
    X.li = 1e6 * np.linspace(1, 1e-3, N) ** 1.5
    X.rhoi = 1000.0 * np.ones(N)
    X.m_rot_calc = 0.0
    X.relax_to_shape()

    assert not X.simpson_in_use
    assert X.opts["use_simpson"] is True  # what the user asked for

    X.opts["use_simpson"] = False  # user override still takes effect
    assert not X.simpson_in_use


def test_user_can_switch_simpson_off_after_construction():
    X = jupiter_polytrope(4, 64, use_simpson=False)
    assert not X.simpson_in_use


# ---------------------------------------------------------- documentation


def test_every_option_is_documented_and_grouped():
    table = Options.describe()
    for name in Options.names():
        assert f"`{name}`" in table
    from dataclasses import fields

    for f in fields(Options):
        assert f.metadata["doc"] and f.metadata["group"], f.name
