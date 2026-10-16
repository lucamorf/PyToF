"""The Options object: defaults, dict interface, validation, state."""

import pickle

import numpy as np
import pytest

from PyToF import ClassToF
from PyToF.messages import RenamedWarning
from PyToF.options import Options
from tests.cases import jupiter_polytrope

# -------------------------------------------------------------- defaults


def test_defaults_are_the_wisdom_hubbard_jupiter():
    opts = Options()
    assert opts["N"] == 1024
    assert opts["order"] == 4
    assert opts["R_phys"] == [71492000, "equatorial"]
    assert opts["M_phys"] == 126686536.1 * (1000) ** 3 / 6.6738480e-11
    assert opts["baro_param_init"] is None
    assert opts["alphas"].shape == (12,)


def test_array_defaults_are_not_shared_between_instances():
    a, b = Options(), Options()
    a["alphas"][0] = 1.0
    a["R_phys"][0] = 1.0
    assert b["alphas"][0] == 0.0
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
        ({"max_iter_barotrope": 1}, ValueError),
        ({"max_iter_density": 0}, ValueError),
        ({"max_iter_update": 1}, ValueError),
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


# ---------------------------------------------------------- renamed options

RENAMED = {
    "MaxIterShape": "max_iter_shape",
    "MaxIterUpdate": "max_iter_update",
    "MaxIterBar": "max_iter_barotrope",
    "MaxIterDen": "max_iter_density",
}


@pytest.mark.parametrize("old, new", RENAMED.items())
def test_old_option_names_still_work_but_warn(old, new):
    opts = Options()

    with pytest.warns(RenamedWarning, match=new) as record:
        value = opts[old]
    assert record[0].filename == __file__  # points at the user's line
    np.testing.assert_array_equal(value, opts[new])

    with pytest.warns(RenamedWarning):
        opts[old] = opts[new]

    with pytest.warns(RenamedWarning):
        getattr(opts, old)


def test_old_names_work_as_constructor_keywords():
    with pytest.warns(RenamedWarning, match="max_iter_barotrope") as record:
        X = ClassToF.ToF(N=64, MaxIterBar=7)
    assert record[0].filename == __file__
    assert X.opts["max_iter_barotrope"] == 7


def test_setting_an_old_name_changes_the_new_option():
    opts = Options()
    with pytest.warns(RenamedWarning):
        opts["MaxIterShape"] = 5
    assert opts["max_iter_shape"] == 5


def test_old_and_new_name_together_are_rejected():
    with pytest.warns(RenamedWarning):
        with pytest.raises(TypeError, match="same option"):
            Options.from_kwargs(MaxIterBar=3, max_iter_barotrope=4)


def test_membership_accepts_old_names_quietly():
    assert "MaxIterBar" in Options()
    assert "NoSuchOption" not in Options()


def test_new_names_do_not_warn():
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        opts = Options.from_kwargs(max_iter_barotrope=3, max_iter_update=4)
        opts["max_iter_density"] = 5
        assert opts.max_iter_barotrope == 3


# -------------------------------------------------- renamed attributes


def test_reset_restores_a_fresh_object():
    fresh = ClassToF.ToF(N=64, order=4, R_ref=71492000)
    X = jupiter_polytrope(4, 64)
    assert not np.array_equal(X.Js, fresh.Js)  # it really was used

    X.reset()

    for name in ["li", "rhoi", "Pi", "Js", "Js_errors", "A0", "U"]:
        np.testing.assert_array_equal(getattr(X, name), getattr(fresh, name))
    for name in ["m_rot_calc", "R_calc", "R_eq_to_R_m", "R_po_to_R_m"]:
        assert getattr(X, name) == getattr(fresh, name)


def test_old_method_name_still_works_but_warns():
    X = ClassToF.ToF(N=32)
    with pytest.warns(RenamedWarning, match="reset"):
        X._set_IC()


@pytest.mark.parametrize(
    "old, new",
    [
        ("Js_error", "Js_errors"),
        ("ss", "figure_functions"),
        ("SS", "volume_integrals"),
    ],
)
def test_old_attribute_names_still_work_but_warn(old, new):
    X = jupiter_polytrope(4, 32)

    with pytest.warns(RenamedWarning, match=new) as record:
        value = getattr(X, old)
    assert record[0].filename == __file__  # points at the user's line
    assert value is getattr(X, new)

    replacement = [np.zeros(3)]
    with pytest.warns(RenamedWarning):
        setattr(X, old, replacement)
    assert getattr(X, new) is replacement


def test_figure_functions_and_volume_integrals_are_filled_by_relaxing():
    X = jupiter_polytrope(7, 64)
    assert len(X.figure_functions) == 8  # s_n for n = 0, 2, ..., 2 * order
    assert len(X.volume_integrals) == 16  # S_n followed by the primed S'_n
    for quantity in (X.figure_functions, X.volume_integrals):
        assert all(len(entry) == 64 for entry in quantity)
    assert np.any(X.figure_functions[1] != 0)
