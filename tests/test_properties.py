"""Invariants that must hold for any correct implementation."""

import numpy as np
import pytest

from PyToF import ClassToF
from PyToF.FunctionsToF import _mass_int
from tests.cases import WH16_BESSEL, jupiter_polytrope


def test_error_decreases_with_resolution():
    coarse = jupiter_polytrope(7, N=128)
    fine = jupiter_polytrope(7, N=1024)
    err = lambda X: abs(X.Js[1] / WH16_BESSEL[1] - 1)  # noqa: E731
    assert err(fine) < err(coarse)


def test_simpson_is_more_accurate_than_trapezoid():
    simpson = jupiter_polytrope(7, N=512, use_simpson=True)
    trapezoid = jupiter_polytrope(7, N=512, use_simpson=False)
    err = lambda X: abs(X.Js[1] / WH16_BESSEL[1] - 1)  # noqa: E731
    assert err(simpson) < err(trapezoid)


def test_no_binning_equals_full_binning():
    N = 256
    a = jupiter_polytrope(4, N=N, n_bin=-1)
    b = jupiter_polytrope(4, N=N, n_bin=N)
    np.testing.assert_allclose(a.Js, b.Js, rtol=1e-12)


def test_binning_stays_close_to_unbinned():
    full = jupiter_polytrope(7, N=512)
    binned = jupiter_polytrope(7, N=512, n_bin=128)
    assert abs(binned.Js[1] / full.Js[1] - 1) < 1e-4


def test_relaxation_is_deterministic():
    a = jupiter_polytrope(4, N=128)
    b = jupiter_polytrope(4, N=128)
    np.testing.assert_array_equal(a.Js, b.Js)
    np.testing.assert_array_equal(a.rhoi, b.rhoi)


def test_mass_and_radius_are_conserved():
    X = jupiter_polytrope(7, N=512)
    assert _mass_int(X) == pytest.approx(X.opts["M_phys"], rel=1e-8)
    assert X.R_eq_to_R_m * X.li[0] == pytest.approx(
        X.opts["R_phys"][0], rel=1e-8
    )


def test_profiles_are_physical():
    X = jupiter_polytrope(4, N=256)
    assert np.all(np.diff(X.li) < 0), "li must run from the surface inwards"
    assert np.all(X.rhoi >= 0)
    assert np.all(np.diff(X.rhoi) >= 0), "density must grow inwards"
    assert np.all(np.diff(X.Pi) >= 0), "pressure must grow inwards"


def test_gravitational_harmonics_alternate_in_sign():
    X = jupiter_polytrope(7, N=256)
    signs = np.sign(X.Js[1:])
    assert np.all(signs == [(-1) ** k for k in range(len(signs))])


def test_js_errors_are_finite_and_positive():
    X = jupiter_polytrope(4, N=256)
    X.get_Js_errors()
    assert np.all(np.isfinite(X.Js_error[1:]))
    assert np.all(X.Js_error[1:] > 0)


def test_get_r_l_mu_shape_and_poles():
    X = jupiter_polytrope(4, N=64)
    r = X.get_r_l_mu(np.array([0.0, 1.0]))  # equator, pole
    assert r.shape == (64, 2)
    assert np.all(r[:, 0] > r[:, 1]), "equatorial radius exceeds polar radius"


def test_public_api_surface():
    # Guards the refactor of ClassToF (step 03): nothing may disappear.
    X = ClassToF.ToF()
    methods = [
        "relax_to_shape",
        "relax_to_barotrope",
        "relax_to_density",
        "set_barotrope",
        "set_density_function",
        "set_check_param",
        "get_Js_errors",
        "get_NMoI",
        "get_Kn",
        "get_r_l_mu",
        "get_U_l_mu",
        "run_baro_MC",
        "run_dens_MC",
        "baro_cost_function",
        "dens_cost_function",
        "classify_and_save_state",
    ]
    for name in methods:
        assert callable(getattr(X, name)), name
    for name in [
        "li",
        "rhoi",
        "Pi",
        "Js",
        "Js_error",
        "R_calc",
        "ss",
        "SS",
        "A0",
        "R_eq_to_R_m",
        "R_po_to_R_m",
        "m_rot_calc",
        "opts",
    ]:
        assert hasattr(X, name), name
