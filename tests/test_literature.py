"""
Compare against the closed-form Bessel solution for a rotating n=1 polytrope
(Wisdom & Hubbard 2016, Icarus 267, 315) and their CMS differential-rotation
result. Tolerances are ~3-4x the errors measured at N=1024 when the tests were
written, so they catch regressions without being flaky.
"""

import numpy as np
import pytest

from tests.cases import (
    ALPHAS_WH16,
    WH16_BESSEL,
    WH16_CMS_DR2_J2_J4_J6,
    jupiter_polytrope,
)

# Max allowed relative error of J_{2k} (k = 1, 2, ...) per ToF order.
# Higher harmonics are intentionally absent: a finite-order ToF truncates
# them (see accuracy/accuracy.ipynb).
JS_RTOL = {
    4: [2e-5, 1e-3],
    7: [5e-6, 5e-6, 1e-5, 4e-4],
    10: [5e-6, 5e-6, 1e-6, 2e-6, 2e-5],
}


@pytest.fixture(scope="module", params=[4, 7, 10])
def bessel_run(request):
    order = request.param
    return order, jupiter_polytrope(order, N=1024)


def test_js_match_bessel_solution(bessel_run):
    order, X = bessel_run
    for k, tol in enumerate(JS_RTOL[order], start=1):
        rel = abs(X.Js[k] / WH16_BESSEL[k] - 1)
        assert rel < tol, f"order {order}: J{2 * k} rel. error {rel:.2e}"


def test_higher_order_is_more_accurate_for_J6():
    err = {
        order: abs(jupiter_polytrope(order, 1024).Js[3] / WH16_BESSEL[3] - 1)
        for order in (4, 7, 10)
    }
    assert err[10] < err[7] < err[4]


def test_differential_rotation_matches_cms():
    X = jupiter_polytrope(7, N=1024, alphas=ALPHAS_WH16)
    rel = np.abs(X.Js[1:4] / WH16_CMS_DR2_J2_J4_J6 - 1)
    assert np.all(rel < 5e-4), rel


def test_differential_rotation_changes_the_solution():
    # Guards against `alphas` silently being ignored (their scale is
    # ~1e5 m^2/s^2; much smaller values have no visible effect).
    rigid = jupiter_polytrope(7, 256)
    differential = jupiter_polytrope(7, 256, alphas=ALPHAS_WH16)
    assert abs(differential.Js[1] / rigid.Js[1] - 1) > 1e-5


def test_orders_7_and_10_agree_with_differential_rotation():
    # Without differential rotation orders 7 and 10 agree to ~1e-9 in J2 and
    # J4. Dropping any differential-rotation term at one order breaks this
    # (an order-10 A0 term was once missing and shifted J2 by 1e-4).
    o7 = jupiter_polytrope(7, N=512, alphas=ALPHAS_WH16)
    o10 = jupiter_polytrope(10, N=512, alphas=ALPHAS_WH16)
    rel = np.abs(o10.Js[1:3] / o7.Js[1:3] - 1)
    assert np.all(rel < 1e-6), rel


def test_mean_radius_definition_explains_the_bessel_radius_ratio():
    # PyToF's R_m (X.R_eq_to_R_m) is the volume-equivalent radius; Wisdom &
    # Hubbard (2016, Eqs. 3-5) use R with no P_0 term in the shape function,
    # i.e. the angular mean radius R_m * (1 + s0). The 3.9e-4 gap to their
    # Table 3 value is exactly s0.
    X = jupiter_polytrope(7, N=1024)
    s0 = X.figure_functions[0][0]
    assert abs(X.R_eq_to_R_m / WH16_BESSEL[0] - 1) > 3e-4
    assert abs(X.R_eq_to_R_m / (1 + s0) / WH16_BESSEL[0] - 1) < 5e-6
