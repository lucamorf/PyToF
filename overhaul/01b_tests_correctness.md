# 01b — Tests: correctness against theory

These tests say the numbers are **right**, independent of today's code.

**Tolerances** are ~3–4× the error I measured at N=1024 (table below), so a regression trips them but noise doesn't.

| order | J2 | J4 | J6 | J8 | J10 |
|---|---|---|---|---|---|
| 4 | 5e-6 | 2.5e-4 | – | – | – |
| 7 | 8e-7 | 6e-7 | 2.9e-6 | 9.5e-5 | – |
| 10 | 8e-7 | 6e-7 | 1.2e-7 | 3e-7 | 4.9e-6 |

Higher harmonics are left out on purpose: a finite-order ToF truncates them (that is what the accuracy notebook shows).

### `tests/test_analytical.py`
```python
"""Closed-form checks: no reference code or literature table needed."""

import numpy as np
import pytest

from tests.cases import uniform_sphere


def test_sphere_nmoi_constant_density():
    X = uniform_sphere(N=1024, period=np.inf)
    assert X.get_NMoI() == pytest.approx(2 / 5, rel=1e-7)


def test_sphere_nmoi_linear_density():
    N, radius = 1024, 1e6
    radii = radius * np.linspace(1, 1e-3, N)
    rho = np.linspace(0, 1000, N)  # same convention as the accuracy notebook
    mass = -4 * np.pi * np.trapezoid(rho * radii**2, radii)
    from PyToF import ClassToF

    X = ClassToF.ToF(
        N=N, M_phys=mass, R_phys=[radius, "mean"], Period=np.inf, order=4
    )
    X.li, X.rhoi = radii, rho
    X.m_rot_calc = 0.0
    X.relax_to_shape()
    assert X.get_NMoI() == pytest.approx(4 / 15, rel=1e-7)


def test_nonrotating_body_has_no_gravitational_harmonics():
    X = uniform_sphere(period=np.inf)
    assert np.allclose(X.Js[1:], 0.0, atol=1e-14)
    assert X.R_eq_to_R_m == pytest.approx(1.0, abs=1e-14)
    assert X.R_po_to_R_m == pytest.approx(1.0, abs=1e-14)


def test_rotating_uniform_spheroid_nmoi_matches_shape_formula():
    # For a constant-density spheroid the NMoI is 2/5 times (R_eq/R_m)^2,
    # exactly the comparison used in PyToF_Accuracy_and_Convergence.ipynb.
    X = uniform_sphere(N=1024, period=24 * 3600)
    expected = 2 / 5 * X.R_eq_to_R_m**2
    assert X.get_NMoI() == pytest.approx(expected, rel=1e-7)


def test_rotation_flattens_the_planet():
    X = uniform_sphere(period=24 * 3600)
    assert X.R_eq_to_R_m > 1.0 > X.R_po_to_R_m


@pytest.mark.parametrize("order", [4, 7, 10])
def test_J2_of_uniform_spheroid_is_positive_and_J4_negative(order):
    X = uniform_sphere(N=256, period=24 * 3600, order=order)
    assert X.Js[1] > 0 > X.Js[2]
```

### `tests/test_literature.py`
```python
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
# them (see the accuracy notebook).
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
```
