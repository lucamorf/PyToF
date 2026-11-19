"""Closed-form checks: no reference code or literature table needed."""

import numpy as np
import pytest
from scipy.integrate import trapezoid

from tests.cases import uniform_sphere


def test_sphere_nmoi_constant_density():
    X = uniform_sphere(N=1024, period=np.inf)
    assert X.get_NMoI() == pytest.approx(2 / 5, rel=1e-7)


def test_sphere_nmoi_linear_density():
    N, radius = 1024, 1e6
    radii = radius * np.linspace(1, 1e-3, N)
    rho = np.linspace(0, 1000, N)  # same convention as accuracy/accuracy.ipynb
    mass = -4 * np.pi * trapezoid(rho * radii**2, x=radii)
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
    # exactly the comparison used in accuracy/accuracy.ipynb.
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
