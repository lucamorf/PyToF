"""
Love numbers: density jumps against the analytical two-layer planet of
Consorzi, Melini & Spada (2023), A&A 676, A21, doi:10.1051/0004-6361/202346352
(Eq. 2, see accuracy/references.py::two_layer_k2).
"""

import numpy as np
import pytest
from scipy.integrate import trapezoid

from PyToF import ToF
from tests.cases import jupiter_polytrope, references

pytestmark = pytest.mark.filterwarnings(
    "ignore::PyToF.messages.AccuracyWarning"
)

RADIUS = 1e6


def _planet(density_of_radius, N=2048):
    radii = RADIUS * np.linspace(1, 1e-3, N)
    dens = density_of_radius(radii)
    mass = -4 * np.pi * trapezoid(dens * radii**2, radii)
    X = ToF(
        N=N,
        order=4,
        M_phys=mass,
        R_phys=[RADIUS, "mean"],
        Period=np.inf,
        R_ref=RADIUS,
    )
    X.li, X.rhoi, X.m_rot_calc = radii, dens, 0.0
    X.relax_to_shape()
    return X


def _two_layers(r_core, rho_core, rho_mantle):
    return lambda radii: np.where(
        radii < r_core * RADIUS, rho_core, rho_mantle
    )


@pytest.mark.parametrize(
    "r_core, rho_core, rho_mantle",
    [(0.5, 3000.0, 1000.0), (0.3, 5000.0, 1000.0), (0.8, 1500.0, 1000.0)],
)
def test_two_layer_planet_with_density_jump_matches_the_paper(
    r_core, rho_core, rho_mantle
):
    X = _planet(_two_layers(r_core, rho_core, rho_mantle))
    k = X.get_Kn(2, bs=[r_core * RADIUS])[2][-1]
    exact = references.two_layer_k2(r_core, rho_core, rho_mantle)
    assert k == pytest.approx(exact, rel=3e-3)


def test_surface_jump_is_taken_into_account():
    # Homogeneous planet: k_2 = 3/2 (the paper's formula for equal densities),
    # which exists only through the density jump at the surface.
    X = _planet(lambda radii: 1000.0 * np.ones_like(radii))
    exact = references.two_layer_k2(0.5, 1000.0, 1000.0)
    assert exact == pytest.approx(1.5, rel=1e-12)
    assert X.get_Kn(2)[2][-1] == pytest.approx(exact, rel=1e-3)


def test_jump_without_bs_is_integrated_over_as_a_steep_gradient():
    # Without bs, the steep gradient on the grid makes the jump count
    # approximately; the explicit boundary condition is the accurate way.
    X = _planet(_two_layers(0.5, 3000.0, 1000.0))
    exact = references.two_layer_k2(0.5, 3000.0, 1000.0)
    k_bs = X.get_Kn(2, bs=[0.5 * RADIUS])[2][-1]
    k_no_bs = X.get_Kn(2)[2][-1]
    assert abs(k_bs / exact - 1) < abs(k_no_bs / exact - 1)


def test_polytrope_without_surface_density_is_unchanged():
    # The n = 1 polytrope has no density at the surface, so the surface term
    # is zero and k_2 = 15 / pi^2 - 1 (Gavrilov et al. 1976):
    X = jupiter_polytrope(4, N=1024, Period=np.inf)
    assert X.rhoi[0] == 0
    assert X.get_Kn(2)[2][-1] == pytest.approx(15 / np.pi**2 - 1, rel=1e-4)
