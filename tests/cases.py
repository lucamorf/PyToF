"""
Model setups shared by the tests and by make_golden.py.

Keeping them in one place guarantees that the golden-master file and the
regression test always describe exactly the same calculation.
"""

import numpy as np
from scipy.integrate import trapezoid

from PyToF import ClassToF

R_JUPITER = 71492000.0

# Wisdom & Hubbard (2016), Icarus 267, 315, doi:10.1016/j.icarus.2015.12.030:
# closed-form (Bessel function) solution for a rotating n=1 polytrope with
# P = rho^2, order of entries: [R_eq/R_m, J2, J4, ..., J20].
WH16_BESSEL = np.array(
    [
        1.022875431133174e0,
        1.398851089834702e-2,
        -5.318281001092907e-4,
        3.011832290533641e-5,
        -2.132115710725050e-6,
        1.740671195871128e-7,
        -1.568219505602588e-8,
        1.518099230068580e-9,
        -1.551985393081485e-10,
        1.655948019619652e-11,
        -1.829544870258362e-12,
    ]
)


def polytrope_n1(P, param=None):
    """Barotrope rho = sqrt(P) (n=1 polytrope in the units of W&H 2016)."""
    return np.sqrt(P)


def jupiter_polytrope(order=4, N=256, **kwargs):
    """Relaxed Jupiter-like n=1 polytrope, the W&H 2016 benchmark setup."""
    X = ClassToF.ToF(N=N, order=order, R_ref=R_JUPITER, **kwargs)
    X.set_barotrope(polytrope_n1)
    X.relax_to_barotrope()
    return X


def uniform_sphere(N=1024, period=np.inf, radius=1e6, rho=1000.0, order=4):
    """Relaxed constant-density body (sphere for period=inf)."""
    radii = radius * np.linspace(1, 1e-3, N)
    mass = -4 * np.pi * trapezoid(rho * np.ones(N) * radii**2, x=radii)
    X = ClassToF.ToF(
        N=N, M_phys=mass, R_phys=[radius, "mean"], Period=period, order=order
    )
    X.li = radii
    X.rhoi = rho * np.ones(N)
    X.m_rot_calc = (
        (2 * np.pi / period) ** 2 * X.li[0] ** 3 / (X.opts["G"] * mass)
    )
    X.relax_to_shape()
    return X


# ----------------------------------------------------------------------
# Golden-master cases. Each entry builds a relaxed ToF object.
# ----------------------------------------------------------------------

# Differential-rotation coefficients of Wisdom & Hubbard (2016), in m^2 s^-2,
# as used in PyToF_Accuracy_and_Convergence.ipynb:
ALPHAS_WH16 = np.zeros(12)
ALPHAS_WH16[:4] = (
    np.array(
        [
            -5.8954166431924311100e-2,
            4.6123389620922838894e-1,
            -1.0426642770099761037e0,
            8.3030636453130002295e-1,
        ]
    )
    * 1e6
)

# Concentric Maclaurin Spheroid result with the differential rotation above
# ("CMSDR2" in the accuracy notebook), W&H 2016: [J2, J4, J6]:
WH16_CMS_DR2_J2_J4_J6 = np.array(
    [1.399685529663848e-2, -5.358080750264726e-4, 3.159100841281628e-5]
)


def _atmosphere(norm_radii, pressure, K=1e2):
    return (pressure / K) ** (1 / 2)


def _barotrope_K(P, K):
    return np.sqrt(P / K)


def _dens_piecewise(radii, mass, param=None):
    x = radii / np.max(radii)
    param = np.concatenate(([0], param))
    rho = np.interp(x, np.linspace(0, 1, len(param)), np.cumsum(param)[::-1])
    return rho / (-4 * np.pi * trapezoid(rho * radii**2, x=radii)) * mass


def _density_case():
    N = 256
    radius, rho0 = 1e6, 1000.0
    radii = radius * np.linspace(1, 1e-3, N)
    mass = -4 * np.pi * trapezoid(rho0 * np.ones(N) * radii**2, x=radii)
    X = ClassToF.ToF(
        N=N,
        M_phys=mass,
        R_phys=[radius, "mean"],
        Period=24 * 3600,
        order=4,
        dens_param_init=[1.0] * 5,
    )
    X.li = radii
    X.rhoi = rho0 * np.ones(N)
    X.m_rot_calc = (
        (2 * np.pi / X.opts["Period"]) ** 2
        * X.li[0] ** 3
        / (X.opts["G"] * mass)
    )
    # relax_to_density() keeps the densities it is given (it only updates
    # shape and pressure), so we set them from the density function here.
    X.set_density_function(_dens_piecewise)
    X.rhoi = _dens_piecewise(X.li, mass, X.opts["dens_param_init"])
    X.relax_to_density()
    return X


def _atmosphere_case():
    # Same small test planet as in the tutorial (constant-density mass/radius).
    N, radius = 256, 1e6
    radii = radius * np.linspace(1, 1e-3, N)
    mass = -4 * np.pi * trapezoid(1000.0 * np.ones(N) * radii**2, x=radii)
    X = ClassToF.ToF(
        N=N,
        M_phys=mass,
        R_phys=[radius, "mean"],
        Period=24 * 3600,
        order=4,
        use_atmosphere=True,
        atmosphere=_atmosphere,
        atmosphere_until=1e6,
        baro_param_init=[1.0],
    )
    X.set_barotrope(_barotrope_K)
    X.relax_to_barotrope()
    return X


GOLDEN_CASES = {
    "jupiter_o4": lambda: jupiter_polytrope(4, 256),
    "jupiter_o7": lambda: jupiter_polytrope(7, 256),
    "jupiter_o10": lambda: jupiter_polytrope(10, 256),
    "jupiter_o7_trapezoid": lambda: jupiter_polytrope(
        7, 256, use_simpson=False
    ),
    "jupiter_o7_binned": lambda: jupiter_polytrope(7, 512, n_bin=128),
    "jupiter_o7_diffrot": lambda: jupiter_polytrope(
        7, 256, alphas=ALPHAS_WH16
    ),
    "jupiter_o10_diffrot": lambda: jupiter_polytrope(
        10, 256, alphas=ALPHAS_WH16
    ),
    "spheroid_o4": lambda: uniform_sphere(N=256, period=24 * 3600, order=4),
    "density_o4": _density_case,
    "atmosphere_o4": _atmosphere_case,
}


def summarize(X):
    """Numbers that must not change under a behaviour-preserving refactor."""
    idx = np.linspace(0, len(X.li) - 1, 12).astype(int)
    return {
        "Js": X.Js.tolist(),
        "NMoI": float(X.get_NMoI()),
        "R_eq_to_R_m": float(X.R_eq_to_R_m),
        "R_po_to_R_m": float(X.R_po_to_R_m),
        "m_rot_calc": float(X.m_rot_calc),
        "li_sample": X.li[idx].tolist(),
        "rhoi_sample": X.rhoi[idx].tolist(),
        "Pi_sample": X.Pi[idx].tolist(),
    }
