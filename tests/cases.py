"""
Model setups shared by the tests and by make_golden.py.

Keeping them in one place guarantees that the golden-master file and the
regression test always describe exactly the same calculation.
"""

import importlib.util
from pathlib import Path

import numpy as np
from scipy.integrate import trapezoid

from PyToF import ClassToF

# Literature reference values live in accuracy/references.py, shared with the
# accuracy notebooks (so notebooks and tests cannot disagree). Wisdom &
# Hubbard (2016), Icarus 267, 315, doi:10.1016/j.icarus.2015.12.030:
# closed-form (Bessel function) solution for a rotating n=1 polytrope with
# P = rho^2, order of entries: [R_eq/R_m, J2, J4, ..., J20].
_spec = importlib.util.spec_from_file_location(
    "pytof_references",
    Path(__file__).resolve().parents[1] / "accuracy" / "references.py",
)
references = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(references)
WH16_BESSEL = references.Bessel
R_JUPITER = references.JUPITER_RADIUS


def polytrope_n1(P, param=None):
    """Barotrope rho = sqrt(P) (n=1 polytrope in the units of W&H 2016)."""
    return np.sqrt(P)


def jupiter_polytrope(order=4, N=256, **kwargs):
    """Relaxed Jupiter-like n=1 polytrope, the W&H 2016 benchmark setup."""
    X = ClassToF.ToF(N=N, order=order, R_ref=R_JUPITER, **kwargs)
    X.set_barotrope(polytrope_n1)
    X.relax_to_barotrope()
    return X


def uniform_sphere(
    N=1024, period=np.inf, radius=1e6, rho=1000.0, order=4, **kwargs
):
    """
    Relaxed constant-density body (sphere for period=inf). Further keyword
    arguments are passed on to ToF and override the defaults set here.
    """
    radii = radius * np.linspace(1, 1e-3, N)
    mass = -4 * np.pi * trapezoid(rho * np.ones(N) * radii**2, x=radii)
    options = dict(
        N=N, M_phys=mass, R_phys=[radius, "mean"], Period=period, order=order
    )
    X = ClassToF.ToF(**{**options, **kwargs})
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

# Differential-rotation coefficients of Wisdom & Hubbard (2016), in m^2 s^-2:
ALPHAS_WH16 = references.alphas_array(references.ALPHAS_DR2)

# Concentric Maclaurin Spheroid result with the differential rotation above
# ("CMSDR2"), W&H 2016: [J2, J4, J6]:
WH16_CMS_DR2_J2_J4_J6 = references.CMSDR2[1:4]


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
    )
    X.li = radii
    X.rhoi = rho0 * np.ones(N)
    X.m_rot_calc = (
        (2 * np.pi / X.opts["Period"]) ** 2
        * X.li[0] ** 3
        / (X.opts["G"] * mass)
    )
    # relax_to_density() keeps the densities it is given (it only updates
    # shape and pressure), so we set them from a density profile here.
    X.rhoi = _dens_piecewise(X.li, mass, [1.0] * 5)
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
