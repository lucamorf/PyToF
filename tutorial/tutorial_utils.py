"""
Small helpers shared by the tutorial notebooks. Nothing in here is needed to
use PyToF; the functions only save a few lines of repeated setup.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "accuracy"))
from references import JUPITER_RADIUS, Bessel  # noqa: E402,F401

from PyToF import ToF  # noqa: E402


def jupiter(N=1024, order=4, **options):
    """
    The Wisdom & Hubbard (2016) benchmark: a Jupiter-like single polytrope
    (P = rho^2, i.e. rho = sqrt(P)), relaxed to hydrostatic equilibrium.
    Extra keyword arguments are passed on to ToF().
    """
    X = ToF(N=N, order=order, R_ref=JUPITER_RADIUS, **options)
    X.set_barotrope(lambda P, param: np.sqrt(P))
    X.relax_to_barotrope()
    return X


def profiles(X):
    """
    The four per-shell quantities that are usually plotted, as a dict of
    arrays, ordered from the outermost shell inwards, like
    X.li, X.rhoi and X.Pi.

        "r":   mean radius of the shell divided by the outermost one
        "m":   mass enclosed by the shell divided by the total mass
        "rho": density in g/cm^3
        "P":   pressure in bar
    """
    # Mass of every shell between two level surfaces, and the mass inside
    # each shell (cumulative sum from the centre outwards, O(N)):
    shell_mass = (
        4 / 3 * np.pi * (X.li**3 - np.append(X.li[1:], 0) ** 3) * X.rhoi
    )
    enclosed = np.cumsum(shell_mass[::-1])[::-1]
    return {
        "r": X.li / X.li[0],
        "m": enclosed / enclosed[0],
        "rho": X.rhoi / 1000,
        "P": X.Pi / 1e5,
    }


LABELS = {
    "r": r"mean radius $l / l_\mathrm{max}$",
    "m": r"enclosed mass $m / M$",
    "rho": r"density $\rho$ [g cm$^{-3}$]",
    "P": r"pressure $P$ [bar]",
}
