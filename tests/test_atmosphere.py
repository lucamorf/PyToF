"""The transition between atmosphere model and interior."""

import warnings

import numpy as np
import pytest

from PyToF import ToF
from PyToF.FunctionsToF import _check_atmosphere
from PyToF.messages import AccuracyWarning, ConvergenceWarning


def _planet(until=1e8, K_atmosphere=5e6):
    X = ToF(
        N=512,
        order=4,
        use_atmosphere=True,
        atmosphere=lambda level, P: np.sqrt(P / K_atmosphere),
        atmosphere_until=until,
    )
    X.set_barotrope(lambda P, param: np.sqrt(P / 2e5))
    return X


def test_transition_that_flips_between_two_level_surfaces_converges():
    # Following the pressures freely, the transition alternated between two
    # level surfaces for ever (2500 density updates, no convergence); once it
    # reverses direction it is now locked.
    X = _planet()
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        warnings.simplefilter("ignore", AccuracyWarning)  # Simpson fallback
        X.relax_to_barotrope()
    assert X._atmosphere_locked
    # ... at most one level surface deeper than requested (grid resolution):
    assert X.Pi[X.atmosphere_index - 2] <= X.opts["atmosphere_until"]


def test_reset_unlocks_the_transition():
    X = _planet()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        X.relax_to_barotrope()
    index = X.atmosphere_index
    X.reset()
    assert X.atmosphere_index == 0
    assert not X._atmosphere_locked
    assert X._atmosphere_last_step == 0
    assert index > 0


def test_atmosphere_deeper_than_requested_is_reported():
    X = _planet(until=1e6)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        X.relax_to_barotrope()
    _check_atmosphere(X)  # fine
    X.atmosphere_index += 5  # as if an early overshoot had been locked in
    with pytest.warns(AccuracyWarning, match="deeper than atmosphere_until"):
        _check_atmosphere(X)
