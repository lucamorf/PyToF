"""ToF.history: the convergence record of relax_to_barotrope/density."""

import warnings

import numpy as np

from PyToF import ClassToF
from PyToF.history import ConvergenceHistory
from tests.cases import R_JUPITER, jupiter_polytrope, polytrope_n1


def test_history_is_none_until_a_relaxation_and_after_reset():
    X = ClassToF.ToF(N=64, R_ref=R_JUPITER)
    assert X.history is None
    X.set_barotrope(lambda P, param=None: np.sqrt(P))
    X.relax_to_barotrope()
    assert isinstance(X.history, ConvergenceHistory)
    X.reset()
    assert X.history is None


def test_history_matches_the_returned_iterations_and_final_result():
    X = ClassToF.ToF(N=128, order=4, R_ref=R_JUPITER)
    X.set_barotrope(polytrope_n1)
    n_iterations = X.relax_to_barotrope()
    h = X.history
    assert h.name == "barotrope"
    assert h.converged
    # One entry per iteration; the one that converged is the returned count:
    assert len(h) == len(h.Js) == len(h.dJs) == len(h.drho) == n_iterations
    np.testing.assert_array_equal(h.Js[-1], X.Js)
    arrays = h.as_arrays()
    assert arrays["Js"].shape == (n_iterations, len(X.Js))
    assert arrays["dJs"][-1] < X.opts["dJ_tol"]
    assert arrays["drot"][-1] < X.opts["drot_tol"]
    assert arrays["drho"][-1] < X.opts["drho_tol"]
    assert arrays["dJs"][0] > arrays["dJs"][-1]


def test_history_of_a_run_that_hits_the_limit_is_not_converged():
    X = ClassToF.ToF(N=64, R_ref=R_JUPITER, max_iter_relax_barotrope=3)
    X.set_barotrope(lambda P, param=None: np.sqrt(P))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        X.relax_to_barotrope()
    assert not X.history.converged
    assert len(X.history) == 2  # iterations 1 and 2 of max_iter = 3


def test_relax_to_density_has_its_own_history():
    X = jupiter_polytrope(4, 128)
    rho = X.rhoi.copy()
    X.relax_to_density()
    assert X.history.name == "density"
    assert X.history.converged
    np.testing.assert_allclose(X.rhoi, rho)
