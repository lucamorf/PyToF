"""Run the MCMC entry points end to end on tiny problems."""

import numpy as np
import pytest

from PyToF import ClassToF


def _barotrope(P, K):
    return np.sqrt(P / K)


def _check_param(param):
    return bool(np.any(np.asarray(param) <= 0))


@pytest.fixture
def mc_model():
    X = ClassToF.ToF(
        N=64,
        order=4,
        R_ref=71492000,
        baro_param_init=[1.0],
        Target_Js=np.array([1.3988e-2, -5.318e-4]),
        Sigma_Js=np.array([1e-4, 1e-5]),
    )
    X.set_barotrope(_barotrope)
    X.set_check_param(_check_param)
    return X


def test_baro_mcmc_runs_serial(mc_model):
    np.random.seed(0)
    state, autocorr = mc_model.run_baro_MC(6, 5, parallelize=False)
    assert state.shape == (6, 1)
    assert np.all(np.isfinite(state))
    assert np.all(state > 0)


def test_classify_and_save_state(mc_model):
    np.random.seed(0)
    state, _ = mc_model.run_baro_MC(6, 5, parallelize=False)
    matches, rho_max = mc_model.classify_and_save_state(state, "baro")
    assert matches.shape == (6,)
    assert rho_max.shape == (6,)


@pytest.mark.slow
def test_baro_mcmc_runs_parallel(mc_model):
    np.random.seed(0)
    state, _ = mc_model.run_baro_MC(6, 5, Ncores=2, parallelize=True)
    assert state.shape == (6, 1)
