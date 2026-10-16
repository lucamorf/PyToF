"""
The MCMC driver: convergence, saving fits during the chain, robustness.

Most of the logic (convergence checks, selection of distinct models, saving
and resuming) does not know about planets and is tested with an analytic
model that is evaluated in microseconds. A few small tests then run the real
ToF model. One evaluation of it takes ~0.05 s even for tiny grids, hence the
loose solver tolerances and the few walkers and steps there.
"""

import csv
import functools

import numpy as np
import pytest

from PyToF import ClassToF, ConvergenceCriteria, MCMCResult, SaveFits
from PyToF.MonteCarloToF import _write_profile
from PyToF.sampling import FitSaver, diagnose, run_sampler, select_distinct
from tests.cases import (
    R_JUPITER,
    _barotrope_K,
    _is_not_positive,
    jupiter_polytrope,
)

# ------------------------------------------------- synthetic model helpers

CENTER = np.array([3.0, -2.0])


def gaussian(param, center):
    """Unit Gaussian; the blob is the distance from the centre in sigma."""

    d = param - center
    return -0.5 * np.sum(d**2), float(np.sqrt(np.sum(d**2)))


def fake_save(center, job):
    """Stands in for writing a model: returns the worst deviation in sigma."""

    param, _ = job
    return float(np.sqrt(np.sum((param - center) ** 2)))


def start_positions(nwalkers=16):
    np.random.seed(0)
    return CENTER + 3 * np.random.RandomState(1).normal(size=(nwalkers, 2))


def run_synthetic(directory, steps=400, **settings):
    """Gaussian chain with a FitSaver; returns (saver, run_sampler result)."""

    options = dict(burn_in=100, n_sigma=1.5, min_distance=0.5)
    options.update(settings)
    saver = FitSaver(SaveFits(directory, **options), 2)
    save_model = functools.partial(fake_save, CENTER)

    outcome = run_sampler(
        gaussian,
        [CENTER],
        start_positions(),
        steps,
        ConvergenceCriteria(check_every=100),
        callback=lambda sampler, diagnostics: saver.update(
            sampler, diagnostics, save_model, None, 0
        ),
    )

    return saver, outcome


def read_index(directory):
    with open(directory / "index.csv", newline="") as f:
        return list(csv.DictReader(f))


# -------------------------------------------------------- select_distinct


def test_select_distinct_prefers_the_best_and_skips_close_ones():
    params = np.array([[0.0], [0.05], [1.0], [1.02], [2.0]])
    quality = np.array([0.5, 0.1, 0.3, 0.2, 0.9])
    kept = select_distinct(params, quality, np.empty((0, 1)), [1.0], 0.1)
    # best first: 0.05 (0.1), then 1.02 (0.2), then 1.0 too close to 1.02,
    # then 0.0 too close to 0.05; 2.0 is far from both
    assert kept == [1, 3, 4]


def test_select_distinct_respects_models_already_saved_and_the_limit():
    params = np.array([[0.0], [1.0], [2.0], [3.0]])
    quality = np.array([0.1, 0.2, 0.3, 0.4])
    kept = select_distinct(params, quality, [[1.0]], [1.0], 0.5, limit=2)
    assert kept == [0, 2]  # 1.0 is saved already, the limit stops at two


def test_select_distinct_uses_the_scale_of_each_parameter():
    params = np.array([[0.0, 0.0], [0.0, 10.0]])
    quality = np.array([0.1, 0.2])
    # far apart in the second parameter, but that one varies by +-100:
    assert select_distinct(params, quality, [[0, 0]], [1, 100], 0.5) == []
    assert select_distinct(params, quality, [[0, 0]], [1, 1], 0.5) == [1]


def test_select_distinct_never_keeps_exact_repeats_even_for_zero_distance():
    params = np.array([[1.0], [1.0], [2.0]])
    quality = np.array([0.1, 0.2, 0.3])
    kept = select_distinct(params, quality, np.empty((0, 1)), [1.0], 0.0)
    assert kept == [0, 2]


# --------------------------------------------------------------- settings


def test_convergence_criteria_validation():
    with pytest.raises(ValueError):
        ConvergenceCriteria(check_every=0)
    with pytest.raises(ValueError):
        ConvergenceCriteria(acc_range=(0.7, 0.1))


def test_save_fits_validation(tmp_path):
    with pytest.raises(ValueError):
        SaveFits(tmp_path, n_sigma=0)
    with pytest.raises(ValueError):
        SaveFits(tmp_path, min_distance=-1)


# ------------------------------------------------------ convergence checks


def test_a_well_behaved_chain_converges_and_stops():
    # (The tau estimate of a short chain fluctuates by a few per cent.)
    criteria = ConvergenceCriteria(
        check_every=200, n_eff_target=200, tau_rel_change=0.1
    )
    sampler, diagnostics, history, autocorr, stopped = run_sampler(
        gaussian, [CENTER], start_positions(20), 6000, criteria
    )
    assert diagnostics["converged"]
    assert all(diagnostics["checks"].values())
    assert sampler.iteration < 6000  # stopped early
    assert len(history) == len(autocorr) == sampler.iteration // 200
    assert not stopped  # not stopped by a callback


def test_a_short_chain_is_not_converged_and_says_why():
    _, diagnostics, history, _, _ = run_sampler(
        gaussian,
        [CENTER],
        start_positions(),
        100,
        ConvergenceCriteria(check_every=50),
    )
    assert not diagnostics["converged"]
    assert not diagnostics["checks"]["length"]
    assert set(diagnostics["checks"]) == {
        "length",
        "tau stable",
        "ESS",
        "acceptance",
        "stationary",
    }
    assert history.shape == (2, 5)


class StubSampler:
    """Just enough of an emcee sampler to feed the convergence checks."""

    def __init__(self, log_prob, tau=5.0, acceptance=0.4):
        self.log_prob = log_prob
        self.iteration, self.nwalkers = log_prob.shape
        self.tau = tau
        self.acceptance_fraction = np.full(self.nwalkers, acceptance)

    def get_autocorr_time(self, tol=0):
        return np.array([self.tau, 0.5 * self.tau])

    def get_log_prob(self):
        return self.log_prob


def stationary_noise(steps=400, walkers=10, seed=0):
    return np.random.RandomState(seed).normal(size=(steps, walkers))


def test_diagnose_passes_a_healthy_chain():
    sampler = StubSampler(stationary_noise())
    criteria = ConvergenceCriteria(n_eff_target=100)
    result = diagnose(sampler, old_tau_max=5.0, criteria=criteria)
    assert result["checks"] == dict.fromkeys(result["checks"], True)
    assert result["converged"]
    assert result["burn"] == 10  # twice the autocorrelation time
    assert result["ess"] == pytest.approx(10 * (400 - 10) / 5.0)


@pytest.mark.parametrize(
    "check, change",
    [
        ("length", dict(tau=20.0)),  # 400 steps < 50 * 20
        ("tau stable", dict(old_tau_max=8.0)),  # tau moved by 60%
        ("ESS", dict(n_eff_target=1e6)),
        ("acceptance", dict(acceptance=0.02)),
        ("stationary", dict(drift=True)),
    ],
)
def test_each_convergence_check_can_fail_on_its_own(check, change):
    log_prob = stationary_noise()
    if change.get("drift"):
        # a trend of 20 sigma over the run:
        log_prob = log_prob + np.linspace(0, 20, 400)[:, None]

    sampler = StubSampler(
        log_prob,
        tau=change.get("tau", 5.0),
        acceptance=change.get("acceptance", 0.4),
    )
    criteria = ConvergenceCriteria(
        n_eff_target=change.get("n_eff_target", 100.0)
    )
    result = diagnose(sampler, change.get("old_tau_max", 5.0), criteria)

    assert result["checks"][check] is False
    others = {k: v for k, v in result["checks"].items() if k != check}
    # (a longer tau also lowers the ESS, and a trend also lengthens... only
    # the check under test has to fail, the rest is allowed to follow)
    assert not result["converged"]
    if check in ("tau stable", "acceptance", "stationary"):
        assert all(others.values())


def test_diagnose_without_a_finite_tau_never_converges():
    sampler = StubSampler(stationary_noise(), tau=np.inf)
    result = diagnose(sampler, 5.0, ConvergenceCriteria())
    assert not result["converged"]
    assert result["ess"] == 0.0 and result["shift"] == np.inf


# ------------------------------------------------------ saving (synthetic)


def test_fits_are_saved_during_the_chain_and_are_distinct(tmp_path):
    saver, _ = run_synthetic(tmp_path / "fits")
    rows = read_index(tmp_path / "fits")
    params = np.array([[float(r["p0"]), float(r["p1"])] for r in rows])
    steps = [int(r["step"]) for r in rows]

    assert saver.n_saved == len(rows) > 5
    assert all(float(r["worst_sigma"]) < 1.5 for r in rows)  # all fit
    assert min(steps) > 100  # nothing from the burn-in
    assert steps == sorted(steps)

    # Distinct: the posterior is a unit Gaussian, so a distance of 0.5 sigma
    # is about 0.5 in these units (the scale is the chain's own spread).
    distances = np.sqrt(
        np.mean((params[:, None, :] - params[None, :, :]) ** 2, axis=-1)
    )
    np.fill_diagonal(distances, np.inf)
    assert distances.min() > 0.25


def test_fits_are_saved_at_several_checks(tmp_path):
    run_synthetic(tmp_path / "fits")
    steps = {int(r["step"]) for r in read_index(tmp_path / "fits")}
    assert len(steps) >= 2  # models arrive while the chain is running


def test_larger_min_distance_saves_fewer_models(tmp_path):
    loose, _ = run_synthetic(tmp_path / "a", min_distance=0.0)
    medium, _ = run_synthetic(tmp_path / "b", min_distance=0.5)
    strict, _ = run_synthetic(tmp_path / "c", min_distance=2.0)
    assert strict.n_saved < medium.n_saved < loose.n_saved
    assert strict.n_saved >= 1


def test_models_that_do_not_fit_are_not_saved(tmp_path):
    saver, _ = run_synthetic(tmp_path / "fits", n_sigma=1e-6)
    assert saver.n_saved == 0
    assert not (tmp_path / "fits" / "index.csv").exists()


def test_max_models_stops_the_run(tmp_path):
    saver, (sampler, _, _, _, stopped) = run_synthetic(
        tmp_path / "fits", min_distance=0.0, max_models=3
    )
    assert saver.n_saved == 3
    assert stopped
    assert sampler.iteration < 400


def test_models_already_on_disk_count_and_are_not_repeated(tmp_path):
    directory = tmp_path / "fits"
    run_synthetic(directory, min_distance=0.0, max_models=3)
    before = read_index(directory)
    assert len(before) == 3

    # The same chain again finds the same fits, but the quota is full:
    again, _ = run_synthetic(directory, min_distance=0.0, max_models=3)
    assert read_index(directory) == before
    assert again.n_saved == 3

    # With a larger quota only new models are added; numbering goes on:
    run_synthetic(directory, min_distance=0.0, max_models=6)
    rows = read_index(directory)
    assert len(rows) == 6
    assert len({r["name"] for r in rows}) == 6
    assert len({(r["p0"], r["p1"]) for r in rows}) == 6


def test_models_that_fail_to_save_are_not_counted(tmp_path):
    saver = FitSaver(SaveFits(tmp_path, burn_in=0, min_distance=0.0), 2)
    run_sampler(
        gaussian,
        [CENTER],
        start_positions(),
        100,
        ConvergenceCriteria(check_every=100),
        callback=lambda s, d: saver.update(s, d, lambda job: None, None, 0),
    )
    assert saver.n_saved == 0  # every re-evaluation "failed"


# ---------------------------------------------------- the real ToF model


def make_tof(**options):
    reference = jupiter_polytrope(4, 16)
    target = reference.Js[1:3]
    X = ClassToF.ToF(
        N=16,
        order=4,
        R_ref=R_JUPITER,
        baro_param_init=[1.0],
        target_Js=target,
        sigma_Js=0.05 * np.abs(target),
        dJ_tol=1e-4,
        drot_tol=1e-4,
        drho_tol=1e-4,
        **options,
    )
    X.set_barotrope(_barotrope_K)
    X.set_check_param(_is_not_positive)
    return X


FAST = ConvergenceCriteria(check_every=20)


@pytest.fixture(scope="module")
def real_run(tmp_path_factory):
    """One small run on the real model that saves its fits."""

    directory = tmp_path_factory.mktemp("fits")
    # The Js of this model hardly depend on K, so every sample fits.
    save = SaveFits(directory, n_sigma=3.0, min_distance=0.0, burn_in=20)
    X = make_tof()
    result = X.run_mcmc(
        "baro", 2, 60, seed=3, convergence=FAST, save_fits=save
    )
    return X, directory, result


def test_run_mcmc_returns_a_complete_result(real_run):
    X, _, result = real_run
    assert isinstance(result, MCMCResult)
    assert result.chain.shape == (60, 2, 1)
    assert result.log_prob.shape == (60, 2)
    assert result.blobs.shape == (60, 2)
    np.testing.assert_array_equal(result.state, result.chain[-1])
    assert len(result.autocorr) == 3  # checks at steps 20, 40 and 60
    assert result.history.shape == (3, 5)
    assert result.failed_checks == [
        name for name, ok in result.diagnostics["checks"].items() if not ok
    ]


def test_blobs_are_the_worst_deviation_in_sigma(real_run):
    X, _, result = real_run
    param = result.chain[-1, 0]
    costJs = X.baro_cost_function(param, return_sum=False)
    assert result.blobs[-1, 0] == pytest.approx(np.sqrt(np.max(-costJs)))


def test_real_fits_are_saved_as_files_with_an_index(real_run):
    X, directory, result = real_run
    rows = read_index(directory)

    assert result.n_fits_saved == len(rows) > 0
    assert result.save_directory == str(directory)
    assert len({r["p0"] for r in rows}) == len(rows)  # no repeats

    for row in rows:
        index = row["name"].split("_")[-1]
        data = np.loadtxt(directory / f"model_SI_{index}.txt")
        assert data.shape == (16, 3)  # radii, densities, pressures
        assert np.all(np.diff(data[:, 0]) < 0)  # from the surface inwards
        assert np.all(np.diff(data[:, 1]) >= 0)  # density grows inwards
        assert np.all(np.diff(data[:, 2]) >= 0)  # pressure grows inwards
        assert float(row["worst_sigma"]) < 3.0
        assert int(row["step"]) > 20  # nothing from the burn-in


def test_saved_real_models_reproduce_their_fit(real_run):
    X, directory, _ = real_run
    for row in read_index(directory)[:3]:
        costJs = X.baro_cost_function(
            np.array([float(row["p0"])]), return_sum=False
        )
        assert np.sqrt(np.max(-costJs)) == pytest.approx(
            float(row["worst_sigma"]), rel=1e-3, abs=1e-6
        )


def test_result_can_be_saved(real_run, tmp_path):
    _, _, result = real_run
    result.save(tmp_path / "chain.npz")
    np.testing.assert_array_equal(
        np.load(tmp_path / "chain.npz")["chain"], result.chain
    )


def test_cgs_profile_units(tmp_path):
    X = jupiter_polytrope(4, 16)
    _write_profile(X, str(tmp_path / "m"), True, 7)
    data = np.loadtxt(tmp_path / "m_CGS_7.txt")
    np.testing.assert_allclose(data[:, 0], X.li * 100)
    np.testing.assert_allclose(data[:, 1], np.log10(X.rhoi) - 3)
    np.testing.assert_allclose(data[:, 2], np.log10(X.Pi) + 1)


def test_seeded_runs_are_reproducible_and_the_old_interface_agrees():
    a = make_tof().run_mcmc("baro", 2, 10, seed=7)
    b = make_tof().run_mcmc("baro", 2, 10, seed=8)
    state, autocorr = make_tof().run_baro_MC(2, 10, seed=7)  # old interface

    np.testing.assert_array_equal(state, a.state)
    np.testing.assert_array_equal(autocorr, a.autocorr)
    assert not np.array_equal(a.chain, b.chain)


def test_saving_works_with_parallel_walkers(tmp_path):
    result = make_tof().run_mcmc(
        "baro",
        2,
        20,
        Ncores=2,
        parallelize=True,
        seed=3,
        convergence=ConvergenceCriteria(check_every=10),
        save_fits=SaveFits(tmp_path, n_sigma=3.0, min_distance=0.0, burn_in=0),
    )
    assert result.n_fits_saved > 0
    assert len(read_index(tmp_path)) == result.n_fits_saved


def test_progress_is_reported_at_verbosity_one(capsys):
    X = make_tof(verbosity=1)
    X.run_mcmc("baro", 2, 20, seed=1, convergence=ConvergenceCriteria(10))
    out = capsys.readouterr().out
    assert "[MCMC] step 10" in out and "[MCMC] step 20" in out
    assert "Generated initial conditions for walker number: 1" in out


# ------------------------------------------------------------- robustness


def test_impossible_starting_points_raise_instead_of_hanging():
    X = make_tof()
    X.set_check_param(lambda param: True)
    with pytest.raises(RuntimeError, match="No physical starting point"):
        X.run_mcmc("baro", 2, 20, max_start_attempts=5)


def test_too_few_walkers():
    with pytest.raises(ValueError, match="twice as many walkers"):
        make_tof().run_mcmc("baro", 1, 20)


def test_missing_settings_are_explained():
    X = ClassToF.ToF(N=16, order=4, R_ref=R_JUPITER)
    X.set_barotrope(_barotrope_K)
    with pytest.raises(ValueError, match="baro_param_init"):
        X.run_mcmc("baro", 2, 20)

    X = ClassToF.ToF(N=16, order=4, R_ref=R_JUPITER, baro_param_init=[1.0])
    with pytest.raises(ValueError, match="set_check_param"):
        X.run_mcmc("baro", 2, 20)


def test_unknown_model_name():
    with pytest.raises(KeyError, match="baro"):
        make_tof().run_mcmc("nonsense", 2, 20)


def test_unpicklable_functions_are_explained_before_starting_a_pool():
    X = make_tof()
    X.set_barotrope(lambda P, K: np.sqrt(P / K))
    with pytest.raises(TypeError, match="cannot be pickled"):
        X.run_mcmc("baro", 2, 20, parallelize=True)


def test_importing_pytof_does_not_change_the_multiprocessing_start_method():
    import subprocess
    import sys

    code = (
        "import multiprocessing as m; before = m.get_start_method(); "
        "import PyToF; assert m.get_start_method() == before, 'changed'"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


# -------------------------------------------------------------- checkpoint


def test_chain_can_be_resumed_from_a_checkpoint(tmp_path):
    pytest.importorskip("h5py")
    path = tmp_path / "chain.h5"

    first = make_tof().run_mcmc(
        "baro", 2, 20, seed=5, convergence=FAST, checkpoint=path
    )
    assert first.chain.shape[0] == 20

    resumed = make_tof().run_mcmc(
        "baro", 2, 40, seed=99, convergence=FAST, checkpoint=path
    )
    assert resumed.chain.shape[0] == 40
    np.testing.assert_array_equal(resumed.chain[:20], first.chain)

    # ... and it is the chain an uninterrupted run would have produced:
    straight = make_tof().run_mcmc("baro", 2, 40, seed=5, convergence=FAST)
    np.testing.assert_allclose(resumed.chain, straight.chain, rtol=1e-12)


def test_checkpoint_of_a_different_shape_is_not_resumed(tmp_path):
    pytest.importorskip("h5py")
    path = tmp_path / "chain.h5"
    make_tof().run_mcmc("baro", 2, 10, seed=5, checkpoint=path)
    other = make_tof().run_mcmc("baro", 4, 10, seed=5, checkpoint=path)
    assert other.chain.shape == (10, 4, 1)  # started afresh


def test_missing_h5py_is_explained(tmp_path, monkeypatch):
    import builtins

    real_import = builtins.__import__

    def refuse_h5py(name, *args, **kwargs):
        if name == "h5py":
            raise ImportError("no h5py")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", refuse_h5py)
    with pytest.raises(ImportError, match="pip install h5py"):
        make_tof().run_mcmc("baro", 2, 10, checkpoint=tmp_path / "x.h5")
