########################################################
# Author of this version: Luca Morf - luca.morf@uzh.ch #
########################################################


import functools
import multiprocessing
import os
import pickle
from collections.abc import Callable
from typing import NamedTuple

import numpy as np

from PyToF.messages import paint, say
from PyToF.sampling import (
    ConvergenceCriteria,
    FitSaver,
    MCMCResult,
    open_checkpoint,
    run_sampler,
)


def _check_phys(tof):
    """
    This function returns True if the current solutions
    (tof.rhoi, tof.Pi, tof.Js) are unphysical or consisting
    of NaN values. Returns False if everything is ok.
    """

    # Density inversions are considered to be unphysical:
    if np.any(np.diff(tof.rhoi) < 0):
        say(
            tof.opts["verbosity"],
            2,
            "Physicality warning: density inversion!",
            kind="warn",
        )

        return True

    # There should be no NaN values:
    elif (
        np.isnan(tof.rhoi).any()
        or np.isnan(tof.Pi).any()
        or np.isnan(tof.Js).any()
    ):
        say(
            tof.opts["verbosity"],
            2,
            "Physicality warning: NaN values!",
            kind="warn",
        )

        return True

    # There should be no negative density or pressure values:
    elif (np.min(tof.rhoi) < 0) or (np.min(tof.Pi) < 0):
        say(
            tof.opts["verbosity"],
            2,
            "Physicality warning: negative density or pressure values!",
            kind="warn",
        )

        return True

    # There should be no density or pressure values above the given maxima:
    elif (np.max(tof.rhoi) > tof.opts["rho_max"]) or (
        np.max(tof.Pi) > tof.opts["P_max"]
    ):
        say(
            tof.opts["verbosity"],
            2,
            "Physicality warning: density or pressure values above the "
            "given maxima!",
            kind="warn",
        )

        return True

    # We passed all checks, we can return False:
    else:
        return False


def set_check_param(self, fun):
    """
    This function allows the user to set the function that checks the
    parameters for the barotrope or density function. The function fun should
    have the form fun(param) and should return True if the parameters are
    nonsense (i.e. out of bounds).

    fun: Function handle for the density model.
    """

    # Set function:
    self.check_param = fun


class _Model(NamedTuple):
    """
    What differs between the barotrope model and the density model.

    param_init:     Option holding the starting values of the parameters.
    param_calc:     ToF attribute holding the current parameters.
    relax:          ToF method relaxing the model to hydrostatic equilibrium.
    max_iter:       Option limiting the iterations of that method.
    prepare:        Function(tof, param) run after reset(), before relaxing.
    """

    param_init: str
    param_calc: str
    relax: str
    max_iter: str
    prepare: Callable


def _prepare_barotrope(self, param):
    # The densities follow from the barotrope while relaxing.
    pass


def _prepare_density(self, param):
    # Get densities according to self.density_function():
    self.rhoi = self.density_function(
        self.li, self.opts["M_phys"], param=param
    )


_BAROTROPE = _Model(
    "baro_param_init",
    "baro_param_calc",
    "relax_to_barotrope",
    "max_iter_barotrope",
    _prepare_barotrope,
)

_DENSITY = _Model(
    "dens_param_init",
    "dens_param_calc",
    "relax_to_density",
    "max_iter_density",
    _prepare_density,
)


def _reset_and_relax(self, model, param):
    """
    Resets the ToF object, applies the model parameters and relaxes. Returns
    the number of iterations; raises ValueError for unphysical parameters.
    """

    # Reset to initial conditions:
    self.reset()

    model.prepare(self, param)

    # Update the internal parameters:
    setattr(self, model.param_calc, param)

    return getattr(self, model.relax)()


def _evaluate(self, model, param):
    """
    Relaxes the model with these parameters and returns the cost of every J,
    -((J - target) / sigma)^2, or None if the parameters are out of bounds or
    the result is unphysical or did not converge.
    """

    if self.check_param(param):
        return None

    try:
        it = _reset_and_relax(self, model, param)

    except ValueError:
        return None

    if _check_phys(self) or it == self.opts[model.max_iter]:
        return None

    target = np.asarray(self.opts["target_Js"])
    sigma = np.asarray(self.opts["sigma_Js"])

    return -((self.Js[1 : len(target) + 1] - target) ** 2) / sigma**2


def _cost(self, model, param, return_sum):
    """Shared implementation of baro_cost_function and dens_cost_function."""

    costJs = _evaluate(self, model, param)

    if costJs is None:
        if return_sum:
            return -np.inf

        return [-np.inf] * len(self.opts["target_Js"])

    if return_sum:
        return np.sum(costJs)

    else:
        return costJs


def baro_cost_function(self, param, return_sum=True):
    """
    Calls relax_to_barotrope() and returns values specifying how far off the
    calculated Js are from opts['target_Js'] given opts['sigma_Js'].

    param: Array of parameters for the barotrope function.
    return_sum: If True, returns the sum of the costs for all Js. If False,
                returns an array of costs for each J separately.
    """

    return _cost(self, _BAROTROPE, param, return_sum)


def dens_cost_function(self, param, return_sum=True):
    """
    Calls relax_to_barotrope() and returns values specifying how far off the
    calculated Js are from opts['target_Js'] given opts['sigma_Js'].

    param: Array of parameters for the density function.
    return_sum: If True, returns the sum of the costs for all Js. If False,
                returns an array of costs for each J separately.
    """

    return _cost(self, _DENSITY, param, return_sum)


def _log_prob(self, model, param):
    """
    What emcee samples: the log probability (the summed cost) and, as a blob,
    the worst deviation of any J from its target in sigma (inf if the model
    is out of bounds or unphysical). A model fits if the blob is small.
    """

    costJs = _evaluate(self, model, param)

    if costJs is None:
        return -np.inf, np.inf

    return np.sum(costJs), float(np.sqrt(np.max(-costJs)))


def baro_log_prob(param, tof):
    """
    Changes order of arguments for the emcee algorithm.
    """

    return _log_prob(tof, _BAROTROPE, param)


def dens_log_prob(param, tof):
    """
    Changes order of arguments for the emcee algorithm.
    """

    return _log_prob(tof, _DENSITY, param)


def _initial_parameters(self, model, nwalkers, max_attempts):
    """
    Draws random starting parameters (as fractions of the initial values)
    until nwalkers of them are accepted by check_param() and yield a physical
    solution. Gives up with an error after max_attempts draws for one walker.
    """

    initial = self.opts[model.param_init]
    param_0 = np.zeros((nwalkers, len(initial)))
    rejected = dict(out_of_bounds=0, unphysical=0)

    for i in range(nwalkers):
        for _ in range(max_attempts):
            candidate = np.random.rand(1, len(initial)) * initial

            if self.check_param(candidate[0]):
                rejected["out_of_bounds"] += 1
                continue

            try:
                it = _reset_and_relax(self, model, candidate[0])

            except ValueError:
                rejected["unphysical"] += 1
                continue

            # Only proceed if the candidate also yields a physical solution:
            if _check_phys(self) or it >= self.opts[model.max_iter]:
                rejected["unphysical"] += 1
                continue

            param_0[i, :] = candidate[0]
            say(
                self.opts["verbosity"],
                1,
                "Generated initial conditions for walker number: {i}",
                i=i,
            )
            break

        else:
            raise RuntimeError(
                f"No physical starting point found for walker {i} in "
                f"{max_attempts} attempts ({rejected['out_of_bounds']} draws "
                f"rejected by check_param, {rejected['unphysical']} gave "
                "unphysical or unconverged models). Starting values are "
                f"drawn uniformly between 0 and {model.param_init}: check "
                "that range and your barotrope/density function, or raise "
                "max_start_attempts."
            )

    return param_0


def _write_profile(self, path_stem, log_CGS_units, index):
    """
    Writes radii, densities and pressures of the current model to
    <path_stem>_SI_<index>.txt, or <path_stem>_CGS_<index>.txt with radii in
    cm and log10 of the density in g/cm^3 and of the pressure in bar.
    """

    if log_CGS_units:
        columns = [
            self.li * 100,
            np.log10(self.rhoi) - 3,
            np.log10(self.Pi) + 1,
        ]
        unit = "CGS"

    else:
        columns = [self.li, self.rhoi, self.Pi]
        unit = "SI"

    np.savetxt(f"{path_stem}_{unit}_{index}.txt", np.transpose(columns))


def _save_fit(self, model, settings, job):
    """
    Re-evaluates one sample of the chain and writes the model to disk if it
    still fits. Returns its worst deviation in sigma, or None if not saved.
    """

    param, index = job
    costJs = _evaluate(self, model, param)

    if costJs is None:
        return None

    worst = float(np.sqrt(np.max(-costJs)))
    if worst >= settings.n_sigma:
        return None

    stem = os.path.join(settings.directory, settings.file_name)
    _write_profile(self, stem, settings.log_CGS_units, index)

    return worst


def _make_pool(self, Ncores):
    """A process pool; explains the usual reason why one cannot be made."""

    try:
        pickle.dumps(self)

    except (pickle.PicklingError, AttributeError, TypeError) as error:
        raise TypeError(
            "parallelize=True sends the ToF object to other processes, but "
            "something it holds cannot be pickled, most likely a barotrope, "
            "density, atmosphere or check_param function defined inside "
            "another function (or a lambda). Define them at the top level "
            "of a module or notebook cell, or use parallelize=False. "
            f"({error})"
        ) from error

    methods = multiprocessing.get_all_start_methods()
    context = multiprocessing.get_context(
        "fork" if "fork" in methods else None
    )

    return context.Pool(processes=Ncores)


def run_mcmc(
    self,
    what_model,
    nwalkers,
    steps,
    Ncores=8,
    parallelize=False,
    *,
    convergence=None,
    save_fits=None,
    checkpoint=None,
    seed=None,
    max_start_attempts=1000,
):
    """
    Explores the parameters of the barotrope (what_model="baro") or the
    density function (what_model="dens") with the emcee algorithm: nwalkers
    walkers take up to `steps` steps, constrained by check_phys() and
    check_param(). Returns an MCMCResult.

    The chain is examined every convergence.check_every steps and stops early
    once it has converged, see ConvergenceCriteria.

    convergence     ConvergenceCriteria, or None for the defaults.
    save_fits       SaveFits: while the chain runs, write the models that fit
                    the data and differ enough from each other to disk, so
                    that a run that does not converge still delivers models.
    checkpoint      Path of an HDF5 file (needs h5py) the chain is written to
                    after every step. If the file holds a chain of this shape
                    the run continues it (up to `steps` in total).
    seed            Seeds NumPy's random numbers for reproducible runs.
    max_start_attempts
                    Random draws allowed per walker when looking for a
                    physical starting point before giving up.
    """

    models = dict(
        baro=(_BAROTROPE, baro_log_prob), dens=(_DENSITY, dens_log_prob)
    )
    if what_model not in models:
        raise KeyError("Invalid keyword for what_model! Use 'baro' or 'dens'.")
    model, log_prob = models[what_model]

    if self.opts[model.param_init] is None:
        raise ValueError(
            f"The option {model.param_init} (the starting values of the "
            "parameters) has to be set."
        )
    if getattr(self, "check_param", None) is None:
        raise ValueError("Call set_check_param() first.")

    ndim = len(self.opts[model.param_init])
    if nwalkers < 2 * ndim:
        raise ValueError(
            f"emcee needs at least twice as many walkers as parameters: "
            f"nwalkers >= {2 * ndim} for {ndim} parameters, not {nwalkers}."
        )
    if steps < 1:
        raise ValueError("steps must be at least 1.")

    criteria = (
        convergence if convergence is not None else ConvergenceCriteria()
    )
    verbosity = self.opts["verbosity"]

    if seed is not None:
        np.random.seed(seed)

    backend, resumed = open_checkpoint(checkpoint, nwalkers, ndim, verbosity)
    if resumed:
        param_0 = np.empty((nwalkers, ndim))  # the chain on disk is used
    else:
        param_0 = _initial_parameters(
            self, model, nwalkers, max_start_attempts
        )

        # Average cost of the starting positions:
        average = sum(_cost(self, model, p, True) for p in param_0) / nwalkers
        say(
            verbosity,
            1,
            "Average cost of a walker with the generated initial "
            "parameters: {average:.2e}",
            average=average,
        )

    saver = FitSaver(save_fits, ndim) if save_fits is not None else None
    save_model = (
        functools.partial(_save_fit, self, model, save_fits)
        if save_fits is not None
        else None
    )

    pool = _make_pool(self, Ncores) if parallelize else None
    try:
        callback = None
        if saver is not None:

            def callback(sampler, diagnostics):
                return saver.update(
                    sampler, diagnostics, save_model, pool, verbosity
                )

        sampler, diagnostics, history, autocorr, stopped_early = run_sampler(
            log_prob,
            [self],
            param_0,
            steps,
            criteria,
            pool=pool,
            backend=backend,
            resumed=resumed,
            callback=callback,
            verbosity=verbosity,
        )

    finally:
        if pool is not None:
            pool.close()
            pool.join()

    failed = [name for name, ok in diagnostics["checks"].items() if not ok]
    say(
        verbosity,
        1,
        "Average cost of a walker after the Markov Chain Monte Carlo walk: "
        "{average:.2e}. The chain {status}.",
        average=np.mean(sampler.get_last_sample().log_prob),
        status="converged"
        if diagnostics["converged"]
        else "did not converge (not yet: " + ", ".join(failed) + ")",
    )

    return MCMCResult(
        state=sampler.get_last_sample().coords,
        autocorr=autocorr,
        converged=diagnostics["converged"],
        failed_checks=failed,
        diagnostics=diagnostics,
        history=history,
        chain=sampler.get_chain(),
        log_prob=sampler.get_log_prob(),
        blobs=sampler.get_blobs(),
        n_fits_saved=saver.n_saved if saver is not None else 0,
        stopped_early=stopped_early,
        save_directory=save_fits.directory if save_fits is not None else None,
    )


def run_baro_MC(self, nwalkers, steps, Ncores=8, parallelize=False, **kwargs):
    """
    Uses the emcee algorithm for the given amount of steps and creates nwalkers
    many parameter candidates (i.e. walkers) that will explore the allowed
    parameter space (constrained by check_phys() and check_param()).

    Returns the final positions of the walkers and the autocorrelation times;
    see run_mcmc() for the options and for a richer result.
    """

    result = run_mcmc(
        self, "baro", nwalkers, steps, Ncores, parallelize, **kwargs
    )

    return result.state, result.autocorr


def run_dens_MC(self, nwalkers, steps, Ncores=8, parallelize=False, **kwargs):
    """
    Uses the emcee algorithm for the given amount of steps and creates nwalkers
    many parameter candidates (i.e. walkers) that will explore the allowed
    parameter space (constrained by check_phys() and check_param()).

    Returns the final positions of the walkers and the autocorrelation times;
    see run_mcmc() for the options and for a richer result.
    """

    result = run_mcmc(
        self, "dens", nwalkers, steps, Ncores, parallelize, **kwargs
    )

    return result.state, result.autocorr


def classify_and_save_state(
    self,
    state,
    what_model,
    what_save="none",
    log_CGS_units=False,
    path_name=None,
    file_name="walker",
):
    """
    Classifies the nwalkers many parameter candidates according to their
    physicality, i.e. by checking if they agree with the physical data within 1
    sigma. Also allows for saving .txt files with the radii, densities and
    pressures within the planet generated by the walker parameters.
    """

    if path_name is None:
        path_name = os.getcwd()

    # Storage arrays:
    matches_observed_Js = []
    rho_maxs = []

    # Loop through walkers:
    for i in range(len(state[:, 0])):
        # Calculate the physical cost:
        if what_model == "baro":
            costJs = baro_cost_function(self, state[i, :], return_sum=False)

        elif what_model == "dens":
            costJs = dens_cost_function(self, state[i, :], return_sum=False)

        else:
            raise KeyError(
                "Invalid keyword for what_model! Use 'baro' or 'dens'."
            )

        # Check if there is agreement with physical data within 1 sigma:
        if (np.abs(costJs) < 1e0).all():
            kind = "good"
            matches_observed_Js.append(True)

        else:
            kind = "warn"
            matches_observed_Js.append(False)

        # Save .txt file if wanted by the user:
        stem = path_name + "/" + file_name
        unit = "CGS" if log_CGS_units else "SI"
        exists = os.path.exists(f"{stem}_{unit}_{i}.txt")

        if what_save == "all":
            if not exists:
                _write_profile(self, stem, log_CGS_units, i)

        elif what_save == "good" and (np.abs(costJs) < 1e0).all():
            _write_profile(self, stem, log_CGS_units, i)

        elif what_save not in ("good", "none"):
            raise KeyError(
                "Invalid keyword for what_save! Use 'all' or 'good' or 'none'."
            )

        # Update storage arrays:
        rho_maxs.append(np.max(self.rhoi))

        # Verbosity output:
        verbosity = self.opts["verbosity"]
        if verbosity > 0:
            header = " " * 9
            mine = ""
            target = ""

            for j, cost_j in enumerate(costJs):
                label = f"     J{2 * (j + 1)}     "
                header += paint(label, "good" if abs(cost_j) < 1e0 else "warn")
                mine += f"{self.Js[j + 1]:.5e} "
                target += f"{self.opts['target_Js'][j]:.5e} "

            say(
                verbosity,
                1,
                "\nWalker #{i} yields a total cost of {total_cost:.2e}.",
                kind=kind,
                i=i,
                total_cost=np.sum(costJs),
            )
            print(header)
            say(verbosity, 1, "My code: {values}", values=mine)
            say(verbosity, 1, "Target:  {values}\n", values=target)

    return np.array(matches_observed_Js), np.array(rho_maxs)
