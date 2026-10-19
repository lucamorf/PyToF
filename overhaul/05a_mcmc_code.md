# 05a — MCMC code (verbatim)

## `PyToF/sampling.py` (new, knows nothing about planets)
```python
"""
Model-independent MCMC machinery: convergence checks, the sampling loop,
a resumable chain on disk, and saving of distinct models that fit.

Nothing in here knows about planets. The model enters through the function
``log_prob(param, ...) -> (log probability, blob)``, where the blob is the
*worst deviation in sigma* of the model from the data (``inf`` if the model
could not be built). A model counts as fitting if its blob is below
``SaveFits.n_sigma``.

The convergence checks and the idea to save good models while the chain is
still running are taken from the MCMC of the Uranus/Neptune interior models
(newUranusNeptune/mcmc.py).
"""

import csv
import os
from dataclasses import dataclass
from pathlib import Path

import emcee
import numpy as np

from PyToF.messages import say

# ---------------------------------------------------------------- settings


@dataclass
class ConvergenceCriteria:
    """
    The chain counts as converged if all five checks pass (they are evaluated
    every ``check_every`` steps):

    length          the chain is longer than ``tau_factor`` autocorrelation
                    times (emcee recommends 50)
    tau stable      the longest autocorrelation time changed by less than
                    ``tau_rel_change`` (relative) since the previous check
    ESS             at least ``n_eff_target`` effective independent samples
                    after the burn-in (twice the autocorrelation time)
    acceptance      the mean acceptance fraction is inside ``acc_range``
                    (with few parameters the stretch move accepts 0.7-0.8)
    stationary      the median log probability of the third and fourth
                    quarter of the chain agree within ``stationarity_sigma``
                    standard errors, i.e. the chain no longer drifts
    """

    check_every: int = 100
    tau_factor: float = 50.0
    tau_rel_change: float = 0.01
    n_eff_target: float = 1000.0
    acc_range: tuple = (0.1, 0.9)
    stationarity_sigma: float = 4.0

    def __post_init__(self):
        if self.check_every < 1:
            raise ValueError("check_every must be at least 1.")
        low, high = self.acc_range
        if not 0 <= low < high <= 1:
            raise ValueError("acc_range must satisfy 0 <= low < high <= 1.")


@dataclass
class SaveFits:
    """
    Save models that fit the data *while the chain is running*, so that a run
    that never converges still delivers something.

    Every ``check_every`` steps, the samples drawn since the last check that
    fit (every J within ``n_sigma`` of its target) are looked at, best fit
    first. A sample is saved only if it is different enough from all models
    saved so far: the root-mean-square distance of the parameters, each in
    units of its standard deviation in the chain, must be at least
    ``min_distance``.

    directory       Where the models and ``index.csv`` are written.
    n_sigma         A model fits if each J is within this many sigma.
    min_distance    Minimum distance between saved models, see above. 0
                    saves every fitting sample except exact repeats.
    max_models      Stop the run once this many models are saved (counting
                    those already in ``directory``). None: no limit.
    burn_in         Steps at the start of the chain that are never saved from.
    log_CGS_units   Write radii in cm and log10 of density and pressure in
                    g/cm^3 and bar (as classify_and_save_state does).
    file_name       Prefix of the model files.
    """

    directory: str
    n_sigma: float = 1.0
    min_distance: float = 0.1
    max_models: int = None
    burn_in: int = 100
    log_CGS_units: bool = False
    file_name: str = "model"

    def __post_init__(self):
        if self.n_sigma <= 0:
            raise ValueError("n_sigma must be positive.")
        if self.min_distance < 0:
            raise ValueError("min_distance must not be negative.")
        self.directory = os.fspath(self.directory)


@dataclass
class MCMCResult:
    """
    Outcome of a run. ``state`` and ``autocorr`` are what run_baro_MC and
    run_dens_MC return; the rest says how the run went.

    state           Positions of the walkers after the last step
                    (nwalkers, nparameters).
    autocorr        Mean autocorrelation time at every multiple of
                    ``check_every`` steps.
    converged       Whether all convergence checks passed.
    failed_checks   Names of the checks that did not pass (empty if converged).
    diagnostics     The numbers of the last check, see :func:`diagnose`.
    history         Array with one row per check: step, longest
                    autocorrelation time, acceptance fraction, effective
                    sample size, drift of the log probability in sigma.
    chain, log_prob, blobs
                    The complete chain (steps, walkers, ...), its log
                    probabilities, and the worst deviation in sigma of every
                    sample.
    n_fits_saved    Models in ``save_fits.directory`` after the run.
    stopped_early   True if the run ended because ``max_models`` were saved.
    """

    state: np.ndarray
    autocorr: np.ndarray
    converged: bool
    failed_checks: list
    diagnostics: dict
    history: np.ndarray
    chain: np.ndarray
    log_prob: np.ndarray
    blobs: np.ndarray
    n_fits_saved: int = 0
    stopped_early: bool = False
    save_directory: str = None

    def save(self, path):
        """Write the chain and the diagnostics to a ``.npz`` file."""

        np.savez(
            path,
            chain=self.chain,
            log_prob=self.log_prob,
            blobs=self.blobs,
            state=self.state,
            autocorr=self.autocorr,
            history=self.history,
            tau=self.diagnostics["tau"],
            converged=self.converged,
            failed_checks=np.array(self.failed_checks),
        )


# ------------------------------------------------------------- diagnostics


def diagnose(sampler, old_tau_max, criteria):
    """The five convergence checks of the chain so far, with their numbers."""

    n = sampler.iteration
    tau = sampler.get_autocorr_time(tol=0)
    tau_max = np.max(tau)
    acc = np.mean(sampler.acceptance_fraction)

    finite = bool(np.isfinite(tau_max))
    burn = int(np.ceil(2 * tau_max)) if finite else n
    ess = sampler.nwalkers * max(n - burn, 0) / tau_max if finite else 0.0

    # Stationarity: ensemble-median log probability of the third and fourth
    # quarter of the chain, compared via the standard error of their means
    # (corrected for the autocorrelation):
    log_prob = np.median(sampler.get_log_prob(), axis=1)
    quarter = n // 4
    if quarter > 0 and finite:
        a = log_prob[2 * quarter : 3 * quarter]
        b = log_prob[3 * quarter : 4 * quarter]
        se = np.sqrt((a.var() + b.var()) / max(quarter / tau_max, 1.0))
        if se > 0:
            shift = abs(a.mean() - b.mean()) / se
        else:
            shift = 0.0 if a.mean() == b.mean() else np.inf
    else:
        shift = np.inf

    with np.errstate(all="ignore"):
        tau_change = abs(old_tau_max - tau_max) / tau_max

    checks = {
        "length": bool(n > criteria.tau_factor * tau_max),
        "tau stable": bool(tau_change < criteria.tau_rel_change),
        "ESS": bool(ess >= criteria.n_eff_target),
        "acceptance": bool(
            criteria.acc_range[0] <= acc <= criteria.acc_range[1]
        ),
        "stationary": bool(shift < criteria.stationarity_sigma),
    }

    return dict(
        iteration=n,
        tau=tau,
        tau_max=tau_max,
        acc=acc,
        ess=ess,
        burn=burn,
        shift=shift,
        checks=checks,
        converged=all(checks.values()),
    )


# --------------------------------------------------------- distinct models


def select_distinct(params, quality, saved, scale, min_distance, limit=None):
    """
    Choose which candidate models to keep.

    Candidates are visited best first (lowest ``quality``). One is kept if its
    root-mean-square distance, with every parameter divided by ``scale``, to
    all kept candidates and to all ``saved`` models is at least
    ``min_distance``. Exact repeats (distance 0) are never kept.

    Returns the indices of the kept candidates, best first.
    """

    params = np.atleast_2d(params)
    saved = np.asarray(saved, dtype=float).reshape(-1, params.shape[1])
    scale = np.asarray(scale, dtype=float)
    kept = []
    reference = saved / scale

    for i in np.argsort(quality, kind="stable"):
        if limit is not None and len(kept) >= limit:
            break

        point = params[i] / scale
        if len(reference):
            distance = np.sqrt(np.mean((reference - point) ** 2, axis=1))
            if np.min(distance) < max(min_distance, np.finfo(float).tiny):
                continue

        kept.append(int(i))
        reference = np.vstack([reference, point])

    return kept


class FitSaver:
    """
    Saves distinct fitting models after every convergence check, see
    :class:`SaveFits`. The models already in the directory (from an earlier,
    interrupted run) count and are respected when judging distinctness.
    """

    def __init__(self, settings, ndim):
        self.settings = settings
        self.ndim = ndim
        self.directory = Path(settings.directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.index_path = self.directory / "index.csv"
        self.next_step = 0
        self.counter = 0
        self.saved = np.empty((0, ndim))

        if self.index_path.exists():
            with open(self.index_path, newline="") as f:
                rows = list(csv.DictReader(f))
            self.counter = len(rows)
            self.saved = np.array(
                [[float(r[f"p{k}"]) for k in range(ndim)] for r in rows]
            ).reshape(-1, ndim)

    @property
    def n_saved(self):
        return len(self.saved)

    def full(self):
        limit = self.settings.max_models
        return limit is not None and self.n_saved >= limit

    def update(self, sampler, diagnostics, save_model, pool, verbosity):
        """
        Look at the samples since the last call, save the distinct fitting
        ones, and return True once ``max_models`` models are saved.

        save_model((param, k)) -> worst deviation in sigma, or None if the
        model could not be saved (e.g. it does not fit when re-evaluated).
        """

        settings = self.settings
        n_steps = sampler.iteration
        start = max(self.next_step, settings.burn_in)
        self.next_step = n_steps

        if start >= n_steps or self.full():
            return self.full()

        chain = sampler.get_chain(discard=start).reshape(-1, self.ndim)
        worst = sampler.get_blobs(discard=start).reshape(-1).astype(float)
        fits = np.where(worst < settings.n_sigma)[0]

        n_new = 0
        if len(fits):
            everything = sampler.get_chain(discard=settings.burn_in)
            scale = everything.reshape(-1, self.ndim).std(axis=0)
            scale[scale == 0] = 1.0

            limit = None
            if settings.max_models is not None:
                limit = settings.max_models - self.n_saved

            order = select_distinct(
                chain[fits],
                worst[fits],
                self.saved,
                scale,
                settings.min_distance,
                limit=limit,
            )

            mapper = pool.map if pool is not None else map
            jobs = [
                (chain[fits][i], self.counter + k) for k, i in enumerate(order)
            ]
            self.counter += len(jobs)
            results = list(mapper(save_model, jobs))

            rows = []
            for (param, k), sigma in zip(jobs, results, strict=True):
                if sigma is None:
                    continue
                rows.append(
                    [
                        f"{settings.file_name}_{k}",
                        n_steps,
                        f"{sigma:.6g}",
                        diagnostics["converged"],
                        f"{diagnostics['ess']:.6g}",
                        f"{diagnostics['tau_max']:.6g}",
                        *[repr(float(p)) for p in param],
                    ]
                )
                self.saved = np.vstack([self.saved, param])
            n_new = len(rows)
            self._append_rows(rows)

        say(
            verbosity,
            1,
            "[MCMC] step {step}: {n_fit} fitting samples, {n_new} new "
            "distinct models saved ({total} in {directory})",
            n_fit=len(fits),
            n_new=n_new,
            total=self.n_saved,
            directory=str(self.directory),
            step=n_steps,
        )

        return self.full()

    def _append_rows(self, rows):
        if not rows:
            return

        new_file = not self.index_path.exists()
        with open(self.index_path, "a", newline="") as f:
            writer = csv.writer(f)
            if new_file:
                writer.writerow(
                    [
                        "name",
                        "step",
                        "worst_sigma",
                        "converged",
                        "ess",
                        "tau_max",
                    ]
                    + [f"p{k}" for k in range(self.ndim)]
                )
            writer.writerows(rows)


# ------------------------------------------------------------ the sampling


def open_checkpoint(path, nwalkers, ndim, verbosity):
    """
    Open the chain on disk. Returns (backend, resumed): resumed is True if
    the file already holds a chain of exactly this shape, which is then
    continued.
    """

    if path is None:
        return None, False

    try:
        import h5py  # noqa: F401
    except ImportError:
        raise ImportError(
            "Writing the chain to disk (checkpoint=...) needs h5py: "
            "pip install h5py"
        ) from None

    backend = emcee.backends.HDFBackend(os.fspath(path))
    resumed = (
        backend.initialized
        and backend.iteration > 0
        and backend.shape == (nwalkers, ndim)
    )
    if resumed:
        say(
            verbosity,
            1,
            "Resuming the chain from {path} at step {step}",
            path=os.fspath(path),
            step=backend.iteration,
        )
    else:
        backend.reset(nwalkers, ndim)

    return backend, resumed


def run_sampler(
    log_prob,
    log_prob_args,
    param_0,
    steps,
    criteria,
    pool=None,
    progress=False,
    backend=None,
    resumed=False,
    callback=None,
    verbosity=0,
):
    """
    Run emcee in chunks of ``criteria.check_every`` steps until all
    convergence checks pass, ``steps`` is reached, or the callback returns
    True. Returns (sampler, diagnostics, history, autocorr).

    ``callback(sampler, diagnostics)`` is called after every chunk.
    """

    nwalkers, ndim = param_0.shape
    sampler = emcee.EnsembleSampler(
        nwalkers,
        ndim,
        log_prob,
        args=log_prob_args,
        pool=pool,
        backend=backend,
    )

    state = None if resumed else param_0
    old_tau_max = np.inf
    diagnostics = None
    history = []
    autocorr = []
    stopped_early = False

    while sampler.iteration < steps:
        n_chunk = min(criteria.check_every, steps - sampler.iteration)
        state = sampler.run_mcmc(state, n_chunk, progress=progress)
        diagnostics = diagnose(sampler, old_tau_max, criteria)
        old_tau_max = diagnostics["tau_max"]

        history.append(
            (
                diagnostics["iteration"],
                diagnostics["tau_max"],
                diagnostics["acc"],
                diagnostics["ess"],
                diagnostics["shift"],
            )
        )
        if sampler.iteration % criteria.check_every == 0:
            autocorr.append(np.mean(diagnostics["tau"]))

        failed = [k for k, ok in diagnostics["checks"].items() if not ok]
        say(
            verbosity,
            1,
            "[MCMC] step {step}: tau_max={tau:.1f}, acceptance={acc:.2f}, "
            "ESS={ess:.0f}, drift={shift:.1f} sigma | {status}",
            kind="info" if diagnostics["converged"] else "warn",
            step=diagnostics["iteration"],
            tau=diagnostics["tau_max"],
            acc=diagnostics["acc"],
            ess=diagnostics["ess"],
            shift=diagnostics["shift"],
            status="converged"
            if not failed
            else "not yet: " + ", ".join(failed),
        )

        if callback is not None and callback(sampler, diagnostics):
            stopped_early = True
            break
        if diagnostics["converged"]:
            break

    if diagnostics is None:  # a resumed chain that already has enough steps
        diagnostics = diagnose(sampler, old_tau_max, criteria)

    return (
        sampler,
        diagnostics,
        np.array(history),
        np.array(autocorr),
        stopped_early,
    )
```

## `PyToF/MonteCarloToF.py` (the ToF-specific part)
```python
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
```
