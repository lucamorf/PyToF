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
