# 05 — MCMC (done): works properly, and saves good models while it runs

Commit `38e2032`. Code in `05a_mcmc_code.md`. Tests: `tests/test_sampling.py` (40) and the unchanged golden `tests/test_mcmc.py`.

## What you asked for, in one example
```python
from PyToF import ToF, ConvergenceCriteria, SaveFits

result = X.run_mcmc(
    "baro", nwalkers=20, steps=5000, Ncores=8, parallelize=True, seed=1,
    save_fits=SaveFits("fits/", n_sigma=1.0, min_distance=0.2, max_models=100),
    checkpoint="chain.h5",                      # needs h5py; resumable
    convergence=ConvergenceCriteria(check_every=100),   # optional
)
result.converged, result.failed_checks          # did it converge, and if not why
result.chain, result.log_prob, result.blobs     # everything, also if it never converged
```
`run_baro_MC` / `run_dens_MC` still exist and still return `(state, autocorr)`; they accept the same new keywords.

**Saving good models during the chain** (`SaveFits`): after every check (every `check_every` steps), the samples drawn since the last check that *fit* (every J within `n_sigma` of its target) are considered, best fit first. A model is saved only if it is **different enough from every model saved so far**: the RMS distance of its parameters, each divided by that parameter's standard deviation in the chain, is at least `min_distance`. Each saved model is re-evaluated (in parallel if a pool exists) and only written if it still fits. Output: `model_SI_<k>.txt` (radii, densities, pressures — same format as `classify_and_save_state`) plus `index.csv` (name, step, worst sigma, whether the chain had converged at that moment, ESS, tau, and the parameters). `max_models` stops the run once enough are saved; models already in the folder count and are respected, so an interrupted run can simply be started again.

## Real test: Neptune, 3 polytropes, 8 parameters (the tutorial's setup)
20 walkers, 300 steps, 8 cores, 234 s. The chain had **not** converged (it says so: length, tau stable, ESS, acceptance) — but 27 distinct models within 1 sigma (best 0.09 sigma) were on disk after the first 100 steps already: 14 after step 100, 22 after 200, 27 after 300. Their parameters differ by 27-53 % (std/mean) from each other.

## What was weak, and what is different now
| Before | Now |
|---|---|
| Stops when `tau*100 < steps` and tau moved <1 %; no information on why a run did not converge | Five checks every `check_every` steps (taken from your Uranus/Neptune code); `result.converged`, `result.failed_checks`, `result.history` |
| Nothing is kept except the last positions | Full chain, log-probabilities and blobs in the result; `result.save("chain.npz")`; fits written while running |
| Seeding loop could hang forever | `max_start_attempts` per walker, then `RuntimeError` saying how many draws failed and why |
| `multiprocessing.set_start_method("fork", force=True)` at **import** (global side effect) | gone; the pool uses a local "fork" context where available (test: importing PyToF leaves the start method alone) |
| `parallelize=True` with a lambda/closure: obscure `Can't pickle local object` deep inside emcee | checked **before** the pool starts, with an explanation |
| Not reproducible | `seed=` |
| No restart after a crash | `checkpoint="chain.h5"`: same file again continues; larger `steps` extends the chain (verified identical to an uninterrupted run) |
| `nwalkers < 2*ndim` crashed in emcee with an unrelated message | explicit `ValueError` |
| missing `set_check_param` / start values: `AttributeError`/`TypeError` | explicit `ValueError` saying what to set |

## What I took from newUranusNeptune, and what not
Taken: the five convergence checks and their meaning, chunked running with a check after each chunk, HDF5 checkpoint/resume, saving accepted samples while the chain is still running ("ignore convergence mode"), the blob idea (store the fit quality with each sample so no model has to be recomputed to decide), bounded start-point search.
Changed: *"different enough"* is new (that code only skips exact repeats); acceptance range widened to (0.1, 0.9) because with few parameters the stretch move accepts 0.7-0.8 and a 0.7 limit would forbid convergence (verified on a 2-parameter Gaussian).
Not taken (specific to that project): prior hash and repeated runs until `n_models`, re-running a fresh chain, the chi-squared ellipse in (J2, J4) (here: each J within `n_sigma`, as `classify_and_save_state` already did).

## Decisions I made — please check
1. **"Different enough" = parameter space**, scaled by the chain's own spread, default `min_distance=0.1`. Cheap and model-independent. Alternative: compare the density profiles of the models (more physical, but needs the model evaluated before deciding). Say if you want that.
2. **"Fits" = every J within `n_sigma` (default 1)**, i.e. `max_i |dJ_i|/sigma_i < n_sigma` — the same criterion as `classify_and_save_state`.
3. **Defaults** `check_every=100, tau_factor=50, tau_rel_change=0.01, n_eff_target=1000, acc_range=(0.1, 0.9), stationarity_sigma=4`. (Your Uranus code uses ESS 2000 and acc 0.7.)
4. **A run that does not converge is not a warning**, it is the normal case; see `result.converged` or `verbosity=1` (`[MCMC] step ...` lines).
5. **The checkpoint file is kept** after the run (it is the chain). It only checks the *shape* (walkers x parameters) before resuming, not that the model is the same; the Uranus code hashes its prior for that reason. Use a new file name when you change the model.
6. `burn_in=100` steps are never saved from, per `SaveFits`.
7. `autocorr` keeps its old meaning (mean tau at every multiple of `check_every`).

## Limits / known
- Starting points are still searched serially (negligible compared to the chain).
- Each saved model costs one extra evaluation.
- The CGS file has `-inf` for the pressure at the surface where P0 = 0 (`log10(0)`; unchanged behaviour of `classify_and_save_state`).
- One model evaluation takes ~0.05-0.15 s even for tiny grids (fixed per-iteration cost of the generated polynomials), so MCMC tests are slow; the whole suite now takes ~90 s.
- Seeded results of the *old* implementation are reproduced exactly (golden test), also on NumPy 1.26 and 2.3.
