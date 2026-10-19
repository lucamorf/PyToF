# 05b — Splitting the MCMC out of PyToF (DONE, merged into master as `99b4e39`)

You are right: PyToF should be the Theory of Figures plus closely related quantities (NMoI, Love numbers, Js errors). Sampling is a *user* of that, so I did the split for real, but without touching your master:

| What | Where |
|---|---|
| PyToF **without** MCMC | git branch **`split-mcmc`** (commit `99b4e39`, on top of master `38e2032`). Diff: `05b_split_mcmc.diff`. Your working tree is untouched. |
| The MCMC as a standalone reference package | **`/home/lumo/Code/PyToF_MCMC/`** (README inside) |

## What leaves PyToF (2 644 lines, 15 added)
| Removed from PyToF | Now lives in PyToF_MCMC |
|---|---|
| `PyToF/MonteCarloToF.py` (662 lines), `PyToF/sampling.py` (516) | `pytof_mcmc/problem.py`, `pytof_mcmc/sampling.py` |
| methods `run_mcmc`, `run_baro_MC`, `run_dens_MC`, `baro_cost_function`, `dens_cost_function`, `set_check_param`, `set_density_function`, `classify_and_save_state` | `run_mcmc(problem, ...)`, `Problem.cost_function`, `Problem(check_param=..., density_function=...)`; `classify_and_save_state` is not ported (`SaveFits` + `result.blobs` replace it) |
| options `target_Js`, `sigma_Js`, `rho_max`, `P_max`, `dens_param_init` (25 options remain) | arguments of `Problem` |
| attributes `check_param`, `density_function`, `dens_param_calc` | inside `Problem` |
| the MCMC plots `plot_state_xy`, `plot_state_corr_xy`, `plot_autocorr` (in PlotToF, which goes in step 06 anyway) | not ported (the tutorial notebook has plain-matplotlib versions) |
| dependencies `emcee`, `tqdm`, `h5py` extra | `emcee` (+ optional `h5py`) of PyToF_MCMC |
| tests: `test_mcmc.py`, `test_sampling.py`, `test_smoke.py`, golden `mcmc.json` | `PyToF_MCMC/tests/` |

**Stays in PyToF** (the interface the MCMC uses, unchanged): `reset()`, `relax_to_shape/barotrope/density()`, `set_barotrope()`, `li, rhoi, Pi, Js, baro_param_calc`, options `baro_param_init, M_phys, verbosity, max_iter_*`. `relax_to_density` stays because it is a legitimate ToF operation (given densities -> shape and pressure).

## How I know the split works
* PyToF on the branch: **96 tests pass in 11 s** (was 151 in 90 s); `import PyToF` no longer loads emcee, tqdm or h5py; golden numbers of every relaxation case unchanged.
* PyToF_MCMC, run against the *stripped* branch (not against the full PyToF): **46 tests pass** (~40 s), among them the **golden test that reproduces the seeded chains of the original implementation exactly** (the file was generated before the move), parallel runs, saving, checkpoint/resume.
* `examples/neptune_three_polytropes.py` (your tutorial's Neptune setup) reproduces the earlier run digit for digit: 14 models after 100 steps, tau_max 11.1, ESS 139.

## What `PyToF_MCMC` looks like
```python
from pytof_mcmc import Problem, run_mcmc, SaveFits
problem = Problem(X, "baro", param_init=[...], target_Js=[...], sigma_Js=[...], check_param=check_param)
result  = run_mcmc(problem, nwalkers=20, steps=5000, parallelize=True, Ncores=8, seed=1,
                   save_fits=SaveFits("fits/", n_sigma=1.0, min_distance=0.2, max_models=100),
                   checkpoint="chain.h5")
```
Everything in one `Problem` object instead of hidden attributes on the planet (clearer: the planet model and the data are no longer mixed into `opts`). Full description, convergence table, migration table old -> new: its `README.md`. 3 files, ~1 000 lines of code.

## What would break if you merge — please read
* **`Morf+2025/RunToF.py`** uses the old MCMC interface heavily (`run_baro_MC`, `run_dens_MC`, `set_check_param`, `set_density_function`, `classify_and_save_state`, `plot_state_*`, `plot_autocorr`, `dens_param_init`, `rho_MAX`, `P_MAX`, `Target_Js`, `Sigma_Js`). It keeps working as long as it imports the **old installed copy** (site-packages holds PyToF 1.7.0, which is what all your other project folders use today). It would break the day you `pip install -e` the new PyToF. **Suggestion:** tag the last version that has the MCMC (`git tag pytof-with-mcmc 38e2032`) so that project can always be reproduced (`pip install git+...@pytof-with-mcmc`).
* The other folders (`dense_subNeptune`, `Morf+2026`, `project_Jason`, `TODO_winds`, `plot_*.py`, `planet_data/`): they import PyToF but do not use the MCMC interface (the `Target_Js`/`Sigma_Js` in `planet_data/` are just keys of data dictionaries, not PyToF options).
* The tutorial notebook's chapter 5 (MCMC) must move to PyToF_MCMC's example — I have not touched notebooks.

## How to adopt
1. (optional) `git tag pytof-with-mcmc 38e2032`
2. `git merge split-mcmc` (the worktree I used lives in my scratch folder; the branch is in your repo; `git worktree prune` cleans the registration).
3. I update step 04/05 notes, the README, and later the tutorial (step 08).

## Decisions for you
1. **Merge the split?** (and tag first?)
2. **Name/location**: `/home/lumo/Code/PyToF_MCMC`, package `pytof_mcmc` — fine? Should I `git init` it?
3. **MCMC plots** (state, correlation with histograms, autocorrelation): drop, or add a small `pytof_mcmc/plots.py` built from the tutorial's plain-matplotlib versions?
4. `Problem(kind="dens")` keeps the density-function model that `relax_to_density` supports; do you still use it, or should the reference only carry the barotrope model?


---

## STATUS (your decisions)
* Merged into master (fast-forward, `99b4e39`); branch and worktree removed. The decision to keep the MCMC out of PyToF is final. No tag was created.
* `PyToF_MCMC` stays at `/home/lumo/Code/PyToF_MCMC` (no `git init`, not asked).
* No MCMC plots to port; tutorials will no longer mention the MCMC.
* **Both models work**: `kind="baro"` and `kind="dens"`, each with serial and parallel runs, saving fits, checkpoint; 5 new tests for the density model (cost/log-prob, "densities are the given profile", run + saved fits, parallel run identical to the serial chain, validation). PyToF_MCMC: **51 tests pass** against the merged PyToF; PyToF itself: 96 pass.
* Reminder: `Morf+2025/RunToF.py` still needs the *old* PyToF (installed 1.7.0 in site-packages has the MCMC); it breaks if you reinstall PyToF from this repo.
