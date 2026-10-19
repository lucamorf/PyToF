# TODO (collected during the overhaul)

## Decided, scheduled
- [x] **Step 06 done** (see `06_plotting.md`, STATUS).
- [ ] **PlotDebugToF, later:** a convergence-history object (Js per iteration) available without plotting; return figures instead of writing files into the cwd.
- [ ] **Accuracy session:** investigate `R_eq_to_R_m` = 1.02248 vs the Wisdom & Hubbard Bessel reference 1.02288 (3.9e-4, independent of N and order) — definition difference or bug? (`09_accuracy_notebook.md`)
- [ ] **Love numbers:** verify the sign change of the density-jump term in `get_Kn` against the analytical Gavrilov et al. cases (`10_love_and_docs.md`), then add tests.
- [ ] Notebooks: switch to the new names (`max_iter_barotrope`, `target_Js`, `reset()`, `figure_functions`, ...) and `scipy.integrate.trapezoid` instead of `np.trapezoid`; strip outputs / shrink images.
- [ ] README: options table from `Options.describe()`, `from PyToF import ToF`, supported versions, changelog.
- [ ] Check whether any published order-10 differential-rotation result was produced with versions 1.4.0-1.7.0 (A0 bug, fixed in `4f7a37d`): J2 was off by ~1e-4.

## MCMC split (done: 05b_split_mcmc.md)
- [x] merged; PyToF_MCMC supports `kind="baro"` and `kind="dens"`.
- [ ] Morf+2025/RunToF.py depends on the old MCMC interface of the installed PyToF 1.7.0 (breaks on reinstall).
- [ ] Tutorial (step 08): remove the MCMC chapter; no MCMC in PyToF docs. README: PyToF_MCMC is mentioned in one line (done).

## Open ideas
- [ ] (PyToF_MCMC) "different enough" by density profile instead of parameters; parallel start-point search; a prior/model hash for checkpoints.
- [ ] Move the functions of `FunctionsToF.py` / `MonteCarloToF.py` physically into the class body (or mixins); they already use `self`.
- [ ] `AlgoToF.default_opts()` duplicates the ToF defaults under other names (`tol`, `maxiter`); give `Algorithm` a real signature.
- [ ] Deduplicate `relax_to_barotrope`/`relax_to_density` and the MCMC pairs (step 04); MCMC robustness (step 05).
- [ ] Speed work (step 07): warm starts, convergence acceleration; benchmarks committed.
- [ ] Defaults of `ToF()` are the Wisdom & Hubbard Jupiter values; consider requiring explicit planet parameters in a future major version.
- [ ] Look at `Mathematica/convert.py` + `ToF.nb`: generate the `A0 +=`/assignment prefixes automatically, so nobody pastes bare expressions again.
- [ ] Unmeasured claim to check: in parallel MCMC the Simpson fallback used to be per worker process (now explicit state on the object).
