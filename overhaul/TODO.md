# TODO (collected during the overhaul)

## Decided, scheduled
- [x] **Step 06 done** (see `06_plotting.md`, STATUS).
- [ ] **PlotDebugToF, later:** a convergence-history object (Js per iteration) available without plotting; return figures instead of writing files into the cwd.
- [ ] **Accuracy session:** investigate `R_eq_to_R_m` = 1.02248 vs the Wisdom & Hubbard Bessel reference 1.02288 (3.9e-4, independent of N and order) — definition difference or bug? (`09_accuracy_notebook.md`)
- [ ] **Love numbers:** verify the sign change of the density-jump term in `get_Kn` against the analytical Gavrilov et al. cases (`10_love_and_docs.md`), then add tests.
- [x] Notebooks redesigned (step 08), README updated.
- [ ] README: options table from `Options.describe()`, `from PyToF import ToF`, supported versions, changelog.
- [ ] Check whether any published order-10 differential-rotation result was produced with versions 1.4.0-1.7.0 (A0 bug, fixed in `4f7a37d`): J2 was off by ~1e-4.

## MCMC split (done: 05b_split_mcmc.md)
- [x] merged; PyToF_MCMC supports `kind="baro"` and `kind="dens"`.
- [ ] Morf+2025/RunToF.py depends on the old MCMC interface of the installed PyToF 1.7.0 (breaks on reinstall).
- [x] Tutorial has no MCMC chapter; README mentions PyToF_MCMC in one line.

## Open ideas
- [ ] (PyToF_MCMC) "different enough" by density profile instead of parameters; parallel start-point search; a prior/model hash for checkpoints.
- [ ] Move the functions of `FunctionsToF.py` / `MonteCarloToF.py` physically into the class body (or mixins); they already use `self`.
- [ ] `AlgoToF.default_opts()` duplicates the ToF defaults under other names (`tol`, `maxiter`); give `Algorithm` a real signature.
- [ ] Deduplicate `relax_to_barotrope`/`relax_to_density` and the MCMC pairs (step 04); MCMC robustness (step 05).
- [ ] Speed work (step 07): warm starts, convergence acceleration; benchmarks committed.
- [ ] Defaults of `ToF()` are the Wisdom & Hubbard Jupiter values; consider requiring explicit planet parameters in a future major version.
- [ ] Look at `Mathematica/convert.py` + `ToF.nb`: generate the `A0 +=`/assignment prefixes automatically, so nobody pastes bare expressions again.
- [ ] Unmeasured claim to check: in parallel MCMC the Simpson fallback used to be per worker process (now explicit state on the object).

## Found during step 08 (tutorial)
- [ ] **Simpson + density gradient at the surface:** for `rho = (P/K)^(n/(n+1))` with n < 1 (e.g. n = 0.8, Jupiter mass/radius/period, N=1024, order 4/7) `relax_to_barotrope` needs the full 100 iterations and ends with a ConvergenceWarning (drot 1e-7, dJ 3e-6); with `use_simpson=False` it converges in 19. n=0.7: 34, 0.75: 56, 0.8: 100, 0.85: 64, 0.9: 29, 0.95: 18, n>=1 fine (17-23). Cause (hypothesis): such profiles have an infinite density gradient at the surface, the Simpson branch integrates `np.gradient(rho)`. See the weak spots in `07_speed.md`.
- [x] `ToF.get_Js_errors` docstring said "called by relax_to_shape()"; fixed (it is not called automatically).
