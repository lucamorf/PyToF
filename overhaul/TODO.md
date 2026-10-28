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
- [x] **Simpson + density gradient at the surface:** warning added (`_check_surface_slope`, option `surface_exponent_warn` = 0.9, 0 = off, tests in `tests/test_messages.py`, tutorial 5). Measured (n=0.8, N=1024): Simpson 100 iterations (cap), trapezoid 19; Simpson iterations grow with N (26 / 39 / 73 / 100 for N = 128 / 256 / 512 / 1024), trapezoid stays at 19. Smoothing the surface away (large P0) removes the problem, which supports the hypothesis; what oscillates is the first cell below the surface (period-2, rho[1] and Pi[1]). The check estimates the exponent in iteration 3 (in iteration 1 it would see 2n/(n+1)); it reads slightly low (0.86 for n=0.8), so 0.9 flags n <~ 0.85. Limits: P0 >~ 1e6 Pa hides the singularity from the three outermost points while convergence stays slow (n=0.8, P0 up to 1e8: still 100 iterations); a P0-independent detector (exponent from P - P0, or fitting deeper than the offset depth) is untested.
- [ ] **Simpson vs trapezoid differ for n=0.8:** J2 = 1.74832e-2 vs 1.74536e-2 (0.17 %), rho[1] 14.28 vs 13.57 (5 %); which one is closer to the truth is not determined.
- [ ] **Crash for n=0.8, N=4096 with Simpson:** `AssertionError: Calculated pressure is not monotonically increasing!`, raised from the recursive `_pressurize` call after the Simpson fallback (N=2048 falls back and converges in 60). Not investigated; the fallback should not end in an assertion.
- [x] `ToF.get_Js_errors` docstring said "called by relax_to_shape()"; fixed (it is not called automatically).
