# PyToF changelog and overhaul notes

Single file replacing the former `overhaul/` folder (31 notes and diffs, still in
git history before the commit that removed them). Newest first.

## 2.0.0 (overhaul of 2026)

### Breaking changes
- **MCMC removed.** PyToF is the Theory of Figures plus closely related quantities (moment of inertia, Love numbers, error estimates); sampling is left to user code, which can call `reset()`, `relax_to_*()` and read `Js`. Removed from `ToF`: `run_mcmc`, `run_baro_MC`, `run_dens_MC`, `baro_cost_function`, `dens_cost_function`, `set_check_param`, `set_density_function`, `classify_and_save_state`; options `target_Js`, `sigma_Js`, `rho_max`, `P_max`, `dens_param_init`. Dependencies `emcee`, `tqdm`, `h5py` are gone.
- **`PlotToF` and the `plot_*` methods deleted.** Plot with plain matplotlib (tutorial 2). matplotlib is optional (`pip install .[plot]`) and only needed for `debug_plot`.
- **Strict keyword arguments:** unknown options raise a clear error instead of being ignored.
- **Renames** (old names still work and emit a `RenamedWarning`, to be removed later):

  | Was | Now |
  |---|---|
  | `MaxIterShape/Update/Bar/Den` | `max_iter_shape/update/barotrope/density` |
  | `baro_param_init`, `baro_param_calc` | `baro_param` (the barotrope reads the option directly) |
  | `X._set_IC()` | `X.reset()` |
  | `X.Js_error` | `X.Js_errors` |
  | `X.ss`, `X.SS` | `X.figure_functions`, `X.volume_integrals` (holds S_n then S'_n) |
  | first parameter `class_obj` | `self` / `tof` |

  Unchanged on purpose: `Period`, `N, G, H, P0, R_ref, R_phys, M_phys`, `li, rhoi, Pi, Js, A0, U, R_eq_to_R_m`, and all names inside `AlgoToF.py` (they match the Mathematica notebook and the papers). `from PyToF import ToF` is the entry point.
- `reset()` also undoes the Simpson fallback, so a result never depends on earlier models computed with the same object.
- `get_Kn`: density jumps are no longer counted twice and the surface jump is included (see Love numbers).
- `U` sign and ordering, messages and option names reworked (`PyToF/messages.py`, `PyToF/options.py`; `Options.describe()` documents every option).

### Fixes and additions
- Python >= 3.10, numpy >= 1.26, scipy >= 1.12 (`cumulative_simpson`). CI matrix: 3.10 (oldest pins), 3.11, 3.12 (reference), 3.13, plus latest unpinned weekly (it caught the SciPy 1.14 `simpson` break).
- `simpson(..., x=)` keyword, 2373 integer literals in the generated code made floats, 116 dead expression statements removed from `AlgoToF.py`, `__version__` from package metadata, `pyproject.toml` replaces `setup.py`.
- `relax_to_barotrope` / `relax_to_density` share one `_relax` worker (identical numerics); max-iteration limits are validated (a limit of 1 crashed).
- Warning for an infinite density slope at the surface when Simpson's rule is active (`_check_surface_slope`, option `surface_exponent_warn`, default 0.9, 0 = off). Measured for n = 0.8: Simpson hits the 100-iteration cap, the trapezoid rule needs 19; iterations of Simpson grow with N.
- `debug_plot` = False / `"show"` / `"save"`: one large figure per `relax_to_*` iteration and per AlgoToF step; options `simpson_check` (False keeps Simpson to see the failure), `debug_label`, `debug_primed`. `PlotDebugToF` imports matplotlib lazily and is O(N).
- Tests (`tests/`): analytical, literature (Wisdom & Hubbard 2016), golden master (`tests/golden/golden.json`, regenerate deliberately with `tests/make_golden.py`), properties, options, messages, atmosphere, Love numbers, debug plot, code hygiene, notebooks (`PYTOF_TEST_NOTEBOOKS=1`). Planting errors in the generated polynomials showed the limit of numerical tests: a term contributing ~1e-7 relative is invisible.
- Tutorials: five new notebooks in `tutorial/` (first model, plotting, own planet, beyond the Js incl. Love numbers, numerics guide).
- Accuracy: one notebook, `accuracy/accuracy.ipynb` with `MODE = "quick" | "full"` (see below).

### Findings
- **`R_eq_to_R_m` vs Bessel (1.02248 vs 1.02288, 3.9e-4): definition, not a bug.** PyToF's `R_m` is the *volume-equivalent* radius (`li[0]`, so 4/3 pi li^3 is the volume). Wisdom & Hubbard (2016) write r = R s [1 + eta(s, mu)] with the shape function eta = sum over l > 0 of a_l P_l(mu) (Eqs. 3-5, no P_0 term; mu = cos colatitude), so their R is the angular mean of the surface, `li (1 + s0)` (they note R_s is only 'approximately' the volume-equivalent radius) with s0 = `figure_functions[0][0]` = -3.9e-4. `R_eq_to_R_m / (1 + s0)` = 1.02287544 against the reference 1.02287543 (1e-8). Movshovitz's code uses the volume-equivalent radius as well. The rotation parameter m is the same in both. Moments are unaffected. the accuracy notebook now reports `R_eq/R_mean` in the reference convention; `tests/test_literature.py` asserts both the match and the gap.
- **Love numbers.** `get_Kn` agrees with the analytical cases of Gavrilov, Zharkov & Leontev (1976) (linear, quadratic, polytropic density; K_n r^n to ~1e-6 at N = 1024) and with the two-layer planet of Consorzi, Melini & Spada (2023, Eq. 2) with and without the density jump (`bs`). Passing `bs` buys robustness against the loose `solve_ivp` default `rtol=1e-3`, not accuracy; both variants converge as 1/N. Not understood: why the `bs` variant has a larger O(1/N) error when the jump lies exactly midway between grid points.
- Speed: nothing changed; tolerances 1e-10 are about one order below the best attainable accuracy (order 10, N = 16384: J2..J6 agree with Bessel to 2e-9..4e-9). Warm starts and convergence acceleration remain ideas.
- `relax_to_density()` never applies the density function (it only entered the removed MCMC cost function); `alphas` are in m^2/s^2 (~1e5), so plausible-looking small values silently do nothing.
- Paper check for the author: Morf+2024 says "trapezoid" for the mass integral, the code uses Simpson.
- Order-10 differential-rotation results of 1.4.0 to 1.7.0 had an A0 bug (fixed in `4f7a37d`, J2 off by ~1e-4); none of them was ever used for a publication.

### Accuracy notebook
`accuracy/accuracy.ipynb` with `MODE = "quick"` (~10 s): analytical checks, Bessel sweep N = 2^5..2^12, DR2 against CMS, comparison with the shipped tables. `MODE = "full"` (default, ~4 min): all N = 2^3..2^15, all binnings, Simpson and trapezoid, DR2/DR3, Movshovitz & Fortney (2022); `WRITE_TABLES = True` overwrites `PyToF/accuracy_data_*.npz` (used by `ToF.get_Js_errors()`). All runs are in one process pool, most expensive first, one BLAS thread per worker, each run computed once. The recomputed tables agree with the shipped ones bit for bit (deviation 0.0), so the numerics of the accuracy tables are unchanged by the rewrite.

### Open
- `PlotDebugToF`: a convergence-history object (Js per iteration) without plotting; return figures instead of showing/saving.
- Simpson vs trapezoid differ for n = 0.8 (J2 by 0.17 %), which is closer is undetermined. `n = 0.8, N = 4096` with Simpson ends in `AssertionError: Calculated pressure is not monotonically increasing!` after the fallback (N = 2048 converges); the fallback should not end in an assertion.
- Surface-slope detector independent of P0 (P0 >~ 1e6 Pa hides the singularity from the three outermost points) is untested.
- `get_Kn`: optional `rtol`/`atol`.
- README: options table generated from `Options.describe()`.
- Ideas: move the `FunctionsToF.py` functions into the class (or mixins); give `AlgoToF.default_opts()` a real signature; require explicit planet parameters instead of the W&H Jupiter defaults in a future major version; generate the `A0 +=` prefixes in `Mathematica/convert.py`.

## Earlier versions
1.7.0 and before: see the git history (the 1.4.0 to 1.7.0 A0 bug above).
