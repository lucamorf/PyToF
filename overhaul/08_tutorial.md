# 08 — Tutorial

## Now
5 chapters, good prose; plotting chapters (2, 5) teach `PlotToF` kwargs, not plotting. File is ~5 MB (embedded 300-dpi PNGs) → unreadable git diffs.

## Proposal
1. Base it on `PyToF_Tutorial_no_PlotToF.ipynb` (already works).
2. **Slim outputs:** images saved to disk and referenced, `dpi=120` inline, outputs stripped in git (`nbstripout`) — keep an executed copy as a release artifact or on ReadTheDocs-style rendering.
3. **Runnable in CI:** a `fast` flag (smaller N, fewer MCMC steps) so `pytest --nbmake` executes it in <2 min.
4. Add short sections that don't exist today: *choosing order/N/n_bin* (with the measured table from 07), *what the warnings mean*, *MCMC pitfalls* (pickling, seeds, convergence via autocorr).
5. Each chapter opens with "what you'll learn" and ends with a one-line takeaway.
6. Consistent `matplotlib` style via one `plt.rcParams.update` cell, not per-module copies.

## Structure
Ch1 first model → Ch2 reading results & plotting → Ch3 custom barotropes/densities → Ch4 atmosphere + differential rotation → Ch5 MCMC → Ch6 (new) performance & accuracy guide.

Sequence: after 03–06 are done, so the API shown is the final one.


---

## STATUS: done (redesigned, not just ported)
Old `PyToF_Tutorial.ipynb` (and the no-PlotToF draft) replaced by five notebooks in `tutorial/`, all executed with outputs and based on the final API (`from PyToF import ToF`, `reset()`, `Js_errors`, no MCMC, no PlotToF):
1. `01_first_model`: create, barotrope, relax, convergence/tolerances/warnings/verbosity, reading `Js`, comparison with W&H 2016, order and N, `get_Js_errors`.
2. `02_plotting`: profiles, comparing models (colorbar, inset), 2D cross-section, shape deviation, figure functions, Js with error estimates.
3. `03_your_own_planet`: physical parameters (R_phys kinds, R_ref), own density profile (`relax_to_shape`), radius consistency warning, density jumps / Simpson fallback, atmosphere, tuning a two-layer Neptune to J2 with a root finder; pointer to PyToF_MCMC (no MCMC chapter).
4. `04_beyond_the_Js`: NMoI (sanity checks, NMoI vs J2), Love numbers (non-rotating check, density jumps with `bs`), differential rotation (`alphas`, CMS comparison).
5. `05_numerics_guide`: order/N cost-accuracy curves, binning, Simpson vs trapezoid (incl. the two cases not caught), tolerances, warnings, `debug_plot`, recommended settings table.
- Every notebook starts with an optional import of a personal `mpl_config.py` (searched in the parent folders, ignored if absent).
- `tutorial/tutorial_utils.py`: `jupiter()`, `profiles()`, labels; reference values come from `accuracy/references.py`.
- Notebooks need PyToF from this repo (the PyToF installed in site-packages is 1.7.0, old); `tests/test_notebooks.py` executes all of them (opt-in: `PYTOF_TEST_NOTEBOOKS=1 pytest tests/test_notebooks.py`).
