# 06 — Plotting (decision made: delete PlotToF, keep and improve PlotDebugToF)

## Your decisions
1. **`PlotToF.py` is deleted outright** (no deprecation period). Plotting happens in user code / the tutorial, as in `PyToF_Tutorial_no_PlotToF.ipynb`.
2. **`PlotDebugToF.py` stays**: its functionality is useful. It goes on the TODO to be made better (below).

## Consequences to handle when deleting PlotToF
- Remove the six `plot_*` methods from `ToF` (`plot_xy, plot_shape, plot_ss, plot_state_xy, plot_state_corr_xy, plot_autocorr`) and their imports; `matplotlib` can become an optional dependency (`PyToF[plot]`) if PlotDebugToF only imports it lazily.
- Tests: the API-surface test never listed plot methods (already removed); `test_smoke.py` has no plot tests.
- `tests/conftest.py` forces the Agg backend; keep (PlotDebugToF needs it headless).
- Notebooks: `PyToF_Tutorial.ipynb` still calls `X.plot_xy(...)`, `X.plot_shape()`, ... -> replaced by the no-PlotToF tutorial in step 08.
- Only `PlotToF` code is deleted: `tof_xy_data` and `plot_correlation` helpers of the tutorial replace what people actually used.
- `Options`: `debug_plot` stays.

## PlotDebugToF — what it is, what to improve (see TODO.md)
It is called only through `debug_plot=True` (`debug_AlgoToF_plot`, `debug_FunctionsToF_plot`) and shows the state of the algorithm per iteration. Known issues:
- 57 of 59 lint findings of the old package (dead imports, 31 long lines, invalid escape `'\p'` that raises a SyntaxWarning on Python 3.12+, no trailing newline).
- Contains the O(N^2) enclosed-mass loop and commented-out debug prints (`print('orange', ...)`).
- Its matplotlib style code is a duplicate of PlotToF's `_default_mpl_opts` (disappears with PlotToF).
- Imported at the top of `AlgoToF.py` and `FunctionsToF.py`, so matplotlib is imported even when `debug_plot=False`.
- No tests (cannot assert plots, but a smoke test with the Agg backend and `debug_plot=True` can).
Ideas to discuss when we get there: make it an optional lazy import; return/accept a `matplotlib` figure instead of saving files into the cwd; a proper "convergence history" (Js per iteration) object so the information is available without plotting; levels of detail.


---

## STATUS: done
- `PyToF/PlotToF.py` deleted; the `plot_xy/plot_shape/plot_ss` methods removed from `ToF`. matplotlib is no longer a dependency (`pip install .[plot]`, included in `test`/`dev`).
- `PlotDebugToF.py` rewritten, same output files: imported lazily (only inside the `debug_plot` branches, tested in a subprocess), lint- and format-clean, O(N) enclosed mass (`cumsum`), debug prints and dead code removed, `'\p'` escape fixed, repeated plot code factored into helpers, figures are closed after saving, no plot style is imposed any more (the rcParams helper and the Ubuntu font are gone; set your own).
- New `tests/test_debug_plot.py` (3 tests). Full suite: 99 passed.
- Not done (left on TODO): convergence-history object, figures instead of files.
- Notebooks/tutorial still call the removed `plot_*` methods: step 08.
- Also: `PyToF_MCMC` reduced to plain code (tests, pyproject, pytest.ini removed; README adjusted).
