# 03 — Class structure, messaging, options (overview)

Four commits, all behaviour-preserving for the numerics (golden tests unchanged); behaviour changes are limited to messages and error handling and are listed in 03a/03b.

| Commit | File | Content |
|---|---|---|
| `1c04672` | `03c_methods.md` | 16 `functools.partial` bindings per instance -> class-level methods |
| `f1a9b5c` | `03a_messages.md` | all prints/colour strings -> `PyToF/messages.py` (`say`, `warn`) |
| `f183999` | `03b_options.md` | `Options` dataclass: typed, validated, strict, documented; solver state out of `opts` |
| `2f1cec5` | — | `.gitattributes`: LF line endings (see below) |
| `d5a8551`, `2cd9e8f`, `8878964`, latest | `03d_renames.md` | option renames, `reset()`, `Js_errors`, `class_obj`->`self`, `ss/SS` -> `figure_functions/volume_integrals` |

Diffs for review: `03_diff_methods.diff`, `03_diff_messages.diff`, `03_diff_options.diff` (line-ending noise removed).
Renames (all approved): `03d_renames.md` — done, old names keep working with a warning.

## Verification
- `pytest`: **100 passed** (was 45). New: `test_messages.py` (levels, colours, warning categories and where they point), `test_options.py` (defaults, dict interface, validation, state).
- The same tests (81 at the time of the first four commits) passed on Python 3.10 / NumPy 1.26.4 / SciPy 1.12.0, 3.11, 3.12 (NumPy 2.4.1) and 3.13.
- The complete executed tutorial (`PyToF_Tutorial_no_PlotToF.ipynb`, incl. 20-walker MCMC) runs without errors on the new code (~5 min).

## Line endings (my mistake, now explicit)
`ClassToF`, `FunctionsToF`, `MonteCarloToF`, `PlotToF`, `PlotDebugToF` and `__init__` had CRLF endings originally. My scripted edits silently converted them to LF (three of them in earlier commits, so those diffs show whole-file changes). The code is unaffected. You approved LF, and `.gitattributes` makes it the policy for the whole repo.

## Found while doing this
- `relax_to_barotrope/density` called the shape step with `maxiter=2` on purpose, which made `Algorithm` look "not converged" on every call (the old `\r` hack hid it). Now separate worker, no false alarms.
- `MaxIterDen` non-convergence was reported as `MaxIterBar`.
- Parallel MCMC (from reading the code, not measured): the Simpson fallback was written into `opts`, and each worker process holds its own copy, so workers could end up disagreeing about it. It is now explicit state on the object; serial runs behave exactly as before.
