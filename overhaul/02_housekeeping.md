# 02 — Housekeeping + multi-version support (overview)

Your two commits are in (`48e093c`: Love-number sign + `get_Js_errors` verbosity).
Details: **02a** = evidence from the version matrix, **02b** = the concrete code changes.

## Answer to "do numpy/scipy versions change the numbers?"
Short: **not in any version the code can run on today — but only because the code refuses to run on the rest.**

- Python 3.10–3.13 × NumPy 1.26–2.4 × SciPy 1.12–1.17: every golden case is **bit-identical** (rel. diff exactly 0), once two small bugs (below) are fixed.
- Love numbers (`get_Kn`, ODE solver): differ by ~1e-15 between NumPy 1.x and 2.x — round-off only.
- What *did* change numbers historically: **SciPy ≤ 1.10 → ≥ 1.11 `simpson` for an even number of points** (every `N = 2^k` run). Measured: 1.3e-10 relative error on a test integral (vs 2e-12 today), i.e. above your 1e-10 solver tolerance. That is almost certainly what you noticed. Nothing at or above SciPy 1.12 shows it.

## Two real compatibility bugs found (hard errors, not drift)
| Bug | Breaks | Fix |
|---|---|---|
| `scipy.integrate.simpson(y, x)` positional; SciPy 1.14.x made `x` keyword-only | **SciPy 1.14.x** (all cases crash) | `simpson(y, x=x)` (5 call sites) |
| 2373 integer literals > 2^63 in generated polynomials; NumPy 1.x makes object arrays | **NumPy 1.x**, orders 7 and 10 | write them as floats (`…875.0`); value after conversion to float64 is identical, so NumPy 2 results do not move |
| `cumulative_simpson` only exists from SciPy 1.12 | SciPy ≤ 1.11 | declare `scipy>=1.12` |

## Proposal (decided where I could, open where not)
1. **Support `python>=3.10`**, `numpy>=1.26`, `scipy>=1.12`, **no upper pins** (setup.py currently pins exact versions, which blocks 3.11 entirely).
2. Fix the two bugs above (02b).
3. CI matrix that would have caught both: oldest / middle / newest, plus a weekly "latest everything" run (02b).
4. Drop dead code (116 no-op statements, found by `ast`), fix `__version__`, untrack `.pyc`, one ruff config.
5. Replace `setup.py` by `pyproject.toml`.

## Your call
- **Floor for NumPy/SciPy**: I verified 1.26/1.12 as the oldest. Lower is possible but untested; OK with `numpy>=1.26`?
- Tests currently use `np.trapezoid` (NumPy ≥ 2). I'd switch the tests to `scipy.integrate.trapezoid` so they run on 1.x too — fine?
