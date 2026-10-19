# 02a — Evidence: version matrix

Method: fresh `uv` virtualenv per row, install exact versions, build all 9 golden cases from `tests/cases.py`, compare every stored quantity (Js, NMoI, radius ratios, m_rot, profile samples) with `tests/golden/golden.json` (generated on 3.10 / NumPy 2.2.6 / SciPy 1.15.3).
Script: `compare.py` (scratchpad, not in repo). Column "now" = current repo code, "fixed" = with the two fixes of 02b.

| Python | NumPy | SciPy | now | fixed |
|---|---|---|---|---|
| 3.10 | 2.2.6 | 1.15.3 | identical | identical |
| 3.10 | 2.0.2 | 1.13.1 | identical | identical |
| 3.10 | 2.1.3 | 1.14.1 | **crash** (`simpson` positional) | identical |
| 3.10 | 1.26.4 | 1.13.1 | **crash**, orders 7/10/diffrot/trapezoid (object dtype) | identical |
| 3.10 | 1.26.4 | 1.12.0 | **crash**, same | identical |
| 3.10 | 1.26.4 | 1.11.4 | **crash** (`cumulative_simpson` missing) | not supported |
| 3.11 | 2.2.6 | 1.15.3 | identical | – |
| 3.12 | 2.2.6 | 1.15.3 | identical | – |
| 3.12 | 2.3.5 | 1.16.3 | identical | – |
| 3.12 | 2.4.1 | 1.17.0 | identical | identical |
| 3.13 | 2.3.5 | 1.16.3 | identical | identical |

"identical" means max |new/golden − 1| = 0.0 exactly — not "within tolerance".

## Other quantities (order 4, N=256, across the same envs)
- `get_Js_errors`: identical everywhere.
- `get_NMoI`: identical everywhere.
- `get_Kn(2)`, `get_Kn(4)` (solve_ivp): ≤ 1.3e-15 relative between NumPy 1.x and 2.x, identical within each major.

## The historical effect (explains what you saw)
`scipy.integrate.simpson` on 1024 (even) points, ∫x³e⁻ˣ over [1/1024, 1]:

| SciPy | result | rel. error vs exact 0.11392894125669567 |
|---|---|---|
| 1.10.1 | 0.11392894127130161 | 1.3e-10 |
| 1.15.3 | 0.11392894125643242 | 2.3e-12 |

SciPy 1.11 rewrote the even-N end correction; results moved by ~1e-10 relative, comparable to the solver tolerance, so Js shift at that level.

## Not tested (be aware)
- macOS/Windows/ARM (FMA, different libm): results could differ at the last bits; golden rtol is 1e-9 for that reason.
- MCMC results: depend on `emcee` version and the RNG; only smoke-tested.
- Python 3.14, NumPy 3: unknown, the weekly "latest" CI job would tell.
