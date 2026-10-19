# 01 — Test suite (the safety net)

**Problem:** no tests exist. Refactoring 5000 lines of physics code blind is how
silent errors happen.

## Proposal
Add `tests/` with `pytest` (+ `pytest` in a `[test]` extra). Four layers, fastest first:

| Layer | Example | Speed |
|---|---|---|
| **Analytical** | NMoI of constant sphere = 2/5; linear-density sphere = 4/15; rotating-spheroid value (already in accuracy notebook cell 3) | <1 s |
| **Golden master** | Jupiter-like default run, orders 4/7/10, N=256: store `Js`, `NMoI`, `R_calc`, `R_eq_to_R_m` in `tests/golden/*.json`, assert `rtol=1e-12` | seconds |
| **Literature** | Wisdom & Hubbard 2016 Jupiter Js (`Target_Js`) reproduced to the documented accuracy per order | seconds |
| **Properties** | Js errors shrink with N; `n_bin=-1` ≡ `n_bin=N`; `use_simpson` False/True agree to expected tol; invalid kwarg raises (after 03) | seconds |

Plus smoke tests: `run_baro_MC` tiny (nwalkers=6, steps=3, order 4, N=64) with a fixed seed;
every plot function runs with the `Agg` backend and doesn't raise.

## Details
- Golden values are generated **from the current code by one script**
  (`tests/make_golden.py`), committed, and only regenerated deliberately.
  This proves refactors don't change numbers; it does *not* prove the numbers are right —
  that's what the analytical + literature layers are for.
- No slow/fast distinction: `pytest` runs everything (~30 s).
- Add a GitHub Actions workflow (`pytest` + `ruff check`) on 3.10 and 3.12.

## Open question
`get_Kn` (Love numbers) has an **uncommitted sign change** in your working tree.
I'd test it against the Gavrilov–Zharkov values in `love_dev.ipynb`, but only after
you decide in 02 whether that change is final.

## Cost / risk
~1 file per layer, no change to package code. Zero risk.

---

## STATUS: implemented (uncommitted) — please review 01a–01d

**Result:** `pytest` → 40 passed in ~30 s (no slow/fast split). `ruff check tests` clean. No file under `PyToF/` was touched.

| File | Content |
|---|---|
| `01a_tests_infrastructure.md` | pytest.ini, conftest, shared cases, golden generator |
| `01b_tests_correctness.md` | analytical + W&H 2016 literature |
| `01c_tests_stability.md` | golden master + invariants |
| `01d_tests_smoke.md` | MCMC (no plot tests, PlotToF is going away) |

**Do the tests actually catch bugs? (mutation check, scratch copy of the package)**

| Planted error | Caught by |
|---|---|
| Simpson → trapezoid in `_mass_int` | analytical NMoI |
| `G` changed in 7th digit | golden master |
| order-10 `f2`: dominant coefficient +1, a sign flip | Bessel literature test |
| order-10 `f2`: one mid-size coefficient 15750→15760 | golden master |
| order-10 `f2`: coefficient of an `s2^7` term ×1.001 | **not caught** — the term contributes ~1e-7 relative; physically invisible at our tolerance. Inherent limit of numerical tests. |

**Findings made while writing tests (nothing changed for them yet):**
1. `relax_to_density()` never applies the density function; only the MC cost function does. Fine, but undocumented.
2. `alphas` are in m²/s² (~1e5); plausible-looking small values do *nothing*, silently. Worth a docstring (step 10) or a warning.
3. `ToF(R_phys=7e7)` gives a bare `TypeError`; step 03 will make it a clear message.
4. **Open physics question:** `X.R_eq_to_R_m` = 1.02248 vs the W&H Bessel entry 1.02288 (rel. diff 3.9e-4, constant in N and order, while J2 agrees to 1e-6). Probably a definition difference of "mean radius", but I have *not* verified that, so no test asserts it. Candidate for the Love/accuracy session.

**Not yet covered** (deliberately): `get_Kn` Love numbers — waits for the sign-check notebook you mentioned; slow N-sweep vs Movshovitz stays in the accuracy notebook (step 09).

**Needs from you:** OK on the layout; do you want these committed now as their own commit (tests only, `golden.json` included)? `__pycache__` noise stays out of it.
