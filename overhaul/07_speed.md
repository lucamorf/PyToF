# 07 — Speed (measure first)

## Already measured (review §3)
- `B1215` + `B1617` = 71% of time at order 10; removing dead lines gave **no** speedup.
- order 10 is 6.4× slower per iteration than order 4; `n_bin=256` gave 3× at N=16384.
- Simpson vs trapezoid: ~4% — not worth trading accuracy.

## Candidate ideas — all *hypotheses* until benchmarked
| Idea | Expected effect | Cost |
|---|---|---|
| **Warm start** `relax_to_barotrope` from previous shape (esp. in MCMC steps) → fewer outer iterations | large for MCMC | low |
| Look at iteration counts: are 18 iterations needed for 1e-10? Better convergence criterion / Anderson acceleration of the fixed-point loop | maybe 2× | medium |
| Common-subexpression elimination of `B1215/B1617` (sympy `cse` already done partially; the `convert.py` step) | unclear — profile first | medium |
| `numba`/`jit` on the polynomial blocks | maybe 3–10× | new heavy dependency; I'd make it optional or skip |
| Cheaper MCMC `Ncores` default / chunking | small | low |

## Proposal
1. Add `benchmarks/bench.py` (fixed cases; prints time + iterations) — committed, so every claim is reproducible.
2. Try warm start + convergence acceleration first (pure Python, no new deps).
3. Only consider numba if 2 isn't enough, and only as an optional extra.
4. Every speed change must keep golden tests within the documented tolerance.

**Decision for you:** is a new optional dependency (numba) acceptable in principle?


---

## MEASURED (nothing in the code was changed)
Jupiter-like n=1 polytrope, `relax_to_barotrope()`, 20-core WSL machine, scripts in the session scratchpad (b1-b8).

**Where the time goes** (`B1215` = update of figure functions, `B1617` = their derivative terms):
| order | N | total | B1215 | B1617 |
|---|---|---|---|---|
| 4 | 256 | 0.10 s | 6% | 3% |
| 7 | 256 | 0.16 s | 28% | 13% |
| 10 | 256 | 0.47 s | 53% | 20% |
| 10 | 4096 | 1.8 s | 53% | 17% |
| 10 | 16384 | 6.2 s | 53% | 17% |
- order 4 is overhead-dominated: N=64 -> 0.09 s, N=1024 -> 0.14 s. The rest is per-pass Python overhead (scipy `simpson` checks, `np.gradient`, ~2000 `dataclasses.fields` calls per run = 8%). No single hotspot.
- order 10: the two generated polynomial blocks are 70%, as in the review. Cost per pass grows ~linearly with N above N~1000.
- Already in place: shape functions are warm-started between the outer iterations (`ss_initial`), each outer iteration runs `Algorithm` with `maxiter=2`.
- ~18 outer iterations for 1e-10, linear convergence (~0.28 per iteration). Loosening the tolerances (order 7, N=1024): 1e-10 -> 18 its / 0.205 s; 1e-8 -> 14 / 0.154 s (error 3e-9); 1e-6 -> 11 / 0.117 s (2e-7); 1e-4 -> 8 / 0.090 s (1e-5). The achieved error is ~ the tolerance.
- Warm start of a *new* model from the previous converged state (not `reset()`): barotrope parameter changed by 0.1% -> 12 instead of 18 iterations (0.049 vs 0.089 s); by 1% -> 15; by 10% -> 16. Results agree to 1e-10. (Needs `baro_param_calc` set by hand; today `reset()` throws the state away.)

**Simpson vs trapezoid**
- Smooth profile (n=1 polytrope), error vs Wisdom & Hubbard Bessel, order 4: J2 at N=512: Simpson 8e-6, trapezoid 7e-5; at N=128: 5e-5 vs 1e-3. Simpson at N=128 is as good as the trapezoid at N=512. Cost of Simpson: +10-20% time. Above the truncation floor of the order (e.g. J6, J8 at order 4) both are the same.
- Density jump (5%-200%) or very steep step: Simpson's S_0-monotonicity check fires and PyToF falls back; result identical to forced trapezoid, no extra time. Works as intended.
- **Weak spot 1:** a steep but smooth step (tanh in pressure, width 5% of the range) does *not* trigger the fallback, and there Simpson is *worse* than the trapezoid at small N: J2 error N=256: 2.2e-4 vs 3.8e-5; N=512: 5.5e-5 vs 1.7e-5; equal only at N>=2048. (Cause: the Simpson branch integrates `integrand * np.gradient(rho)`, the finite-difference gradient of a steep rho is inaccurate; the trapezoid branch integrates `d(rho)` directly.)
- **Weak spot 2:** the fallback is sticky per object and `reset()` does not undo it. A planet object that once saw a jump model keeps using the trapezoid rule for later smooth models: J2 differs by 7.7e-5 from a fresh object (history-dependent results, relevant for serial MCMC chains).

## Suggestions (not done, your call)
1. For MCMC / scans: choose `dJ_tol`, `drot_tol`, `drho_tol` ~ the Js uncertainty you actually need (1e-6 saves ~45%), N=128-256 with Simpson instead of 1024 with trapezoid. Pure settings, no code change.
2. Warm start in `reset()` (keep `li`/`rhoi`/figure functions, only reset what the new parameters invalidate): -15% to -45% per model in a chain. Risk: result depends on the starting state at the 1e-10 level; needs a golden comparison.
3. Simpson: make the fallback state explicit per call (reset it in `reset()`), and consider an additional trigger for steep smooth profiles (e.g. compare the Simpson integral with the trapezoid one and fall back if they differ by more than X). Needs your judgement on X.
4. Order 10 only: the polynomial blocks (70%). Hypotheses, untested: common-subexpression elimination in the generator, fused evaluation (numexpr), or numba. Large effort for a niche case.
5. Convergence acceleration (Anderson) of the outer loop: untested hypothesis, potentially 2x; most invasive.
6. Small stuff: ~8% of the order-4 time is `dataclasses.fields` called by `Options` access in the inner loops (cache the dict).

## STATUS (your decisions)
- `reset()` now undoes the Simpson fallback (`_simpson_failed = False` moved into it; test `test_reset_undoes_simpson_fallback`). 99 tests pass, golden numbers unchanged.
- Steep smooth profiles: left as they are (no extra safeguard). Tolerances: keep 1e-10 as default. Check at N=16384 vs Bessel: order 10 J2..J6 agree to 2e-9..4e-9, order 7 J2 to 2e-9, so 1e-10 is about one order below the best attainable accuracy; no change.
- Speed: considered fine for now; no further changes.
