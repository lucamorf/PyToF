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
