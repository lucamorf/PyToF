# 09 — Accuracy & convergence notebook

## Now
Solid design (analytical → Love → N-sweep vs Movshovitz and Bessel solutions) but **no markdown narration**; two ~80-line near-identical plot functions; `CMSDR2/3` data uncited.

## Proposal
1. Add 2–4 sentences per section: *what is tested, what the reference is, how to read the plot, what "good" looks like*.
2. Merge `plot_res_nobin`/`plot_res_bin` into one function.
3. Cite every literature array (CMSDR2/3, Gavrilov et al. tables).
4. Add a **summary table at the top**: order × recommended N × achieved |ΔJ| — the answer a user wants.
5. Split in two:
   - `accuracy_quick` (<1 min): analytical checks + small sweep; runs in CI.
   - `accuracy_full` (multi-minute): the existing full sweep, run manually before releases.
6. Share its checks with `tests/` (same reference numbers, one place) so notebook and tests can't disagree.
7. Regenerate `accuracy_data_*.npz` through a script (`tools/make_accuracy_data.py`) so the error-estimate files are reproducible; document what they contain.

8. **`R_eq_to_R_m` mystery (first item to investigate):** PyToF gives 1.02248 for the W&H n=1 polytrope, the Bessel reference 1.02288 (rel. diff 3.9e-4, independent of N and order, while J2 agrees to ~1e-6). Plan: (a) compute the reference's radius ratio from its own definition, (b) compare with Movshovitz's `aos` at the same N, (c) check whether the difference is a definition of the "mean" radius (s0 vs. volume-equivalent) or a real bug, (d) turn the outcome into a test.


---

## STATUS: done, except the `R_eq_to_R_m` investigation (still on TODO)
Done (items 1-7 of the proposal):
- `accuracy/references.py`: all literature numbers with citations (Bessel, CMS DR2/DR3, wind parameters, analytical NMoI, analytical Love functions), single source for notebooks and tests (`tests/cases.py` imports from it).
- `accuracy/helpers.py`: run PyToF / Movshovitz, parallel sweeps, ONE plotting function (was two ~80-line copies), `error_tables`, `write_accuracy_data`; also a CLI: `python accuracy/helpers.py --write-accuracy-data`.
- `accuracy/accuracy_quick.ipynb` (~20 s): summary of the error tables, NMoI, Love numbers, small Bessel sweep, DR2 vs CMS, reproducibility of the shipped tables. Narrated.
- `accuracy/accuracy_full.ipynb` (~5 min on 20 cores): the full sweeps (no binning + Movshovitz, binning, DR2, DR3), regenerates `accuracy/images/`; the shipped `PyToF/accuracy_data_*.npz` are only overwritten with `WRITE_ACCURACY_DATA = True` (the old notebook overwrote them unconditionally).
- Result: the recomputed tables (all of N x binning x order, Simpson and trapezoid) agree with the shipped tables bit for bit (largest relative deviation 0.0), so the accuracy tests still work and the tables belong to the current code.
- Images moved to `accuracy/images/`; README updated.
