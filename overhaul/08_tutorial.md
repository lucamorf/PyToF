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
