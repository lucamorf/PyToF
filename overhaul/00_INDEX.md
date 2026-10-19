# PyToF overhaul — roadmap

Each step below has its own short proposal file. **Nothing is changed in code
until you approve that file.** Review one at a time; say "ok", "change X" or "skip".

## Ground rules
1. **Tests before refactors.** Step 01 freezes today's numerical output, so every
   later step can prove "no physics changed".
2. **Behaviour-preserving and behaviour-changing edits never share a commit.**
3. One step = one commit (or a few), always with tests green.
4. Every claim in a proposal is either measured or marked *hypothesis*.

## Steps (in order)

| # | File | What | Risk |
|---|---|---|---|
| 01 | `01_tests.md` | pytest suite: analytical, golden-master, literature | none |
| 02 | `02_housekeeping.md` | versions, setup.py, dead code, ruff, your uncommitted work | low |
| 03 | `03_class_structure.md` | mixins instead of `partial`, split `opts`, strict kwargs | medium |
| 04 | `04_deduplicate.md` | merge `relax_to_*` and MCMC duplicates | medium |
| 05 | `05_mcmc_robustness.md` | retry caps, fork side effect, pickling note | low |
| 06 | `06_plotting.md` | what to do with `PlotToF` / `PlotDebugToF` | low |
| 07 | `07_speed.md` | measure-first speed work | medium |
| 08 | `08_tutorial.md` | tutorial rewrite | none (docs) |
| 09 | `09_accuracy_notebook.md` | accuracy notebook + a fast CI-sized version | none (docs) |
| 10 | `10_love_and_docs.md` | `love_dev` fate, README, docstrings | none |

## Decisions I need from you early
- **Python versions to support** (3.10 only? 3.10–3.13?) — affects 02 and 03.
- **Is the public API frozen?** (`ToF(**opts)`, `X.relax_to_barotrope()` …) I assume yes.
- Keep `PlotToF` as a public module, or deprecate it? — see 06.

Start with `01_tests.md`.
