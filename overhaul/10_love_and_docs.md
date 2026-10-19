# 10 — Love numbers notebook, README, docstrings

## `love_dev.ipynb`
No markdown, uncited tables, overlaps the accuracy notebook's Love cell.
**Proposal:** after you confirm the `get_Kn` change (02), merge into the accuracy notebook as a "Love numbers" section with citations (Gavrilov, Zharkov & Leontev 1976), and delete `love_dev.ipynb` — or keep it under `dev/` clearly marked. Corresponding unit test in 01.

## README
- Install (pip, extras), 10-line quickstart, link to tutorial/accuracy notebooks, supported Python versions, citation (Morf, Müller & Helled 2024, A&A 690, A105), changelog link.
- Move the options table out of a code comment: generate it from the `Options` dataclass (03) so docs can't drift.

## Docstrings & types
- NumPy-style docstrings with Parameters / Returns on all public methods.
- Type hints on public API only (not on the generated polynomial blocks).
- `CHANGELOG.md` starting with this overhaul; note the breaking changes (strict kwargs, plot removal if chosen).

## Paper-text check
One-line check: Morf+2024 says "trapezoid" for the mass integral, code uses Simpson. Needs your eye on the paper.
