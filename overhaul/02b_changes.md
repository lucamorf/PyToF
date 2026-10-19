# 02b — Concrete changes (nothing applied yet)

## A. Compatibility fixes (must keep golden tests bit-identical)
1. **`simpson` keyword** — 5 sites:
```diff
ClassToF.py:172   simpson(rho_pre * self.li**2, self.li)      -> simpson(rho_pre * self.li**2, x=self.li)
FunctionsToF.py:26   simpson(class_obj.rhoi * class_obj.li**2, class_obj.li)  -> ..., x=class_obj.li)
FunctionsToF.py:502  simpson(integrand_l_theta, mu, axis=1)     -> simpson(integrand_l_theta, x=mu, axis=1)
FunctionsToF.py:503  simpson(integrand_l, class_obj.li)         -> simpson(integrand_l, x=class_obj.li)
AlgoToF.py:102       simpson(rho * mean_l**2, mean_l)           -> simpson(rho * mean_l**2, x=mean_l)
```
2. **Huge integer literals → floats**: regex `(?<![\d.eE_])(\d{19,})(?![\d.eE])` → `\1.0` on `AlgoToF.py` (2373 literals). Also make `Mathematica/convert.py` emit floats so regenerated code stays compatible (I'll show that diff when we get there).

## B. Cleanup (no behaviour change)
3. Delete the 116 pure-expression statements in `AlgoToF.py` (lines ~259–283, 341–387, 1336–1398; found by `ast`, listing available). Tests prove nothing moved; review §3 measured no speed change.
4. `PyToF/__init__.py`: `__version__` read from package metadata (currently "1.5.3" vs setup.py 1.7.0).
5. `git rm --cached PyToF/__pycache__/*.pyc`, add `__pycache__/` to `.gitignore`.
6. Tests: `np.trapezoid` → `scipy.integrate.trapezoid`.

## C. Packaging: `pyproject.toml` replaces `setup.py`
```toml
[build-system]
requires = ["setuptools>=64"]
build-backend = "setuptools.build_meta"

[project]
name = "PyToF"
version = "1.7.0"
description = "Theory of Figures (4th, 7th, 10th order) incl. barotropic differential rotation"
readme = "README.md"
license = { text = "MPL-2.0" }
authors = [{ name = "Luca Morf", email = "luca.morf@uzh.ch" }]
requires-python = ">=3.10"
dependencies = ["numpy>=1.26", "scipy>=1.12", "matplotlib>=3.7", "emcee>=3.1", "tqdm>=4.60"]

[project.optional-dependencies]
test = ["pytest>=7"]
dev  = ["pytest>=7", "ruff"]

[tool.setuptools.packages.find]
include = ["PyToF*"]
[tool.setuptools.package-data]
PyToF = ["*.npz"]

[tool.ruff]
target-version = "py310"
```
(existing `PyToF/ruff.toml` settings get merged into `[tool.ruff]` — I'll show that diff.)
`matplotlib` stays a dependency until PlotToF is gone (step 06), then becomes optional.

## D. CI — `.github/workflows/tests.yml`
| Job | Python | NumPy / SciPy | Purpose |
|---|---|---|---|
| oldest | 3.10 | 1.26.4 / 1.12.0 | floor still works |
| middle | 3.12 | 2.2.6 / 1.15.3 | the reference |
| newest | 3.13 | latest / latest | newest |
| weekly (cron) | 3.13 | latest, unpinned | early warning, e.g. the SciPy 1.14 `simpson` break |

Each job: `pip install -e .[test]` + `pytest` + `ruff check`. Golden tolerance stays rtol 1e-9.

## Verification plan
Apply A → rerun `pytest` (39 must pass, golden unchanged) → rerun the version matrix → then B, C. One commit per letter.
