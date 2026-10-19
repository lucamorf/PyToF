# 03b — Options

Commit `f183999`. Full diff: `03_diff_options.diff`.

## What changed
- **One place** (`PyToF/options.py`) for defaults, types, one-line docs and groups of all 31 options; `ToF._default_opts` (110 lines of dict + comments) is gone.
- `X.opts["N"]`, `X.opts["N"] = 5`, `in`, `.get`, `.keys()`, `.items()`, `.copy()` all still work (that is everything the notebooks use).
- **Strict:** `ToF(Verbosity=2)` → `TypeError: 'Verbosity' is not a PyToF option. Did you mean 'verbosity'?` (before: a printed warning and the typo was silently ignored). Same for `X.opts["ordr"] = 7`.
- **Validated, also after construction:** `order` in (4, 7, 10); `R_phys` = `[value, "equatorial"|"mean"|"polar"]`; `verbosity` a non-negative int. (Before: bare `assert`s, which vanish under `python -O`, and `len()` TypeError for `R_phys=7e7`.)
- **State out of options:** the Simpson→trapezoid fallback used to overwrite `opts["use_simpson"]` (and the `n_bin` of `get_Js_errors` overwrote `opts["n_bin"]`). Now `opts` only holds what you asked for; the fallback is `X.simpson_in_use` (a property: option and not failed). The fallback still persists across calls exactly as before (needed for MCMC stability).
- Defaults are **bit-identical** (golden tests unchanged).
- `Options.describe()` prints the README options table, so docs cannot drift.

## Not changed (on purpose)
- Option *names* (see 03d).
- `AlgoToF.default_opts()` still exists for direct `Algorithm(...)` calls; ToF passes every key explicitly, so it cannot disagree. It can go when `Algorithm` gets a proper signature.
- Defaults are still the Wisdom & Hubbard Jupiter values (changing them would silently alter every user's script).

### `PyToF/options.py`
```python
"""
Options of the ToF class.

All options are fields of :class:`Options` with a default, a one-line
description and a group. They are set via the keyword arguments of ``ToF``
and can be read and changed afterwards like dictionary entries::

    X = ToF(N=2**12, order=7)
    X.opts["order"]            # 7
    X.opts["verbosity"] = 2    # validated, unknown names are rejected

``Options.describe()`` returns the table of all options as Markdown.

Options only ever hold what the *user* asked for. Quantities that PyToF
determines while running (e.g. a fallback to the trapezoid rule) are stored
on the ToF object, never here.
"""

import copy
import difflib
from dataclasses import MISSING, dataclass, field, fields

import numpy as np

# Defaults follow the Jupiter benchmark of Wisdom & Hubbard (2016),
# "Differential rotation in Jupiter: A comparison of methods", Icarus 267, 315.
_G = 6.6738480e-11  # m^3 kg^-1 s^-2
_GM = 126686536.1 * (1000) ** 3  # m^3 s^-2
_RADIUS = 71492000  # m, equatorial
_ROTATIONAL_PARAMETER = 0.089195487
_JS = [
    1.398851089834702e-2,
    -5.318281001092907e-4,
    3.011832290533641e-5,
    -2.132115710725050e-6,
    1.740671195871128e-7,
    -1.568219505602588e-8,
    1.518099230068580e-9,
    -1.551985393081485e-10,
    1.655948019619652e-11,
    -1.829544870258362e-12,
]

RADIUS_KINDS = ("equatorial", "mean", "polar")
ORDERS = (4, 7, 10)


def _option(default, doc, group, factory=False):
    if factory:
        return field(
            default_factory=default, metadata={"doc": doc, "group": group}
        )
    return field(default=default, metadata={"doc": doc, "group": group})


@dataclass
class Options:
    # ------------------------------------------------------------ numerics
    N: int = _option(
        2**10, "Number of grid points (level surfaces).", "numerics"
    )
    n_bin: int = _option(
        -1,
        "Number of bins for the figure functions; -1 or N means no binning, "
        "smaller values are faster but less accurate.",
        "numerics",
    )
    order: int = _option(4, "Theory of Figures order: 4, 7 or 10.", "numerics")
    use_simpson: bool = _option(
        True,
        "Use Simpson's rule (otherwise the trapezoid rule) for the pressure "
        "and AlgoToF integrals. PyToF falls back to the trapezoid rule on "
        "its own if Simpson's rule becomes unstable, see ToF.simpson_in_use.",
        "numerics",
    )
    dJ_tol: float = _option(
        1e-10, "Relative tolerance of the gravitational moments.", "numerics"
    )
    drot_tol: float = _option(
        1e-10, "Relative tolerance of the rotational parameter.", "numerics"
    )
    drho_tol: float = _option(
        1e-10, "Relative tolerance of the densities.", "numerics"
    )
    MaxIterShape: int = _option(
        100, "Maximum iterations of relax_to_shape().", "numerics"
    )
    MaxIterUpdate: int = _option(
        100,
        "Maximum iterations of the density update (barotrope, atmosphere) "
        "within one step of relax_to_barotrope()/relax_to_density().",
        "numerics",
    )
    MaxIterBar: int = _option(
        100, "Maximum iterations of relax_to_barotrope().", "numerics"
    )
    MaxIterDen: int = _option(
        100, "Maximum iterations of relax_to_density().", "numerics"
    )
    # -------------------------------------------------------------- output
    verbosity: int = _option(
        0,
        "0: quiet (warnings are always issued); 1: notices; 2: timings and "
        "diagnostics; 3: timing of every shape relaxation.",
        "output",
    )
    debug_plot: bool = _option(
        False, "Save debug plots at every iteration (very slow).", "output"
    )
    # -------------------------------------------------------------- planet
    G: float = _option(_G, "Gravitational constant in SI units.", "planet")
    M_phys: float = _option(
        _GM / _G, "Mass of the planet in SI units.", "planet"
    )
    R_phys: list = _option(
        lambda: [_RADIUS, "equatorial"],
        "Physical radius in SI units and its kind: [value, 'equatorial' | "
        "'mean' | 'polar'].",
        "planet",
        factory=True,
    )
    R_ref: float = _option(
        None,
        "Reference radius for the definition of the Js in SI units; "
        "defaults to the value in R_phys.",
        "planet",
    )
    Period: float = _option(
        2
        * np.pi
        / np.sqrt(
            (126686536.1 * (1000) ** 3 * _ROTATIONAL_PARAMETER) / (_RADIUS**3)
        ),
        "Rotation period in SI units.",
        "planet",
    )
    P0: float = _option(0, "Surface pressure in SI units.", "planet")
    Target_Js: np.ndarray = _option(
        lambda: np.array(_JS),
        "Observed gravitational moments J2, J4, ... used by the cost "
        "function.",
        "planet",
        factory=True,
    )
    Sigma_Js: np.ndarray = _option(
        lambda: np.array([np.spacing(j) for j in _JS]),
        "1-sigma uncertainties of Target_Js.",
        "planet",
        factory=True,
    )
    alphas: np.ndarray = _option(
        lambda: np.zeros(12),
        "Barotropic differential rotation parameters in m^2 s^-2 "
        "(typically ~1e5; much smaller values have no visible effect).",
        "planet",
        factory=True,
    )
    # --------------------------------------------------------------- model
    rho_MAX: float = _option(
        2e4, "Largest physically acceptable density in SI units.", "model"
    )
    P_MAX: float = _option(
        1e13, "Largest physically acceptable pressure in SI units.", "model"
    )
    baro_param_init: list = _option(
        None,
        "Parameters of the barotrope set via set_barotrope().",
        "model",
    )
    dens_param_init: list = _option(
        None,
        "Parameters of the density function set via set_density_function().",
        "model",
    )
    use_atmosphere: bool = _option(
        False,
        "Use an atmospheric model for the outermost densities.",
        "model",
    )
    atmosphere: object = _option(
        None,
        "Function atmosphere(mean_level_surface, pressure) returning the "
        "density of the atmospheric model.",
        "model",
    )
    atmosphere_until: float = _option(
        None,
        "Pressure in SI units at which the atmospheric model may end "
        "earliest.",
        "model",
    )
    H: float = _option(0.0, "WORK IN PROGRESS", "model")

    # ------------------------------------------------------------ creation
    @classmethod
    def from_kwargs(cls, **kwargs):
        """Like ``Options(**kwargs)``, with a helpful unknown-name error."""

        for name in kwargs:
            cls._check_name(name)

        return cls(**kwargs)

    @classmethod
    def names(cls):
        return [f.name for f in fields(cls)]

    @classmethod
    def _check_name(cls, name):
        if name in cls.names():
            return

        message = f"{name!r} is not a PyToF option."
        close = difflib.get_close_matches(name, cls.names(), n=3, cutoff=0.5)
        if close:
            message += " Did you mean " + " or ".join(map(repr, close)) + "?"

        raise TypeError(message)

    # ---------------------------------------------------------- validation
    def __setattr__(self, name, value):
        self._check_name(name)
        validate = getattr(self, "_validate_" + name, None)
        if validate is not None:
            validate(value)

        super().__setattr__(name, value)

    @staticmethod
    def _validate_order(value):
        if value not in ORDERS:
            raise ValueError(f"order must be one of {ORDERS}, not {value!r}.")

    @staticmethod
    def _validate_R_phys(value):
        try:
            radius, kind = value
        except (TypeError, ValueError):
            raise TypeError(
                "R_phys must be a pair [radius in SI units, kind] with kind "
                f"in {RADIUS_KINDS}, not {value!r}."
            ) from None

        if kind not in RADIUS_KINDS:
            raise ValueError(
                f"The second entry of R_phys must be one of {RADIUS_KINDS}, "
                f"not {kind!r}."
            )

    @staticmethod
    def _validate_verbosity(value):
        if not isinstance(value, (int, np.integer)) or value < 0:
            raise ValueError(
                f"verbosity must be a non-negative integer, not {value!r}."
            )

    # ------------------------------------------------------- dict interface
    def __getitem__(self, name):
        self._check_name(name)
        return getattr(self, name)

    def __setitem__(self, name, value):
        setattr(self, name, value)

    def __contains__(self, name):
        return name in self.names()

    def keys(self):
        return self.names()

    def items(self):
        return [(name, getattr(self, name)) for name in self.names()]

    def get(self, name, default=None):
        return getattr(self, name) if name in self else default

    def copy(self):
        return copy.copy(self)

    # ---------------------------------------------------------- documentation
    @classmethod
    def describe(cls):
        """All options as a Markdown table (for the README)."""

        rows = [
            "| group | option | default | description |",
            "|---|---|---|---|",
        ]
        for f in fields(cls):
            default = (
                f.default if f.default is not MISSING else f.default_factory()
            )
            if isinstance(default, np.ndarray):
                text = f"array of length {len(default)}"
            else:
                text = repr(default)
            rows.append(
                f"| {f.metadata['group']} | `{f.name}` | {text[:40]} | "
                f"{f.metadata['doc']} |"
            )

        return "\n".join(rows)
```

### `tests/test_options.py`
```python
"""The Options object: defaults, dict interface, validation, state."""

import pickle

import numpy as np
import pytest

from PyToF import ClassToF
from PyToF.options import Options
from tests.cases import jupiter_polytrope

# -------------------------------------------------------------- defaults


def test_defaults_are_the_wisdom_hubbard_jupiter():
    opts = Options()
    assert opts["N"] == 1024
    assert opts["order"] == 4
    assert opts["R_phys"] == [71492000, "equatorial"]
    assert opts["M_phys"] == 126686536.1 * (1000) ** 3 / 6.6738480e-11
    assert opts["Target_Js"][0] == 1.398851089834702e-2
    assert opts["alphas"].shape == (12,)


def test_array_defaults_are_not_shared_between_instances():
    a, b = Options(), Options()
    a["Target_Js"][0] = 0.0
    a["R_phys"][0] = 1.0
    assert b["Target_Js"][0] != 0.0
    assert b["R_phys"][0] == 71492000


# -------------------------------------------------------- dict interface


def test_dict_interface():
    opts = Options.from_kwargs(N=64, order=7)
    assert opts["N"] == 64 and opts.get("order") == 7
    assert "verbosity" in opts and "nonsense" not in opts
    assert opts.get("nonsense", "fallback") == "fallback"
    assert dict(opts.items())["N"] == 64
    assert set(opts.keys()) == set(Options.names())

    opts["N"] = 128
    assert opts["N"] == 128


def test_copy_is_independent():
    a = Options()
    b = a.copy()
    b["N"] = 8
    assert a["N"] == 1024


def test_options_can_be_pickled():
    # Needed for multiprocessing in the MCMC.
    opts = Options.from_kwargs(N=64, atmosphere=np.sqrt)
    assert pickle.loads(pickle.dumps(opts))["N"] == 64


# ------------------------------------------------------------ validation


def test_unknown_option_suggests_the_right_name():
    with pytest.raises(TypeError, match="Did you mean 'verbosity'"):
        ClassToF.ToF(Verbosity=2)


def test_unknown_name_cannot_be_read_or_written():
    opts = Options()
    with pytest.raises(TypeError, match="not a PyToF option"):
        opts["ordr"]
    with pytest.raises(TypeError, match="Did you mean 'order'"):
        opts["ordr"] = 7


@pytest.mark.parametrize(
    "kwargs, error",
    [
        ({"order": 5}, ValueError),
        ({"R_phys": [7e7, "diameter"]}, ValueError),
        ({"R_phys": 7e7}, TypeError),
        ({"R_phys": [7e7]}, TypeError),
        ({"verbosity": -1}, ValueError),
        ({"verbosity": "loud"}, ValueError),
    ],
)
def test_invalid_values_are_rejected(kwargs, error):
    with pytest.raises(error):
        ClassToF.ToF(**kwargs)


def test_validation_also_applies_after_construction():
    X = ClassToF.ToF()
    with pytest.raises(ValueError):
        X.opts["order"] = 5
    assert X.opts["order"] == 4


# ----------------------------------------------------------------- state


def test_reference_radius_defaults_to_the_physical_radius():
    X = ClassToF.ToF(R_phys=[1e6, "mean"])
    assert X.opts["R_ref"] == 1e6


def test_relaxing_does_not_change_options():
    X = ClassToF.ToF(N=64, order=4)
    before = {k: v for k, v in X.opts.items() if np.isscalar(v)}
    X.set_barotrope(lambda P, p: np.sqrt(P))
    X.relax_to_barotrope()
    X.get_Js_errors()
    after = {k: v for k, v in X.opts.items() if np.isscalar(v)}
    assert after == before  # in particular n_bin stays -1


def test_simpson_fallback_is_state_not_an_option():
    N = 128
    X = ClassToF.ToF(
        N=N, M_phys=1e24, R_phys=[1e6, "mean"], Period=np.inf, order=4
    )
    assert X.simpson_in_use
    # A non-uniform grid forces the trapezoid rule:
    X.li = 1e6 * np.linspace(1, 1e-3, N) ** 1.5
    X.rhoi = 1000.0 * np.ones(N)
    X.m_rot_calc = 0.0
    X.relax_to_shape()

    assert not X.simpson_in_use
    assert X.opts["use_simpson"] is True  # what the user asked for

    X.opts["use_simpson"] = False  # user override still takes effect
    assert not X.simpson_in_use


def test_user_can_switch_simpson_off_after_construction():
    X = jupiter_polytrope(4, 64, use_simpson=False)
    assert not X.simpson_in_use


# ---------------------------------------------------------- documentation


def test_every_option_is_documented_and_grouped():
    table = Options.describe()
    for name in Options.names():
        assert f"`{name}`" in table
    from dataclasses import fields

    for f in fields(Options):
        assert f.metadata["doc"] and f.metadata["group"], f.name
```
