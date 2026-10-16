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

from PyToF.messages import RenamedWarning, warn

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

# Options that were renamed. The old names keep working but warn.
RENAMED = {
    "MaxIterShape": "max_iter_shape",
    "MaxIterUpdate": "max_iter_update",
    "MaxIterBar": "max_iter_barotrope",
    "MaxIterDen": "max_iter_density",
    "Target_Js": "target_Js",
    "Sigma_Js": "sigma_Js",
    "rho_MAX": "rho_max",
    "P_MAX": "P_max",
}

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
    max_iter_shape: int = _option(
        100, "Maximum iterations of relax_to_shape().", "numerics"
    )
    max_iter_update: int = _option(
        100,
        "Maximum iterations of the density update (barotrope, atmosphere) "
        "within one step of relax_to_barotrope()/relax_to_density().",
        "numerics",
    )
    max_iter_barotrope: int = _option(
        100, "Maximum iterations of relax_to_barotrope().", "numerics"
    )
    max_iter_density: int = _option(
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
    target_Js: np.ndarray = _option(
        lambda: np.array(_JS),
        "Observed gravitational moments J2, J4, ... used by the cost "
        "function.",
        "planet",
        factory=True,
    )
    sigma_Js: np.ndarray = _option(
        lambda: np.array([np.spacing(j) for j in _JS]),
        "1-sigma uncertainties of target_Js.",
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
    rho_max: float = _option(
        2e4, "Largest physically acceptable density in SI units.", "model"
    )
    P_max: float = _option(
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
        """Like ``Options(**kwargs)``, with helpful errors for bad names."""

        resolved = {}
        for name, value in kwargs.items():
            new_name = cls._resolve(name, stacklevel=4)
            if new_name in resolved:
                raise TypeError(
                    f"{name!r} and {new_name!r} are the same option."
                )
            cls._check_name(new_name)
            resolved[new_name] = value

        return cls(**resolved)

    @classmethod
    def _resolve(cls, name, stacklevel=3):
        """Translate a renamed option to its new name (with a warning)."""

        if name not in RENAMED:
            return name

        warn(
            "The option {old!r} was renamed to {new!r}. The old name still "
            "works but will be removed in a future version.",
            RenamedWarning,
            stacklevel=stacklevel,
            old=name,
            new=RENAMED[name],
        )

        return RENAMED[name]

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
        name = self._resolve(name)
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
    def _validate_max_iteration_count(name, value):
        # The relaxation loops count from 1 and run while the counter is below
        # the limit, so a limit of 1 would never run them at all.
        if not isinstance(value, (int, np.integer)) or value < 2:
            raise ValueError(f"{name} must be an integer >= 2, not {value!r}.")

    @classmethod
    def _validate_max_iter_update(cls, value):
        cls._validate_max_iteration_count("max_iter_update", value)

    @classmethod
    def _validate_max_iter_barotrope(cls, value):
        cls._validate_max_iteration_count("max_iter_barotrope", value)

    @classmethod
    def _validate_max_iter_density(cls, value):
        cls._validate_max_iteration_count("max_iter_density", value)

    @staticmethod
    def _validate_verbosity(value):
        if not isinstance(value, (int, np.integer)) or value < 0:
            raise ValueError(
                f"verbosity must be a non-negative integer, not {value!r}."
            )

    # ------------------------------------------------------- dict interface
    def __getattr__(self, name):
        # Only called if normal lookup failed: serves renamed options.
        if name in RENAMED:
            return getattr(self, self._resolve(name))
        raise AttributeError(name)

    def __getitem__(self, name):
        name = self._resolve(name)
        self._check_name(name)
        return getattr(self, name)

    def __setitem__(self, name, value):
        setattr(self, name, value)

    def __contains__(self, name):
        return RENAMED.get(name, name) in self.names()

    def keys(self):
        return self.names()

    def items(self):
        return [(name, getattr(self, name)) for name in self.names()]

    def get(self, name, default=None):
        return self[name] if name in self else default

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
