########################################################
# Author of this version: Luca Morf - luca.morf@uzh.ch #
########################################################

import numpy as np
import scipy

from PyToF.FunctionsToF import (
    get_Js_errors,
    get_Kn,
    get_NMoI,
    get_r_l_mu,
    get_U_l_mu,
    relax_to_barotrope,
    relax_to_density,
    relax_to_shape,
    set_barotrope,
)
from PyToF.messages import RenamedWarning, say, warn
from PyToF.options import Options


def _renamed_attribute(old, new):
    """A property making the attribute ``old`` an alias of ``new`` (warns)."""

    def message():
        warn(
            "ToF.{old} was renamed to ToF.{new}. The old name still works but "
            "will be removed in a future version.",
            RenamedWarning,
            stacklevel=3,
            old=old,
            new=new,
        )

    def getter(self):
        message()
        return getattr(self, new)

    def setter(self, value):
        message()
        setattr(self, new, value)

    return property(getter, setter, doc=f"Old name of {new}.")


class ToF:
    """
    This class contains standard routines when using the Theory of Figures.

    The routines themselves live in FunctionsToF.py; they take the ToF
    object as their first argument and are attached below as ordinary
    methods.
    """

    # Shape and density relaxation:
    relax_to_shape = relax_to_shape
    relax_to_barotrope = relax_to_barotrope
    relax_to_density = relax_to_density
    set_barotrope = set_barotrope

    # Derived quantities:
    get_Js_errors = get_Js_errors
    get_NMoI = get_NMoI
    get_Kn = get_Kn
    get_r_l_mu = get_r_l_mu
    get_U_l_mu = get_U_l_mu

    def reset(self):
        """
        Resets all results to the initial conditions given by the options, so
        that a new relaxation starts from scratch. Does not touch the options.
        A fallback from Simpson's rule to the trapezoid rule (simpson_in_use)
        is undone, so the result of a calculation does not depend on the
        models calculated before with the same object.

        Sets...

        - li:               Array, level surfaces l, Eq. (B.1) in
                            arXiv:1708.06177v1
        - rhoi:             Array, densities at the level surfaces li
        - Pi:               Array, pressures at the level surfaces li
        - Js:               Array, gravitational harmonics, Eq. (B.11) in
                            arXiv:1708.06177v1
        - Js_errors:         Array, estimated errors of the gravitational
                            harmonics Js, only if no binning or differential
                            rotation is used
        - R_calc            Float, calculated equatorial radius in SI units
        - m_rot_calc:       Float, dimensionless rotational parameter
        - figure_functions: List of arrays, figure functions s_n introduced in
                            Eq. (B.1) in arXiv:1708.06177v1
        - volume_integrals: List of arrays, dimensionless volume integrals S_n
                            introduced in Eq. (B.7) in arXiv:1708.06177v1,
                            followed by the primed integrals S'_n
        - A0:               Array, important to determine the total potential
                            introduced in Eq. (B.3) in arXiv:1708.06177v1
        - R_eq_to_R_m:      Float, ratio of the equatorial radius to the to
                            outermost mean surface layer
        - R_po_to_R_m:      Float, ratio of the polar radius to the to
                            outermost mean surface layer
        - simpson_in_use:   True again if use_simpson is True (see above)
        - atmosphere_index: Integer, defines the index of atmosphere transition
        """

        self.li = (
            np.linspace(1, 1 / self.opts["N"], self.opts["N"])
            * self.opts["R_phys"][0]
        )
        rho_pre = np.linspace(0, 1, self.opts["N"])
        self.rhoi = (
            rho_pre
            * self.opts["M_phys"]
            / (
                -4
                * np.pi
                * scipy.integrate.simpson(rho_pre * self.li**2, x=self.li)
            )
        )
        self._simpson_failed = False
        self.Pi = np.zeros(self.opts["N"])
        self.Js = np.hstack((-1, np.zeros(self.opts["order"])))
        self.Js_errors = np.hstack(
            (np.nan, np.nan * np.zeros(self.opts["order"]))
        )
        self.R_calc = self.opts["R_phys"][0]
        self.m_rot_calc = (
            (2 * np.pi / self.opts["Period"]) ** 2
            * self.li[0] ** 3
            / (self.opts["G"] * self.opts["M_phys"])
        )
        self.figure_functions = (self.opts["order"] + 1) * [
            np.zeros(self.opts["N"])
        ]
        self.volume_integrals = (self.opts["order"] + 1) * [
            np.zeros(self.opts["N"])
        ]
        self.A0 = np.zeros(self.opts["N"])
        self.U = np.zeros(self.opts["N"])
        self.R_eq_to_R_m = 1.0
        self.R_po_to_R_m = 1.0
        self.atmosphere_index = 0

    def __init__(self, **kwargs):
        """
        Initializes the ToF class. All options, see PyToF.options.Options,
        can be provided as keyword arguments, everything else is set to the
        defaults. Unknown or invalid options raise an error.
        """

        self.opts = Options.from_kwargs(**kwargs)

        if self.opts["R_ref"] is None:
            self.opts["R_ref"] = self.opts["R_phys"][0]

            say(
                self.opts["verbosity"],
                1,
                "No reference radius supplied by the user. PyToF assumes "
                "R_ref = R_phys.",
                kind="warn",
            )

        # Set initial conditions:
        self.reset()

    @property
    def simpson_in_use(self):
        """
        Whether Simpson's rule is currently used for the integrals: the
        option use_simpson, unless PyToF had to give it up for stability.
        """

        return bool(self.opts["use_simpson"]) and not self._simpson_failed

    # ---------------------------------------------------- renamed, still work
    def _set_IC(self):
        """Old name of reset()."""

        warn(
            "ToF._set_IC() was renamed to ToF.reset(). The old name still "
            "works but will be removed in a future version.",
            RenamedWarning,
        )
        self.reset()

    @property
    def baro_param_calc(self):
        """Old name of opts["baro_param"]."""

        warn(
            "ToF.baro_param_calc was replaced by ToF.opts['baro_param']. The "
            "old name still works but will be removed in a future version.",
            RenamedWarning,
            stacklevel=2,
        )
        return self.opts["baro_param"]

    @baro_param_calc.setter
    def baro_param_calc(self, value):
        warn(
            "ToF.baro_param_calc was replaced by ToF.opts['baro_param']. The "
            "old name still works but will be removed in a future version.",
            RenamedWarning,
            stacklevel=2,
        )
        self.opts["baro_param"] = value

    Js_error = _renamed_attribute("Js_error", "Js_errors")
    ss = _renamed_attribute("ss", "figure_functions")
    SS = _renamed_attribute("SS", "volume_integrals")
