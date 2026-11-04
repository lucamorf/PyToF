########################################################
# Author of this version: Luca Morf - luca.morf@uzh.ch #
########################################################

import time
from pathlib import Path

import numpy as np
import scipy

import PyToF.AlgoToF as AlgoToF
from PyToF.messages import (
    AccuracyWarning,
    ConvergenceWarning,
    PyToFWarning,
    say,
    warn,
    warn_simpson_fallback,
    warn_surface_slope,
)

# Iteration of _relax() in which _check_surface_slope() looks at the densities:
SURFACE_CHECK_ITERATION = 3


def _mass_int(tof):
    """
    Calculates and returns the total mass implied by the level surfaces
    tof.li and densities tof.rhoi using the simpson rule.
    """

    # Negative sign because beginning of the array is the outer surface:
    return -4 * np.pi * scipy.integrate.simpson(tof.rhoi * tof.li**2, x=tof.li)


def _fixradius(tof):
    """
    Renormalizes the level surfaces tof.li for consistency with the
    initially provided physical radius.
    """

    # Renormalize the level surfaces in such a way that the newly calculated
    # R_calc equatorial radius is the same as the initial one:
    if tof.opts["R_phys"][1] == "equatorial":
        tof.li = (tof.li / tof.li[0]) * (
            tof.opts["R_phys"][0] / tof.R_eq_to_R_m
        )
        tof.R_calc = tof.li[0] * tof.R_eq_to_R_m

    elif tof.opts["R_phys"][1] == "mean":
        tof.li = (tof.li / tof.li[0]) * tof.opts["R_phys"][0]
        tof.R_calc = tof.li[0]

    elif tof.opts["R_phys"][1] == "polar":
        tof.li = (tof.li / tof.li[0]) * (
            tof.opts["R_phys"][0] / tof.R_po_to_R_m
        )
        tof.R_calc = tof.li[0] * tof.R_po_to_R_m

    else:
        raise KeyError(
            "Invalid R_phys type specification! Valid options: 'equatorial', "
            "'mean', 'polar'"
        )

    assert np.isclose(tof.R_calc, tof.opts["R_phys"][0]), (
        "Renormalizing the level surfaces for consistency with the initially "
        "provided physical radius failed!"
    )


def _fixmass(tof):
    """
    Renormalizes the densities tof.rhoi for consistency with the
    initially provided mass.
    """

    # Renormalize the densities in such a way that the newly calculated mass is
    # the same as the initial one:
    tof.rhoi = tof.rhoi * tof.opts["M_phys"] / _mass_int(tof)

    # Sanity check:
    assert np.isclose(_mass_int(tof), tof.opts["M_phys"]), (
        "Renormalizing the densities for consistency with the initially "
        "provided mass failed!"
    )


def _fixrot(tof):
    """
    Renormalizes the rotational parameter for consistency with the initially
    provided period.
    """

    # We update the m_rot_calc parameter such that it is consistent with the
    # period, the outermost level surface and the calculated mass:
    tof.m_rot_calc = (
        (2 * np.pi / tof.opts["Period"]) ** 2
        * tof.li[0] ** 3
        / (tof.opts["G"] * _mass_int(tof))
    )


def _ensure_consistency(tof):
    """
    This function updates all variables necessary to ensure consistency with
    the initially provided physical values tof.opts['R_phys'] and
    tof.opts['M_phys'].
    """

    trace = tof.opts["verbosity"] >= 3
    if trace:
        radius_before = tof.li[0]

    # Changes the radii to be self-consistent with the provided radius
    # (affects mass and rotational parameter):
    _fixradius(tof)
    if trace:
        mass_factor = tof.opts["M_phys"] / _mass_int(tof)
    # Changes the densities to to be self-consistent with the provided mass
    # (affects rotational parameter):
    _fixmass(tof)
    if trace:
        # Shown as 1 +/- deviation, since the factors soon differ from 1 by
        # less than what a fixed number of decimals can show:
        radius_dev = tof.li[0] / radius_before - 1
        density_dev = mass_factor - 1
        say(
            tof.opts["verbosity"],
            3,
            "Renormalized the radii by a factor 1 {radius_sign} "
            "{radius_dev:.2e} and the densities by a factor 1 "
            "{density_sign} {density_dev:.2e}.",
            radius_sign="-" if radius_dev < 0 else "+",
            radius_dev=abs(radius_dev),
            density_sign="-" if density_dev < 0 else "+",
            density_dev=abs(density_dev),
        )
    # Changes the rotational parameter to be self-consistent
    # (affects nothing else):
    _fixrot(tof)


def _pressurize(tof):
    """
    Calculates the pressure tof.Pi at the level surfaces tof.li
    assuming hydrostatic equilibrium.
    """

    # The potential is positive (GM/r for a point mass), so dP = rho dU:
    tof.U = (
        tof.opts["G"]
        * tof.opts["M_phys"]
        / tof.li[0] ** 3
        * tof.li**2
        * tof.A0
    )

    tof.Pi = _pressure_profile(tof, tof.simpson_in_use)

    invalid = not (tof.Pi >= 0).all() or not (np.diff(tof.Pi) >= 0).all()

    if invalid and tof.opts["debug_plot"]:
        from PyToF.PlotDebugToF import debug_relax_plot

        debug_relax_plot(tof, failure=True)

    if tof.simpson_in_use and tof.opts["simpson_check"] and invalid:
        tof._simpson_failed = True
        warn_simpson_fallback(
            "the calculated pressure contains negative or non-monotonic "
            "entries."
        )
        _pressurize(tof)

    assert (tof.Pi >= 0).all(), (
        "Calculated pressure contains negative entries! If you switched off "
        "the option simpson_check, Simpson's rule may have failed."
    )

    assert (np.diff(tof.Pi) >= 0).all(), (
        "Calculated pressure is not monotonically increasing! If you "
        "switched off the option simpson_check, Simpson's rule may have "
        "failed."
    )


def _pressure_profile(tof, use_simpson):
    """
    Pressure at the level surfaces from the potential tof.U and the densities
    tof.rhoi by hydrostatic equilibrium, dP = rho dU, integrated from the
    surface pressure P0 inwards with Simpson's rule or the trapezoid rule.
    """

    integrand = tof.rhoi * np.gradient(tof.U, tof.li, edge_order=2)

    if use_simpson:
        return tof.opts["P0"] - scipy.integrate.cumulative_simpson(
            integrand, x=tof.li[::-1], initial=0.0
        )

    return tof.opts["P0"] + scipy.integrate.cumulative_trapezoid(
        integrand, x=tof.li, initial=0.0
    )


def _check_surface_slope(tof):
    """
    Warn if the density has an infinite slope at the surface. Near the surface
    rho ~ c s^n with s the distance from it; on the equidistant grid the first
    three points (s = 0, h, 2h) give (rho[2] - rho[1]) / (rho[1] - rho[0]) =
    2^n - 1, so n = log2(1 + that ratio). n = 1 for a finite slope, n < 1 for
    an infinite one, which Simpson's rule handles badly (it integrates
    np.gradient(rho)).

    Called once per relaxation in iteration SURFACE_CHECK_ITERATION, not in
    the first: the first densities follow from the pressure of the starting
    profile and show an exponent 2n / (n + 1) instead of n. By the third the
    estimate has settled; it stays slightly below the true n
    for n < 1 (0.86 for n = 0.8), so the default limit 0.9 flags n of about
    0.85 and below, which are the cases that converge badly. A surface
    pressure P0 much above ~1e6 Pa hides the singularity from the three
    outermost points (the estimate tends to 1), without curing the slow
    convergence.
    """

    threshold = tof.opts["surface_exponent_warn"]
    if not tof.simpson_in_use or threshold <= 0:
        return

    first = tof.rhoi[0] - tof.rhoi[1]
    second = tof.rhoi[1] - tof.rhoi[2]
    # Flat or decreasing outwards (e.g. a zero-density layer): no exponent.
    if not (first < 0 and second < 0):
        return

    exponent = np.log2(1 + second / first)
    if exponent < threshold:
        warn_surface_slope(exponent, threshold)


def _update_densities_barotrope(tof):
    """
    This function is called by relax_to_barotrope() and implements the
    barotrope model density = barotrope(pressure), i.e. tof.rhoi =
    tof.barotrope(tof.Pi, tof.opts["baro_param"]).
    """

    # Calculates the pressure values according to hydrostatic equilibrium:
    _pressurize(tof)

    # Set new densitites:
    tof.rhoi = tof.barotrope(tof.Pi, tof.opts["baro_param"])

    # Ensure physical mass stays unaffacted:
    _fixmass(tof)

    # Check for unphysical density inversions:
    if np.any(np.diff(tof.rhoi) < 0):
        raise ValueError("Barotrope created density inversion!")

    # Optional, use a provided atmospheric model:
    if tof.opts["use_atmosphere"]:
        _apply_atmosphere(tof)

        # Ensure physical mass stays unaffacted:
        _fixmass(tof)


def _apply_atmosphere(tof):
    """
    This function is called by e.g. _update_densities_barotrope() and
    implements the atmosphere model density = atmosphere(argument), i.e.
    tof.rhoi[specified by tof.opts['atmosphere_until']] =
    tof.opts['atmosphere'](tof.li[:index], tof.Pi[:index]).
    """

    # Define index that marks the transition from the atmosphere to the rest of
    # the model:
    index = np.arange(tof.opts["N"])[tof.Pi > tof.opts["atmosphere_until"]][0]
    old_index = tof.atmosphere_index
    # Follow the pressures, but stop following (and only let the transition
    # move inwards) as soon as it reverses direction: a transition that
    # flips between two level surfaces never converges.
    step = index - old_index
    if step * tof._atmosphere_last_step < 0:
        tof._atmosphere_locked = True
    if tof._atmosphere_locked:
        index = max(index, old_index)
    if step != 0:
        tof._atmosphere_last_step = step
    tof.atmosphere_index = index
    if tof.atmosphere_index != old_index:
        say(
            tof.opts["verbosity"],
            2,
            "The atmosphere model now extends to level surface {index} of "
            "{N} at a pressure of {pressure:.2e} Pa.",
            index=tof.atmosphere_index,
            N=tof.opts["N"],
            pressure=tof.Pi[tof.atmosphere_index],
        )

    # Adjust the densities to fit the atmosphere model:
    tof.rhoi[: (tof.atmosphere_index + 1)] = tof.opts["atmosphere"](
        tof.li[: (tof.atmosphere_index + 1)],
        tof.Pi[: (tof.atmosphere_index + 1)],
    )

    # Check for unphysical density inversions:
    if np.any(np.diff(tof.rhoi) < 0):
        raise ValueError("Atmosphere created density inversion!")


def _check_atmosphere(tof):
    """
    Warns if the atmosphere model extends more than one level surface deeper
    than atmosphere_until, which happens if the transition moved inwards early
    on (when the pressures were still inaccurate) and was then locked. One
    level surface of slack is the resolution of the grid.
    """

    index = tof.atmosphere_index
    if index > 1 and tof.Pi[index - 2] > tof.opts["atmosphere_until"]:
        warn(
            "The atmosphere model extends to a pressure of {pressure:.2e} Pa, "
            "deeper than atmosphere_until = {until:.2e} Pa.",
            AccuracyWarning,
            stacklevel=4,
            pressure=tof.Pi[index - 2],
            until=tof.opts["atmosphere_until"],
        )


def _warn_not_converged_drho(tof, drho):
    """
    Warns that the densities did not converge within max_iter_density_update
    iterations.
    """

    warn(
        "drho = {drho:.0e} > {tol:.0e} = drho_tol after "
        "max_iter_density_update = {max_iter} iterations.",
        ConvergenceWarning,
        stacklevel=4,
        drho=drho,
        tol=tof.opts["drho_tol"],
        max_iter=tof.opts["max_iter_density_update"],
    )


def _warn_not_converged(tof, drot, dJs, drho, max_iter_name):
    """
    Warns that the rotational parameter, the Js or the densities did not
    converge within the number of iterations given by opts[max_iter_name].
    """

    quantities = [
        ("drot", drot, tof.opts["drot_tol"]),
        ("dJ", dJs, tof.opts["dJ_tol"]),
        ("drho", drho, tof.opts["drho_tol"]),
    ]
    parts = [
        f"{name} = {value:.0e} {'<' if value < tol else '>'} {tol:.0e}"
        for name, value, tol in quantities
    ]

    warn(
        "{details} after {name} = {max_iter} iterations.",
        ConvergenceWarning,
        stacklevel=4,
        details=", ".join(parts),
        name=max_iter_name,
        max_iter=tof.opts[max_iter_name],
    )


def get_Js_errors(self):
    """
    Fills self.Js_errors with error estimates for the gravitational moments
    Js calculated by the Theory of Figures. They are absolute errors (in the
    units of Js; divide by abs(Js) for the relative errors, which is what the
    tables store), based on the convergence study in
    accuracy/accuracy_full.ipynb (tables PyToF/accuracy_data_*.npz). Not
    called automatically: call it after relax_to_shape(), relax_to_barotrope()
    or relax_to_density(). Not available with differential rotation or a
    non-equidistant grid (warns instead).

    The tables belong to the integration rule that is in use: Simpson's rule,
    unless the option use_simpson is False or PyToF fell back to the
    trapezoid rule (see ToF.simpson_in_use), which is less accurate.
    """

    if max(abs(self.opts["alphas"])) != 0:
        warn(
            "Accuracy when using differential rotation is unknown. "
            "PyToF provides no error estimates."
        )
        return 0

    if not np.allclose(np.diff(self.li), np.diff(self.li)[0]):
        warn(
            "Mean levels surfaces are not equidistant! Accuracy will be "
            "greatly reduced and PyToF provides no error estimates."
        )
        return 0

    HERE = Path(__file__).resolve().parent
    if self.simpson_in_use:
        accuracy_data = np.load(HERE / "accuracy_data_simpson.npz")
    else:
        accuracy_data = np.load(HERE / "accuracy_data_trapezoid.npz")

    Ns_n_bins = accuracy_data["Ns_n_bins"]
    rel_error_04 = accuracy_data["rel_error_04"]
    rel_error_07 = accuracy_data["rel_error_07"]
    rel_error_10 = accuracy_data["rel_error_10"]

    if self.opts["order"] == 4:
        rel_error = rel_error_04
    elif self.opts["order"] == 7:
        rel_error = rel_error_07
    elif self.opts["order"] == 10:
        rel_error = rel_error_10

    # Negative n_bin is equivalent to n_bin = N, see AlgoToF.py:
    n_bin = self.opts["n_bin"]
    if n_bin < 0:
        n_bin = self.opts["N"]

    # n_bin<4 throws an error in the interpolation anyways:
    mask = Ns_n_bins[:, 1] >= 4
    points = Ns_n_bins[mask, :]

    for i, J in enumerate(self.Js):
        if i != 0:
            values = rel_error[:, i - 1, :].reshape(-1)[mask]
            err_func = scipy.interpolate.LinearNDInterpolator(
                np.log2(points), np.log10(values)
            )

            self.Js_errors[i] = abs(
                J
                * 10
                ** err_func(
                    np.log2(self.opts["N"]),
                    np.log2(n_bin),
                )
            )

            if np.isnan(self.Js_errors[i]):
                warn(
                    "Tuple (N, n_bin) outside range that has been tested for "
                    "accuracy. PyToF provides no error estimates."
                )


def get_r_l_mu(self, mu):
    """
    This function returns an array with shape (self.opts['N'], length(mu))
    that stores the values of r_l_mu that is defined in equation (B.1) from
    arXiv:1708.06177v1 for each l and each mu.

    mu: Array of cos(theta) values where theta is the colatitude.
    """

    # Calculate array with r_l(mu):
    r_l_mu = 1

    for i in range(len(self.figure_functions)):
        r_l_mu = r_l_mu + np.outer(
            self.figure_functions[i],
            np.polynomial.legendre.Legendre.basis(2 * i)(mu),
        )

    return r_l_mu * np.outer(self.li, np.ones_like(mu))


def get_U_l_mu(self, mu):
    """
    This function returns an array with shape (self.opts['N'], length(mu))
    that stores the values of the potential U that is defined in equation (B.3)
    from arXiv:1708.06177v1 for each l and each mu. As for self.U, the sign is
    such that the potential of a point mass is +GM/r.

    mu: Array of cos(theta) values where theta is the colatitude.
    """

    return (
        self.opts["G"]
        * self.opts["M_phys"]
        / self.li[0] ** 3
        * np.outer(self.li**2 * self.A0, np.ones_like(mu))
    )


def get_NMoI(self, N=1000):
    """
    This function returns a float that represents the value of the normalized
    moment of inertia.

    N: Number of mu points to use in the integration.
    """

    # Define variables to integrate over:
    mu = np.linspace(-1, 1, N)
    mu_2D = np.outer(np.ones_like(self.li), mu)
    r_l_mu = get_r_l_mu(self, mu)

    # Change of variables in integration:
    dr_dl = np.gradient(r_l_mu, self.li, mu, edge_order=2)[0]

    # Perform integrations:
    integrand_l_theta = (
        2
        * np.pi
        * np.outer(self.rhoi, np.ones_like(mu))
        * r_l_mu**2
        * (1 - mu_2D**2)
        * r_l_mu**2
        * dr_dl
    )  # dmu dl

    if not np.allclose(np.diff(self.li), np.diff(self.li)[0]):
        warn(
            "Mean levels surfaces are not equidistant! NMoI integration "
            "will be inaccurate."
        )

    integrand_l = scipy.integrate.simpson(integrand_l_theta, x=mu, axis=1)
    MoI = -scipy.integrate.simpson(integrand_l, x=self.li)  # minus sign due
    # to integration from outside to inside
    NMoI = MoI / (_mass_int(self) * self.li[0] ** 2)

    return NMoI


def get_Kn(self, n, bs=None):
    """
    This function returns arrays r, T_n and K_n. K_n(r) is the Love function
    of n-th degree based on solving a differential equation for T_n(r).

    n: An integer greater or equal to two, determines Love function degree.
    bs: An array that contains the radii of all density discontinuities
        inside the planet. A discontinuity at the surface (a nonzero density
        at the outermost level surface) is taken into account automatically.

    A density jump is not integrated over as a steep gradient: the grid cells
    around each radius in bs are removed from the density derivative, which
    then belongs to the continuous part of the profile, and the jump enters
    through its boundary condition for T_n'.
    """

    # Assure that the potential has been calculated for the current density
    # distribution:
    _pressurize(self)

    # Get smallest and largest radius:
    r0 = self.li[-1]
    R = self.li[0]
    assert r0 < R

    # Define array with all radii where boundary conditions apply:
    if bs is None:
        bs = np.array([r0, R])
    else:
        bs = np.concatenate(
            (np.atleast_1d(r0), np.sort(np.atleast_1d(bs)), np.atleast_1d(R))
        )

    # Grid index and size of every density jump:
    jumps = []
    for b in bs[1:-1]:
        k = np.argmin(np.abs(self.li - b))
        jumps.append(
            (
                k,
                max(
                    self.rhoi[k] - self.rhoi[k - 1],
                    self.rhoi[k + 1] - self.rhoi[k],
                ),
            )
        )

    # Calculate density and potential derivatives. The derivative of the
    # density is that of its continuous part: the jumps, which np.gradient
    # smears over the neighbouring grid points, enter via the boundary
    # conditions below, and must not be counted twice.
    drho = np.gradient(self.rhoi, self.li)
    for k, _ in jumps:
        drho[k - 1 : k + 2] = 0.0
    rho_prime = scipy.interpolate.interp1d(self.li[::-1], drho[::-1])
    # Gravitational acceleration g = -dU/dl, positive since U decreases
    # outwards:
    g = scipy.interpolate.interp1d(
        self.li[::-1], -np.gradient(self.U, self.li)[::-1]
    )

    # Define right-hand side of the differential equation:
    # (1D, 2nd order) -> (2D, 1st order)
    def rhs(t, y):
        Q = (
            4 * np.pi * self.opts["G"] * rho_prime(t) / g(t)
            + n * (n + 1) / t**2
        )
        return [y[1], Q * y[0] - 2 / t * y[1]]

    # Initial condition:
    y0 = [1.0, n / r0]

    # Solve first part of the differential equation:
    sol = scipy.integrate.solve_ivp(
        rhs,
        (bs[0], bs[1]),
        y0,
        max_step=(R - r0) / self.opts["N"],
    )

    # Store solutions:
    r = sol.t
    T_n = sol.y[0]
    T_n_prime = sol.y[1]

    # Solve further parts of the differential equation, if there are any:
    for i in range(1, len(bs) - 1):
        delta_rho = jumps[i - 1][1]
        say(
            self.opts["verbosity"],
            2,
            "Boundary condition applied at a radius of r/R={radius:.2f} "
            "with a density jump of {jump:.2e} kg/m^3.",
            radius=bs[i] / R,
            jump=delta_rho,
        )

        # Initial condition:
        y0 = [
            T_n[-1],
            T_n_prime[-1]
            - 4 * np.pi * self.opts["G"] * delta_rho * T_n[-1] / g(bs[i]),
        ]

        # Solve further parts of the differential equation:
        sol = scipy.integrate.solve_ivp(
            rhs,
            (bs[i], bs[i + 1]),
            y0,
            max_step=(R - r0) / self.opts["N"],
        )

        # Store solutions:
        r = np.concatenate((r, sol.t))
        T_n = np.concatenate((T_n, sol.y[0]))
        T_n_prime = np.concatenate((T_n_prime, sol.y[1]))

    # Apply final boundary condition. If the density is not zero at the
    # surface, T_n' jumps there as well, by the same condition as inside:
    jump_surface = (
        4 * np.pi * self.opts["G"] * self.rhoi[0] * sol.y[0, -1] / g(R)
    )
    T_n_prime_outside = sol.y[1, -1] - jump_surface
    scale = (
        (2 * n + 1) * g(R) / (T_n_prime_outside + (n + 1) * sol.y[0, -1] / R)
    )
    T_n *= scale
    T_n_prime *= scale

    # Calculate Love function:
    K_n = (T_n / R / g(R)) * (R / r) ** n - 1

    return r, T_n, K_n


def set_barotrope(self, fun):
    """
    This function allows the user to internally set the function relating the
    density self.rhoi to the pressure self.Pi via self.rhoi =
    fun(self.Pi, param).

    fun: Function handle for the barotrope model.
    """

    # Set function:
    self.barotrope = fun


def _check_radius_consistency(tof):
    """
    Warns if the radius provided via opts['R_phys'] differs from the radius of
    the shape calculated by relax_to_shape().
    """

    kind = tof.opts["R_phys"][1]
    ratio = {
        "equatorial": tof.R_eq_to_R_m,
        "polar": tof.R_po_to_R_m,
    }.get(kind)

    if ratio is None:  # 'mean' is consistent by construction
        return

    given = tof.opts["R_phys"][0]
    calculated = ratio * tof.li[0]

    if not np.isclose(calculated, given):
        warn(
            "Your provided {which} radius is not consistent with the shape of "
            "the planet calculated by PyToF! Your value: {given:.5e} / PyToF "
            "value: {calculated:.5e}",
            PyToFWarning,
            stacklevel=3,
            which=kind,
            given=given,
            calculated=calculated,
        )


def _relax_to_shape(tof, maxiter="default"):
    """
    Runs AlgoToF.Algorithm on the current level surfaces and densities and
    stores the results in tof. Returns the Algorithm output, which says
    whether it converged, how many iterations it took and, as out.seconds,
    how long. Does not warn, since relax_to_barotrope() and
    relax_to_density() deliberately call this with a small maxiter.
    """

    # Initialize variables:
    alphas = np.zeros(len(tof.opts["alphas"]))

    if maxiter == "default":
        maxiter = tof.opts["max_iter_relax_shape"]

    # Convert barotropic differential rotation parameters to Theory of Figures
    # logic:
    if np.any(tof.opts["alphas"]):
        for i in range(len(alphas)):
            alphas[i] = (
                2
                * (i + 1)
                * (tof.li[0]) ** (2 * i)
                * tof.opts["alphas"][i]
                / (
                    (tof.m_rot_calc * tof.opts["G"] * tof.opts["M_phys"])
                    / tof.li[0] ** 3
                )
                / tof.opts["R_ref"] ** (2 * (i + 1))
            )

    # Measure ToF performance:
    tic = time.time()

    # Implement the Theory of Figures:
    tof.Js, out = AlgoToF.Algorithm(
        tof.li,
        tof.rhoi,
        tof.m_rot_calc,
        order=tof.opts["order"],
        n_bin=tof.opts["n_bin"],
        tol=tof.opts["dJ_tol"],
        maxiter=maxiter,
        verbosity=tof.opts["verbosity"],
        debug_plot=tof.opts["debug_plot"],
        debug_label=getattr(tof, "_debug_label", None) or "relax_to_shape",
        debug_primed=tof.opts["debug_primed"],
        simpson_check=tof.opts["simpson_check"],
        R_ref=tof.opts["R_ref"],
        # AlgoToF wants the innermost level surface first:
        ss_initial=[np.flip(s) for s in tof.figure_functions],
        alphas=alphas,
        H=tof.opts["H"],
        use_simpson=tof.simpson_in_use,
    )

    # Measure ToF performance (reported by the callers):
    out.seconds = time.time() - tic

    # Save results:
    # AlgoToF orders its arrays from the innermost to the outermost level
    # surface; ToF stores everything from the outermost inwards, like li:
    tof.A0 = np.flip(out.A0)
    tof.figure_functions = [np.flip(s) for s in out.ss]
    tof.volume_integrals = [np.flip(S) for S in out.SS]
    tof.R_eq_to_R_m = out.R_eq_to_R_m
    tof.R_po_to_R_m = out.R_po_to_R_m
    if tof.simpson_in_use and not out.use_simpson:
        tof._simpson_failed = True

    return out


def relax_to_shape(self, check_consistency=True, maxiter="default"):
    """
    Calls Algorithm from AlgoToF until either the accuray given by
    self.opts['dJ_tol'] is fulfilled or maxiter is reached.

    check_consistency:  If True, checks whether the user provided radius is
                        consistent with the shape of the planet calculated by
                        PyToF
    maxiter:            Maximum number of iterations to perform. If "default",
                        uses the self.opts['max_iter_relax_shape'] value.
    """

    out = _relax_to_shape(self, maxiter)

    say(
        self.opts["verbosity"],
        1,
        "Relaxing to shape finished after {iterations} iterations: "
        "dJ = {dJ:.1e}.",
        iterations=out.it + 1,
        dJ=out.dJ_max,
    )
    say(
        self.opts["verbosity"],
        2,
        "Relaxing to shape done in {seconds:.2e} seconds.",
        seconds=out.seconds,
    )

    if not out.converged:
        warn(
            "Theory of Figures did not converge: max(dJs) = {dJ:.0e} > "
            "{tol:.0e} = dJ_tol after maxiter = {maxiter} iterations.",
            ConvergenceWarning,
            dJ=out.dJ_max,
            tol=self.opts["dJ_tol"],
            maxiter=out.it + 1,
        )

    if check_consistency:
        _check_radius_consistency(self)

    return out.it


def _relax(self, update_densities, max_iter_name, name):
    """
    Shared iteration of relax_to_barotrope() and relax_to_density(): calls
    _relax_to_shape() and update_densities(self) alternately until the
    tolerances dJ_tol, drot_tol and drho_tol are fulfilled or
    self.opts[max_iter_name] is reached. Returns the number of iterations.

    update_densities:   Function taking the ToF object and updating the
                        pressure and the densities in place.
    max_iter_name:      Name of the option limiting the iterations.
    name:               Name used in messages, e.g. "barotrope".
    """

    # Measure ToF performance:
    tic = time.time()

    # Only relevant if debug plots are activated:
    self._debug_name = name
    self._debug_history = []
    self._debug_iteration = 0
    self._debug_label = f"relax_to_{name}, start"

    # Call relax_to_shape() and ensure consistency for the first time:
    _relax_to_shape(self, maxiter=2)
    _ensure_consistency(self)
    iteration = 1

    # Converge on gravitational moments:
    while iteration < self.opts[max_iter_name]:
        # Store old gravitational moment values:
        old_Js = self.Js

        # Define iteration counter for density loop:
        IterUpdate = 1
        n_updates = 0

        self._debug_iteration = iteration
        self._debug_label = f"relax_to_{name}, iteration {iteration}"

        # Converge on densities (without an atmosphere this loop ends after
        # one iteration for relax_to_density()):
        while IterUpdate < self.opts["max_iter_density_update"]:
            # Store old values:
            old_rho = self.rhoi

            update_densities(self)
            n_updates += 1
            if iteration == SURFACE_CHECK_ITERATION and n_updates == 1:
                _check_surface_slope(self)

            # Update drho, ignore first entry to avoid possible division by
            # zero:
            drho = np.max(np.abs(self.rhoi[1:] / old_rho[1:] - 1))

            # Check convergence:
            if drho < self.opts["drho_tol"]:
                break

            # Update iteration parameter:
            IterUpdate += 1

        # Warning if not converged:
        if IterUpdate == self.opts["max_iter_density_update"]:
            _warn_not_converged_drho(self, drho)

        # Call relax_to_shape():
        shape = _relax_to_shape(self, maxiter=2)

        # Store old values:
        old_m = self.m_rot_calc
        old_rho = self.rhoi

        # Ensure consistency:
        _ensure_consistency(self)

        # Check convergence, and address division by zero issues:
        if old_m != 0:
            drot = np.abs(self.m_rot_calc / old_m - 1)
        else:
            drot = 0.0

        mask = ~np.logical_and(self.Js == 0, old_Js == 0)

        dJs = np.max(np.abs(self.Js[mask] / old_Js[mask] - 1))
        drho = np.max(np.abs(self.rhoi[1:] / old_rho[1:] - 1))

        if self.opts["debug_plot"]:
            from PyToF.PlotDebugToF import debug_relax_plot

            debug_relax_plot(self, convergence=(drot, dJs, drho))

        say(
            self.opts["verbosity"],
            3,
            "Iteration {iteration}: drot = {drot:.1e}, dJ = {dJs:.1e}, "
            "drho = {drho:.1e}; {n_updates} density updates, {n_shape} shape "
            "iterations in {seconds:.2e} seconds.",
            iteration=iteration,
            drot=drot,
            dJs=dJs,
            drho=drho,
            n_updates=n_updates,
            n_shape=shape.it + 1,
            seconds=shape.seconds,
        )

        if (
            drot < self.opts["drot_tol"]
            and dJs < self.opts["dJ_tol"]
            and drho < self.opts["drho_tol"]
        ):
            break

        # Update iteration parameter:
        iteration += 1

    # Measure ToF performance:
    toc = time.time()
    self._debug_label = None

    # Warning if not converged:
    if iteration == self.opts[max_iter_name]:
        _warn_not_converged(self, drot, dJs, drho, max_iter_name)

    if self.opts["use_atmosphere"]:
        _check_atmosphere(self)

    # Verbosity output:
    say(
        self.opts["verbosity"],
        1,
        "Relaxing to {name} finished after {iterations} iterations: "
        "drot = {drot:.1e}, dJ = {dJs:.1e}, drho = {drho:.1e}.",
        name=name,
        iterations=iteration,
        drot=drot,
        dJs=dJs,
        drho=drho,
    )
    say(
        self.opts["verbosity"],
        2,
        "Relaxing to {name} done in {seconds:.2e} seconds.",
        name=name,
        seconds=toc - tic,
    )

    return iteration


def _update_densities_given(tof):
    """
    Update step of relax_to_density(): the densities are given, so only the
    pressure follows from hydrostatic equilibrium, optionally followed by the
    atmospheric model.
    """

    # Calculates the pressure values according to hydrostatic equilibrium:
    _pressurize(tof)

    # Optional, use a provided atmospheric model:
    if tof.opts["use_atmosphere"]:
        _apply_atmosphere(tof)

    # Ensure physical mass stays unaffacted:
    _fixmass(tof)


def relax_to_barotrope(self):
    """
    Calls relax_to_shape() and _update_densities_barotrope() until either the
    accuray given by self.opts['dJ_tol'], self.opts['drot_tol'] and
    self.opts['drho_tol'] is fulfilled or
    self.opts['max_iter_relax_barotrope'] is reached.
    """

    return _relax(
        self,
        _update_densities_barotrope,
        "max_iter_relax_barotrope",
        "barotrope",
    )


def relax_to_density(self):
    """
    Calls relax_to_shape() until either the accuray given by
    self.opts['dJ_tol'], self.opts['drot_tol'] and
    self.opts['drho_tol'] is fulfilled or
    self.opts['max_iter_relax_density'] is reached.
    """

    return _relax(
        self, _update_densities_given, "max_iter_relax_density", "density"
    )
