########################################################
# Author of this version: Luca Morf - luca.morf@uzh.ch #
########################################################

import time
from pathlib import Path

import numpy as np
import scipy

import PyToF.AlgoToF as AlgoToF
from PyToF.messages import ConvergenceWarning, PyToFWarning, say, warn
from PyToF.PlotDebugToF import debug_FunctionsToF_plot


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

    # Changes the radii to be self-consistent with the provided radius
    # (affects mass and rotational parameter):
    _fixradius(tof)
    # Changes the densities to to be self-consistent with the provided mass
    # (affects rotational parameter):
    _fixmass(tof)
    # Changes the rotational parameter to be self-consistent
    # (affects nothing else):
    _fixrot(tof)


def _pressurize(tof):
    """
    Calculates the pressure tof.Pi at the level surfaces tof.li
    assuming hydrostatic equilibrium.
    """

    tof.U = (
        -tof.opts["G"]
        * tof.opts["M_phys"]
        / tof.li[0] ** 3
        * tof.li**2
        * np.flip(tof.A0)
    )

    integrand = -tof.rhoi * np.gradient(tof.U, tof.li, edge_order=2)

    if tof.simpson_in_use:
        tof.Pi = tof.opts["P0"] - scipy.integrate.cumulative_simpson(
            integrand, x=tof.li[::-1], initial=0.0
        )
    else:
        tof.Pi = tof.opts["P0"] + scipy.integrate.cumulative_trapezoid(
            integrand, x=tof.li, initial=0.0
        )

    if tof.opts["debug_plot"]:
        debug_FunctionsToF_plot(tof, new=False, iteration=tof._debug_iteration)

    if tof.simpson_in_use and (
        not (tof.Pi >= 0).all() or not (np.diff(tof.Pi) >= 0).all()
    ):
        tof._simpson_failed = True
        say(
            tof.opts["verbosity"],
            1,
            "Calculated pressure contains negative or non-monotonic entries! "
            "Reverting to the more stable and less accurate trapezoid rule "
            "for integration to hopefully mitigate the issue.",
            kind="warn",
        )
        _pressurize(tof)

    assert (tof.Pi >= 0).all(), (
        "Calculated pressure contains negative entries!"
    )

    assert (np.diff(tof.Pi) >= 0).all(), (
        "Calculated pressure is not monotonically increasing!"
    )


def _update_densities_barotrope(tof):
    """
    This function is called by relax_to_barotrope() and implements the
    barotrope model density = barotrope(pressure), i.e. tof.rhoi =
    tof.barotrope(tof.Pi, tof.baro_param_calc).
    """

    # Calculates the pressure values according to hydrostatic equilibrium:
    _pressurize(tof)

    # Ensure that the barotrope has an argument in case it needs one:
    if tof.baro_param_calc is None:
        tof.baro_param_calc = tof.opts["baro_param_init"]

    # Set new densitites:
    tof.rhoi = tof.barotrope(tof.Pi, tof.baro_param_calc)

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
    tof.atmosphere_index = max(
        index, tof.atmosphere_index
    )  # prevent index oscillations

    # Adjust the densities to fit the atmosphere model:
    tof.rhoi[: (tof.atmosphere_index + 1)] = tof.opts["atmosphere"](
        tof.li[: (tof.atmosphere_index + 1)],
        tof.Pi[: (tof.atmosphere_index + 1)],
    )

    # Check for unphysical density inversions:
    if np.any(np.diff(tof.rhoi) < 0):
        raise ValueError("Atmosphere created density inversion!")


def _warn_not_converged_drho(tof, drho):
    """
    Warns that the densities did not converge within max_iter_update
    iterations.
    """

    warn(
        "drho = {drho:.0e} > {tol:.0e} = drho_tol after max_iter_update = "
        "{max_iter} iterations.",
        ConvergenceWarning,
        stacklevel=3,
        drho=drho,
        tol=tof.opts["drho_tol"],
        max_iter=tof.opts["max_iter_update"],
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
        stacklevel=3,
        details=", ".join(parts),
        name=max_iter_name,
        max_iter=tof.opts[max_iter_name],
    )


def get_Js_errors(self):
    """
    This function is called by relax_to_shape() and fills self.Js_errors
    error estimates for the gravitational moments Js calculated by the Theory
    of Figures based on the results from PyToF_Accuracy_and_Convergence.ipynb.
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

    if self.simpson_in_use:
        say(
            self.opts["verbosity"],
            1,
            "Simpson's rule is in use. This ensures the highest possible "
            "accuracy.",
        )
        HERE = Path(__file__).resolve().parent
        accuracy_data = np.load(HERE / "accuracy_data_simpson.npz")
    else:
        say(
            self.opts["verbosity"],
            1,
            "The trapezoid rule is in use. This yields a reduced accuracy. "
            "Either the option use_simpson is False or PyToF switched to the "
            "trapezoid rule on its own to mitigate stability issues.",
        )
        HERE = Path(__file__).resolve().parent
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
        # Flip since AlgoToF uses a different ordering logic:
        r_l_mu = r_l_mu + np.outer(
            np.flip(self.figure_functions[i]),
            np.polynomial.legendre.Legendre.basis(2 * i)(mu),
        )

    return r_l_mu * np.outer(self.li, np.ones_like(mu))


def get_U_l_mu(self, mu):
    """
    This function returns an array with shape (self.opts['N'], length(mu))
    that stores the values of the potential U that is defined in equation (B.3)
    from arXiv:1708.06177v1 for each l and each mu.

    mu: Array of cos(theta) values where theta is the colatitude.
    """

    # Flip since AlgoToF uses a different ordering logic:
    return (
        -self.opts["G"]
        * self.opts["M_phys"]
        / self.li[0] ** 3
        * np.outer(self.li**2 * np.flip(self.A0), np.ones_like(mu))
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
        say(
            self.opts["verbosity"],
            1,
            "Mean levels surfaces are not equidistant! NMoI integration "
            "will be inaccurate.",
            kind="warn",
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
    bs: An array that contains the radii of all density discontinuities.
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
            (np.atleast_1d(r0), np.atleast_1d(bs), np.atleast_1d(R))
        )

    # Calculate density and potential derivatives:
    rho_prime = scipy.interpolate.interp1d(
        self.li[::-1], np.gradient(self.rhoi, self.li)[::-1]
    )
    U_prime = scipy.interpolate.interp1d(
        self.li[::-1], np.gradient(self.U, self.li)[::-1]
    )

    # Define right-hand side of the differential equation:
    # (1D, 2nd order) -> (2D, 1st order)
    def rhs(t, y):
        Q = (
            4 * np.pi * self.opts["G"] * rho_prime(t) / U_prime(t)
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
        # Find index closest to boundary:
        index = np.argmin(np.abs(self.li - bs[i]))

        # Calculate jump in density:
        delta_rho = max(
            self.rhoi[index] - self.rhoi[index - 1],
            self.rhoi[index + 1] - self.rhoi[index],
        )
        say(
            self.opts["verbosity"],
            1,
            "Boundary condition applied at a radius of r/R={radius:.2f} "
            "with a density jump of {jump:.2e} kg/m^3.",
            radius=bs[i] / R,
            jump=delta_rho,
        )

        # Initial condition:
        y0 = [
            T_n[-1],
            T_n_prime[-1]
            - 4
            * np.pi
            * self.opts["G"]
            * delta_rho
            * T_n[-1]
            / U_prime(bs[i]),
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

    # Apply final boundary condition:
    scale = (
        (2 * n + 1) * U_prime(R) / (sol.y[1, -1] + (n + 1) * sol.y[0, -1] / R)
    )
    T_n *= scale
    T_n_prime *= scale

    # Calculate Love function:
    K_n = (T_n / R / U_prime(R)) * (R / r) ** n - 1

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


def set_density_function(self, fun):
    """
    This function allows the user to internally set the function relating the
    density self.rhoi to the mean level surfaces self.li via
    self.rhoi = fun(self.li, mass, param).

    fun: Function handle for the density model.
    """

    # Set function:
    self.density_function = fun


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
    whether it converged. Does not warn, since relax_to_barotrope() and
    relax_to_density() deliberately call this with a small maxiter.
    """

    # Initialize variables:
    alphas = np.zeros(len(tof.opts["alphas"]))

    if maxiter == "default":
        maxiter = tof.opts["max_iter_shape"]

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
        R_ref=tof.opts["R_ref"],
        ss_initial=tof.figure_functions,
        alphas=alphas,
        H=tof.opts["H"],
        use_simpson=tof.simpson_in_use,
    )

    # Measure ToF performance:
    toc = time.time()

    say(
        tof.opts["verbosity"],
        3,
        "Relaxing to shape done in {seconds:.2e} seconds.",
        seconds=toc - tic,
    )

    # Save results:
    tof.A0 = out.A0  # inside->outside instead of outside->inside since
    # AlgoToF uses a different ordering logic!
    # inside->outside instead of outside->inside since AlgoToF uses a
    # different ordering logic:
    tof.figure_functions = out.ss
    tof.volume_integrals = out.SS
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
                        uses the self.opts['max_iter_shape'] value.
    """

    out = _relax_to_shape(self, maxiter)

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
    self._debug_iteration = 0

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

        if self.opts["debug_plot"]:
            debug_FunctionsToF_plot(
                self, new=True, iteration=self._debug_iteration
            )

        # Converge on densities (without an atmosphere this loop ends after
        # one iteration for relax_to_density()):
        while IterUpdate < self.opts["max_iter_update"]:
            # Store old values:
            old_rho = self.rhoi

            update_densities(self)

            # Update drho, ignore first entry to avoid possible division by
            # zero:
            drho = np.max(np.abs(self.rhoi[1:] / old_rho[1:] - 1))

            # Check convergence:
            if drho < self.opts["drho_tol"]:
                break

            # Update iteration parameter:
            IterUpdate += 1

        # Only relevant if debug plots are activated:
        self._debug_iteration += 1

        # Warning if not converged:
        if IterUpdate == self.opts["max_iter_update"]:
            _warn_not_converged_drho(self, drho)

        # Call relax_to_shape():
        _relax_to_shape(self, maxiter=2)

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

    # Warning if not converged:
    if iteration == self.opts[max_iter_name]:
        _warn_not_converged(self, drot, dJs, drho, max_iter_name)

    # Verbosity output:
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
    self.opts['max_iter_barotrope'] is reached.
    """

    return _relax(
        self, _update_densities_barotrope, "max_iter_barotrope", "barotrope"
    )


def relax_to_density(self):
    """
    Calls relax_to_shape() until either the accuray given by
    self.opts['dJ_tol'], self.opts['drot_tol'] and
    self.opts['drho_tol'] is fulfilled or
    self.opts['max_iter_density'] is reached.
    """

    return _relax(self, _update_densities_given, "max_iter_density", "density")
