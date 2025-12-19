########################################################
# Author of this version: Luca Morf - luca.morf@uzh.ch #
########################################################

import time

import numpy as np
import scipy

import PyToF.AlgoToF as AlgoToF
from PyToF.color import c
from PyToF.PlotDebugToF import debug_FunctionsToF_plot


def _mass_int(class_obj):
    """
    Calculates and returns the total mass implied by the level surfaces
    class_obj.li and densities class_obj.rhoi using the simpson rule.
    """

    # Negative sign because beginning of the array is the outer surface:
    return (
        -4
        * np.pi
        * scipy.integrate.simpson(
            class_obj.rhoi * class_obj.li**2, class_obj.li
        )
    )


def _fixradius(class_obj):
    """
    Renormalizes the level surfaces class_obj.li for consistency with the
    initially provided physical radius.
    """

    # Renormalize the level surfaces in such a way that the newly calculated
    # R_calc equatorial radius is the same as the initial one:
    if class_obj.opts["R_phys"][1] == "equatorial":
        class_obj.li = (class_obj.li / class_obj.li[0]) * (
            class_obj.opts["R_phys"][0] / class_obj.R_eq_to_R_m
        )
        class_obj.R_calc = class_obj.li[0] * class_obj.R_eq_to_R_m

    elif class_obj.opts["R_phys"][1] == "mean":
        class_obj.li = (class_obj.li / class_obj.li[0]) * class_obj.opts[
            "R_phys"
        ][0]
        class_obj.R_calc = class_obj.li[0]

    elif class_obj.opts["R_phys"][1] == "polar":
        class_obj.li = (class_obj.li / class_obj.li[0]) * (
            class_obj.opts["R_phys"][0] / class_obj.R_po_to_R_m
        )
        class_obj.R_calc = class_obj.li[0] * class_obj.R_po_to_R_m

    else:
        raise KeyError(
            c.WARN
            + "Invalid R_phys type specification! Valid options: 'equatorial',"
            + " 'mean', 'polar'"
            + c.ENDC
        )

    assert np.isclose(class_obj.R_calc, class_obj.opts["R_phys"][0]), (
        c.WARN
        + "Renormalizing the level surfaces for consistency with the initially"
        + " provided physical radius failed!"
        + c.ENDC
    )


def _fixmass(class_obj):
    """
    Renormalizes the densities class_obj.rhoi for consistency with the
    initially provided mass.
    """

    # Renormalize the densities in such a way that the newly calculated mass is
    # the same as the initial one:
    class_obj.rhoi = (
        class_obj.rhoi * class_obj.opts["M_phys"] / _mass_int(class_obj)
    )

    # Sanity check:
    assert np.isclose(_mass_int(class_obj), class_obj.opts["M_phys"]), (
        c.WARN
        + "Renormalizing the densities for consistency with the initially"
        + " provided mass failed!"
        + c.ENDC
    )


def _fixrot(class_obj):
    """
    Renormalizes the rotational parameter for consistency with the initially
    provided period.
    """

    # We update the m_rot_calc parameter such that it is consistent with the
    # period, the outermost level surface and the calculated mass:
    class_obj.m_rot_calc = (
        (2 * np.pi / class_obj.opts["Period"]) ** 2
        * class_obj.li[0] ** 3
        / (class_obj.opts["G"] * _mass_int(class_obj))
    )


def _ensure_consistency(class_obj):
    """
    This function updates all variables necessary to ensure consistency with
    the initially provided physical values class_obj.opts['R_phys'] and
    class_obj.opts['M_phys'].
    """

    # Changes the radii to be self-consistent with the provided radius
    # (affects mass and rotational parameter):
    _fixradius(class_obj)
    # Changes the densities to to be self-consistent with the provided mass
    # (affects rotational parameter):
    _fixmass(class_obj)
    # Changes the rotational parameter to be self-consistent
    # (affects nothing else):
    _fixrot(class_obj)


def _pressurize(class_obj):
    """
    Calculates the pressure class_obj.Pi at the level surfaces class_obj.li
    assuming hydrostatic equilibrium.
    """

    class_obj.U = (
        -class_obj.opts["G"]
        * class_obj.opts["M_phys"]
        / class_obj.li[0] ** 3
        * class_obj.li**2
        * np.flip(class_obj.A0)
    )

    integrand = -class_obj.rhoi * np.gradient(class_obj.U, class_obj.li)

    # Do NOT use cumulative_simpson here, as it does just sum up up local
    # simpsons, which is not equivalent to a global simpson integration:
    class_obj.Pi = class_obj.opts["P0"] + scipy.integrate.cumulative_trapezoid(
        integrand, x=class_obj.li, initial=0.0
    )

    if class_obj.opts["debug_plot"]:
        debug_FunctionsToF_plot(
            class_obj, new=False, iteration=class_obj.bugfix_iter
        )

    assert (class_obj.Pi >= 0).all(), (
        c.WARN + "Calculated pressure contains negative entries!" + c.ENDC
    )
    assert (np.diff(class_obj.Pi) >= 0).all(), (
        c.WARN
        + "Calculated pressure is not monotonically increasing!"
        + c.ENDC
    )


def _update_densities_barotrope(class_obj):
    """
    This function is called by relax_to_barotrope() and implements the
    barotrope model density = barotrope(pressure), i.e. class_obj.rhoi =
    class_obj.barotrope(class_obj.Pi, class_obj.baro_param_calc).
    """

    # Calculates the pressure values according to hydrostatic equilibrium:
    _pressurize(class_obj)

    # Ensure that the barotrope has an argument in case it needs one:
    if class_obj.baro_param_calc is None:
        class_obj.baro_param_calc = class_obj.opts["baro_param_init"]

    # Set new densitites:
    class_obj.rhoi = class_obj.barotrope(
        class_obj.Pi, class_obj.baro_param_calc
    )

    # Ensure physical mass stays unaffacted:
    _fixmass(class_obj)

    # Check for unphysical density inversions:
    if np.any(np.diff(class_obj.rhoi) < 0):
        raise ValueError(
            c.WARN + "Barotrope created density inversion!" + c.ENDC
        )

    # Optional, use a provided atmospheric model:
    if class_obj.opts["use_atmosphere"]:
        _apply_atmosphere(class_obj)

        # Ensure physical mass stays unaffacted:
        _fixmass(class_obj)


def _apply_atmosphere(class_obj):
    """
    This function is called by e.g. _update_densities_barotrope() and
    implements the atmosphere model density = atmosphere(argument), i.e.
    class_obj.rhoi[specified by class_obj.opts['atmosphere_until']] =
    class_obj.opts['atmosphere'](class_obj.li[:index], class_obj.Pi[:index]).
    """

    # Define index that marks the transition from the atmosphere to the rest of
    # the model:
    index = np.arange(class_obj.opts["N"])[
        class_obj.Pi > class_obj.opts["atmosphere_until"]
    ][0]
    class_obj.atmosphere_index = max(
        index, class_obj.atmosphere_index
    )  # prevent index oscillations

    # Adjust the densities to fit the atmosphere model:
    class_obj.rhoi[: (class_obj.atmosphere_index + 1)] = class_obj.opts[
        "atmosphere"
    ](
        class_obj.li[: (class_obj.atmosphere_index + 1)],
        class_obj.Pi[: (class_obj.atmosphere_index + 1)],
    )

    # Check for unphysical density inversions:
    if np.any(np.diff(class_obj.rhoi) < 0):
        raise ValueError(
            c.WARN + "Atmosphere created density inversion!" + c.ENDC
        )


def _get_Js_errors(class_obj):
    """
    This function is called by relax_to_shape() and fills class_obj.Js_error
    error estimates for the gravitational moments Js calculated by the Theory
    of Figures based on the results from PyToF_Accuracy_and_Convergence.ipynb.
    """

    if max(abs(class_obj.opts["alphas"])) != 0:
        print(
            c.WARN
            + "Accuracy when using differential rotation is unknown. "
            + "PyToF provides no error estimates."
            + c.ENDC
        )
        return 0

    if (
        class_obj.opts["n_bin"] > 0
        and class_obj.opts["n_bin"] != class_obj.opts["N"]
    ):
        print(
            c.WARN
            + "Accuracy when interpolation and binning is unknown. "
            + "PyToF provides no error estimates."
            + c.ENDC
        )
        return 0

    # ruff: noqa: E501
    # fmt: off
    Ns = np.array([128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768])

    rel_error_04 = np.array(
        [
            [9.50e-04, 2.15e-03, 5.72e-03, 9.11e-02],
            [2.63e-04, 7.84e-04, 7.83e-03, 8.84e-02],
            [6.97e-05, 4.01e-04, 8.43e-03, 8.77e-02],
            [1.60e-05, 2.94e-04, 8.59e-03, 8.75e-02],
            [1.26e-06, 2.64e-04, 8.64e-03, 8.73e-02],
            [2.76e-06, 2.56e-04, 8.65e-03, 8.73e-02],
            [3.85e-06, 2.54e-04, 8.65e-03, 8.73e-02],
            [4.14e-06, 2.54e-04, 8.65e-03, 8.73e-02],
            [4.22e-06, 2.54e-04, 8.65e-03, 8.73e-02]
        ]
    )

    rel_error_07 = np.array(
        [
            [9.55e-04, 1.89e-03, 2.92e-03, 3.96e-03, 7.03e-03, 1.80e-02, 1.60e-01],
            [2.67e-04, 5.31e-04, 8.20e-04, 1.04e-03, 3.20e-03, 2.29e-02, 1.54e-01],
            [7.39e-05, 1.47e-04, 2.29e-04, 2.16e-04, 2.12e-03, 2.43e-02, 1.53e-01],
            [2.03e-05, 4.02e-05, 6.47e-05, 1.04e-05, 1.83e-03, 2.47e-02, 1.53e-01],
            [5.51e-06, 1.09e-05, 1.97e-05, 7.23e-05, 1.75e-03, 2.48e-02, 1.52e-01],
            [1.49e-06, 2.89e-06, 7.50e-06, 8.91e-05, 1.72e-03, 2.48e-02, 1.52e-01],
            [4.00e-07, 7.13e-07, 3.10e-06, 1.33e-04, 1.61e-03, 2.50e-02, 1.52e-01],
            [1.08e-07, 1.30e-07, 2.21e-06, 1.34e-04, 1.61e-03, 2.50e-02, 1.52e-01],
            [2.93e-08, 2.58e-08, 1.97e-06, 1.34e-04, 1.61e-03, 2.50e-02, 1.52e-01]
        ]
    )

    rel_error_10 = np.array(
        [
            [9.55e-04, 1.89e-03, 2.92e-03, 4.06e-03, 5.31e-03, 6.73e-03, 7.77e-03, 1.53e-02, 3.93e-02, 2.23e-01],
            [2.67e-04, 5.31e-04, 8.17e-04, 1.13e-03, 1.47e-03, 1.88e-03, 1.80e-03, 8.10e-03, 4.82e-02, 2.15e-01],
            [7.39e-05, 1.47e-04, 2.26e-04, 3.11e-04, 4.04e-04, 5.35e-04, 1.49e-04, 6.13e-03, 5.06e-02, 2.13e-01],
            [2.03e-05, 4.03e-05, 6.18e-05, 8.50e-05, 1.09e-04, 1.68e-04, 2.97e-04, 5.60e-03, 5.12e-02, 2.12e-01],
            [5.51e-06, 1.10e-05, 1.68e-05, 2.29e-05, 2.39e-05, 5.08e-05, 4.52e-04, 5.40e-03, 5.15e-02, 2.12e-01],
            [1.49e-06, 2.96e-06, 4.52e-06, 6.08e-06, 2.10e-06, 2.37e-05, 4.85e-04, 5.36e-03, 5.16e-02, 2.12e-01],
            [3.99e-07, 7.96e-07, 1.21e-06, 1.54e-06, 3.76e-06, 1.65e-05, 4.94e-04, 5.35e-03, 5.16e-02, 2.12e-01],
            [1.07e-07, 2.13e-07, 3.21e-07, 3.27e-07, 5.33e-06, 1.45e-05, 4.96e-04, 5.35e-03, 5.16e-02, 2.12e-01],
            [2.84e-08, 5.67e-08, 8.56e-08, 1.09e-07, 5.42e-06, 1.46e-05, 4.96e-04, 5.35e-03, 5.16e-02, 2.12e-01]
        ]
    )
    # ruff: enable=E501
    # fmt: on

    if class_obj.opts["order"] == 4:
        rel_error = rel_error_04
    elif class_obj.opts["order"] == 7:
        rel_error = rel_error_07
    elif class_obj.opts["order"] == 10:
        rel_error = rel_error_10

    for i, J in enumerate(class_obj.Js):
        if i != 0:
            err_func = scipy.interpolate.interp1d(
                np.log2(Ns), np.log10(rel_error[:, i - 1])
            )
            try:
                class_obj.Js_error[i] = abs(
                    J * 10 ** err_func(np.log2(class_obj.opts["N"]))
                )
            except ValueError:
                if i == 1:
                    print(
                        c.WARN
                        + "Provided number of points N outside range "
                        + "that has been tested for accuracy. PyToF provides "
                        + "no error estimates."
                        + c.ENDC
                    )


def _print_convergence_warning_drho(class_obj, drho):
    """
    This function prints a convergence warning message if rho did not converge
    within the allowed number of iterations.
    """

    string = (
        f"{c.WARN}CONVERGENCE WARNING: drho = "
        f"{c.NUMB}{drho:.0e} "
        f"{c.WARN}> "
        f"{c.NUMB}{class_obj.opts['drho_tol']:.0e} "
        f"{c.WARN}= drho_tol "
        f"{c.INFO}after MaxIterUpdate = "
        f"{c.NUMB}{class_obj.opts['MaxIterUpdate']} "
        f"{c.INFO}iterations.{c.ENDC}"
    )

    print("\n" + string)


def _print_convergence_warning(class_obj, drot, dJs, drho):
    """
    Print a convergence warning message if m, the Js, or rho did not converge
    within the allowed number of iterations.
    """

    # Determine boolean convergence for each quantity
    b1, b2, b3 = (
        drot < class_obj.opts["drot_tol"],
        dJs < class_obj.opts["dJ_tol"],
        drho < class_obj.opts["drho_tol"],
    )

    # Get color codes for each convergence status
    c1, c2, c3 = c.get(b1), c.get(b2), c.get(b3)

    # Helper to format colored values and comparison symbols
    def colored_val(start, val, tol, flag, color):
        symbol = "<" if flag else ">"
        return (
            f"{color}{start}{val:.0e}{color} {symbol} {c.NUMB}{tol:.0e}{color}"
        )

    string = (
        f"{c.WARN}CONVERGENCE WARNING: "
        f"{colored_val('drot = ', drot, class_obj.opts['drot_tol'], b1, c1)}, "
        f"{colored_val('dJ = ', dJs, class_obj.opts['dJ_tol'], b2, c2)}, "
        f"{colored_val('drho = ', drho, class_obj.opts['drho_tol'], b3, c3)} "
        f"{c.INFO}after MaxIterBar = {c.NUMB}{class_obj.opts['MaxIterBar']} "
        f"{c.INFO}iterations.{c.ENDC}"
    )

    print("\n" + string)


def get_r_l_mu(class_obj, mu):
    """
    This function returns an array with shape (class_obj.opts['N'], length(mu))
    that stores the values of r_l_mu that is defined in equation (B.1) from
    arXiv:1708.06177v1 for each l and each mu.

    mu: Array of cos(theta) values where theta is the colatitude.
    """

    # Calculate array with r_l(mu):
    r_l_mu = 1

    for i in range(len(class_obj.ss)):
        # Flip since AlgoToF uses a different ordering logic:
        r_l_mu = r_l_mu + np.outer(
            np.flip(class_obj.ss[i]),
            np.polynomial.legendre.Legendre.basis(2 * i)(mu),
        )

    return r_l_mu * np.outer(class_obj.li, np.ones_like(mu))


def get_U_l_mu(class_obj, mu):
    """
    This function returns an array with shape (class_obj.opts['N'], length(mu))
    that stores the values of the potential U that is defined in equation (B.3)
    from arXiv:1708.06177v1 for each l and each mu.

    mu: Array of cos(theta) values where theta is the colatitude.
    """

    # Flip since AlgoToF uses a different ordering logic:
    return (
        -class_obj.opts["G"]
        * class_obj.opts["M_phys"]
        / class_obj.li[0] ** 3
        * np.outer(class_obj.li**2 * np.flip(class_obj.A0), np.ones_like(mu))
    )


def get_NMoI(class_obj, N=1000):
    """
    This function returns a float that represents the value of the normalized
    moment of inertia.

    N: Number of mu points to use in the integration.
    """

    # Define variables to integrate over:
    mu = np.linspace(-1, 1, N)
    mu_2D = np.outer(np.ones_like(class_obj.li), mu)
    r_l_mu = get_r_l_mu(class_obj, mu)

    # Change of variables in integration:
    dr_dl = np.gradient(r_l_mu, class_obj.li, mu)[0]

    # Perform integrations:
    integrand_l_theta = (
        2
        * np.pi
        * np.outer(class_obj.rhoi, np.ones_like(mu))
        * r_l_mu**2
        * (1 - mu_2D**2)
        * r_l_mu**2
        * dr_dl
    )  # dmu dl
    integrand_l = scipy.integrate.simpson(integrand_l_theta, mu, axis=1)
    MoI = -scipy.integrate.simpson(integrand_l, class_obj.li)  # minus sign due
    # to integration from outside to inside
    NMoI = MoI / (_mass_int(class_obj) * class_obj.li[0] ** 2)

    return NMoI


def set_barotrope(class_obj, fun):
    """
    This function allows the user to internally set the function relating the
    density class_obj.rhoi to the pressure class_obj.Pi via class_obj.rhoi =
    fun(class_obj.Pi, param).

    fun: Function handle for the barotrope model.
    """

    # Set function:
    class_obj.barotrope = fun


def set_density_function(class_obj, fun):
    """
    This function allows the user to internally set the function relating the
    density class_obj.rhoi to the mean level surfaces class_obj.li via
    class_obj.rhoi = fun(class_obj.li, mass, param).

    fun: Function handle for the density model.
    """

    # Set function:
    class_obj.density_function = fun


def relax_to_shape(class_obj, check_consistency=True, maxiter="default"):
    """
    Calls Algorithm from AlgoToF until either the accuray given by
    class_obj.opts['dJ_tol'] is fulfilled or maxiter is reached.

    check_consistency:  If True, checks whether the user provided radius is
                        consistent with the shape of the planet calculated by
                        PyToF
    maxiter:            Maximum number of iterations to perform. If "default",
                        uses the class_obj.opts['MaxIterShape'] value.
    """

    # Initialize variables:
    alphas = np.zeros(len(class_obj.opts["alphas"]))

    if maxiter == "default":
        maxiter = class_obj.opts["MaxIterShape"]

    # Convert barotropic differential rotation parameters to Theory of Figures
    # logic:
    if np.any(class_obj.opts["alphas"]):
        for i in range(len(alphas)):
            alphas[i] = (
                2
                * (i + 1)
                * (class_obj.li[0]) ** (2 * i)
                * class_obj.opts["alphas"][i]
                / (
                    (
                        class_obj.m_rot_calc
                        * class_obj.opts["G"]
                        * class_obj.opts["M_phys"]
                    )
                    / class_obj.li[0] ** 3
                )
                / class_obj.opts["R_ref"] ** (2 * (i + 1))
            )

    # Measure ToF performance:
    tic = time.time()

    # Implement the Theory of Figures:
    class_obj.Js, out = AlgoToF.Algorithm(
        class_obj.li,
        class_obj.rhoi,
        class_obj.m_rot_calc,
        order=class_obj.opts["order"],
        n_bin=class_obj.opts["n_bin"],
        tol=class_obj.opts["dJ_tol"],
        maxiter=maxiter,
        verbosity=class_obj.opts["verbosity"],
        debug_plot=class_obj.opts["debug_plot"],
        R_ref=class_obj.opts["R_ref"],
        ss_initial=class_obj.ss,
        alphas=alphas,
        H=class_obj.opts["H"],
    )

    # Measure ToF performance:
    toc = time.time()

    # Verbosity output:
    if class_obj.opts["verbosity"] > 2:
        print(
            "\n"
            f"{c.INFO}Relaxing to shape done in "
            f"{c.NUMB}{toc - tic:.2e} "
            f"{c.INFO}seconds.{c.ENDC}"
        )

    # Save results:
    class_obj.A0 = out.A0  # inside->outside instead of outside->inside since
    # AlgoToF uses a different ordering logic!
    class_obj.ss = out.ss  # inside->outside instead of outside->inside since
    # AlgoToF uses a different ordering logic!
    class_obj.SS = out.SS  # inside->outside instead of outside->inside since
    # AlgoToF uses a different ordering logic!
    class_obj.R_eq_to_R_m = out.R_eq_to_R_m
    class_obj.R_po_to_R_m = out.R_po_to_R_m

    _get_Js_errors(class_obj)

    if check_consistency:
        # Check equatorial radius consistency
        if class_obj.opts["R_phys"][1] == "equatorial" and not np.isclose(
            class_obj.R_eq_to_R_m * class_obj.li[0],
            class_obj.opts["R_phys"][0],
        ):
            print(
                f"{c.WARN}WARNING: "
                f"{c.INFO}Your provided equatorial radius is not consistent "
                f"with the shape of the planet calculated by PyToF!{c.ENDC}"
            )
            print(
                f"{c.INFO}Your value: {c.NUMB}"
                f"{class_obj.opts['R_phys'][0]:.5e} {c.INFO}/ PyToF value: "
                f"{c.NUMB}{class_obj.R_eq_to_R_m * class_obj.li[0]:.5e}"
                f"{c.ENDC}"
            )

        # Check polar radius consistency
        if class_obj.opts["R_phys"][1] == "polar" and not np.isclose(
            class_obj.R_po_to_R_m * class_obj.li[0],
            class_obj.opts["R_phys"][0],
        ):
            print(
                f"{c.WARN}WARNING: "
                f"{c.INFO}Your provided polar radius is not consistent with "
                f"the shape of the planet calculated by PyToF!{c.ENDC}"
            )
            print(
                f"{c.INFO}Your value: {c.NUMB}"
                f"{class_obj.opts['R_phys'][0]:.5e} {c.INFO}/ PyToF value: "
                f"{c.NUMB}{class_obj.R_po_to_R_m * class_obj.li[0]:.5e}"
                f"{c.ENDC}"
            )

    return out.it


def relax_to_barotrope(class_obj):
    """
    Calls relax_to_shape() and _update_densities_barotrope() until either the
    accuray given by class_obj.opts['dJ_tol'], class_obj.opts['drot_tol'] and
    class_obj.opts['drho_tol'] is fulfilled or class_obj.opts['MaxIterBar'] is
    reached.
    """

    # Measure ToF performance:
    tic = time.time()

    # Only relevant if debug plots are activated:
    class_obj.bugfix_iter = 0

    # Call relax_to_shape() and ensure consistency for the first time:
    relax_to_shape(class_obj, check_consistency=False, maxiter=2)
    _ensure_consistency(class_obj)
    IterBar = 1

    # Converge on gravitational moments:
    while IterBar < class_obj.opts["MaxIterBar"]:
        # Store old gravitational moment values:
        old_Js = class_obj.Js

        # Define iteration counter for density loop:
        IterUpdate = 1

        if class_obj.opts["debug_plot"]:
            debug_FunctionsToF_plot(
                class_obj, new=True, iteration=class_obj.bugfix_iter
            )

        # Converge on densities:
        while IterUpdate < class_obj.opts["MaxIterUpdate"]:
            # Store old values:
            old_rho = class_obj.rhoi

            # Call _update_densities_barotrope():
            _update_densities_barotrope(class_obj)

            # Update drho, ignore first entry to avoid possible division by
            # zero:
            drho = np.max(np.abs(class_obj.rhoi[1:] / old_rho[1:] - 1))

            # Check convergence:
            if drho < class_obj.opts["drho_tol"]:
                break

            # Update iteration parameter:
            IterUpdate += 1

        # Only relevant if debug plots are activated:
        class_obj.bugfix_iter += 1

        # Warning if not converged:
        if IterUpdate == class_obj.opts["MaxIterUpdate"]:
            _print_convergence_warning_drho(class_obj, drho)

        # Call relax_to_shape():
        relax_to_shape(class_obj, check_consistency=False, maxiter=2)

        # Store old values:
        old_m = class_obj.m_rot_calc
        old_rho = class_obj.rhoi

        # Ensure consistency:
        _ensure_consistency(class_obj)

        # Check convergence, and address division by zero issues:
        if old_m != 0:
            drot = np.abs(class_obj.m_rot_calc / old_m - 1)
        else:
            drot = 0.0

        mask = ~np.logical_and(class_obj.Js == 0, old_Js == 0)

        dJs = np.max(np.abs(class_obj.Js[mask] / old_Js[mask] - 1))
        drho = np.max(np.abs(class_obj.rhoi[1:] / old_rho[1:] - 1))

        if (
            drot < class_obj.opts["drot_tol"]
            and dJs < class_obj.opts["dJ_tol"]
            and drho < class_obj.opts["drho_tol"]
        ):
            break

        # Update iteration parameter:
        IterBar += 1

    # Measure ToF performance:
    toc = time.time()

    # Warning if not converged:
    if IterBar == class_obj.opts["MaxIterBar"]:
        _print_convergence_warning(class_obj, drot, dJs, drho)

    # Verbosity output:
    if class_obj.opts["verbosity"] > 1:
        print(
            "\n"
            f"{c.INFO}Relaxing to barotrope done in "
            f"{c.NUMB}{toc - tic:.2e} "
            f"{c.INFO}seconds.{c.ENDC}"
        )

    return IterBar


def relax_to_density(class_obj):
    """
    Calls relax_to_shape() until either the accuray given by
    class_obj.opts['dJ_tol'], class_obj.opts['drot_tol'] and
    class_obj.opts['drho_tol'] is fulfilled or class_obj.opts['MaxIterDen'] is
    reached.
    """

    # Measure ToF performance:
    tic = time.time()

    # Only relevant if debug plots are activated:
    class_obj.bugfix_iter = 0

    # Call relax_to_shape() and ensure consistency for the first time:
    relax_to_shape(class_obj, check_consistency=False, maxiter=2)
    _ensure_consistency(class_obj)
    IterDen = 1

    # Converge on gravitational moments:
    while IterDen < class_obj.opts["MaxIterDen"]:
        # Store old gravitational moment values:
        old_Js = class_obj.Js

        # Define iteration counter for density loop:
        IterUpdate = 1

        if class_obj.opts["debug_plot"]:
            debug_FunctionsToF_plot(
                class_obj, new=True, iteration=class_obj.bugfix_iter
            )

        # Converge on densities, this loop terminates after one iteration if no
        # atmosphere is provided:
        while IterUpdate < class_obj.opts["MaxIterUpdate"]:
            # Store old values:
            old_rho = class_obj.rhoi

            # Calculates the pressure values according to hydrostatic
            # equilibrium:
            _pressurize(class_obj)

            # Optional, use a provided atmospheric model:
            if class_obj.opts["use_atmosphere"]:
                _apply_atmosphere(class_obj)

            # Ensure physical mass stays unaffacted:
            _fixmass(class_obj)

            # Update drho, ignore first entry to avoid possible division by
            # zero:
            drho = np.max(np.abs(class_obj.rhoi[1:] / old_rho[1:] - 1))

            # Check convergence:
            if drho < class_obj.opts["drho_tol"]:
                break

            # Update iteration parameter:
            IterUpdate += 1

        # Only relevant if debug plots are activated:
        class_obj.bugfix_iter += 1

        # Warning if not converged:
        if IterUpdate == class_obj.opts["MaxIterUpdate"]:
            _print_convergence_warning_drho(class_obj, drho)

        # Call relax_to_shape():
        relax_to_shape(class_obj, check_consistency=False, maxiter=2)

        # Store old values:
        old_m = class_obj.m_rot_calc
        old_rho = class_obj.rhoi

        # Ensure consistency:
        _ensure_consistency(class_obj)

        # Check convergence, and address division by zero issues:
        if old_m != 0:
            drot = np.abs(class_obj.m_rot_calc / old_m - 1)
        else:
            drot = 0.0

        mask = ~np.logical_and(class_obj.Js == 0, old_Js == 0)

        dJs = np.max(np.abs(class_obj.Js[mask] / old_Js[mask] - 1))
        drho = np.max(np.abs(class_obj.rhoi[1:] / old_rho[1:] - 1))

        if (
            drot < class_obj.opts["drot_tol"]
            and dJs < class_obj.opts["dJ_tol"]
            and drho < class_obj.opts["drho_tol"]
        ):
            break

        # Update iteration parameter:
        IterDen += 1

    # Measure ToF performance:
    toc = time.time()

    # Warning if not converged:
    if IterDen == class_obj.opts["MaxIterDen"]:
        _print_convergence_warning(class_obj, drot, dJs, drho)

    # Verbosity output:
    if class_obj.opts["verbosity"] > 1:
        print(
            "\n"
            f"{c.INFO}Relaxing to density done in "
            f"{c.NUMB}{toc - tic:.2e} "
            f"{c.INFO}seconds.{c.ENDC}"
        )

    return IterDen
