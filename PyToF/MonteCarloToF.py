########################################################
# Author of this version: Luca Morf - luca.morf@uzh.ch #
########################################################


import multiprocessing
import os

import emcee
import numpy as np

from PyToF.messages import paint, say

multiprocessing.set_start_method("fork", force=True)


def _check_phys(tof):
    """
    This function returns True if the current solutions
    (tof.rhoi, tof.Pi, tof.Js) are unphysical or consisting
    of NaN values. Returns False if everything is ok.
    """

    # Density inversions are considered to be unphysical:
    if np.any(np.diff(tof.rhoi) < 0):
        say(
            tof.opts["verbosity"],
            2,
            "Physicality warning: density inversion!",
            kind="warn",
        )

        return True

    # There should be no NaN values:
    elif (
        np.isnan(tof.rhoi).any()
        or np.isnan(tof.Pi).any()
        or np.isnan(tof.Js).any()
    ):
        say(
            tof.opts["verbosity"],
            2,
            "Physicality warning: NaN values!",
            kind="warn",
        )

        return True

    # There should be no negative density or pressure values:
    elif (np.min(tof.rhoi) < 0) or (np.min(tof.Pi) < 0):
        say(
            tof.opts["verbosity"],
            2,
            "Physicality warning: negative density or pressure values!",
            kind="warn",
        )

        return True

    # There should be no density or pressure values above the given maxima:
    elif (np.max(tof.rhoi) > tof.opts["rho_max"]) or (
        np.max(tof.Pi) > tof.opts["P_max"]
    ):
        say(
            tof.opts["verbosity"],
            2,
            "Physicality warning: density or pressure values above the "
            "given maxima!",
            kind="warn",
        )

        return True

    # We passed all checks, we can return False:
    else:
        return False


def set_check_param(self, fun):
    """
    This function allows the user to set the function that checks the
    parameters for the barotrope or density function. The function fun should
    have the form fun(param) and should return True if the parameters are
    nonsense (i.e. out of bounds).

    fun: Function handle for the density model.
    """

    # Set function:
    self.check_param = fun


def baro_cost_function(self, param, return_sum=True):
    """
    Calls relax_to_barotrope() and returns values specifying how far off the
    calculated Js are from opts['target_Js'] given opts['sigma_Js'].

    param: Array of parameters for the barotrope function.
    return_sum: If True, returns the sum of the costs for all Js. If False,
                returns an array of costs for each J separately.
    """

    # Infinite cost if the parameters are out of bounds:
    if self.check_param(param):
        if return_sum:
            return -np.inf

        else:
            return [-np.inf] * len(self.opts["target_Js"])

    # Reset to initial conditions:
    self.reset()

    # Update the internal parameters:
    self.baro_param_calc = param

    # Call relax_to_barotrope():
    try:
        it = self.relax_to_barotrope()

    except ValueError:
        if return_sum:
            return -np.inf

        else:
            return [-np.inf] * len(self.opts["target_Js"])

    # Infinite cost if the result is unphysical:
    if _check_phys(self) or it == self.opts["max_iter_barotrope"]:
        if return_sum:
            return -np.inf

        else:
            return [-np.inf] * len(self.opts["target_Js"])

    # Calculate the cost, i.e. the deviation from the target values:
    costJs = np.zeros_like(self.opts["target_Js"])

    for i in range(len(costJs)):
        costJs[i] = (
            -((self.Js[i + 1] - self.opts["target_Js"][i]) ** 2)
            / self.opts["sigma_Js"][i] ** 2
        )

    # Return the cost value:
    if return_sum:
        return np.sum(costJs)

    else:
        return costJs


def dens_cost_function(self, param, return_sum=True):
    """
    Calls relax_to_barotrope() and returns values specifying how far off the
    calculated Js are from opts['target_Js'] given opts['sigma_Js'].

    param: Array of parameters for the density function.
    return_sum: If True, returns the sum of the costs for all Js. If False,
                returns an array of costs for each J separately.
    """

    # Infinite cost if the parameters are out of bounds:
    if self.check_param(param):
        if return_sum:
            return -np.inf

        else:
            return [-np.inf] * len(self.opts["target_Js"])

    # Reset to initial conditions:
    self.reset()

    # Get densities according to self.density_function():
    self.rhoi = self.density_function(
        self.li, self.opts["M_phys"], param=param
    )

    # Update the internal parameters:
    self.dens_param_calc = param

    # Call relax_to_density():
    try:
        it = self.relax_to_density()

    except ValueError:
        if return_sum:
            return -np.inf

        else:
            return [-np.inf] * len(self.opts["target_Js"])

    # Infinite cost if the result is unphysical:
    if _check_phys(self) or it == self.opts["max_iter_density"]:
        if return_sum:
            return -np.inf

        else:
            return [-np.inf] * len(self.opts["target_Js"])

    # Calculate the cost, i.e. the deviation from the target values:
    costJs = np.zeros_like(self.opts["target_Js"])

    for i in range(len(costJs)):
        costJs[i] = (
            -((self.Js[i + 1] - self.opts["target_Js"][i]) ** 2)
            / self.opts["sigma_Js"][i] ** 2
        )

    # Return the cost value:
    if return_sum:
        return np.sum(costJs)

    else:
        return costJs


def baro_log_prob(param, tof):
    """
    Changes order of arguments for the emcee algorithm.
    """

    return baro_cost_function(tof, param)


def dens_log_prob(param, tof):
    """
    Changes order of arguments for the emcee algorithm.
    """

    return dens_cost_function(tof, param)


def run_baro_MC(self, nwalkers, steps, Ncores=8, parallelize=False):
    """
    Uses the emcee algorithm for the given amount of steps and creates nwalkers
    many parameter candidates (i.e. walkers) that will explore the allowed
    parameter space (constrained by check_phys() and check_param()).
    """

    # Find a random set of starting parameters:
    param_0 = np.zeros((nwalkers, len(self.opts["baro_param_init"])))

    # Set up iterative procedure:
    i = 0

    while i < nwalkers:
        # Propose a new candidate:
        param_0[i, :] = (
            np.random.rand(1, len(self.opts["baro_param_init"]))
            * self.opts["baro_param_init"]
        )

        # Propose new candidates as long the previous one is out of bounds:
        while self.check_param(param_0[i, :]):
            param_0[i, :] = (
                np.random.rand(1, len(self.opts["baro_param_init"]))
                * self.opts["baro_param_init"]
            )

        # Reset to initial conditions:
        self.reset()

        # Update the internal parameters:
        self.baro_param_calc = param_0[i, :]

        # Call relax_to_barotrope():
        try:
            it = self.relax_to_barotrope()

        except ValueError:
            continue

        # Only proceed if the candidate also yields a physical solution:
        if (not _check_phys(self)) and it < self.opts["max_iter_barotrope"]:
            say(
                self.opts["verbosity"],
                1,
                "Generated initial conditions for walker number: {i}",
                i=i,
            )

            i = i + 1

    if self.opts["verbosity"] > 0:
        progress = True
        average = 0

        for i in range(nwalkers):
            average += baro_cost_function(self, param_0[i, :]) / nwalkers

        say(
            self.opts["verbosity"],
            1,
            "Average cost of a walker with the generated initial "
            "parameters: {average:.2e}",
            average=average,
        )

    else:
        progress = False

    # Do the MCMC algorithm in a parallel manner on multiple cores:
    if parallelize:
        with multiprocessing.Pool(processes=Ncores) as pool:
            sampler = emcee.EnsembleSampler(
                nwalkers,
                len(self.opts["baro_param_init"]),
                baro_log_prob,
                args=[self],
                pool=pool,
            )

            # Run the Markov Chain Monte Carlo class from emcee:
            index = 0
            autocorr = np.empty(steps)
            old_tau = np.inf

            for _sample in sampler.sample(
                param_0, iterations=steps, progress=progress
            ):
                if sampler.iteration % 100:
                    continue

                tau = sampler.get_autocorr_time(tol=0)
                autocorr[index] = np.mean(tau)
                index = index + 1

                converged = np.all(tau * 100 < sampler.iteration) and np.all(
                    np.abs(old_tau - tau) / tau < 0.01
                )
                if converged:
                    break
                old_tau = tau

            state = _sample

    # Do the MCMC algorithm on a single core:
    else:
        # Set up the Markov Chain Monte Carlo class from emcee:
        sampler = emcee.EnsembleSampler(
            nwalkers,
            len(self.opts["baro_param_init"]),
            baro_log_prob,
            args=[self],
        )

        # Run the Markov Chain Monte Carlo class from emcee:
        index = 0
        autocorr = np.empty(steps)
        old_tau = np.inf

        for _sample in sampler.sample(
            param_0, iterations=steps, progress=progress
        ):
            if sampler.iteration % 100:
                continue

            tau = sampler.get_autocorr_time(tol=0)
            autocorr[index] = np.mean(tau)
            index = index + 1

            converged = np.all(tau * 100 < sampler.iteration) and np.all(
                np.abs(old_tau - tau) / tau < 0.01
            )
            if converged:
                break
            old_tau = tau

        state = _sample

    say(
        self.opts["verbosity"],
        1,
        "Average cost of a walker after the Markov Chain Monte Carlo walk: "
        "{average:.2e}",
        average=np.sum(state[1]) / nwalkers,
    )

    return state[0], autocorr[:index]


def run_dens_MC(self, nwalkers, steps, Ncores=8, parallelize=False):
    """
    Uses the emcee algorithm for the given amount of steps and creates nwalkers
    many parameter candidates (i.e. walkers) that will explore the allowed
    parameter space (constrained by check_phys() and check_param()).
    """

    # Find a random set of starting parameters:
    param_0 = np.zeros((nwalkers, len(self.opts["dens_param_init"])))

    # Set up iterative procedure:
    i = 0

    while i < nwalkers:
        # Propose a new candidate:
        param_0[i, :] = (
            np.random.rand(1, len(self.opts["dens_param_init"]))
            * self.opts["dens_param_init"]
        )

        # Propose new candidates as long the previous one is out of bounds:
        while self.check_param(param_0[i, :]):
            param_0[i, :] = (
                np.random.rand(1, len(self.opts["dens_param_init"]))
                * self.opts["dens_param_init"]
            )

        # Reset to initial conditions:
        self.reset()

        # Get densities according to self.density_function():
        self.rhoi = self.density_function(
            self.li, self.opts["M_phys"], param=param_0[i, :]
        )

        # Update the internal parameters:
        self.dens_param_calc = param_0[i, :]

        # Call relax_to_density():
        try:
            it = self.relax_to_density()

        except ValueError:
            continue

        # Only proceed if the candidate also yields a physical solution:
        if (not _check_phys(self)) and it < self.opts["max_iter_density"]:
            say(
                self.opts["verbosity"],
                1,
                "Generated initial conditions for walker number: {i}",
                i=i,
            )

            i = i + 1

    if self.opts["verbosity"] > 0:
        progress = True
        average = 0

        for i in range(nwalkers):
            average += dens_cost_function(self, param_0[i, :]) / nwalkers

        say(
            self.opts["verbosity"],
            1,
            "Average cost of a walker with the generated initial "
            "parameters: {average:.2e}",
            average=average,
        )

    else:
        progress = False

    # Do the MCMC algorithm in a parallel manner on multiple cores:
    if parallelize:
        with multiprocessing.Pool(processes=Ncores) as pool:
            sampler = emcee.EnsembleSampler(
                nwalkers,
                len(self.opts["dens_param_init"]),
                dens_log_prob,
                args=[self],
                pool=pool,
            )

            # Run the Markov Chain Monte Carlo class from emcee:
            index = 0
            autocorr = np.empty(steps)
            old_tau = np.inf

            for _sample in sampler.sample(
                param_0, iterations=steps, progress=progress
            ):
                if sampler.iteration % 100:
                    continue

                tau = sampler.get_autocorr_time(tol=0)
                autocorr[index] = np.mean(tau)
                index = index + 1

                converged = np.all(tau * 100 < sampler.iteration) and np.all(
                    np.abs(old_tau - tau) / tau < 0.01
                )
                if converged:
                    break
                old_tau = tau

            state = _sample

    # Do the MCMC algorithm on a single core:
    else:
        # Set up the Markov Chain Monte Carlo class from emcee:
        sampler = emcee.EnsembleSampler(
            nwalkers,
            len(self.opts["dens_param_init"]),
            dens_log_prob,
            args=[self],
        )

        # Run the Markov Chain Monte Carlo class from emcee:
        index = 0
        autocorr = np.empty(steps)
        old_tau = np.inf

        for _sample in sampler.sample(
            param_0, iterations=steps, progress=progress
        ):
            if sampler.iteration % 100:
                continue

            tau = sampler.get_autocorr_time(tol=0)
            autocorr[index] = np.mean(tau)
            index = index + 1

            converged = np.all(tau * 100 < sampler.iteration) and np.all(
                np.abs(old_tau - tau) / tau < 0.01
            )
            if converged:
                break
            old_tau = tau

        state = _sample

    say(
        self.opts["verbosity"],
        1,
        "Average cost of a walker after the Markov Chain Monte Carlo walk: "
        "{average:.2e}",
        average=np.sum(state[1]) / nwalkers,
    )

    return state[0], autocorr[:index]


def classify_and_save_state(
    self,
    state,
    what_model,
    what_save="none",
    log_CGS_units=False,
    path_name=None,
    file_name="walker",
):
    """
    Classifies the nwalkers many parameter candidates according to their
    physicality, i.e. by checking if they agree with the physical data within 1
    sigma. Also allows for saving .txt files with the radii, densities and
    pressures within the planet generated by the walker parameters.
    """

    if path_name is None:
        path_name = os.getcwd()

    # Storage arrays:
    matches_observed_Js = []
    rho_maxs = []

    # Loop through walkers:
    for i in range(len(state[:, 0])):
        # Calculate the physical cost:
        if what_model == "baro":
            costJs = baro_cost_function(self, state[i, :], return_sum=False)

        elif what_model == "dens":
            costJs = dens_cost_function(self, state[i, :], return_sum=False)

        else:
            raise KeyError(
                "Invalid keyword for what_model! Use 'baro' or 'dens'."
            )

        # Check if there is agreement with physical data within 1 sigma:
        if (np.abs(costJs) < 1e0).all():
            kind = "good"
            matches_observed_Js.append(True)

        else:
            kind = "warn"
            matches_observed_Js.append(False)

        # Save .txt file if wanted by the user:
        if what_save == "all":
            if log_CGS_units and not os.path.exists(
                path_name + "/" + file_name + "_CGS_" + str(i) + ".txt"
            ):
                np.savetxt(
                    path_name + "/" + file_name + "_CGS_" + str(i) + ".txt",
                    np.transpose(
                        np.array(
                            [
                                self.li * 100,
                                np.log10(self.rhoi) - 3,
                                np.log10(self.Pi) + 1,
                            ]
                        )
                    ),
                )

            if not log_CGS_units and not os.path.exists(
                path_name + "/" + file_name + "_SI_" + str(i) + ".txt"
            ):
                np.savetxt(
                    path_name + "/" + file_name + "_SI_" + str(i) + ".txt",
                    np.transpose(np.array([self.li, self.rhoi, self.Pi])),
                )

        elif what_save == "good" and (np.abs(costJs) < 1e0).all():
            if log_CGS_units:
                np.savetxt(
                    path_name + "/" + file_name + "_CGS_" + str(i) + ".txt",
                    np.transpose(
                        np.array(
                            [
                                self.li * 100,
                                np.log10(self.rhoi) - 3,
                                np.log10(self.Pi) + 1,
                            ]
                        )
                    ),
                )

            else:
                np.savetxt(
                    path_name + "/" + file_name + "_SI_" + str(i) + ".txt",
                    np.transpose(np.array([self.li, self.rhoi, self.Pi])),
                )

        elif what_save == "none":
            pass

        elif not what_save == "good":
            raise KeyError(
                "Invalid keyword for what_save! Use 'all' or 'good' or 'none'."
            )

        # Update storage arrays:
        rho_maxs.append(np.max(self.rhoi))

        # Verbosity output:
        verbosity = self.opts["verbosity"]
        if verbosity > 0:
            header = " " * 9
            mine = ""
            target = ""

            for j, cost_j in enumerate(costJs):
                label = f"     J{2 * (j + 1)}     "
                header += paint(label, "good" if abs(cost_j) < 1e0 else "warn")
                mine += f"{self.Js[j + 1]:.5e} "
                target += f"{self.opts['target_Js'][j]:.5e} "

            say(
                verbosity,
                1,
                "\nWalker #{i} yields a total cost of {total_cost:.2e}.",
                kind=kind,
                i=i,
                total_cost=np.sum(costJs),
            )
            print(header)
            say(verbosity, 1, "My code: {values}", values=mine)
            say(verbosity, 1, "Target:  {values}\n", values=target)

    return np.array(matches_observed_Js), np.array(rho_maxs)
