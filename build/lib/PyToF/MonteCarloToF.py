########################################################
# Author of this version: Luca Morf - luca.morf@uzh.ch #
########################################################

import numpy as np
import os
import functools
import emcee 
from multiprocessing import Pool 

from PyToF.color import c

def _check_phys(class_obj):

    """
    This function returns True if the current solutions (class_obj.rhoi, class_obj.Pi, class_obj.Js) are
    unphysical or consisting of NaN values. Returns False if everything is ok.
    """

    #Density inversions are considered to be unphysical:
    if (np.any(np.diff(class_obj.rhoi) < 0)):

        if (class_obj.opts['verbosity'] > 1):

            print(c.WARN + 'Physicality warning: density inversion!' + c.ENDC)

        return True

    #There should be no NaN values:
    elif np.isnan(class_obj.rhoi).any() or np.isnan(class_obj.Pi).any() or np.isnan(class_obj.Js).any():

        if (class_obj.opts['verbosity'] > 1):

            print(c.WARN + 'Physicality warning: NaN values!' + c.ENDC)

        return True
    
    #There should be no negative density or pressure values:
    elif (np.min(class_obj.rhoi)<0) or (np.min(class_obj.Pi)<0):

        if (class_obj.opts['verbosity'] > 1):

            print(c.WARN + 'Physicality warning: negative density or pressure values!' + c.ENDC)

        return True

    #There should be no density or pressure values above the given maxima:
    elif (np.max(class_obj.rhoi)>class_obj.opts['rho_MAX']) or (np.max(class_obj.Pi)>class_obj.opts['P_MAX']):

        if (class_obj.opts['verbosity'] > 1):

            print(c.WARN + 'Physicality warning: density or pressure values above the given maxima!' + c.ENDC)

        return True
    
    #We passed all checks, we can return False:
    else:

        return False

def set_check_param(class_obj, fun):
    
    """
    This function allows the user to set the function that checks the parameters for the barotrope or density function.
    The function fun should have the form fun(param, give_atmosphere_index=False) and should:

    - return True if the parameters are nonsense (i.e. out of bounds)
    - return only the parameter specifiyng the location of the atmosphere if give_atmosphere_index==True
    """
    
    #Set function:
    class_obj.check_param = fun       

def baro_cost_function(class_obj, param, return_sum=True):

    """
    Calls relax_to_barotrope() and returns values specifying how far off the calculated Js are from opts['Target_Js'] given opts['Sigma_Js'].
    """

    #Infinite cost if the parameters are out of bounds:
    if class_obj.check_param(param):

        if return_sum:

            return -np.inf

        else:

            return [-np.inf]*len(class_obj.opts['Target_Js'])

    #Reset to initial conditions:
    class_obj._set_IC()

    #Update the internal parameters:
    class_obj.baro_param_calc = param

    #Call relax_to_barotrope():
    try:

        it = class_obj.relax_to_barotrope(fixradius=True, fixmass=True, fixrot=True, pressurize=True)

    except:

        if return_sum:

            return -np.inf

        else:

            return [-np.inf]*len(class_obj.opts['Target_Js'])

    #Infinite cost if the result is unphysical:
    if _check_phys(class_obj) or it == class_obj.opts['MaxIterBar']:

        if return_sum:

            return -np.inf

        else:

            return [-np.inf]*len(class_obj.opts['Target_Js'])

    #Calculate the cost, i.e. the deviation from the target values:
    costJs = np.zeros_like(class_obj.opts['Target_Js'])

    for i in range(len(costJs)):

        costJs[i] = -(class_obj.Js[i+1] - class_obj.opts['Target_Js'][i])**2/class_obj.opts['Sigma_Js'][i]**2

    #Return the cost value:
    if return_sum:

        return np.sum(costJs)

    else:

        return costJs

def dens_cost_function(class_obj, param, return_sum=True):

    """
    Calls relax_to_barotrope() and returns values specifying how far off the calculated Js are from opts['Target_Js'] given opts['Sigma_Js'].
    """

    #Infinite cost if the parameters are out of bounds:
    if class_obj.check_param(param):

        if return_sum:

            return -np.inf

        else:

            return [-np.inf]*len(class_obj.opts['Target_Js'])

    #Reset to initial conditions:
    class_obj._set_IC()

    #Get densities according to class_obj.density_function():
    class_obj.rhoi = class_obj.density_function(class_obj.li, param=param)

    #Update the internal parameters:
    class_obj.dens_param_calc = param

    #Call relax_to_density():
    try:

        it = class_obj.relax_to_density(fixradius=True, fixmass=True, fixrot=True, pressurize=True)
    
    except:

        if return_sum:

            return -np.inf

        else:

            return [-np.inf]*len(class_obj.opts['Target_Js'])

    #Infinite cost if the result is unphysical:
    if _check_phys(class_obj) or it == class_obj.opts['MaxIterDen']:

        if return_sum:

            return -np.inf

        else:

            return [-np.inf]*len(class_obj.opts['Target_Js'])

    #Calculate the cost, i.e. the deviation from the target values:
    costJs = np.zeros_like(class_obj.opts['Target_Js'])

    for i in range(len(costJs)):

        costJs[i] = -(class_obj.Js[i+1] - class_obj.opts['Target_Js'][i])**2/class_obj.opts['Sigma_Js'][i]**2

    #Return the cost value:
    if return_sum:

        return np.sum(costJs)

    else:

        return costJs

def baro_log_prob(param, class_obj):

    """
    Changes order of arguments for the emcee algorithm.
    """

    return baro_cost_function(class_obj, param)

def dens_log_prob(param, class_obj):

    """
    Changes order of arguments for the emcee algorithm.
    """

    return dens_cost_function(class_obj, param)

def run_baro_MC(class_obj, nwalkers, steps, Ncores=8, parallelize=False):

    """
    Uses the emcee algorithm for the given amount of steps and creates nwalkers many parameter candidates
    (i.e. walkers) that will explore the allowed parameter space (constrained by check_phys() and check_param()).
    """

    #Find a random set of starting parameters:
    param_0     = np.zeros((nwalkers,len(class_obj.opts['baro_param_init'])))

    #Set up iterative procedure:
    i = 0

    while i < nwalkers:

        #Propose a new candidate:
        param_0[i,:]        = np.random.rand(1, len(class_obj.opts['baro_param_init'])) * class_obj.opts['baro_param_init']

        #Propose new candidates as long the previous one is out of bounds:
        while class_obj.check_param(param_0[i,:]):

            param_0[i,:]    = np.random.rand(1, len(class_obj.opts['baro_param_init'])) * class_obj.opts['baro_param_init']
        
        #Reset to initial conditions:
        class_obj._set_IC()

        #Update the internal parameters:
        class_obj.baro_param_calc = param_0[i,:]

        #Call relax_to_barotrope():
        try:

            it = class_obj.relax_to_barotrope(fixradius=True, fixmass=True, fixrot=True, pressurize=True)

        except:

            continue

        #Only proceed if the candidate also yields a physical solution:
        if (not _check_phys(class_obj)) and it < class_obj.opts['MaxIterBar']:

            if (class_obj.opts['verbosity'] > 0):

                print(c.INFO + 'Generated initial conditions for walker number: ' + c.NUMB + str(i) + c.ENDC)

            i = i+1

    if (class_obj.opts['verbosity'] > 0):

        progress = True
        average = 0

        for i in range(nwalkers):

                average += baro_cost_function(class_obj, param_0[i,:])/nwalkers

        print(c.INFO + 'Average cost of a walker with the generated initial parameters: ' + c.NUMB + '{:.2e}'.format(average) + c.ENDC)

    else:

        progress = False

    #Do the MCMC algorithm in a parallel manner on multiple cores:
    if parallelize:

        with Pool(processes = Ncores) as pool:

            sampler  = emcee.EnsembleSampler(nwalkers, len(class_obj.opts['baro_param_init']), baro_log_prob, args=[class_obj], pool=pool)

            #Run the Markov Chain Monte Carlo class from emcee:
            index    = 0; autocorr = np.empty(steps); old_tau  = np.inf

            for sample in sampler.sample(param_0, iterations=steps, progress=progress):

                if sampler.iteration % 100:
                    continue

                tau             = sampler.get_autocorr_time(tol=0)
                autocorr[index] = np.mean(tau)
                index           = index + 1

                converged       = np.all(tau*100 < sampler.iteration) and np.all(np.abs(old_tau - tau) / tau < 0.01)
                if converged:
                    break
                old_tau = tau
            
            state = sample

    #Do the MCMC algorithm on a single core:
    else:

        #Set up the Markov Chain Monte Carlo class from emcee:
        sampler  = emcee.EnsembleSampler(nwalkers, len(class_obj.opts['baro_param_init']), baro_log_prob, args=[class_obj])

        #Run the Markov Chain Monte Carlo class from emcee:
        index    = 0; autocorr = np.empty(steps); old_tau  = np.inf

        for sample in sampler.sample(param_0, iterations=steps, progress=progress):

            if sampler.iteration % 100:
                continue

            tau             = sampler.get_autocorr_time(tol=0)
            autocorr[index] = np.mean(tau)
            index           = index + 1

            converged       = np.all(tau*100 < sampler.iteration) and np.all(np.abs(old_tau - tau) / tau < 0.01)
            if converged:
                break
            old_tau = tau
        
        state = sample
        
    if (class_obj.opts['verbosity'] > 0):

        print(c.INFO + 'Average cost of a walker after the Markov Chain Monte Carlo walk: ' + c.NUMB + '{:.2e}'.format(np.sum(state[1])/nwalkers) + c.ENDC)
        
    return state[0], autocorr[:index]

def run_dens_MC(class_obj, nwalkers, steps, Ncores=8, parallelize=False):

    """
    Uses the emcee algorithm for the given amount of steps and creates nwalkers many parameter candidates
    (i.e. walkers) that will explore the allowed parameter space (constrained by check_phys() and check_param()).
    """

    #Find a random set of starting parameters:
    param_0     = np.zeros((nwalkers,len(class_obj.opts['dens_param_init'])))

    #Set up iterative procedure:
    i = 0

    while i < nwalkers:

        #Propose a new candidate:
        param_0[i,:]        = np.random.rand(1, len(class_obj.opts['dens_param_init'])) * class_obj.opts['dens_param_init']

        #Propose new candidates as long the previous one is out of bounds:
        while class_obj.check_param(param_0[i,:]):

            param_0[i,:]    = np.random.rand(1, len(class_obj.opts['dens_param_init'])) * class_obj.opts['dens_param_init']
        
        #Reset to initial conditions:
        class_obj._set_IC()

        #Get densities according to class_obj.density_function():
        class_obj.rhoi = class_obj.density_function(class_obj.li, param=param_0[i,:])

        #Update the internal parameters:
        class_obj.dens_param_calc = param_0[i,:]

        #Call relax_to_density():
        try:

            it = class_obj.relax_to_density(fixradius=True, fixmass=True, fixrot=True, pressurize=True)

        except:

            continue

        #Only proceed if the candidate also yields a physical solution:
        if (not _check_phys(class_obj)) and it < class_obj.opts['MaxIterDen']:

            if (class_obj.opts['verbosity'] > 0):

                print(c.INFO + 'Generated initial conditions for walker number: ' + c.NUMB + str(i) + c.ENDC)

            i = i+1

    if (class_obj.opts['verbosity'] > 0):

        progress = True
        average = 0

        for i in range(nwalkers):

                average += dens_cost_function(class_obj, param_0[i,:])/nwalkers

        print(c.INFO + 'Average cost of a walker with the generated initial parameters: ' + c.NUMB + '{:.2e}'.format(average) + c.ENDC)

    else:

        progress = False

    #Do the MCMC algorithm in a parallel manner on multiple cores:
    if parallelize:

        with Pool(processes = Ncores) as pool:

            sampler  = emcee.EnsembleSampler(nwalkers, len(class_obj.opts['dens_param_init']), dens_log_prob, args=[class_obj], pool=pool)

            #Run the Markov Chain Monte Carlo class from emcee:
            index    = 0; autocorr = np.empty(steps); old_tau  = np.inf

            for sample in sampler.sample(param_0, iterations=steps, progress=progress):

                if sampler.iteration % 100:
                    continue

                tau             = sampler.get_autocorr_time(tol=0)
                autocorr[index] = np.mean(tau)
                index           = index + 1

                converged       = np.all(tau*100 < sampler.iteration) and np.all(np.abs(old_tau - tau) / tau < 0.01)
                if converged:
                    break
                old_tau = tau
            
            state = sample

    #Do the MCMC algorithm on a single core:
    else:

        #Set up the Markov Chain Monte Carlo class from emcee:
        sampler  = emcee.EnsembleSampler(nwalkers, len(class_obj.opts['dens_param_init']), dens_log_prob, args=[class_obj])

        #Run the Markov Chain Monte Carlo class from emcee:
        index    = 0; autocorr = np.empty(steps); old_tau  = np.inf

        for sample in sampler.sample(param_0, iterations=steps, progress=progress):

            if sampler.iteration % 100:
                continue

            tau             = sampler.get_autocorr_time(tol=0)
            autocorr[index] = np.mean(tau)
            index           = index + 1

            converged       = np.all(tau*100 < sampler.iteration) and np.all(np.abs(old_tau - tau) / tau < 0.01)
            if converged:
                break
            old_tau = tau
        
        state = sample
        
    if (class_obj.opts['verbosity'] > 0):

        print(c.INFO + 'Average cost of a walker after the Markov Chain Monte Carlo walk: ' + c.NUMB + '{:.2e}'.format(np.sum(state[1])/nwalkers) + c.ENDC)
        
    return state[0], autocorr[:index]

def classify_and_save_state(class_obj, state, what_model, what_save='none', log_CGS_units=False, path_name=os.getcwd(), file_name='walker'):

    """
    Classifies the nwalkers many parameter candidates according to their physicality, i.e. by checking if they agree with the physical data within 1 sigma.
    Also allows for saving .txt files with the radii, densities and pressures within the planet generated by the walker parameters.
    """

    #Storage arrays: 
    matches_observed_Js = []
    rho_maxs            = []

    #Loop through walkers:
    for i in range(len(state[:,0])):

        #Calculate the physical cost:
        if what_model == 'baro':

            costJs = baro_cost_function(class_obj, state[i,:], return_sum=False)

        elif what_model == 'dens':

            costJs = dens_cost_function(class_obj, state[i,:], return_sum=False)

        else:

            raise KeyError(c.WARN + 'Invalid keyword for what_model! Use \'baro\' or \'dens\'.' + c.ENDC)

        #Check if there is agreement with physical data within 1 sigma:
        if (np.abs(costJs)<1e0).all():

            output_color = c.GOOD
            matches_observed_Js.append(True)
            
        else:

            output_color = c.WARN
            matches_observed_Js.append(False)

        #Save .txt file if wanted by the user:
        if what_save=='all':

            if log_CGS_units:

                np.savetxt(path_name + '/' + file_name + '_CGS_' + str(i) + '.txt', np.transpose(np.array([ class_obj.li * 100, 
                                                                                                            np.log10(class_obj.rhoi) - 3, 
                                                                                                            np.log10(class_obj.Pi) + 1])))

            else:

                np.savetxt(path_name + '/' + file_name + '_SI_' + str(i) + '.txt',  np.transpose(np.array([ class_obj.li, 
                                                                                                            class_obj.rhoi, 
                                                                                                            class_obj.Pi])))
        
        elif what_save=='good' and (np.abs(costJs)<1e0).all(): 

            if log_CGS_units:

                np.savetxt(path_name + '/' + file_name + '_CGS_' + str(i) + '.txt', np.transpose(np.array([ class_obj.li * 100, 
                                                                                                            np.log10(class_obj.rhoi) - 3, 
                                                                                                            np.log10(class_obj.Pi) + 1])))

            else:

                np.savetxt(path_name + '/' + file_name + '_SI_' + str(i) + '.txt',  np.transpose(np.array([ class_obj.li, 
                                                                                                            class_obj.rhoi, 
                                                                                                            class_obj.Pi])))

        elif what_save=='none':

            pass
        
        else:

            raise KeyError(c.WARN + 'Invalid keyword for what_save! Use \'all\' or \'good\' or \'none\'.' + c.ENDC)

        #Update storage arrays:
        rho_maxs.append(np.max(class_obj.rhoi))

        #Verbositiy output:
        if class_obj.opts['verbosity'] > 0:

            print()
            print(output_color + 'Walker #' + c.NUMB  + str(i) + output_color + ' yields a total cost of ' + c.NUMB + '{:.2e}'.format(np.sum(costJs)) + output_color + '.' + c.ENDC)
            string1 = '         '
            string2 = ''
            string3 = ''
            for j in range(len(costJs)):
                string1 += c.get(np.abs(costJs[j])<1e0) + '     J'+str(2*(j+1))+'     '
                string2 += '{:.5e}'.format(class_obj.Js[j+1]) + ' '
                string3 += '{:.5e}'.format(class_obj.opts['Target_Js'][j]) + ' '
            print(c.INFO + string1 + c.ENDC)
            print(c.INFO + 'My code: ' + c.NUMB + string2 + c.ENDC)
            print(c.INFO + 'Target:  ' + c.NUMB + string3 + c.ENDC)
            print()

    return np.array(matches_observed_Js), np.array(rho_maxs)
