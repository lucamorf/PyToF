########################################################
# Author of this version: Luca Morf - luca.morf@uzh.ch #
########################################################

import numpy as np
import scipy
import time

import matplotlib.pyplot as plt
import matplotlib

import PyToF.AlgoToF as AlgoToF
from PyToF.color import c

def _mass_int(class_obj):

    """
    Calculates and returns the total mass implied by the level surfaces class_obj.li and densities class_obj.rhoi using the trapezoid rule.
    """

    #Negative sign because beginning of the array is the outer surface:
    return -4*np.pi*np.trapz(class_obj.rhoi*class_obj.li**2, class_obj.li) 

def _fixradius(class_obj):

    """
    Renormalizes the level surfaces class_obj.li for consistency with the initially provided equatorial radius.
    """

    #Renormalize the level surfaces in such a way that...
    class_obj.li        = (class_obj.li/class_obj.li[0])*(class_obj.opts['R_init']/class_obj.R_ratio)

    #...the newly calculated R_calc equatorial radius is the same as the initial one:
    class_obj.R_calc    = class_obj.li[0]*class_obj.R_ratio
    assert np.isclose(class_obj.R_calc, class_obj.opts['R_init'])

    #Update the mass:
    class_obj.M_calc    = _mass_int(class_obj)

def _fixmass(class_obj):

    """
    Renormalizes the densities class_obj.rhoi for consistency with the initially provided mass.
    """

    #Renormalize the densities in such a way that...
    class_obj.rhoi      = class_obj.rhoi*class_obj.opts['M_init']/class_obj.M_calc

    #...the newly calculated M_calc mass is the same as the initial one:
    class_obj.M_calc   = _mass_int(class_obj)
    assert np.isclose(class_obj.M_calc, class_obj.opts['M_init'])

def _fixrot(class_obj):

    """
    Renormalizes the rotational parameter for consistency with the initially provided period.
    """

    #We update the m_rot_calc parameter such that it is consistent with the period, the outermost level surface and the calculated mass:
    class_obj.m_rot_calc = (2*np.pi/class_obj.opts['Period'])**2*class_obj.li[0]**3/(class_obj.opts['G']*class_obj.M_calc)

def _pressurize(class_obj):

    """
    Calculates the pressure class_obj.Pi at the level surfaces class_obj.li assuming hydrostatic equilibrium.
    """

    #Initialize the pressure boundary condition:
    class_obj.Pi[0] = class_obj.opts['P0']

    #See (B.3) in arXiv:1708.06177v1, flip since AlgoToF uses a different ordering logic:
    U               = -class_obj.opts['G']*class_obj.opts['M_init']/class_obj.li[0]**3*class_obj.li**2*np.flip(class_obj.A0)

    #Approximate the gradient of U:
    gradU           = np.zeros_like(class_obj.li)
    gradU[0]        = (U[0]    - U[1])  / (class_obj.li[0]    - class_obj.li[1])
    gradU[1:-1]     = (U[0:-2] - U[2:]) / (class_obj.li[0:-2] - class_obj.li[2:])
    gradU[-1]       = (U[-2]   - U[-1]) / (class_obj.li[-2]   - class_obj.li[-1])

    #Calculate the pressure according to gradP = -rho*gradU:
    integrand   = class_obj.rhoi*gradU

    for k in range(np.size(class_obj.Pi)-1):

        #We integrate downward:
        class_obj.Pi[k+1]  = class_obj.Pi[k] + 0.5*(integrand[k] + integrand[k+1])*(class_obj.li[k] - class_obj.li[k+1]) 

    class_obj.U = U

def _update_densities(class_obj, fixradius=True, fixmass=True, fixrot=True):

    """
    This function is called by relax_to_barotrope() and implements the barotrope model density = barotrope(pressure), 
    i.e. class_obj.rhoi = class_obj.barotrope(class_obj.Pi, class_obj.baro_param_calc).
    """

    if class_obj.baro_param_calc is None:

        class_obj.baro_param_calc = class_obj.opts['baro_param_init']

    #Call _pressurize() and set the new densities:
    _pressurize(class_obj)
    class_obj.rhoi = class_obj.barotrope(class_obj.Pi, class_obj.baro_param_calc)

    #Check for unphysical density inversions:
    if np.any(np.diff(class_obj.rhoi) < 0):
        
        raise Exception(c.WARN + 'Barotrope created density inversion!' + c.ENDC)

    if fixradius:
        _fixradius(class_obj)

    if fixmass:

        _fixmass(class_obj)

    if fixrot:

        _fixrot(class_obj)

    #Optional: use the provided atmospheric model:
    if class_obj.opts['use_atmosphere']:

        _apply_atmosphere(class_obj)

def _apply_atmosphere(class_obj, baro=True):

    """
    This function is called by e.g. _update_densities() and implements the atmosphere model density = atmosphere(argument), i.e. 
    class_obj.rhoi[specified by class_obj.opts['atmosphere_until']] = class_obj.opts['atmosphere'](class_obj.li[:index], class_obj.Pi[:index]). 
    """

    if not hasattr(class_obj, 'check_param'):

        def check_param(param, give_atmosphere_index=False):
            return 0

        class_obj.check_param = check_param

    #Define index that marks the transition from the atmosphere to the rest of the model:
    if baro:
        index = max(np.abs(class_obj.Pi - class_obj.opts['atmosphere_until']).argmin()+1, round(class_obj.check_param(class_obj.baro_param_calc, give_atmosphere_index=True)))
    else:
        index = max(np.abs(class_obj.Pi - class_obj.opts['atmosphere_until']).argmin()+1, round(class_obj.check_param(class_obj.dens_param_calc, give_atmosphere_index=True)))
   
    #Adjust the densities to fit the atmosphere model:
    class_obj.rhoi[:index] = class_obj.opts['atmosphere'](class_obj.li[:index], class_obj.Pi[:index])

    #Make sure the total mass stays unaffected:
    class_obj.rhoi = class_obj.rhoi/_mass_int(class_obj)*class_obj.M_calc

    #Check for unphysical density inversions:
    if np.any(np.diff(class_obj.rhoi) < 0):

        raise Exception(c.WARN + 'Atmosphere created density inversion!' + c.ENDC)

def get_r_l_mu(class_obj, mu):

    """
    This function returns an array with shape (class_obj.opts['N'], length(mu)) that stores the values
    of r_l_mu that is defined in equation (B.1) from arXiv:1708.06177v1 for each l and each mu.
    """

    #Calculate array with r_l(mu):
    r_l_mu = 1

    for i in range(len(class_obj.ss)):

        #Flip since AlgoToF uses a different ordering logic:
        r_l_mu  = r_l_mu + np.outer(np.flip(class_obj.ss[i]), np.polynomial.legendre.Legendre.basis(2*i)(mu))

    return r_l_mu*np.outer(class_obj.li, np.ones_like(mu))

def get_U_l_mu(class_obj, mu):

    """
    This function returns an array with shape (class_obj.opts['N'], length(mu)) that stores the values
    of the potential U that is defined in equation (B.3) from arXiv:1708.06177v1 for each l and each mu.
    """

    #Calculate array with U(l,mu), flip since AlgoToF uses a different ordering logic:
    U = np.outer(np.flip(class_obj.A0), np.ones_like(mu))
    
    for i in range(len(class_obj.ss)-1):

            #Flip since AlgoToF uses a different ordering logic:
            U += np.outer(np.flip(class_obj.As[i,:]), np.polynomial.legendre.Legendre.basis(2*i)(mu))

    return -class_obj.opts['G']*class_obj.opts['M_init']/class_obj.li[0]**3*np.outer(class_obj.li**2, np.ones_like(mu))*U

def get_NMoI(class_obj, N=1000):

    """
    This function returns a float that represents the value of the normalized moment of inertia.
    """

    #Define cosine of angles to integrate over:
    mu = np.linspace(-1, 1, N)

    #Perform integrations:
    integrand_r_theta   = 2*np.pi * np.outer(class_obj.rhoi, np.ones_like(mu)) * get_r_l_mu(class_obj, mu)**4 * (1-np.outer(np.ones_like(class_obj.li), mu)**2)
    integrand_r         = np.trapz(integrand_r_theta, mu, axis=1)
    MoI                 = np.trapz(integrand_r, class_obj.li)*(-1) #minus sign due to integration from outside to inside
    NMoI                = MoI/(class_obj.M_calc*class_obj.li[0]**2)

    return NMoI

def set_barotrope(class_obj, fun):

    """
    This function allows the user to internally set the function relating the density 
    class_obj.rhoi to the pressure class_obj.Pi via class_obj.rhoi = fun(class_obj.Pi, param).

    param: Parameters of the barotrope model
    """

    #Set function:
    class_obj.barotrope = fun

def set_density_function(class_obj, fun):

    """
    This function allows the user to internally set the function relating the density 
    class_obj.rhoi to the mean level surfaces class_obj.li via class_obj.rhoi = fun(class_obj.li, param).

    param: Parameters of the density function model
    """

    #Set function:
    class_obj.density_function = fun
 
def relax_to_HE(class_obj, fixradius=True, fixmass=True, fixrot=True, pressurize=True):

    """
    Calls Algorithm from AlgoToF until either the accuray given by class_obj.opts['dJ_tol'] is fulfilled 
    or class_obj.opts['MaxIterHE'] is reached.
    """

    alphas = np.zeros(len(class_obj.opts['alphas']))

    #Convert barotropic differential rotation parameters to Theory of Figures logic:
    if np.any(class_obj.opts['alphas']):

        for i in range(len(alphas)):

            alphas[i] = 2*(i+1) * (class_obj.li[0])**(2*i) * class_obj.opts['alphas'][i] / ( ( class_obj.m_rot_calc*class_obj.opts['G']*class_obj.opts['M_init'] ) / class_obj.li[0]**3 ) / class_obj.opts['R_init']**(2*(i+1))

    tic = time.time()

    #Implement the Theory of Figures: 
    class_obj.Js, out = AlgoToF.Algorithm(  class_obj.li,
                                            class_obj.rhoi,
                                            class_obj.m_rot_calc,
                                            order       = class_obj.opts['order'],
                                            nx          = class_obj.opts['nx'],
                                            tol         = class_obj.opts['dJ_tol'],
                                            maxiter     = class_obj.opts['MaxIterHE'],
                                            verbosity   = class_obj.opts['verbosity'],
                                            alphas      = alphas,
                                            H           = class_obj.opts['H'],
                                            ss_guesses  = class_obj.ss
                                            )
    
    toc = time.time()

    #Verbosity output:
    if (class_obj.opts['verbosity'] > 2):
        
        print()
        print(c.INFO + 'Relaxing to hydrostatic equilibrium done in ' + c.NUMB + '{:.2e}'.format(toc-tic) + c.INFO + ' seconds.' + c.ENDC)

    #Save results, flipped since AlgoToF uses a different ordering logic:
    class_obj.A0        = out.A0
    class_obj.As        = out.As
    class_obj.ss        = out.ss
    class_obj.SS        = out.SS
    class_obj.R_ratio   = out.R_ratio

    if fixradius:

        _fixradius(class_obj)

    if fixmass:

        _fixmass(class_obj)

    if fixrot:

        _fixrot(class_obj)

    if pressurize:

        _pressurize(class_obj)

    return out.it

def relax_to_barotrope(class_obj, fixradius=True, fixmass=True, fixrot=True, pressurize=True):

    """
    Calls relax_to_HE() and _update_densities() until either the accuray given by class_obj.opts['dJ_tol'], 
    class_obj.opts['drot_tol'] and class_obj.opts['drho_tol'] is fulfilled or class_obj.opts['MaxIterBar'] is reached.
    """

    tic = time.time()

    #Set up iterative procedure:
    it = 1

    while (it < class_obj.opts['MaxIterBar']):

        #Update iteration parameter:
        it          = it + 1

        #Store old values:
        old_Js      = class_obj.Js
        old_m       = class_obj.m_rot_calc
        old_rho     = class_obj.rhoi

        #Call relax_to_HE():
        relax_to_HE(class_obj, fixradius=fixradius, fixmass=fixmass, fixrot=fixrot, pressurize=pressurize)

        #Call update_densities():
        _update_densities(class_obj, fixradius=fixradius, fixmass=fixmass, fixrot=fixrot)

        #Check convergence:
        old_Js[old_Js==0]   = np.spacing(1) #Smallest numerically resolvable non-zero number
        dJs                 = np.max(np.abs((class_obj.Js                       /old_Js         - 1)))
        drot                =        np.abs( class_obj.m_rot_calc /old_m          - 1)
        drho                = np.max(np.abs( class_obj.rhoi[1:]                 /old_rho[1:]    - 1))
        
        if (drot    < class_obj.opts['drot_tol']    and
            dJs     < class_obj.opts['dJ_tol']      and
            drho    < class_obj.opts['drho_tol']):

            break

    toc = time.time()

    #Warning if not converged:
    if it == class_obj.opts['MaxIterBar']:

        b1, b2, b3 = drot < class_obj.opts['drot_tol'], dJs < class_obj.opts['dJ_tol'], drho < class_obj.opts['drho_tol']
        c1, c2, c3 = c.get(b1), c.get(b2), c.get(b3)
        
        string  = c.WARN + 'Convergence warning: '
        string += c1 + 'drot_tol = ' + c.NUMB + "{:.0e}".format(drot)   + c1 + [' > ', ' < '][b1] + c.NUMB + "{:.0e}".format(class_obj.opts['drot_tol'])
        string += c2 + ', dJ_tol = ' + c.NUMB + "{:.0e}".format(dJs)    + c2 + [' > ', ' < '][b2] + c.NUMB + "{:.0e}".format(class_obj.opts['dJ_tol'])
        string += c3 + ', drho_tol = ' + c.NUMB + "{:.0e}".format(drho) + c3 + [' > ', ' < '][b3] + c.NUMB + "{:.0e}".format(class_obj.opts['drho_tol'])
        string += c.WARN + ' after MaxIterBar = ' + c.NUMB + str(class_obj.opts['MaxIterBar']) + c.WARN + ' iterations.' + c.ENDC

        print()
        print(string)
        
    #Verbosity output:
    if (class_obj.opts['verbosity'] > 1):
        
        print()
        print(c.INFO + 'Relaxing to barotrope done in ' + c.NUMB + '{:.2e}'.format(toc-tic) + c.INFO + ' seconds.' + c.ENDC)

    return it

def relax_to_density(class_obj, fixradius=True, fixmass=True, fixrot=True, pressurize=True):

    """
    Calls relax_to_HE() and _update_densities() until either the accuray given by class_obj.opts['dJ_tol'], 
    class_obj.opts['drot_tol'] and class_obj.opts['drho_tol'] is fulfilled or class_obj.opts['MaxIterDen'] is reached.
    """

    tic = time.time()

    #Set up iterative procedure:
    it = 1

    while (it < class_obj.opts['MaxIterDen']):

        #Update iteration parameter:
        it          = it + 1

        #Store old values:
        old_Js      = class_obj.Js
        old_m       = class_obj.m_rot_calc
        old_rho     = class_obj.rhoi

        #Call relax_to_HE():
        relax_to_HE(class_obj, fixradius=fixradius, fixmass=fixmass, fixrot=fixrot, pressurize=pressurize)

        #Call _pressurize():
        _pressurize(class_obj)

        if fixradius:

            _fixradius(class_obj)

        if fixmass:

            _fixmass(class_obj)

        if fixrot:

            _fixrot(class_obj)

        #Optional: use the provided atmospheric model:
        if class_obj.opts['use_atmosphere']:

            _apply_atmosphere(class_obj, baro=False)

        #Check convergence:
        old_Js[old_Js==0]   = np.spacing(1) #Smallest numerically resolvable non-zero number
        dJs                 = np.max(np.abs((class_obj.Js                       /old_Js         - 1)))
        drot                =        np.abs( class_obj.m_rot_calc /old_m          - 1)
        drho                = np.max(np.abs( class_obj.rhoi[1:]                 /old_rho[1:]    - 1))

        if (drot    < class_obj.opts['drot_tol']    and
            dJs     < class_obj.opts['dJ_tol']      and
            drho    < class_obj.opts['drho_tol']):

            break

    toc = time.time()

    #Warning if not converged:
    if it == class_obj.opts['MaxIterDen']:

        b1, b2, b3 = drot < class_obj.opts['drot_tol'], dJs < class_obj.opts['dJ_tol'], drho < class_obj.opts['drho_tol']
        c1, c2, c3 = c.get(b1), c.get(b2), c.get(b3)
        
        string  = c.WARN + 'Convergence warning: '
        string += c1 + 'drot_tol = ' + c.NUMB + "{:.0e}".format(drot)   + c1 + [' > ', ' < '][b1] + c.NUMB + "{:.0e}".format(class_obj.opts['drot_tol'])
        string += c2 + ', dJ_tol = ' + c.NUMB + "{:.0e}".format(dJs)    + c2 + [' > ', ' < '][b2] + c.NUMB + "{:.0e}".format(class_obj.opts['dJ_tol'])
        string += c3 + ', drho_tol = ' + c.NUMB + "{:.0e}".format(drho) + c3 + [' > ', ' < '][b3] + c.NUMB + "{:.0e}".format(class_obj.opts['drho_tol'])
        string += c.WARN + ' after MaxIterDen = ' + c.NUMB + str(class_obj.opts['MaxIterDen']) + c.WARN + ' iterations.' + c.ENDC

        print()
        print(string)
        
    #Verbosity output:
    if (class_obj.opts['verbosity'] > 1):
        
        print()
        print(c.INFO + 'Relaxing to density done in ' + c.NUMB + '{:.2e}'.format(toc-tic) + c.INFO + ' seconds.' + c.ENDC)

    return it
