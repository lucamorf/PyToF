###########################################################
# Author of this version: Stefano Wirth - stwirth@ethz.ch #
###########################################################

"""
             . . .......... .                                                                                                                                    
           .......................... ...     .       .         ..... ...            ...................... ..    ..           .                                 
          ....................... ...  .. .    ... ..          ...... ......         ........................... ................................::...           
         ...........................  ...   ..  .. .  .  ...................         ...........................  .......... .........................           
         ........................;+x++++xx+++xxx+++;++:     .................        ................................... ...................:.........           
        ....................:+xxXxxXXXxXx++;:.;+++++xxxx+;    ...............        ...................... ;++++++++;++x++++++:......................           
       .................:xXXXXXXXXXXXXxxx++;::  ::;++++++xxxx:...............       ....................;++xxxxxxxxx++:::;++;++++++;;:.......:.:::.:...          
       .............  .+XXXXXXXXXXXXXXXXxxxx++++;;;;;;++++++xx+;.............        ................+xXXxxxxXxxxxx+++;::::::;;;;;++++;:.....:::::.:...          
       ............ :XXXXXXXXXXXXXXXXXxxxx+++++;;::::::::;;+++xx+ ...........        ...............;xxXxXXxxxxxxxxxx++++;;;:::::::;;+++x:..::::::::...          
       .............xXXXXXXXXXXXXXXXXxxx++++++;;:::....::;+++++xx; ..........        .............:xXXXXXXXXXXxxxxxx++++;;;::::...::;;+++x:.::::::::.:..         
        .........  ;XXXXXXXXXXXXXXXXXXXXXXXxx+;:::.. ...::;+++xxxx: .........        .............+XXXXXXXXXXXXXxx+++++;;::::.. ...::;++++x;:::::.::::..         
      .........   :XXXXXXXXXXXXXXXXxx+;;;+++x+++;;;:;+xxxxxXXXxxx+...........        .............+XXXXXXXXXxXXXXx+xxxxxxxx+;:::::;++++++++;..::.:..:...         
      .... ..... .+XXXXXXXXXXXXXXXXXXxXx;.;++XX+;::;++xx+;;;++xxx+...........        ............ ;XXXXXXXXXxx+xxXXXXXXx. :+++;;;+xxxxx+++;;........::.:.        
     ..........  ;XXXXXXXXXXXXXXx+++++++++++xXXx+;;++;;++;:+xx+++:...........       ............:xx++xXXXXXXXXx+: .XXXXXXXxXxX+XXXXXXXXXXx+++x+..:::::::.        
     ........... .XXxxXXXXXXXXXXXx+++;;;;;+xXXXxx+++++;::;;;;;+++:...........        ............+XXXx;XXXXXXXXXxxXXXXXXXXXXXx++;XxXXXXXXXXXx+....::.::::        
    ..............+XXXXXXXXXXXXxx+;;;;;;++XX++++;::;;;;:....:;+++............        ..............+XXXXXXXXXXXXxx+++xx+xXX+;+;:.;;++++xxxxX;..:...::::::.       
    ..............xXXXXXXXXXXXXXXx+x+++xXXXXXXXX+++xx+;+;;:::;++: ...........        .............:xXXXXXXXXXXXXx+++;;;;;+xxxx+;++;::::;;;+;.:....:::::::.       
    ............:XXXXXXXXXXXXXXXxxxxxXXXXXx+++;;;;+xxXxxxxxx+xx; ............        .............xXXXXXXXXXXXXXXXx++xxxxXXx+;;;;++x+;;;;+;....:::::::::::       
   .............;XXXXXXXXXXXXXXXXXxxxxX+;::.. .: :;+xXx++++xxx; ..............       .............;XXXXXXXXXXXXXXXXXXXx++xXx++++++++++++++:......:::::::::.      
   .........:++XXXXXXXXXXXXXXXXXXXXXxxXXXx++;:  ..:;++++++xx+................        ..........:xXXXXXXXXXXXXXXXXXXXXXXXXXXx+::::::;++x++::::::::::::::::::      
   ......:xXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXx+;::::;;+++xXX+...................       ........+XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXxxXx++x+xxx::::::::::::::::::::      
     ..+XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXxxxxxXXXx++++++xx:....................        ....:+XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX+::::::::::::::::::::::.     
  .;+XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXxxXXXXXx:.......................        ;xXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX$XXXXXXXXXXX:::::::::::::::::::::::::     
 :XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX$XXXXXXXXXXXXXx:.........................      :XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXx:.:::::::::::::::::;:::     
 ;XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX+.......................       :XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX+:::::::::::::::::;::.    
 +XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX;  ...................       :XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXx;:::.::::::::;:;;;:    
 XXXXXXXXXXXXXXXXXXXXXXXXXXX$XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXx+;...............       :XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX+:::::::::;;;:    
:XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXx+;: ......       .XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX+:::::;:;::.   
+XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXx;...       .XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX+:::;;;::.   
XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX+:       .XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX+:::::::   
XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX;      .XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX::;;::   
;;;;;;;;;;;;++;+;++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++:       +++++++++++++xxxxxxxxxxxxxxxxxxXXXxxxXXxxxxxxxxXXxXXXXXXXXXXXXXXXXXX;::::   
===============================================================================================================================================================          
     __  __             _        ____           _     _____     _____                      _____ _                            _                             
    |  \/  | ___  _ __ | |_ ___ / ___|__ _ _ __| | __|_   _|__ |  ___|                    |_   _| |__   ___    ___ ___   ___ | | ___ _ __                   
    | |\/| |/ _ \| '_ \| __/ _ \ |   / _` | '__| |/ _ \| |/ _ \| |_                         | | | '_ \ / _ \  / __/ _ \ / _ \| |/ _ \ '__|                  
    | |  | | (_) | | | | ||  __/ |__| (_| | |  | | (_) | | (_) |  _|                        | | | | | |  __/ | (_| (_) | (_) | |  __/ |                     
    |_|  |_|\___/|_| |_|\__\___|\____\__,_|_|  |_|\___/|_|\___/|_|                         _|_|_|_| |_|\___|  \___\___/_\___/|_|\___|_|     _____     _____ 
                                                                                          |  \/  | ___  _ __ | |_ ___ / ___|__ _ _ __| | __|_   _|__ |  ___|
                                                                                          | |\/| |/ _ \| '_ \| __/ _ \ |   / _` | '__| |/ _ \| |/ _ \| |_   
                                                                                          | |  | | (_) | | | | ||  __/ |__| (_| | |  | | (_) | | (_) |  _|  
                                                                                          |_|  |_|\___/|_| |_|\__\___|\____\__,_|_|  |_|\___/|_|\___/|_|    
"""
import numpy as np
import PyToF.AlgoToF as AlgoToF
import scipy
import time
import random
import math
from PyToF.FunctionsToF import _pressurize

#TODO: Note: when you wanna turn this all into a package, replace all the references back to PyToF.reference again

from PyToF.color import c

def fix_params(params, weights, min):
    #weirdly, this distinction is unnecessary, since numpy can deal with both equally... funny, huh?
    if np.isscalar(min):
        return np.maximum(params, np.log(weights*min)) #to enforce drho >= min, we basically undo drho = exp(p_i)/w_i >= min to p_i >= log(min*w_i)
    else:
        return np.maximum(params, np.log(weights*min))

def param_to_rho_exp_fixed(OptToF, ToF, params):

    """
    Calculates rho based on the params, respecting the order of rhoi and li, that is, param represents the jumps going inward.
    The formula is: rho_i = ∑_j=1 ^i  exp(p_j)/w_j
    It respects the mass via fixing.
    """

    rho = np.zeros(len(params)+1) #params represent rho-jumps (drho), so we need one more rho
    rho[1:] = np.cumsum(np.exp(params)/OptToF.weights) #first rho is always zero
    MassFixFactor = ToF.opts['M_phys']/(-4*np.pi*scipy.integrate.simpson(rho*ToF.li**2, ToF.li))
    rho *= MassFixFactor
    OptToF.mass_fix_factor_running_average = OptToF.update_running_average(OptToF.mass_fix_factor_running_average, MassFixFactor)
    if random.random() < OptToF.opts['DBGshowchance']: print("MassFixFactor was: " + str(MassFixFactor))
    return rho

def gradient_vector_exp_fixed(ToF, params, gamma, weights):
    """Calculates the gradient vector in a fast way by reordering terms. May you achieve nirvana."""
    nto1vec = np.arange(len(params), 0, -1)
    #easy way to write sum of sum of e^pi/wi by reordering terms
    return nto1vec - gamma*np.dot(nto1vec,np.exp(params)/weights)*weights

def full_gradient(OptToF, ToF, params):

    """
    Calculates the full gradient.
    Phase 1: Calculate prefactors
    Phase 2: Calls ToF to find J values for the given parameters
    Phase 3: Calculates gradients based on those J values

    Not all those who wander are lost.
    """

    #Phase 1: Preliminaries
    #=============================
    time0 = time.perf_counter()

    #rhoi: first generate rhoi without mass normalisation p_α
    unnormalised_rhoi = np.zeros(len(params)+1)
    unnormalised_rhoi[1:] = np.cumsum(np.exp(params)/OptToF.weights)

    #calculate the real rho
    ToF.rhoi = param_to_rho_exp_fixed(OptToF, ToF, params)
    #assert the mass is correct
    #assert(abs(abs(-4*np.pi*scipy.integrate.simpson(ToF.rhoi*ToF.li**2, ToF.li)/ToF.opts['M_phys'])-1)<0.01), "MassIntError: Density curve does not fit Planet Mass within 1% margin"

    time1 = time.perf_counter()

    integral = (-scipy.integrate.simpson(unnormalised_rhoi*ToF.li**2, ToF.li))
    #assert(integral > 0), "Integrated the wrong way round!"

    time2 = time.perf_counter()

    #gamma = (dl/∫)
    gamma = abs(ToF.li[2]-ToF.li[1])/(integral)
    #factor for mass cost gradient, 2*(4π∫/M-1)*4π/M*∇∫
    fourpioverm = 4*np.pi/ToF.opts['M_phys']
    masscostfactor = 2*fourpioverm*(fourpioverm*integral-1)

    #Phase 2: ToF
    #=============================
    time3 = time.perf_counter()
    #Calculate Js and SS for gradients
    call_ToF(OptToF, ToF)

    time4 = time.perf_counter()

    #Phase 2.5: Atmosphere
    #=============================
    if ToF.opts['use_atmosphere']:
        _pressurize(ToF)
        #Define index that marks the transition from the atmosphere to the rest of the model:
        if not hasattr(ToF, 'check_param'):
            index = np.abs(ToF.Pi - ToF.opts['atmosphere_until']).argmin()+1
        else:
            index = max(np.abs(ToF.Pi - ToF.opts['atmosphere_until']).argmin()+1, round(ToF.check_param(ToF.rhoi, give_atmosphere_index=True)))

        #Adjust the parameters to fit the atmosphere model:
        goal_rhoi = ToF.opts['atmosphere'](ToF.li[:index], ToF.Pi[:index])
        for i in range(len(goal_rhoi) - 1):
            if goal_rhoi[i+1] <= goal_rhoi[i]:
                params[i] = -100+math.log(OptToF.weights[i]) #a number very close to zero or negative, log would be -∞
                if ToF.opts['verbosity'] > 0: print(c.WARN + 'Warning: Atmosphere contained nonincreasing step. Fudged to avoid log(0) = -∞' + c.ENDC)
                continue
            params[i] = math.log(abs(OptToF.weights[i]*(goal_rhoi[i+1]-goal_rhoi[i]))) #ensure no negatives either

        ToF.rhoi = param_to_rho_exp_fixed(OptToF, ToF, params)

        #Check if the densities are roughly equal:
        if not np.all(np.isclose(goal_rhoi, ToF.rhoi[:index], rtol=1e-2)):

            print()
            print('ToF.rhoi[:index]', ToF.rhoi[:index])
            print('goal_rhoi', goal_rhoi)
            print()

            raise Exception(c.WARN + 'Atmosphere enforcement failed!' + c.ENDC)

    #Phase 3: Gradients
    #=============================
    gradient = np.zeros_like(params)

    objective_gradient = np.zeros_like(params)

    # Objective Gradient

    # This is really just in range of order / number of Js were given. 
    # Target Js can be any length, start at J2. Our Js start at J0, thats why we can go as far as length of Target_Js but no longer than length of Js-1, because we have J0
    # We have SSs till order + 1 but again skip the first
    n = min(ToF.opts['order'] + 1 - 1, len(ToF.opts['Target_Js']))
    Flag = False
    if random.random() <  OptToF.opts['DBGshowchance']: Flag = True
    gradvec = gradient_vector_exp_fixed(ToF, params, gamma, OptToF.weights)

    for i in range(n):
        temp = -(1/n)*(ToF.R_eq_to_R_m**(2*(i+1)))*((ToF.Js[i+1] - ToF.opts['Target_Js'][i])/ToF.opts['Sigma_Js'][i]**2)*(ToF.SS[i+1][-1]-ToF.SS[i+1][-2])*gradvec
        objective_gradient += temp
        if Flag:
            print("Magnitude of J gradient Nr " + str(2*(i+1)))
            print('{:.4e}'.format(np.linalg.norm(temp)))
            print("SSdiff:" + '{:.4e}'.format(ToF.SS[i+1][-1]-ToF.SS[i+1][-2]))

    # Mass Gradient
    #∫ ~!~ M 
    #∇∫ = e^p_i/w_i* ∑_j=i ^n   l_j^2 == e^p_i
    #this was the point of weight correction
    mass_gradient = masscostfactor*np.exp(params)

    # Distance Gradient
    #dont go too far
    #f = 1/2||params - start_params||^2, ∇f = (params-start_params)*ONES
    local_gradient = (params - OptToF.start_params)

    #this simply conditions the mass_gradient to be of same magnitude as the objective gradient
    #it is equivalent to choosing the masscostfactor very well each time
    mass_gradient *= np.linalg.norm(objective_gradient)

    #combine
    gradient = objective_gradient + OptToF.costfactor*mass_gradient + OptToF.localfactor*local_gradient

    if ToF.opts['use_atmosphere']:
        gradient[:index - 1] = 0 #we avoid changing these atmospheric gradient parameters since these must remain fixed. -1 because 1 less params than rho

    time5 = time.perf_counter()

    if OptToF.opts['verbosity'] > 3 and random.random() < OptToF.opts['DBGshowchance']:
        print("I'm currently at:")
        print(params[::100])
        print("With rhos equal to:")
        print(ToF.rhoi[::100])
        print("Current Objective Gradient is:")
        print(objective_gradient[::100])
        print("Current Mass Corrector Gradient is:")
        print(OptToF.costfactor*mass_gradient[::100]) 
        print("Current Locality Gradient is:")
        print(OptToF.localfactor*local_gradient[::100])
        print("Current Full Gradient is:")
        print(gradient[::100])

    OptToF.timing[0] += time1-time0
    OptToF.timing[1] += time2-time1
    OptToF.timing[2] += time3-time2
    OptToF.timing[3] += time4-time3
    OptToF.timing[4] += time5-time4

    return gradient

def opt_step(Optimiser, OptToF, ToF, params):
    return Optimiser.update(params, full_gradient(OptToF, ToF, params))

def calc_cost(ToF):
    costJs = np.zeros_like(ToF.opts['Target_Js'])
    for i in range(min(len(ToF.opts['Target_Js']),len(ToF.Js)-1)):
        costJs[i] = (ToF.Js[i+1] - ToF.opts['Target_Js'][i])**2/ToF.opts['Sigma_Js'][i]**2
    return sum(costJs)

def call_ToF(OptToF, ToF):

    """
    Calls Algorithm from AlgoToF until either the accuray given by OptToF.ToF_convergence_tolerance is fulfilled 
    or ToF.opts['MaxIterShape'] is reached.
    """

    alphas = np.zeros(len(ToF.opts['alphas']))

    #Convert barotropic differential rotation parameters to Theory of Figures logic:
    if np.any(ToF.opts['alphas']):

        for i in range(len(alphas)):

            alphas[i] = 2*(i+1) * (ToF.li[0])**(2*i) * ToF.opts['alphas'][i] / ( ( ToF.m_rot_calc*ToF.opts['G']*ToF.opts['M_phys'] ) / ToF.li[0]**3 ) / ToF.opts['R_init']**(2*(i+1))

    #Implement the Theory of Figures: 
    ToF.Js, out = AlgoToF.Algorithm(        ToF.li,
                                            ToF.rhoi,
                                            ToF.m_rot_calc,
                                            order       = ToF.opts['order'],
                                            n_bin       = ToF.opts['nx'],
                                            tol         = OptToF.ToF_convergence_tolerance,
                                            maxiter     = ToF.opts['MaxIterShape'],
                                            verbosity   = ToF.opts['verbosity'],
                                            R_ref       = ToF.opts['R_ref'],
                                            ss_initial  = ToF.ss,
                                            alphas      = alphas,
                                            H           = ToF.opts['H'])
    
    #Save results, flipped since AlgoToF uses a different ordering logic:
    
    ToF.A0          = out.A0 #technically,
    ToF.As          = out.As #dont need these
    ToF.ss          = out.ss
    ToF.SS          = out.SS
    ToF.R_eq_to_R_m = out.R_eq_to_R_m

    return