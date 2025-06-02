###########################################################
# Author of this version: Stefano Wirth - stwirth@ethz.ch #
###########################################################

import numpy as np
import matplotlib.pyplot as plt
from PyToF.StenOpt.ClassAdam import Adam
from PyToF.StenOpt.OptToF import *
from PyToF.StenOpt.StartGen import *
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import time
import random
from os import cpu_count, getpid, linesep
from colorhash import ColorHash
import signal
import h5py
import uuid

from PyToF.color import c

class OptToF:

    """
    This class contains the optimisation routine to run a descent method on planet data using the Theory of Figures.
    Create an instance and pass your desired kwargs through Optimiser = OptToF(**kwargs). When ready, call Optimiser.run(X), where X is an instance of ToF.
    We highlight the following important hyperparameters that must be tuned:
        learning rate:              the learning rate passed to Adam. Adam default: 0.001. Too low and convergence is slow, too high and chaos™. I used 0.05.
        costfactor:                 the cost factor of the mass part of the gradient. Too low and mass cost is ignored. Too high and it doesnt see anything else in the gradient. Err on the side of too high. I used 1e5.
        ToF convergence tolerance:  passed to ToF. Will be at least 1/10th of best sigma. Can be gung-ho with this. I used 1e-6
        kitty:                      extremely important. set as high as it will possibly go.
    Note: Only optimisation concerns are wrapped here, for ToF one should configure their instance of ClassToF.
    """

    def _default_opts(self, kwargs):

        """
        This function implements the standard options used for the OptToF class,
        except for the kwargs given by the user.
        """

        opts      =    {'verbosity':                    1,              #Higher numbers lead to more verbosity output in the console
                        'steps':                        1000,           #How many steps to take total
                        'epoch size':                   50,             #How big one epoch should be. This only matters for verbose output and convergence checks. (ideally steps is divisible by this)
                        'cores':                        cpu_count() - 1,#How many cpu cores the process occupies in parallel
                        'parallelize':                  True,           #Run in parallel
                        'time':                         True,           #Time the operations
                        'figures':                      True,           #Display figures
                        'early stopping':               True,           #Should the optimisation stop early if convergence is detected?
                        'convergence limit':            0.0005,         #Limit below which average increase is considered converged
                        'ToF convergence tolerance':    1e-8,          
                        'DBGshowchance':                0,              #Chance to show certain debug status prints
                        'continuous running':           False,          #Start new runs when old ones finished
                        'write to file':                False,          #Save results to file
                        'file location':                'results.hdf5', #Name of said file
                        'learning rate':                0.01,           #Adam learning rate
                        'costfactor':                   1e8,            #A multiplier for the costfactor of the gradient to stay within the mass region of the target   
                        'localfactor':                  0,              #A multiplier to stay local to the starting distribution. Unsupported
                        'rolling average forgetfulness':0.7,            #How aggressively trackers should forget the past (0 is immediate, must be <1)
                        'minimum increase':             0,              #minimum increase of rho at each step, can be a scalar or an np.array of shape params
                        'kitty':                        0.001           #🐈
                        }

        #Update the standard numerical parameters with the user input provided via the kwargs
        for kw, v in kwargs.items():

            if kw in opts:

                opts[kw] = v

            else:

                print(str(kw) + c.WARN + ' is an invalid keyword!' + c.ENDC)

        return opts

    def _set_IC(self):

        """
        Initializes...
        - improvement_running_average       Float, running average of the cost improvement ratio
        - mass_fix_factor_running_average   Float, running average of the mass fix factor
        - runtime_running_average           Float, running average of runtime. Used solely for estimated time remaining
        - timing                            Array, time taken for each subtask in seconds: Rho | Integral | Factors | ToF | Gradients
        - convergence_strikes               Int,   tracks number of times we determined to have converged. We need this because by sheer chance, improvement_running_average could be very close to 1 once (by, say, strong improvement and then worsening)
        - learning_rate                     Float, learning rate for Adam. Here because some optimisation schemes required scheduling learning rate decay
        - costfactor                        Float, cost factor for the mass fix gradient. Here because some optimisation schemes required adapting this factor during optimisation
        - localfactor                       Float, cost factor for the locality gradient. Not used during my work but still here
        - start_params                      Array, this doesnt need to be here, and yet it is. curious...
        """

        self.improvement_running_average    = 1
        self.mass_fix_factor_running_average= 1
        self.runtime_running_average        = 10
        self.timing                         = [0, 0, 0, 0, 0] # Rho | Integral | Factors | ToF | Gradients
        self.convergence_strikes            = 0
        self.learning_rate                  = self.opts['learning rate']
        self.costfactor                     = self.opts['costfactor']
        self.localfactor                    = self.opts['localfactor']
        self.start_params                   = None

    def __init__(self, **kwargs):

        """
        Initializes the OptToF class. Options can be provided via **kwargs, otherwise the default options from _default_opts() will be implemented.
        """

        #Set initial values:
        self.opts  = self._default_opts(kwargs)
        self._set_IC()

    def run(self, ToF):

        """
        Runs the Adam optimisation algorithm on the given instance of ToF.
        """

        #Set weights for rhoi
        self.weights = np.zeros(ToF.opts['N']-1)
        li2 = ToF.li**2 #∑_j=i ^n   l_j^2
        for i in range(len(self.weights)):
            self.weights[:i+1] += li2[i+1]*np.ones(i+1)

        #ToF convergence tolerance will always be at least 10 times better than our highest resolution Sigma J
        self.ToF_convergence_tolerance = min(self.opts['ToF convergence tolerance'], 0.1*min(ToF.opts['Sigma_Js']))

        #Do the optimisation algorithm in a parallel manner on multiple cores:
        if self.opts['parallelize']:
            with ProcessPoolExecutor() as executor:
                m = multiprocessing.Manager()
                lock = m.Lock()
                futures = [executor.submit(categoriser, self, ToF, lock) for core in range(self.opts['cores'])]
                try:
                    for future in as_completed(futures):
                        future.result()
                except KeyboardInterrupt: #KeyboardInterrupts are hijacked to finish up current run.
                    if (self.opts['verbosity'] > 0): print(c.WARN + 'KeyboardInterrupt caught.'+ c.ENDC)

        #Do the optimisation algorithm on a single core:
        else:
            m = multiprocessing.Manager()
            lock = m.Lock()
            categoriser(self, ToF, lock)

    def update_running_average(self, average, new):
        return self.opts['rolling average forgetfulness']*average + (1-self.opts['rolling average forgetfulness'])*new

    def huh(self):
        print()
        print(c.ENDC + 'A confused little kitten has stumbled onto the terminal!' + c.ENDC)
        print()
        print()
        print('                                                     ／l、             ')
        print('                                                   （ﾟ､ ｡ ７         ')
        print('                                                     |   ~ヽ       ')
        print('                                                     じしf_,)ノ')
        print('\033[92m'+'wwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwww' + c.ENDC)
        print()

    def interrupt_handler(self, sig, frame):
        self.opts['continuous running'] = False
        print(c.WARN + 'Termination request raised. Finishing current run.'+ c.ENDC)

def categoriser(OptToF, ToF, lock):

    """
    Runs one instance of optimisation and categorises results.
    Note the reason this function is defined outside of class scope is because of multithreading.
    If it was part of the class, every process would be working on the same class (I think I'm just doing what some dude on stackoverflow told some other dude)
    """
    
    signal.signal(signal.SIGINT, OptToF.interrupt_handler)

    while True: #This is always a good idea

        #Run Optimisation
        starting_distr, result_distr, ToF = run_opt(OptToF, ToF)

        #Categorise success/fail
        are_Js_explained = True
        is_rho_MAX_respected = True

        #check if Js are explained within one sigma
        for i in range(min(len(ToF.opts['Target_Js']),len(ToF.Js)-1)):
            if abs((ToF.Js[i+1] - ToF.opts['Target_Js'][i])/ToF.opts['Sigma_Js'][i]) >= 1:
                are_Js_explained = False

        #check if maximum density is explained
        if np.max(result_distr)>ToF.opts['rho_MAX']:
            is_rho_MAX_respected = False

        result = np.array([starting_distr, result_distr, ToF.li])

        if OptToF.opts['write to file'] == True:
            with lock: #lock out other processes from writing to file
                f = h5py.File(OptToF.opts['file location'], 'a')
                dset = f.create_dataset(str(uuid.uuid4()), data = result)
                dset.attrs['Js explained'] = are_Js_explained
                dset.attrs['rho_MAX respected'] = is_rho_MAX_respected
                dset.attrs['m_rot_calc'] = ToF.m_rot_calc
                f.close()

        if OptToF.opts['continuous running'] == False:
            break
        
        #reset to initial conditions and LETS GO AGAIN WHEEEEEEEEEEE
        OptToF._set_IC()

    #break gets you here
    if (OptToF.opts['verbosity'] > 0):
        r, g, b = ColorHash(getpid()).rgb
        cPID = '\033[38;2;' + str(r) + ';' + str(g) + ';' + str(b) + 'm'
        print(c.INFO + 'Process with ID ' + cPID + str(getpid()) + c.INFO + ' finished.'+ c.ENDC)

    return #this is just for my spiritual fulfilment

def run_opt(OptToF, ToF):

    """
    Congrats, you've finally made it down the daisychain of nested wrappers to the function that actually does the damn thing, except not really, this is mostly caretaking.
    Essentially this is just:

    PREPARE (starting parameters, optimiser etc)
    (verbosity)
    EPOCH LOOP:
        (verbosity)
        convergence check
        MAIN OPTIMISATION LOOP:
            a few steps of adam
    (verbosity)
    return

    """

    #random.seed(3)
    np.set_printoptions(formatter={'float_kind':'{:.5e}'.format})

    #Find a random set of starting parameters:
    #Note there are many starting generators available in StartGen. create_starting_point is just the only good one.
    tic = time.perf_counter()
    OptToF.start_params = create_starting_point(ToF, OptToF.weights)
    toc = time.perf_counter()

    if (OptToF.opts['verbosity'] > 2) and OptToF.opts['time']:
        print(c.INFO + 'Starting distribution generated in ' + c.NUMB + '{:.6f}'.format(toc-tic) + c.INFO + ' seconds.' +c.ENDC)

    params = OptToF.start_params.copy()

    #Set up the Adam Optimiser class
    AdamOptimiser = Adam(learning_rate=OptToF.learning_rate)

    #Verbosity -------------------------------------------------------------------------------------
    if (OptToF.opts['verbosity'] > 0) and OptToF.opts['parallelize'] == False:
        print()
        print(c.INFO + '==============================================================================' + c.ENDC)
        print(c.INFO + '                           Beginning optimisation                           ' + c.ENDC)
        print(c.INFO + '==============================================================================' + c.ENDC)
        print()
    
    if (OptToF.opts['verbosity'] > 0) and OptToF.opts['parallelize'] == True:
        r, g, b = ColorHash(getpid()).rgb
        cPID = '\033[38;2;' + str(r) + ';' + str(g) + ';' + str(b) + 'm'
        print()
        print(c.INFO + '==============================================================================' + c.ENDC)
        print(c.INFO + '              Beginning optimisation for process ID ' + cPID + str(getpid()) +  c.ENDC)
        print(c.INFO + '==============================================================================' + c.ENDC)
        print()
    #Verbosity over---------------------------------------------------------------------------------

    #divide steps into epochs of epoch size each
    epochs = OptToF.opts['steps'] // OptToF.opts['epoch size']
    steps = OptToF.opts['epoch size'] #this is now steps per epoch

    #setup plots
    if OptToF.opts['figures']:
        figure, (devax, perax) = plt.subplots(1, 2)
        devax.plot(param_to_rho_exp_fixed(OptToF, ToF, OptToF.start_params), color = 'red')
    
    #calculate first cost
    ToF.rhoi = param_to_rho_exp_fixed(OptToF, ToF, OptToF.start_params)
    call_ToF(OptToF, ToF)
    new_cost = calc_cost(ToF)
    cost_vector = [new_cost]

    total_start_time = time.perf_counter()

    # ======================================= THE LOOP™ ========================================
    for epoch in range(epochs):

        old_cost = new_cost
        new_cost = calc_cost(ToF)
        
        if (epoch + 1) % 3 == 0:
            if OptToF.opts['figures']: devax.plot(param_to_rho_exp_fixed(OptToF, ToF, params), color = 'b', alpha = (epoch + 1)/epochs)
        
        #Verbosity -------------------------------------------------------------------------------------
        if (OptToF.opts['verbosity'] > 0):
            print()
            r, g, b = ColorHash(getpid()).rgb
            cPID = '\033[38;2;' + str(r) + ';' + str(g) + ';' + str(b) + 'm'
            if OptToF.opts['parallelize'] == True: print(c.INFO + '                             Process ID: ' + cPID + str(getpid()) + c.ENDC)
            print(c.INFO + '                  Starting epoch number: ' + c.NUMB + str(epoch+1) + '/' + str(epochs) + c.ENDC)
        if (OptToF.opts['verbosity'] > 1):
            print(c.INFO + '                           Current cost: ' + c.NUMB + '{:.2e}'.format(new_cost) + c.INFO + '  A reduction of: ' + c.NUMB + '{:.2%}'.format(old_cost/new_cost-1) + c.ENDC)
            if epoch > 0 and OptToF.opts['time']:
                OptToF.runtime_running_average = OptToF.update_running_average(OptToF.runtime_running_average, (toc-tic))
                print(c.INFO + '                        Last epoch took: ' + c.NUMB + '{:.2f}'.format(toc-tic) + 's' + c.INFO + '     Time remaining: ' + c.NUMB + '{:.2f}'.format(OptToF.runtime_running_average*(epochs-epoch)) + 's' +c.ENDC)
        if OptToF.opts['verbosity'] > 2:
            print(c.INFO + '                  Current learning rate: ' + c.NUMB + str(AdamOptimiser.learning_rate) + c.ENDC)
            print(c.INFO + '               Current mass cost factor: ' + c.NUMB + '{:.0e}'.format((OptToF.costfactor)) + c.ENDC)
            print(c.INFO + '        Mass fix factor running average: ' + c.NUMB + '{:.6f}'.format((OptToF.mass_fix_factor_running_average)) + c.ENDC)
            print(c.INFO + '            Improvement running average: ' + c.NUMB + '{:.6f}'.format((OptToF.improvement_running_average)) + c.ENDC)
        if OptToF.opts['verbosity'] > 1:
            print(c.INFO + '                 J explanation strength:'
                + c.INFO + ' J2: ' + c.NUMB + '{:.3f}'.format((ToF.Js[1]-ToF.opts['Target_Js'][0])**2/ToF.opts['Sigma_Js'][0]**2)
                + c.INFO + ' J4: ' + c.NUMB + '{:.3f}'.format((ToF.Js[2]-ToF.opts['Target_Js'][1])**2/ToF.opts['Sigma_Js'][1]**2)
#                + c.INFO + ' J6: ' + c.NUMB + '{:.3f}'.format((ToF.Js[3]-ToF.opts['Target_Js'][2])**2/ToF.opts['Sigma_Js'][2]**2)
#                + c.INFO + ' J8: ' + c.NUMB + '{:.3f}'.format((ToF.Js[4]-ToF.opts['Target_Js'][3])**2/ToF.opts['Sigma_Js'][3]**2)
                + c.ENDC)
        if abs(new_cost/old_cost-1) < OptToF.opts['convergence limit']: OptToF.convergence_strikes += 1
        else: OptToF.convergence_strikes = 0
        if OptToF.opts['early stopping'] and OptToF.convergence_strikes >= 2:
                if OptToF.opts['verbosity'] > 0: print(c.INFO + 'Convergence detected. Terminating.' + c.ENDC)
                break
        if random.random() < OptToF.opts['kitty'] and OptToF.opts['verbosity'] > 0: OptToF.huh()
        #Verbosity over---------------------------------------------------------------------------------

        tic = time.perf_counter()

        # ====================================== MAIN OPTIMISATION LOOP ======================================

        for step in range(steps):

            params = opt_step(AdamOptimiser, OptToF, ToF, params)
            params = fix_params(params, OptToF.weights, OptToF.opts['minimum increase'])
            cost_vector.append(calc_cost(ToF))
            OptToF.improvement_running_average = OptToF.update_running_average(OptToF.improvement_running_average, cost_vector[-1]/cost_vector[-2])

        # ====================================== MAIN OPTIMISATION LOOP ======================================

        toc = time.perf_counter()

    # ======================================= THE LOOP™ ENDED =======================================

    total_stop_time = time.perf_counter()

    #Verbosity -------------------------------------------------------------------------------------
    if (OptToF.opts['verbosity'] > 0):
        if OptToF.opts['parallelize'] == False:
            print()
            print(c.INFO + '==============================================================================' + c.ENDC)
            print(c.INFO + '                            Optimisation ended                                ' + c.ENDC)
            print(c.INFO + '==============================================================================' + c.ENDC)
            print()
        
        if OptToF.opts['parallelize'] == True:
            r, g, b = ColorHash(getpid()).rgb
            cPID = '\033[38;2;' + str(r) + ';' + str(g) + ';' + str(b) + 'm'
            print()
            print(c.INFO + '==============================================================================' + c.ENDC)
            print(c.INFO + '                  Optimisation ended for process ID ' + cPID + str(getpid()) +  c.ENDC)
            print(c.INFO + '==============================================================================' + c.ENDC)
            print()

        print(c.INFO + 'Final cost: ' + c.NUMB + '{:.2e}'.format(calc_cost(ToF)) + c.ENDC)
        print()

        print(c.INFO + 'Final Js:   ' + c.NUMB, end = ' ')
        print(ToF.Js[1:len(ToF.opts['Target_Js'])+1])
        print(c.INFO + 'Target Js:  ' + c.NUMB, end = ' ')
        print(ToF.opts['Target_Js'])
        print(c.INFO + 'Difference: ' + c.NUMB, end = ' ')
        print(ToF.Js[1:len(ToF.opts['Target_Js'])+1]-ToF.opts['Target_Js'])
        print(c.INFO + 'Tolerance:  ' + c.NUMB, end = ' ')
        print(ToF.opts['Sigma_Js'])

        print(c.ENDC)
        # Rho | Integral | Factors | ToF | Gradients
        if OptToF.opts['time']:
            print(c.INFO + 'Total time:                         ' + c.NUMB + '{:.2f}'.format(total_stop_time-total_start_time) + c.INFO + ' seconds.' +c.ENDC)
            print(c.INFO + 'Time for rho calculation:           ' + c.NUMB + '{:.2f}'.format(OptToF.timing[0]) + c.INFO + ' seconds.' +c.ENDC)
            print(c.INFO + 'Time for integral calculations:     ' + c.NUMB + '{:.2f}'.format(OptToF.timing[1]) + c.INFO + ' seconds.' +c.ENDC)
            print(c.INFO + 'Time for factor calculations:       ' + c.NUMB + '{:.2f}'.format(OptToF.timing[2]) + c.INFO + ' seconds.' +c.ENDC)
            print(c.INFO + 'Time for ToF calculations:          ' + c.NUMB + '{:.2f}'.format(OptToF.timing[3]) + c.INFO + ' seconds.' +c.ENDC)
            print(c.INFO + 'Time for gradient calculations:     ' + c.NUMB + '{:.2f}'.format(OptToF.timing[4]) + c.INFO + ' seconds.' +c.ENDC)

        if OptToF.opts['figures']: perax.semilogy(np.array(cost_vector))
        if OptToF.opts['figures']: plt.show()
    #Verbosity over---------------------------------------------------------------------------------

    return param_to_rho_exp_fixed(OptToF, ToF, OptToF.start_params), param_to_rho_exp_fixed(OptToF, ToF, params), ToF