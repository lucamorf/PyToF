import numpy as np
import h5py
import time
import signal
import pickle

from PyToF.color import c
from PyToF.FunctionsToF import _pressurize
import PyToF.AlgoToF as AlgoToF
from PyToF import ClassToF

"""
Template file to read results from the hdf5 file using h5py

file structure:

read_file():                    main function to call that reads file
interrupt_handler():            helper function to interrupt result generation. check keeprunning in your loop
read_datasets(f):               given a file, generates a dictionary results[] with all the results you want in it
analyse_dataset(name, object):  function visited within generate results per dataset that analyses that dataset and adds the found attributes to the results vectors
                                analyse dataset must share a lot of variables with read_datasets (since it writes to them) but due to visititems requirements, must keep its signature
                                if one could be bothered, one could wrap this in a class or, or, one simply declares all relevant variables as globals (guess which one we do)
initialisation and call of read_file()
"""

N_samples = 1000
starttime = time.perf_counter()
keeprunning = True

is_neptune=False

kwargs = {}

if is_neptune:

    kwargs['N']      = 2**10
    kwargs['G']      = 6.6743e-11
    kwargs['M_phys'] = 6836525.21*(1000)**3/kwargs['G'] #https://doi.org/10.1051/0004-6361/202244537
    kwargs['R_phys'] = [25225*1e3, 'wrong']                        #https://iopscience.iop.org/article/10.1088/0004-6256/137/5/4322
    kwargs['Period'] = 15.9663*60*60                    #https://www.sciencedirect.com/science/article/pii/S0019103511001783?via%3Dihub
    kwargs['P0']     = 0
    kwargs['Target_Js'] = [3401.655e-6, -33.294e-6] #https://doi.org/10.1051/0004-6361/202244537
    kwargs['Sigma_Js']  = [   3.994e-6,  10.000e-6] #https://doi.org/10.1051/0004-6361/202244537
    kwargs['verbosity'] = 0
    kwargs['MaxIterShape'] = 100

else:

    kwargs['N']      = 2**10
    kwargs['G']      = 6.6743e-11
    kwargs['M_phys'] = 86.8127e24    #http://dx.doi.org/10.1098/rsta.2019.0474
    kwargs['R_phys'] = [25559*1e3, 'equatorial']      #http://dx.doi.org/10.1098/rsta.2019.0474
    kwargs['Period'] = 62080         #http://dx.doi.org/10.1098/rsta.2019.0474, Voyager
    kwargs['P0']     = 0
    kwargs['Target_Js'] = [3509.291e-6, -35.522e-6]  #https://doi.org/10.1016/j.icarus.2024.115957, French
    kwargs['Sigma_Js']  = [0.412e-6, 0.466e-6]       #https://doi.org/10.1016/j.icarus.2024.115957, French
    kwargs['verbosity'] = 0
    kwargs['MaxIterShape'] = 100

X = ClassToF.ToF(**kwargs)

def read_file():
    """
    Main file reading function. Manipulation of all results together goes here (plotting etc).
    """

    tic = time.perf_counter()

    #neptune
    if is_neptune:
        filename = 'bigrun_neptune_uniform.hdf5'
    #uranus
    else:
        filename = 'bigrun_uranus_uniform.hdf5'

    #this with is load-bearing because it neatly closes the filestream after we leave scope. close it manually if you dont want to use with
    with h5py.File(filename, 'r') as f:
        results = read_datasets(f)

    #@LUCA If saving all results as one file, handle here
    if is_neptune:
        with open('result_dict_neptune.pkl', 'xb') as f:
            pickle.dump(results, f)
    else:
        with open('result_dict_uranus.pkl', 'xb') as f:
            pickle.dump(results, f)

    toc = time.perf_counter()

    print(c.INFO + "Total time to generate results:  " + c.NUMB + time.strftime("%M:%S", time.gmtime((toc-tic))) + c.ENDC)

    return

def interrupt_handler(sig, frame):
    global keeprunning
    keeprunning = False
    print(c.WARN + 'Termination request raised.' + c.ENDC)

def read_datasets(f):
    """
    Interface function. Not necessary and can be combined with read_file, but wraps the messy global hack into a nice result dict.
    Wrapping of results into dict goes here.
    Returns a dictionary containing certain results read from the hdf5 file supplied. Structure is:
    results: result dictionary. indexed as results['attribute'] for the following:
    n:                          INT number of samples
    """

    #set the interrupt handler at this level
    signal.signal(signal.SIGINT, interrupt_handler)

    global n

    n = 0

    #main loop
    # note we use visititems because of the significant speed gain.
    # the disadvantage is that visititems must be given a callable (function) of exact signature callable(name, object)
    # if callable returns None, the visiting continues. Returning value will immediately stop visiting and return that value
    # note: because this callable has exact signature, no extra variables can be explicitly passed.
    #       Your options are:
    #           use global variables (easy and works but is mcterrible™) 
    #           wrap it in a class (untested but should work)
    #           some other great option I havent thought of
    # note: this CANNOT be multithreaded
    # note: if you wish to time the performance of callable, some hidden time will be obscured within the visititems process.
    #        darktime measures that. She's also very ominous and loves to loom in the shadowiest booth of the tavern.


    f.visititems(analyse_dataset)

    results = {}

    results['n'] = n

    return results

def analyse_dataset(name, object):

    """
    Function called once per dataset in the hdf5 file. By default, analyse_dataset has no memory and is re-initialised for each dataset.
    It would be wise to ensure large objects that are used repeatedly are not initialised every time analyse_dataset is called.
    When called by visititem, analyse_dataset is given the name fo the dataset (its key, I chose a UUID), and the object itself (i.e. the actual data).
    Data is formatted as explained 10 lines below.
    Actual work to be done for each resulting distribution goes here.
    """

    global n

    #the filestructure of the hdf5 file. each dataset is composed of two arrays. Array at object[0] is the starting density distribution, object[1] the result density distribution.
    #attributes contain rho_MAX respected (if rho_MAX was respected) and Js explained (if the Js were explained by the result)
    #the arrays behave like numpy arrays
    starting_rho = object[0]
    X.rhoi = object[1]
    X.li = object[2]
    rhoexplained = object.attrs['rho_MAX respected']
    Jsexplained = object.attrs['Js explained']
    X.m_rot_calc = object.attrs['m_rot_calc']

    #example (trivial) dataset analysis
    n += 1

    #@LUCA if saving each individually, save here

    X.Js, out = AlgoToF.Algorithm(          X.li,
                                            X.rhoi,
                                            X.m_rot_calc,
                                            order       = X.opts['order'],
                                            n_bin       = X.opts['nx'],
                                            tol         = 1e-5,
                                            maxiter     = X.opts['MaxIterShape'],
                                            verbosity   = X.opts['verbosity'],
                                            R_ref       = X.opts['R_ref'],
                                            ss_initial  = X.ss,
                                            alphas      = np.zeros(12),
                                            H           = X.opts['H'])
    _pressurize(X)

    np.savetxt('test_'+str(n)+'.txt', np.array([X.li, X.rhoi, X.Pi]))

    #progress tracker (sort of like tqdm)
    if n % 100 == 0:
        duration = time.perf_counter() - starttime
        rate = (n/duration)
        esttimerem = (N_samples - n)/rate
        days = int(esttimerem // (60*60*24))
        esttimerem = esttimerem%(60*60*24)
        print('  Nr: ' + str(n) +'. Time since start: ' + time.strftime("%H:%M:%S", time.gmtime(duration)) + '. Rate: ' + '{:.0f}'.format(rate) + ' it/s. Estimated time remaining: ' + str(days) + "d "+ time.strftime("%H:%M:%S", time.gmtime(esttimerem)) + '  ' , end='\r') 

    if keeprunning == False: return "Interrupted" #anything except None here

    return None

# the actual call




read_file()