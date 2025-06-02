###########################################################
# Author of this version: Stefano Wirth - stwirth@ethz.ch #
###########################################################

import numpy as np
import math
import scipy
import random
import functools

from PyToF.color import c

def parameterise_starting_points(ToF, weights, ResultFunction):
    #note: second density entry should be nonzero
    assert(ResultFunction[1]>0), "Error: Outermost nonendpoint density is zero"
    assert(ToF.opts['N'] == len(ResultFunction)), "Error: Generated density function is not of correct length."
    #after the raw resultfunction has been generated, it needs to be preconditioned for total mass agreeing
    MassFixFactor = ToF.opts['M_phys']/(-4*np.pi*scipy.integrate.simpson(ResultFunction*ToF.li**2, ToF.li))
    ResultFunction *= MassFixFactor
    if ToF.opts['verbosity'] > 2: print(c.INFO + "MassFixFactor for initial generation was: " + c.NUMB + '{:.2f}'.format(MassFixFactor) + c.ENDC)
    #then, we need to obtain parameters from the conditioned function (note fudging factor p_alpha should be 1 now)
    params = np.zeros(ToF.opts['N']-1)
    for i in range(len(params)):
        if ResultFunction[i+1] <= ResultFunction[i]:
            params[i] = -100+math.log(weights[i]) #a number very close to zero or negative, log would be -∞
            if ToF.opts['verbosity'] > 0: print(c.WARN + 'Warning: Starting Function contained nonincreasing step. Fudged to avoid log(0) = -∞' + c.ENDC)
            continue
        params[i] = math.log(abs(weights[i]*(ResultFunction[i+1]-ResultFunction[i]))) #ensure no negatives either
    return params

def create_starting_point_fixed_jupiter(ToF, weights):
    ResultFunction = 100*np.concatenate((np.linspace(0,0.5,20),np.linspace(0.51,1,80),np.linspace(1.1,3,300),np.linspace(3.1,6,450),np.linspace(6.1,30,24),np.linspace(30.1,50,150)))
    return parameterise_starting_points(ToF, weights, ResultFunction)

def create_starting_point_fixed_earth(ToF, weights):
    ResultFunction = np.concatenate((np.linspace(0,3000,12),np.linspace(3001,4500,100),np.linspace(4501,5000,400),np.linspace(5001,10000,12),np.linspace(10001,13000,400),np.linspace(13200,13250,100)))
    return parameterise_starting_points(ToF, weights, ResultFunction)

def density(r):
    earth_radius = 6.3710e6
    radii = (1.2215e6, 3.4800e6, 5.7010e6, 5.7710e6, 5.9710e6,
            6.1510e6, 6.3466e6, 6.3560e6, 6.3680e6, earth_radius)
    densities = (
        lambda x: 13.0885 - 8.8381*x**2,
        lambda x: 12.5815 - 1.2638*x - 3.6426*x**2 - 5.5281*x**3,
        lambda x: 7.9565 - 6.4761*x + 5.5283*x**2 - 3.0807*x**3,
        lambda x: 5.3197 - 1.4836*x,
        lambda x: 11.2494 - 8.0298*x,
        lambda x: 7.1089 - 3.8045*x,
        lambda x: 2.691 + 0.6924*x,
        2.9,
        2.6,
        1.02
    )   
    r = np.array(r)
    radius_bounds = np.concatenate(([0], radii))
    conditions = list((lower<=r) & (r<upper) for lower, upper in
                        zip(radius_bounds[:-1], radius_bounds[1:]))
    return np.piecewise(r/earth_radius, conditions, densities)

def create_starting_point_fixed_earth_better(ToF, weights):
    earth_radius = 6.3710e6
    ResultFunction = density(np.linspace(earth_radius,0,1024))
    print(ResultFunction)
    return parameterise_starting_points(ToF, weights, ResultFunction)

def subdivide(ResultFunction, lower_x, upper_x, lower_y, upper_y):
    if upper_x == lower_x:
        ResultFunction[lower_x] = random.uniform(lower_y,upper_y)
        return
    x = random.randint(lower_x ,upper_x)
    y = random.uniform(lower_y,upper_y)
    ResultFunction[x] = y
    if lower_x < x:
        subdivide(ResultFunction, lower_x, x - 1, lower_y, y)
    if x < upper_x:
        subdivide(ResultFunction, x + 1, upper_x, y, upper_y)

def create_starting_point_uneven(ToF, weights):
    N = ToF.opts['N']
    R = 6000
    ResultFunction = np.zeros(N)
    ResultFunction[-1] = R
    subdivide(ResultFunction, 1, N - 2, np.spacing(0), R)
    return parameterise_starting_points(ToF, weights, ResultFunction)

def subdivide_binary(ResultFunction, pivot, depth, power, lower_y, upper_y):
    y = random.uniform(lower_y,upper_y)
    ResultFunction[pivot] = y
    pivot_step = int(2**(power-2-depth))
    if pivot_step == 0:
        return
    subdivide_binary(ResultFunction, pivot-pivot_step, depth + 1, power, lower_y, y)
    subdivide_binary(ResultFunction, pivot+pivot_step, depth + 1, power, y, upper_y)

def create_starting_point(ToF, weights):
    N = ToF.opts['N']
    assert(N.bit_count() == 1), "Error: N must be a power of 2."
    power = int(math.log2(N))
    R = 6000
    ResultFunction = np.zeros(N)
    subdivide_binary(ResultFunction, 2**(power-1), 0, power, np.spacing(0), R) #should be N/2
    #note this leaves ResultFunction[0] alone because ∑_i<n 2^i = 2^n-1
    #np.spacing avoids ResultFunction[1] = 0
    #ResultFunction[-1] < R doesnt really matter
    return parameterise_starting_points(ToF, weights, ResultFunction)

@functools.cache
def multiset_nr(k, n):
    if k == 0 and n == 0: return 1
    return math.comb(k+n-1,k)

def find_permutation_nonrec(Balls2Place, NBins):
    ResultFunction = np.zeros(NBins)
    Number_of_functions = int(multiset_nr(Balls2Place, NBins))
    index = random.randint(0,Number_of_functions-1)
    Balls2Bin = 0
    Balls2OtherBins = Balls2Place
    Bin = 0
    while Bin < NBins:
        ways2place = multiset_nr(Balls2OtherBins, NBins - Bin - 1)
        if index < ways2place:
            ResultFunction[Bin] = Balls2Bin
            Balls2Bin = 0
            Bin += 1
        else:
            index -= ways2place
            Balls2Bin += 1
            Balls2OtherBins -= 1
    assert(Balls2OtherBins == 0), "Placement Error: Not all increases distributed"
    return ResultFunction

#creates a starting point chosen uniformly at random from all possible starting points, converts it into its respective parameters.
def create_starting_point_uniform_normalised(ToF, weights):
    N = ToF.opts['N']
    R = 6000
    #Resolution, ie target max density. Value chosen to be somewhat realistic. 
    ResultFunction = find_permutation_nonrec(R, N)
    for i in range(N-1):
        ResultFunction[i+1] = ResultFunction[i+1]+ResultFunction[i]
    return parameterise_starting_points(ToF, weights, ResultFunction)
