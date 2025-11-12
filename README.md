# PyToF
Numerical implementation of the Theory of Figures algorithm (4th, 7th, 10th order) including barotropic differential rotation.

**You are free to use this code for your own work if you cite**

Morf, L., Müller, S., and Helled, R., "The interior of Uranus: Thermal profile, bulk composition, and the distribution of rock, water, and hydrogen and helium", <i>Astronomy and Astrophysics</i>, vol. 690, Art. no. A105, EDP, 2024. doi:10.1051/0004-6361/202450698. 

## Installation

Download or clone this repository, navigate into the directory and execute
```
pip install .
```

Note that the package requires numpy, scipy, matplotlib, emcee and tqdm.

## Basic usage

```python
from PyToF import ClassToF

X = ClassToF.ToF()
```
**There is an extensive tutorial in PyToF_Tutorial.ipynb that explains all functionalities associated with this class.**

The most important section within this tutorial can be found at the beginning of chapter 3

```python
N         = 2**10                                            #number 
density   = 1000                                             #constant density in SI units (kg/m^3)
densities = density*np.ones(N)                               #density array in SI units (kg/m^3)
radius    = 1e6                                              #outermost radius in SI units (m)
radii     = radius*np.logspace(0, -2, N)                     #radius array in SI units (m), arrays must start with the outer surface
mass      = -4*np.pi*np.trapezoid(densities*radii**2, radii) #calculated mass in SI units (kg), negative sign because array starts with the outer surface
period    = 24*60*60                                         #rotation period in SI units (s)

X = ClassToF.ToF(N=N, M_phys=mass, R_phys=[radius, 'mean'], Period=period) #all radius options: 'equatorial', 'mean', 'polar'

X.li         = radii
X.rhoi       = densities
X.m_rot_calc = (2*np.pi/period)**2*X.li[0]**3/(X.opts['G']*mass)

number_of_iterations = X.relax_to_shape()
print('Number of iterations used by the algorithm:', number_of_iterations)

X.plot_xy(0, 2)

print('PyToF solutions:', ['J_'+str(2*i)   +' = ' + "{:.4e}".format(X.Js[i]) + ' +/- ' + "{:.1e}".format(X.Js_error[i]) for i in range(1,5)])
```

and contains a minimal working example of how to obtain gravitational moments given an interior planetary profile.

## Plotting capabilities

Below you can find a few figures that illustrate PyToF's capabilities, in partiular when it comes to built-in plotting routines:

![plot_xy](plot_xy.png "X.plot_xy()")

## Accuracy and Convergence

See the folder PyToF_Accuracy_and_Convergence_Images for plots that demonstrate the accuracy of PyToF when compared against 

Wisdom, J. and Hubbard, W. B., "Differential rotation in Jupiter: A comparison of methods", <i>Icarus</i>, vol. 267, pp. 315-322, 2016. doi:10.1016/j.icarus.2015.12.030.

The plots have been generated using PyToF_Accuracy_and_Convergence.ipynb
