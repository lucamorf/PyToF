# PyToF
Numerical implementation of the Theory of Figures algorithm (4th, 7th, 10th order) including barotropic differential rotation.

**You are free to use this code for your own work if you cite**

Morf, L., Müller, S., and Helled, R., "The interior of Uranus: Thermal profile, bulk composition, and the distribution of rock, water, and hydrogen and helium", <i>Astronomy and Astrophysics</i>, vol. 690, Art. no. A105, EDP, 2024. doi:10.1051/0004-6361/202450698. 

## Installation

Download or clone this repository, navigate into the directory and execute

```console
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
N         = 2**10 #number of gridpoints
n_bin     = N #OPTIONAL, if n_bin < N, interpolation is used to speed up
#calculations but with reduced accuracy
order     = 4 #OPTIONAL, 4 is fast and inaccurate; 10 is slow and accurate, 
#7 is an intermediate option
densities = 1000*np.ones(N) #density array in SI units (kg/m^3)
radius    = 1e6 #outermost radius in SI units (m)
radii     = radius*np.linspace(1, 1e-3, N) #radius array in SI units (m), 
#arrays must start with the outer surface
mass      = -4*np.pi*np.trapezoid(densities*radii**2, radii) #calculated mass 
#in SI units (kg), negative sign because array starts with the outer surface
period    = 24*60*60 #rotation period in SI units (s)

X = ClassToF.ToF(
    N=N, 
    M_phys=mass, 
    R_phys=[radius, 'mean'], #all radius options: 'equatorial', 'mean', 'polar'
    Period=period, 
    order=order, 
    n_bin=n_bin
    ) 

X.li         = radii
X.rhoi       = densities
X.m_rot_calc = (2*np.pi/period)**2*X.li[0]**3/(X.opts['G']*mass)

number_of_iterations = X.relax_to_shape()
print('Number of iterations used by the algorithm:', number_of_iterations)

X.get_Js_errors()
print('PyToF solutions:')
for i in range(1,5):
    print('J_'+str(2*i) 
    + ' = ' 
    + "{:.4e}".format(X.Js[i]) 
    + ' +/- ' 
    + "{:.1e}".format(X.Js_error[i]) 
    )
```

and contains a minimal working example of how to obtain gravitational moments given an interior planetary profile. Output of the above code snippet:

```console
Number of iterations used by the algorithm: 49
PyToF solutions:
J_2 = 9.5477e-03 +/- 4.9e-08
J_4 = -1.9534e-04 +/- 4.9e-08
J_6 = 5.1743e-06 +/- 4.5e-08
J_8 = -1.6401e-07 +/- 1.4e-08
```

## Plotting capabilities

Below you can find a few figures that illustrate PyToF's capabilities, in partiular when it comes to built-in plotting routines. For more, consider chapters 2 and 5 in the tutorial.

### X.plot_shape()

![plot_shape_polar](/PyToF_Tutorial_images/plot_shape_polar_1_6_0.png "X.plot_shape()")
![plot_shape_cartesian](/PyToF_Tutorial_images/plot_shape_cartesian_1_6_0.png "X.plot_shape()")

### X.plot_state_xy()

![plot_state_xy](/PyToF_Tutorial_images/plot_state_xy_1_6_0.png "X.plot_state_xy()")

### X.plot_state_xy_corr()

![plot_state_xy_corr](/PyToF_Tutorial_images/plot_state_corr_xy_1_6_0.png "X.plot_state_xy_corr()")

## Accuracy and Convergence

We demonstrate the accuracy of PyToF when compared against 

Wisdom, J. and Hubbard, W. B., "Differential rotation in Jupiter: A comparison of methods", <i>Icarus</i>, vol. 267, pp. 315-322, 2016. doi:10.1016/j.icarus.2015.12.030.

with the plots that are stored in the folder PyToF_Accuracy_and_Convergence_Images and have been generated using PyToF_Accuracy_and_Convergence.ipynb. Some examples:

### Gravitational moment J2

No binning and compared against Movshovitz, N. and Fortney, J. J., “The Promise and Limitations of Precision Gravity: Application to the Interior Structure of Uranus and Neptune”, <i>The Planetary Science Journal</i>, vol. 3, no. 4, Art. no. 88, IOP, 2022. doi:10.3847/PSJ/ac60ff:

![J_2_Bessel](/PyToF_Accuracy_and_Convergence_images/Bessel_J_2_1_6_0.png)

### Gravitational moment J8

With binning and just with the results from PyToF:

![J_8_Bessel](/PyToF_Accuracy_and_Convergence_images/Binning_Bessel_J_8_1_6_0.png)

### Runtime comparison

No binning and compared against Movshovitz, N. and Fortney, J. J., “The Promise and Limitations of Precision Gravity: Application to the Interior Structure of Uranus and Neptune”, <i>The Planetary Science Journal</i>, vol. 3, no. 4, Art. no. 88, IOP, 2022. doi:10.3847/PSJ/ac60ff:

![time_bessel](/PyToF_Accuracy_and_Convergence_images/Bessel_time_1_6_0.png)
