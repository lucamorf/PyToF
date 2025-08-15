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

## Accuracy and Convergence

See the folder PyToF_Accuracy_and_Convergence_Images for plots that demonstrate the accuracy of PyToF when compared against 

Wisdom, J. and Hubbard, W. B., "Differential rotation in Jupiter: A comparison of methods", <i>Icarus</i>, vol. 267, pp. 315-322, 2016. doi:10.1016/j.icarus.2015.12.030.

The plots have been generated using PyToF_Accuracy_and_Convergence.ipynb
