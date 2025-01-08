# PyToF
numerical implementation of the Theory of Figures algorithm (4th, 7th, 10th order) including barotropic differential rotation

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
There is an extensive tutorial in PyToF_Tutorial.ipynb that explains all functionalities associated with this class.

## Accuracy and Convergence
See the folder PyToF_Accuracy_and_Convergence_Images for plots that demonstrate the accuracy of PyToF when compared against Wisdom & Hubbard 2016.
The plots have been generated using PyToF_Accuracy_and_Convergence.ipynb
