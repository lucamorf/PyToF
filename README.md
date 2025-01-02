# PyToF
numerical implementation of the Theory of Figures algorithm (4th, 7th, 10th order) including barotropic differential rotation

## Installation
Download or clone this repository, navigate into the directory and execute
```
pip install .
```

Note that the package requires numpy, scipy, matplotlib and emcee.

## Basic usage
```python
from PyToF import ClassToF

X = ClassToF.ToF()
```
There is an extensive tutorial in $\texttt{PyToF\_Tutorial.ipynb}$ that explains everything.

## Accuracy and Convergence
See the folder $\texttt{PyToF\_Accuracy\_and\_Convergence\_Images}$ for plots that demonstrate the accuracy of PyToF when compared against Wisdom & Hubbard 2016.
The plots have been generated using $\texttt{PyToF\_Accuracy\_and\_Convergence.ipynb}$
