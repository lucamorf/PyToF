########################################################
# Author of this version: Luca Morf - luca.morf@uzh.ch #
########################################################

from setuptools import setup, find_packages

setup(
    name="PyToF",
    version="1.0.0",
    description="numerical implementation of the Theory of Figures algorithm (4th, 7th, 10th order) including barotropic differential rotation",
    url="",
    author="Luca Morf",
    author_email="luca.morf@uzh.ch",
    license="Mozilla Public License Version 2.0",
    packages=find_packages(include=["PyToF", "PyToF.*"]),
    install_requires=["numpy", "scipy", "matplotlib", "emcee"],
)