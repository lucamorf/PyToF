########################################################
# Author of this version: Luca Morf - luca.morf@uzh.ch #
########################################################

from setuptools import setup, find_packages

setup(
    name="PyToF",
    version="1.4.2",
    description="numerical implementation of the Theory of Figures algorithm (4th, 7th, 10th order) including barotropic differential rotation",
    url="",
    author="Luca Morf",
    author_email="luca.morf@uzh.ch",
    license="Mozilla Public License Version 2.0",
    packages=find_packages(include=["PyToF", "PyToF.*"]),
    install_requires=["numpy==2.2.6", "scipy==1.15.3", "matplotlib==3.10.1", "emcee==3.1.6", "tqdm==4.67.1", "colorhash==2.0.0", "h5py==3.14.0"],
)