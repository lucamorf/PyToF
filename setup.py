########################################################
# Author of this version: Luca Morf - luca.morf@uzh.ch #
########################################################

from setuptools import setup, find_packages

USE_PYTHON3_10 = True

if USE_PYTHON3_10:
    install_requires=["numpy==2.2.6", "scipy==1.15.3", "matplotlib==3.10.1", "emcee==3.1.6", "tqdm==4.67.1"]
    python_requires=">=3.10.0, <3.11"
else:
    install_requires=["numpy==2.4.1", "scipy==1.17.0", "matplotlib==3.10.8", "emcee==3.1.6", "tqdm==4.67.1"]
    python_requires=">=3.12.0, <3.13"

setup(
    name="PyToF",
    version="1.6.1",
    description="numerical implementation of the Theory of Figures algorithm (4th, 7th, 10th order) including barotropic differential rotation",
    url="",
    author="Luca Morf",
    author_email="luca.morf@uzh.ch",
    license="Mozilla Public License Version 2.0",
    packages=find_packages(include=["PyToF", "PyToF.*"]),
    include_package_data=True,
    package_data={"PyToF": ["*.npz"]},
    install_requires=install_requires,
    python_requires=python_requires,
)


    

