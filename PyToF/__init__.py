########################################################
# Author of this version: Luca Morf - luca.morf@uzh.ch #
########################################################

from importlib.metadata import PackageNotFoundError, version

from PyToF.ClassToF import ToF as ToF
from PyToF.sampling import ConvergenceCriteria as ConvergenceCriteria
from PyToF.sampling import MCMCResult as MCMCResult
from PyToF.sampling import SaveFits as SaveFits

try:
    __version__ = version("PyToF")
except (
    PackageNotFoundError
):  # running from a source tree that is not installed
    __version__ = "unknown"
