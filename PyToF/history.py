"""Record of the convergence of relax_to_barotrope() / relax_to_density()."""

from dataclasses import dataclass, field

import numpy as np

__all__ = ["ConvergenceHistory"]


@dataclass
class ConvergenceHistory:
    """
    What happened in the last call of relax_to_barotrope() or
    relax_to_density(), available as ``ToF.history`` (None before the first
    call and after ``reset()``). One entry per iteration of the outer loop,
    in this order:

    name:       "barotrope" or "density"
    Js:         list of arrays, the gravitational moments after the iteration
                (like ``ToF.Js``, so ``Js[i][1]`` is J2)
    drot:       relative change of the rotational parameter m_rot_calc
    dJs:        largest relative change of the Js
    drho:       largest relative change of the densities
    converged:  whether all tolerances were reached before the iteration limit
    simpson_failed: whether PyToF had to leave Simpson's rule on the way

    The arrays ``drot``, ``dJs`` and ``drho`` are the quantities the
    tolerances ``drot_tol``, ``dJ_tol`` and ``drho_tol`` apply to. Use
    ``as_arrays()`` for numpy arrays, e.g. to plot ``dJs`` against the
    iteration number.
    """

    name: str
    Js: list = field(default_factory=list)
    drot: list = field(default_factory=list)
    dJs: list = field(default_factory=list)
    drho: list = field(default_factory=list)
    converged: bool = False
    simpson_failed: bool = False

    def __len__(self):
        return len(self.drot)

    def record(self, Js, drot, dJs, drho):
        """Append the result of one iteration."""

        self.Js.append(np.array(Js, dtype=float))
        self.drot.append(float(drot))
        self.dJs.append(float(dJs))
        self.drho.append(float(drho))

    def as_arrays(self):
        """Dictionary of numpy arrays: Js (iterations x entries), drot, dJs,
        drho."""

        return {
            "Js": np.array(self.Js),
            "drot": np.array(self.drot),
            "dJs": np.array(self.dJs),
            "drho": np.array(self.drho),
        }
