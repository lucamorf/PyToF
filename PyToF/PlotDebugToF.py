########################################################
# Author of this version: Luca Morf - luca.morf@uzh.ch #
########################################################

"""
Debug plots, shown/saved when the option ``debug_plot`` is True.

This module is imported lazily (only when ``debug_plot`` is True), so
matplotlib is not needed for normal use. Figures are written to the current
working directory. The plot style is matplotlib's current one (set your own
rcParams if you want a different look).
"""

import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

_LINESTYLES = ["-", "--", ":", "-."]


def _normalized(y):
    """y divided by its largest absolute value (y itself if that is zero)."""
    scale = np.max(abs(y))
    return y / scale if scale > 0 else y


def _autosave_fig(fig, base_name):
    """Save ``fig`` as base_name_<i>.png with the first unused i; close it."""
    i = 0
    while os.path.exists(f"{base_name}_{i}.png"):
        i += 1
    fig.savefig(f"{base_name}_{i}.png")
    plt.close(fig)


def debug_FunctionsToF_plot(tof, what_x=0, new=True, iteration=0):
    """
    Plot the density and pressure profiles during relax_to_barotrope() /
    relax_to_density().

    The normalized profiles are drawn against a chosen x-axis quantity.
    Successive calls are overlaid on the same figures with a vertical offset
    and a different color, so that the iterations can be compared. The
    figures are saved after every call.

    Parameters
    ----------
    tof:        PyToF object, see ClassToF.py.
    what_x:     Index of the quantity along the x axis: 0: average radius
                r/R, 1: enclosed mass m/M, 2: density, 3: pressure.
    new:        If True, create new figures and reset the offset. If False,
                add the current profiles to the existing figures.
    iteration:  Iteration counter used in the output file names.
    """
    # Mass of each layer and enclosed mass (sum over this layer and all
    # layers below it):
    shell_m = (
        4 / 3 * np.pi * (tof.li**3 - np.append(tof.li[1:], 0) ** 3) * tof.rhoi
    )
    enclosed_m = np.cumsum(shell_m[::-1])[::-1]

    xs = [
        tof.li / tof.li[0],
        enclosed_m / enclosed_m[0],
        tof.rhoi / 1000,
        tof.Pi / 1e5,
    ]
    label_xs = [
        r"average $r/R$",
        r"enclosed $m/M$",
        r"$\rho$ [g/cm$^3$]",
        r"$P$ [bar]",
    ]
    scale_xs = ["linear", "linear", "linear", "log"]
    offset = 0.05

    if new:
        tof.bugfix_offset = 0
        for name, label in (("rho", r"\rho"), ("P", "P")):
            fig, ax = plt.subplots(layout="constrained")
            ax.set_xscale(scale_xs[what_x])
            ax.set_xlabel(label_xs[what_x])
            ax.set_ylabel(rf"normalized ${label}$ + offset")
            ax.set_yticks([])
            setattr(tof, f"bugfix_{name}_fig", fig)
            setattr(tof, f"bugfix_{name}_ax", ax)

    k = tof.bugfix_offset
    for name, y in (("rho", tof.rhoi), ("P", tof.Pi)):
        ax = getattr(tof, f"bugfix_{name}_ax")
        fig = getattr(tof, f"bugfix_{name}_fig")
        y = _normalized(y) + offset * (k - 1)
        ax.plot(xs[what_x], y, color=f"C{k}")
        ax.annotate(str(k), (xs[what_x][0], y[0]))
        fig.savefig(
            f"debug_FunctionsToF_{name}_{iteration}.png",
            format="png",
            transparent=False,
        )

    tof.bugfix_offset += 1


def _method_legends(ax, names):
    """Legend for the quantities (colors) and the methods (line styles)."""
    patches = [
        Patch(color=to_rgb(f"C{i}"), label=label)
        for i, label in enumerate([r"$I_{2j}$", r"$S_{2j}$"])
    ]
    ax.add_artist(
        ax.legend(
            handles=patches,
            ncol=2,
            loc="best",
            bbox_transform=ax.transAxes,
            bbox_to_anchor=(0.0, 0.0, 0.5, 1.0),
            handleheight=1.0,
            handlelength=1.0,
        )
    )
    lines = [
        Line2D(
            [0],
            [0],
            color="black",
            label=name,
            linestyle=_LINESTYLES[i % len(_LINESTYLES)],
        )
        for i, name in enumerate(names)
    ]
    ax.legend(
        handles=lines,
        ncol=1,
        loc="best",
        bbox_transform=ax.transAxes,
        bbox_to_anchor=(0.5, 0.0, 0.5, 1.0),
        frameon=False,
    )


def _integral_figure(z, n_j, curves, names, base_name):
    """
    One panel per j, with for every method (line style) the integral
    (color C0) and the shape function S_2j (color C1).

    curves: function (method index i, j) -> (I_2j(z), S_2j(z)); n_j panels.
    """
    n_methods = len(names)
    ncols = int(np.ceil(n_j / 2))
    fig, axs = plt.subplots(
        2,
        ncols,
        figsize=[ncols * 6.4, 2 * 4.8],
        layout="constrained",
        squeeze=False,
    )
    axs = axs.reshape(-1)
    for j in range(n_j):
        for i in range(n_methods):
            ls = _LINESTYLES[i % len(_LINESTYLES)]
            integral, shape = curves(i, j)
            axs[j].plot(z, integral, color="C0", alpha=0.7, linestyle=ls)
            axs[j].plot(z, shape, color="C1", alpha=0.7, linestyle=ls)
        axs[j].set_xlabel(r"$z$")
        axs[j].set_title(rf"$j={j}$", fontsize=16)
    _method_legends(axs[0], names)
    _autosave_fig(fig, base_name)


def _integrand_figure(z, integrand, label, base_name):
    fig, ax = plt.subplots(layout="constrained")
    for j, y in enumerate(integrand):
        ax.plot(z, _normalized(y), label=label(j))
    ax.set_xlabel(r"$z$")
    ax.legend()
    _autosave_fig(fig, base_name)


def debug_AlgoToF_plot(
    ToF_order,
    z,
    domain,
    fs,
    integrand,
    integrand_p,
    integral,
    integral_p,
    names,
):
    """
    Save debug figures of the intermediate quantities of AlgoToF.Algorithm,
    comparing several integration methods. The files are called
    debug_AlgoToF_<what>_<i>.png with the first unused i, so nothing is
    overwritten.

    The figures are:
      integral:    I_2j(z) and S_2j(z) for every method.
      integral_p:  the same for the primed (outer) integrals.
      integrand:   normalized integrands z^(2j+3) f_2j of the I_2j.
      integrand_p: normalized integrands z^(2-2j) f'_2j.

    Parameters
    ----------
    ToF_order:    Order of the theory of figures.
    z:            Normalized radius values.
    domain:       Normalized density values.
    fs:           Shape functions, see arXiv:1708.06177v1.
    integrand:    Integrands of the I_2j, one array per j.
    integrand_p:  Integrands of the primed integrals, one array per j.
    integral:     Integrals, shape (number of methods, ToF_order + 1, len(z)).
    integral_p:   Primed integrals, same shape as ``integral``.
    names:        Label of each method.
    """
    N = len(z) - 1
    n_j = len(integral[0])

    def inner(i, j):
        integral_ij = integral[i][j]
        return integral_ij, domain * fs[j] - z ** -(2 * j + 3) * integral_ij

    def outer(i, j):
        integral_ij = integral_p[i][j]
        f = fs[ToF_order + 1 + j]
        shape = -domain * f + z ** -(2 - 2 * j) * (
            domain[N] * f[N] - (integral_ij[N] - integral_ij)
        )
        return integral_ij, shape

    _integral_figure(z, n_j, inner, names, "debug_AlgoToF_integral")
    _integral_figure(z, n_j, outer, names, "debug_AlgoToF_integral_p")
    _integrand_figure(
        z,
        integrand[:n_j],
        lambda j: rf"$z^{{{2 * j + 3}}} f_{{{2 * j}}}$",
        "debug_AlgoToF_integrand",
    )
    _integrand_figure(
        z,
        integrand_p[:n_j],
        lambda j: rf"$z^{{{2 - 2 * j}}} f^\prime_{{{2 * j}}}$",
        "debug_AlgoToF_integrand_p",
    )
