########################################################
# Author of this version: Luca Morf - luca.morf@uzh.ch #
########################################################

import math
import os

import matplotlib.pyplot as plt
import numpy as np
import scipy
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from PyToF.color import c


def _default_mpl_opts():
    """
    Return default Matplotlib rcParams options for all plots.

    These settings control visual aspects such as line width, marker size,
    tick style, font size, and figure dimensions. They serve as a base for
    all plotting functions in this module.
    """

    opts = {}

    # Lines and errobar:
    opts["lines.linewidth"] = 3.0
    opts["lines.markersize"] = 9
    opts["errorbar.capsize"] = 0.0

    # Axes and margins:
    opts["axes.labelsize"] = 18
    opts["axes.xmargin"] = 0.0
    opts["axes.ymargin"] = 0.0
    opts["axes.formatter.useoffset"] = False

    # Ticks:
    opts["xtick.labelsize"] = 16
    opts["ytick.labelsize"] = 16
    opts["xtick.direction"] = "in"
    opts["ytick.direction"] = "in"
    opts["xtick.top"] = True
    opts["ytick.right"] = True
    opts["xtick.minor.visible"] = True
    opts["ytick.minor.visible"] = True
    opts["xtick.major.size"] = 7.0
    opts["ytick.major.size"] = 7.0
    opts["xtick.minor.size"] = 4.0
    opts["ytick.minor.size"] = 4.0
    opts["xtick.major.width"] = 1.6
    opts["ytick.major.width"] = 1.6
    opts["xtick.major.pad"] = 7.0
    opts["ytick.major.pad"] = 7.0

    # Legend:
    opts["legend.title_fontsize"] = 16
    opts["legend.fontsize"] = 16

    # Figure properties:
    opts["figure.figsize"] = [6.4, 4.8]
    opts["figure.dpi"] = 200
    opts["font.family"] = "Ubuntu"

    return opts


def _apply_mpl_opts(opts):
    """
    Apply Matplotlib rcParams options from a dictionary.

    opts:   Dictionary of rcParams-style key-value pairs to apply globally.
    """

    for kwd, value in opts.items():
        try:
            plt.rcParams[kwd] = value

        except (KeyError, ValueError):
            continue


def debug_FunctionsToF_plot(class_obj, what_x=0, new=True, iteration=0):
    """
    TODO
    """

    # Apply default and user supplied options:
    mpl_opts = {**_default_mpl_opts()}
    _apply_mpl_opts(mpl_opts)

    # Calculate mass of each layer and enclosed mass:
    shell_m = (
        4
        / 3
        * np.pi
        * (class_obj.li**3 - np.append(class_obj.li[1:], 0) ** 3)
        * class_obj.rhoi
    )
    summed_m = np.array([sum(shell_m[i:]) for i in range(len(shell_m))])

    # Data for plotting:
    xs = [
        class_obj.li / class_obj.li[0],
        summed_m / summed_m[0],
        class_obj.rhoi / 1000,
        class_obj.Pi / 1e5,
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

        class_obj.bugfix_offset = 0
        
        class_obj.bugfix_rho_fig, class_obj.bugfix_rho_ax = plt.subplots(layout="constrained")
        class_obj.bugfix_rho_ax.set_xscale(scale_xs[what_x])
        class_obj.bugfix_rho_ax.set_xlabel(label_xs[what_x])
        class_obj.bugfix_rho_ax.set_ylabel(r"normalized $\rho$ + offset")
        class_obj.bugfix_rho_ax.set_yticks([])

        class_obj.bugfix_P_fig, class_obj.bugfix_P_ax = plt.subplots(layout="constrained")
        class_obj.bugfix_P_ax.set_xscale(scale_xs[what_x])
        class_obj.bugfix_P_ax.set_xlabel(label_xs[what_x])
        class_obj.bugfix_P_ax.set_ylabel(r"normalized $P$ + offset")
        class_obj.bugfix_P_ax.set_yticks([])

    
    class_obj.bugfix_rho_ax.plot(       xs[what_x], class_obj.rhoi/np.max(abs(class_obj.rhoi))+offset*(class_obj.bugfix_offset-1), color='C'+str(class_obj.bugfix_offset))
    class_obj.bugfix_P_ax.plot(         xs[what_x], class_obj.Pi  /np.max(abs(class_obj.Pi))  +offset*(class_obj.bugfix_offset-1), color='C'+str(class_obj.bugfix_offset))

    class_obj.bugfix_rho_ax.annotate(   str(class_obj.bugfix_offset), (xs[what_x][0], class_obj.rhoi[0]/np.max(abs(class_obj.rhoi))+offset*(class_obj.bugfix_offset-1)))
    class_obj.bugfix_P_ax.annotate(     str(class_obj.bugfix_offset), (xs[what_x][0], class_obj.Pi[0]  /np.max(abs(class_obj.Pi))  +offset*(class_obj.bugfix_offset-1)))

    class_obj.bugfix_rho_fig.savefig(
            f"debug_FunctionsToF_rho_{iteration}.png", format="png", transparent=False
    )
    class_obj.bugfix_P_fig.savefig(
        f"debug_FunctionsToF_P_{iteration}.png", format="png", transparent=False
    )

    class_obj.bugfix_offset += 1


def debug_AlgoToF_plot(z, domain, integrand, fs, Is, names):
    """
    TODO
    """

    # Apply default and user supplied options:
    mpl_opts = {**_default_mpl_opts()}
    _apply_mpl_opts(mpl_opts)
    linestyles = ['-', '--', ':']

    def autosave_fig(fig, base_name):
        
        i = 0

        # While filename exists, increment the suffix
        while os.path.exists(f"{base_name}_{i}.png"):
            i += 1
            
        fig.savefig(f"{base_name}_{i}.png")

    ### Integrals:

    fig, axs = plt.subplots(2, int(np.ceil(len(Is[0])/2)), figsize=[int(np.ceil(len(Is[0])/2))*6.4, 2*4.8], layout='constrained')
    axs = axs.reshape(-1)

    for j in range(len(Is[0])): #Is

        for i in range(len(Is)): #methods
                        
            axs[j].plot(z, Is[i][j], label=names[i], linestyle=linestyles[i-3*(i//3)])

        axs[j].set_xlabel(r'$z$')
        axs[j].set_ylabel(rf'$I_{2*j}$')

    axs[0].legend()
    autosave_fig(fig, f"debug_AlgoToF_Is")

    ### Shape functions:

    fig, axs = plt.subplots(2, int(np.ceil(len(Is[0])/2)), figsize=[int(np.ceil(len(Is[0])/2))*6.4, 2*4.8], layout='constrained')
    axs = axs.reshape(-1)

    for j in range(len(Is[0])): #Is

        for i in range(len(Is)): #methods
                        
            axs[j].plot(z, domain*fs[j] - z**-(2*j+3)*Is[i][j], label=names[i], linestyle=linestyles[i-3*(i//3)])

        axs[j].set_xlabel(r'$z$')
        axs[j].set_ylabel(rf'$S_{2*j}$')

    axs[0].legend()
    autosave_fig(fig, f"debug_AlgoToF_Ss")

    ### Z to some power:

    fig, ax = plt.subplots(layout='constrained')

    for j in range(len(Is[0])): #z_powers
          
        ax.plot(domain, z**(2*j+3), label=r'$z^{'+str(2*j+3)+'}$')

    ax.set_xlabel(r'$\rho/\bar{\rho}$')
    ax.legend()
    fig.savefig(f"debug_AlgoToF_z_power")

    ### Integrands:

    fig, ax = plt.subplots(layout='constrained')

    for j in range(len(Is[0])): #domains
          
        ax.plot(domain, integrand[j]/np.max(abs(integrand[j])), label=r'$z^{'+str(2*j+3)+'} f_{'+str(2*j)+'}$')

    ax.set_xlabel(r'$\rho/\bar{\rho}$')
    ax.legend()
    autosave_fig(fig, f"debug_AlgoToF_integrand")

    ### fs:

    fig, ax = plt.subplots(layout='constrained')

    for j in range(len(Is[0])): #domains
          
        ax.plot(domain, fs[j]/np.max(abs(fs[j])), label=r'$f_{'+str(2*j)+'}$')

    ax.set_xlabel(r'$\rho/\bar{\rho}$')
    ax.legend()
    autosave_fig(fig, f"debug_AlgoToF_fs")