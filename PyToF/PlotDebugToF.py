########################################################
# Author of this version: Luca Morf - luca.morf@uzh.ch #
########################################################

import math
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy

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
    opts["axes.xmargin"] = 0.02
    opts["axes.ymargin"] = 0.02
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
    Generate and update diagnostic plots for density and pressure profiles
    during FunctionsToF execution.

    This function plots normalized density (rho) and pressure (P) profiles
    as a function of a chosen x-axis quantity (e.g. radius, enclosed mass,
    density, or pressure). Multiple calls to this function can be overlaid
    on the same figures using vertical offsets and distinct colors, allowing
    visual comparison across iterations or function calls for debugging
    purposes.

    Parameters
    ----------
    class_obj:      PyToF object, see ClassToF.py.
    what_x:         Integer index selecting the quantity to plot along the x
                    axis.
    new:            If True, initialize new figures and reset the plotting 
                    offset. If False, overlay the current profiles onto
                    existing figures.
    iteration:      Iteration counter used in the output filename when saving 
                    figures.
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

    # Initialize figures on first call:
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


def debug_AlgoToF_plot(ToF_order, z, domain, fs, integrand, integrand_p, integral, integral_p, names):
    """
    This function produces and saves a collection of debug figures that
    visualize intermediate quantities involved in the AlgoToF procedure,
    allowing comparison between multiple iterations or methods.

    The following plots are generated:
      1. Integrals I_{2j}(z) for a given method.
      2. Shape functions S_{2j}(z) constructed from the integrals and fs.
      3. Powers of z (z^{2j+3}) over the domain.
      4. Normalized integrands used in the computation of I_{2j}.
      5. Normalized shape functions f_{2j}.

    Figures are automatically saved to disk with incremented filenames
    to avoid overwriting existing files.

    Parameters
    ----------
    z:          array_like, corresponds to normalised radius values.
    domain:     array_like, corresponds to normalised density values.
    integrand:  sequence of array_like, integrand values for the Is.
    fs:         sequence of array_like, see arXiv:1708.06177v1.
    Is :        sequence of sequence of array_like, expected shape is 
                (number of methods, ToF_order + 1, len(z)).
    names:      labels corresponding to each method.
    """

    # Apply default and user supplied options:
    mpl_opts = {**_default_mpl_opts()}
    _apply_mpl_opts(mpl_opts)
    linestyles = ['-', '--', ':', '-.']
    
    # Initialize relevant variables:
    N = len(z) - 1

    def autosave_fig(fig, base_name):
        
        i = 0

        # While filename exists, increment the suffix
        while os.path.exists(f"{base_name}_{i}.png"):
            i += 1
            
        fig.savefig(f"{base_name}_{i}.png")

    # integral:

    fig, axs = plt.subplots(2, int(np.ceil(len(integral[0])/2)), figsize=[int(np.ceil(len(integral[0])/2))*6.4, 2*4.8], layout='constrained')
    axs = axs.reshape(-1)

    for j in range(len(integral[0])): #Is

        for i in range(len(integral)): #methods
                        
            axs[j].plot(z, integral[i][j], color='C0', alpha=0.7, linestyle=linestyles[i-4*(i//4)])
            axs[j].plot(z, domain*fs[j] - z**-(2*j+3)*integral[i][j], color='C1', alpha=0.7, linestyle=linestyles[i-4*(i//4)])
            
        axs[j].set_xlabel(r'$z$')
        axs[j].set_title(rf'$j={j}$', fontsize=16)

    legend_patches = []
    for i in range(2):
        legend_patches.append(matplotlib.patches.Patch(color=matplotlib.colors.to_rgb('C'+str(i)), label=[r'$I_{2j}$', r'$S_{2j}$'][i]))
    axs[0].add_artist(axs[0].legend(handles=legend_patches, ncol=2, loc='best', bbox_transform=axs[0].transAxes, bbox_to_anchor=(0.0, 0.0, 0.5, 1.0), handleheight=1.0, handlelength=1.0))
    
    legend_patches = []
    for i in range(len(names)):
        legend_patches.append(matplotlib.lines.Line2D([0], [0], color='black', label=names[i], linestyle=linestyles[i-4*(i//4)]))
    axs[0].legend(handles=legend_patches, ncol=1, loc='best', bbox_transform=axs[0].transAxes, bbox_to_anchor=(0.5, 0.0, 0.5, 1.0), frameon=False)
    
    autosave_fig(fig, f"debug_AlgoToF_integral")

    # integral_p:

    fig, axs = plt.subplots(2, int(np.ceil(len(integral[0])/2)), figsize=[int(np.ceil(len(integral[0])/2))*6.4, 2*4.8], layout='constrained')
    axs = axs.reshape(-1)

    for j in range(len(integral[0])): #Is

        for i in range(len(integral)): #methods
            
            #print('z', z)
            axs[j].plot(z, integral_p[i][j], color='C0', alpha=0.7, linestyle=linestyles[i-4*(i//4)])
            axs[j].plot(z, -domain*fs[ToF_order+1+j] + z**-(2-2*j) * (domain[N] * (fs[ToF_order + 1 + j])[N] - (integral_p[i][j][N] - integral_p[i][j])), color='C1', alpha=0.7, linestyle=linestyles[i-4*(i//4)])
            """
            orange       = -domain*fs[ToF_order+1+j] + z**-(2-2*j) * (domain[N] * (fs[ToF_order + 1 + j])[N] - (integral_p[i][j][N] - integral_p[i][j]))
            print('orange', orange)
            axs[j].plot(z, -domain*fs[ToF_order+1+j]                                                                                                    , color='C2', alpha=0.7, linestyle=linestyles[i-4*(i//4)])
            green        = -domain*fs[ToF_order+1+j]
            print('green', green)
            axs[j].plot(z,                             z**-(2-2*j) * (domain[N] * (fs[ToF_order + 1 + j])[N]                                           ), color='C3', alpha=0.7, linestyle=linestyles[i-4*(i//4)])
            red                                      = z**-(2-2*j) * (domain[N] * (fs[ToF_order + 1 + j])[N]                                           )
            print('red', red)
            axs[j].plot(z,                             z**-(2-2*j) * (                                       - (integral_p[i][j][N] - integral_p[i][j])), color='C4', alpha=0.7, linestyle=linestyles[i-4*(i//4)])
            purple                                   = z**-(2-2*j) * (                                       - (integral_p[i][j][N] - integral_p[i][j]))
            print('purple', purple)
            """

        axs[j].set_xlabel(r'$z$')
        axs[j].set_title(rf'$j={j}$', fontsize=16)

    legend_patches = []
    for i in range(2):
        legend_patches.append(matplotlib.patches.Patch(color=matplotlib.colors.to_rgb('C'+str(i)), label=[r'$I_{2j}$', r'$S_{2j}$'][i]))
    axs[0].add_artist(axs[0].legend(handles=legend_patches, ncol=2, loc='best', bbox_transform=axs[0].transAxes, bbox_to_anchor=(0.0, 0.0, 0.5, 1.0), handleheight=1.0, handlelength=1.0))
    
    legend_patches = []
    for i in range(len(names)):
        legend_patches.append(matplotlib.lines.Line2D([0], [0], color='black', label=names[i], linestyle=linestyles[i-4*(i//4)]))
    axs[0].legend(handles=legend_patches, ncol=1, loc='best', bbox_transform=axs[0].transAxes, bbox_to_anchor=(0.5, 0.0, 0.5, 1.0), frameon=False)
    
    autosave_fig(fig, f"debug_AlgoToF_integral_p")

    # integrand:

    fig, ax = plt.subplots(layout='constrained')

    for j in range(len(integral[0])): #domains
          
        ax.plot(z, integrand[j]/np.max(abs(integrand[j])), label=r'$z^{'+str(2*j+3)+'} f_{'+str(2*j)+'}$')

    ax.set_xlabel(r'$z$')
    ax.legend()
    autosave_fig(fig, f"debug_AlgoToF_integrand")

    # integrand_p:

    fig, ax = plt.subplots(layout='constrained')

    for j in range(len(integral[0])): #domains
          
        ax.plot(z, integrand_p[j]/np.max(abs(integrand_p[j])), label=r'$z^{'+str(2-2*j)+'} f^\prime_{'+str(2*j)+'}$')

    ax.set_xlabel(r'$z$')
    ax.legend()
    autosave_fig(fig, f"debug_AlgoToF_integrand_p")