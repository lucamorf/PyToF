########################################################
# Author of this version: Luca Morf - luca.morf@uzh.ch #
########################################################

import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


def _default_mpl_opts():
    """
    Return default Matplotlib rcParams options for all plots.

    These settings control visual aspects such as line width, marker size,
    tick style, font size, and figure dimensions. They serve as a base for
    all plotting functions in this module.
    """

    opts = {}

    tick_major_size = 7.0
    tick_minor_size = 4.0
    tick_widths = 1.6

    # Lines and errobar:
    opts["lines.linewidth"] = 3.0
    opts["lines.markersize"] = 7.5
    opts["errorbar.capsize"] = 0.0

    # Axes and margins:
    opts["axes.linewidth"] = 1.6
    opts["axes.labelsize"] = 16
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
    opts["xtick.major.size"] = tick_major_size
    opts["ytick.major.size"] = tick_major_size
    opts["xtick.minor.size"] = tick_minor_size
    opts["ytick.minor.size"] = tick_minor_size
    opts["xtick.major.width"] = tick_widths
    opts["ytick.major.width"] = tick_widths
    opts["xtick.minor.width"] = tick_widths
    opts["ytick.minor.width"] = tick_widths
    opts["xtick.major.pad"] = 7.0
    opts["ytick.major.pad"] = 7.0

    # Legend:
    opts["legend.title_fontsize"] = 14
    opts["legend.fontsize"] = 14

    # Figure properties:
    opts["figure.figsize"] = [6.4, 4.8]
    opts["figure.dpi"] = 300
    if [f for f in matplotlib.font_manager.findSystemFonts() if "Ubuntu" in f]:
        opts["font.family"] = "Ubuntu"
    else:
        opts["font.family"] = "DejaVu Sans"  # very safe fallback

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


def _default_plot_xy_opts():
    """
    Return default keyword arguments for plot_xy(), plot_state_xy(), and
    plot_state_corr_xy().

    These defaults define line appearance, legend configuration, and
    output options for saving figures.
    """

    opts = {}

    opts["color"] = "C0"
    opts["linestyle"] = "-"
    opts["label"] = ""
    opts["zorder"] = 0

    opts["do_legend"] = False
    opts["loc_legend"] = "best"
    opts["ncol_legend"] = 1
    opts["frameon_legend"] = True
    opts["bbox_to_anchor_legend"] = (0.0, 0.0, 1.0, 1.0)

    opts["do_save"] = False
    opts["path_name"] = os.getcwd()
    opts["fig_name"] = "figure"
    opts["format"] = "png"
    opts["transparent"] = False

    return opts


def _default_plot_shape_opts():
    """
    Return default keyword arguments for plot_shape().

    These defaults control contour levels, color maps, and figure export
    settings for shape visualization.
    """

    opts = {}

    opts["contourf_levels"] = 50
    opts["contourf_cmap"] = "viridis"
    opts["contour_levels"] = [0.05, 0.20, 0.35, 0.50, 0.65, 0.80, 0.95]
    opts["contour_colors"] = "white"
    opts["contour_fontsize"] = 15
    opts["colorbar_ticks"] = [0, 0.2, 0.4, 0.6, 0.8, 1.0]

    opts["do_save"] = False
    opts["path_name"] = os.getcwd()
    opts["fig_name"] = "figure"
    opts["format"] = "png"
    opts["transparent"] = False

    return opts


def _default_plot_ss():
    """
    Return default keyword arguments for plot_ss().

    The returned parameters define legend and save options for plotting
    figure functions or spherical harmonics.
    """

    opts = {}

    opts["do_legend"] = True
    opts["loc_legend"] = "best"
    opts["ncol_legend"] = 1
    opts["frameon_legend"] = False
    opts["bbox_to_anchor_legend"] = (1.0, 1.0)

    opts["do_save"] = False
    opts["path_name"] = os.getcwd()
    opts["fig_name"] = "figure"
    opts["format"] = "png"
    opts["transparent"] = False

    return opts


def plot_xy(
    self,
    x,
    y,
    do_new_figure=True,
    literature=None,
    inset_plot_xy=None,
    colorbar=None,
    **kwargs,
):
    """
    Plot selected properties of a model (for example radius vs. density).

    Parameters
    ----------
    self:      PyToF object, does not need to be provided by the user,
                    see ClassToF.py.
    x, y:           Integer indices selecting which quantities to plot along x
                    and y axes.
    do_new_figure:  bool, whether to create a new figure, by default True.
    literature:     object, reference data object with attributes lis, rhois,
                    Pis, etc.
    inset_plot_xy:  object, definition of an inset subplot with limits and
                    axes configuration.
    colorbar:       object, defining a Matplotlib colorbar, with attributes
                    such as mappable, label, width, stretch, etc.
    **kwargs:       Additional keyword arguments to override default plot
                    settings.
    """

    # Apply default and user supplied options:
    mpl_opts = {**_default_mpl_opts(), **kwargs}
    _apply_mpl_opts(mpl_opts)
    opts = {**_default_plot_xy_opts(), **kwargs}

    # Calculate mass of each layer and enclosed mass:
    shell_m = (
        4
        / 3
        * np.pi
        * (self.li**3 - np.append(self.li[1:], 0) ** 3)
        * self.rhoi
    )
    summed_m = np.array([sum(shell_m[i:]) for i in range(len(shell_m))])

    # Data for plotting:
    xy = [
        self.li / self.li[0],
        summed_m / summed_m[0],
        self.rhoi / 1000,
        self.Pi / 1e5,
    ]
    label_xy = [
        r"average $r/R$",
        r"enclosed $m/M$",
        r"$\rho$ [g/cm$^3$]",
        r"$P$ [bar]",
    ]
    scale_xy = ["linear", "linear", "linear", "log"]

    # Append radius zero entry for completeness:
    xy[0] = np.append(xy[0], 0)
    xy[1] = np.append(xy[1], 0)
    xy[2] = np.append(xy[2], xy[2][-1])
    xy[3] = np.append(xy[3], xy[3][-1])

    # Additional data for plotting supplied by the user:
    if literature is not None:
        literature_xys = []

        for i in range(len(literature.lis)):
            # Get radii and sort them:
            li = literature.lis[i]
            p = np.argsort(li)[::-1]

            # Calculate mass of each layer and enclosed mass:
            shell_m = (
                4
                / 3
                * np.pi
                * (li[p] ** 3 - np.append(li[p][1:], 0) ** 3)
                * literature.rhois[i][p]
            )
            summed_m = np.array(
                [sum(shell_m[i:]) for i in range(len(shell_m))]
            )

            # Data for plotting:
            literature_xys.append(
                [
                    li[p] / li[p][0],
                    summed_m / summed_m[0],
                    literature.rhois[i][p] / 1000,
                    literature.Pis[i][p] / 1e5,
                ]
            )

    # Create new figure if needed:
    if do_new_figure:
        self.fig, [self.ax, self.cax] = plt.subplots(
            1, 2, layout="constrained", gridspec_kw={"width_ratios": [1, 0.01]}
        )
        self.cax.axis("off")
        self.ax.set_xlabel(label_xy[x])
        self.ax.set_ylabel(label_xy[y])
        self.ax.set_xscale(scale_xy[x])
        self.ax.set_yscale(scale_xy[y])
        if x == 2 or x == 3:
            self.ax.invert_xaxis()

        # Add inset plot within figure if needed:
        if inset_plot_xy is not None:
            self.ax_ins = inset_axes(
                self.ax,
                width=inset_plot_xy.width,
                height=inset_plot_xy.height,
                loc=inset_plot_xy.loc,
                borderpad=inset_plot_xy.borderpad,
                bbox_to_anchor=inset_plot_xy.bbox_to_anchor,
                bbox_transform=self.ax.transAxes,
                axes_kwargs={
                    "xlim": inset_plot_xy.xlim,
                    "ylim": inset_plot_xy.ylim,
                    "xscale": scale_xy[inset_plot_xy.x],
                    "yscale": scale_xy[inset_plot_xy.y],
                },
            )
            self.ax_ins.set_xlabel(label_xy[inset_plot_xy.x])
            self.ax_ins.set_ylabel(label_xy[inset_plot_xy.y])
            if inset_plot_xy.x == 2 or inset_plot_xy.x == 3:
                self.ax_ins.invert_xaxis()

    # Do the plotting:
    self.ax.plot(
        xy[x],
        xy[y],
        color=opts["color"],
        linestyle=opts["linestyle"],
        label=opts["label"],
        zorder=opts["zorder"],
    )

    # Do the plotting (inset plot):
    if inset_plot_xy is not None:
        self.ax_ins.plot(
            xy[inset_plot_xy.x],
            xy[inset_plot_xy.y],
            color=opts["color"],
            linestyle=opts["linestyle"],
            label=opts["label"],
            zorder=opts["zorder"],
        )

    # Do the plotting (user provided data):
    if literature is not None:
        for i in range(len(literature_xys)):
            # Do the plotting:
            self.ax.plot(
                literature_xys[i][x],
                literature_xys[i][y],
                color=literature.colors[i],
                linestyle=literature.linestyles[i],
                label=literature.labels[i],
                zorder=literature.zorders[i],
            )

            # Do the plotting (inset plot):
            if inset_plot_xy is not None:
                self.ax_ins.plot(
                    literature_xys[i][inset_plot_xy.x],
                    literature_xys[i][inset_plot_xy.y],
                    linestyle=literature.linestyles[i],
                    label=literature.labels[i],
                    zorder=literature.zorders[i],
                )

    # Add a colorbar if needed:
    if colorbar is not None:
        gs = self.ax.get_subplotspec().get_gridspec()
        gs.set_width_ratios([1, colorbar.width])
        self.cax.axis("on")
        self.fig.set_size_inches(
            mpl_opts["figure.figsize"][0] * colorbar.stretch,
            mpl_opts["figure.figsize"][1],
            forward=True,
        )

        cbar = self.fig.colorbar(
            colorbar.mappable, cax=self.cax, orientation="vertical"
        )
        cbar.set_label(colorbar.label)
        cbar.ax.tick_params(
            which="major",
            direction="out",
            right=True,
            left=False,
            width=mpl_opts["ytick.major.width"],
            length=mpl_opts["ytick.major.size"],
        )

        if hasattr(colorbar, "tick_lab"):
            cbar.set_ticks(colorbar.tick_pos)
            cbar.set_ticklabels(colorbar.tick_lab)

        if hasattr(colorbar, "tick_show"):
            if not colorbar.tick_show:
                cbar.ax.tick_params(left=False, right=False)

        cbar.ax.tick_params(
            which="minor",
            direction="out",
            right=True,
            left=False,
            width=mpl_opts["ytick.minor.width"],
            length=mpl_opts["ytick.minor.size"],
        )

    # Do the legend if needed:
    if opts["do_legend"]:
        self.ax.legend(
            loc=opts["loc_legend"],
            ncol=opts["ncol_legend"],
            frameon=opts["frameon_legend"],
            bbox_to_anchor=opts["bbox_to_anchor_legend"],
        )

    # Save the plot if needed:
    if opts["do_save"]:
        self.fig.savefig(
            opts["path_name"] + "/" + opts["fig_name"] + "." + opts["format"],
            format=opts["format"],
            transparent=opts["transparent"],
        )


def plot_shape(self, **kwargs):
    """
    Plot the internal density structure and shape of a rotating spheroid.

    Generates both Cartesian and polar contour plots of normalized density
    distributions based on model results.

    Parameters
    ----------
    self:      PyToF object, does not need to be provided by the user,
                    see ClassToF.py.
    **kwargs:       Additional keyword arguments to override default plot
                    settings.
    """

    # Apply default and user supplied options:
    mpl_opts = {**_default_mpl_opts(), **kwargs}
    _apply_mpl_opts(mpl_opts)
    opts = {**_default_plot_shape_opts(), **kwargs}

    # Define coordinates:
    theta = np.linspace(0, 2 * np.pi, 1000)
    r_l_mu = self.get_r_l_mu(np.cos(theta))
    N = np.shape(r_l_mu)[0]

    # Define variables for plotting:
    X = np.zeros((N + 1, 1000))
    Y = np.ones((N + 1, 1000))
    Z = np.ones((N + 1, 1000))

    # Ensures a radius 0 entry is present:
    X[:N, :] = r_l_mu / np.max(self.li)
    Y *= theta / (2 * np.pi) * 360
    Z[:N, :] = np.outer(self.rhoi / np.max(self.rhoi), np.ones_like(theta))

    # Create new figure:
    fig, axs = plt.subplots(
        1, 2, layout="constrained", gridspec_kw={"width_ratios": [1, 0.07]}
    )

    # Do the plotting:
    con = axs[0].contourf(
        X, Y, Z, levels=opts["contourf_levels"], cmap=opts["contourf_cmap"]
    )
    con_line = axs[0].contour(
        X, Y, Z, levels=opts["contour_levels"], colors=opts["contour_colors"]
    )
    cbar = fig.colorbar(
        con, ticks=opts["colorbar_ticks"], cax=axs[1], pad=0.02
    )

    # Do labeling:
    axs[0].clabel(con_line, inline=True, fontsize=opts["contour_fontsize"])
    cbar.ax.tick_params(
        which="major",
        direction="out",
        right=True,
        left=False,
        width=mpl_opts["ytick.major.width"],
        length=mpl_opts["ytick.major.size"],
    )
    cbar.ax.tick_params(
        which="minor",
        direction="out",
        right=True,
        left=False,
        width=mpl_opts["ytick.minor.width"],
        length=mpl_opts["ytick.minor.size"],
    )
    cbar.set_label(
        r"$\rho$ [" + str(np.round(np.max(self.rhoi / 1000), 2)) + " g/cm$^3$]"
    )
    axs[0].set_yticks(
        [0, 45, 90, 135, 180, 225, 270, 315, 360],
        labels=["N", "NW", "W", "SW", "S", "SE", "E", "NE", "N"],
    )
    axs[0].set_xlabel(r"spheroid shell radius / average surface radius")

    # Save the plot if needed:
    if opts["do_save"]:
        fig.savefig(
            opts["path_name"]
            + "/"
            + opts["fig_name"]
            + "_cartesian."
            + opts["format"],
            format=opts["format"],
            transparent=opts["transparent"],
        )

    # Create new figure:
    fig, axs = plt.subplots(
        1, 2, layout="constrained", gridspec_kw={"width_ratios": [1, 0.07]}
    )
    axs[0].remove()
    axs[0] = fig.add_subplot(1, 2, 1, projection="polar")

    # Do the plotting:
    con = axs[0].contourf(
        Y * (2 * np.pi) / 360,
        X,
        Z,
        levels=opts["contourf_levels"],
        cmap=opts["contourf_cmap"],
    )
    cbar = fig.colorbar(
        con, ticks=opts["colorbar_ticks"], cax=axs[1], pad=0.02
    )
    line = axs[0].plot(
        theta,
        np.ones_like(theta),
        color=(0.8, 0.8, 0.8, 1.0),
        label=r"average surface radius",
    )

    # Do labeling:
    cbar.ax.tick_params(
        which="major",
        direction="out",
        right=True,
        left=False,
        width=mpl_opts["ytick.major.width"],
        length=mpl_opts["ytick.major.size"],
    )
    cbar.ax.tick_params(
        which="minor",
        direction="out",
        right=True,
        left=False,
        width=mpl_opts["ytick.minor.width"],
        length=mpl_opts["ytick.minor.size"],
    )
    cbar.set_label(
        r"$\rho$ [" + str(np.round(np.max(self.rhoi / 1000), 2)) + " g/cm$^3$]"
    )
    axs[0].set_theta_zero_location("N")
    axs[0].set_thetagrids(
        [0, 45, 90, 135, 180, 225, 270, 315, 360],
        labels=["N", "NW", "W", "SW", "S", "SE", "E", "NE", "N"],
    )
    axs[0].set_rgrids([], labels=[])
    axs[0].spines["polar"].set_visible(False)
    axs[0].legend(
        handles=line,
        loc="center",
        bbox_to_anchor=(0.5, -0.15),
        borderaxespad=0,
        frameon=False,
    )

    # Save the plot if needed:
    if opts["do_save"]:
        fig.savefig(
            opts["path_name"]
            + "/"
            + opts["fig_name"]
            + "_polar."
            + opts["format"],
            format=opts["format"],
            transparent=opts["transparent"],
        )


def plot_ss(self, **kwargs):
    """
    Plot normalized figure functions (spherical shape coefficients).

    Parameters
    ----------
    self:      PyToF object, does not need to be provided by the user,
                    see ClassToF.py.
    **kwargs:       Additional keyword arguments to override default plot
                    settings.
    """

    # Apply default and user supplied options:
    mpl_opts = {**_default_mpl_opts(), **kwargs}
    _apply_mpl_opts(mpl_opts)
    opts = {**_default_plot_ss(), **kwargs}

    # Create new figure:
    self.fig, self.ax = plt.subplots(layout="constrained")

    # Get data for plotting:
    figure_functions = self.figure_functions

    handles = []
    labels = []
    for i in range(len(figure_functions)):
        if i != 0:
            x = np.append(self.li / self.li[0], 0)
            f = figure_functions[i]
            y = np.append(
                np.flip(f / np.max(abs(f))),
                f[0] / np.max(abs(f)),
            )

            # Do the plotting:
            val, exp = f"{np.max(abs(figure_functions[i])):.1e}".split("e")
            self.ax.plot(
                x,
                y,
            )
            labels.append(
                rf"${float(val):.1f}"
                rf" \cdot 10^{{{int(exp)}}}"
                rf" s_{{{2 * i}}}$"
            )
            handles.append(
                matplotlib.patches.Patch(
                    color=matplotlib.colors.to_rgb("C" + str(i - 1))
                )
            )

    # Do the labeling:
    self.ax.set_xlabel(r"average $r/R$")
    self.ax.set_ylabel(r"normalised figure functions")

    # Do the legend if needed:
    if opts["do_legend"]:
        self.ax.legend(
            handles,
            labels,
            handletextpad=0.3,
            labelspacing=0.3,
            columnspacing=1.0,
            handleheight=1.0,
            handlelength=1.0,
            loc=opts["loc_legend"],
            ncol=opts["ncol_legend"],
            frameon=opts["frameon_legend"],
            bbox_to_anchor=opts["bbox_to_anchor_legend"],
        )

    # Save the plot if needed:
    if opts["do_save"]:
        self.fig.savefig(
            opts["path_name"] + "/" + opts["fig_name"] + "." + opts["format"],
            format=opts["format"],
            transparent=opts["transparent"],
        )
