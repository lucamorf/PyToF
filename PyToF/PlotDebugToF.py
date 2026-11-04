########################################################
# Author of this version: Luca Morf - luca.morf@uzh.ch #
########################################################

"""
Debug plots, drawn when the option ``debug_plot`` is "show" or "save" (True
means "save").

There are two kinds of figures, each one a single figure with a few panels so
that everything about one iteration can be checked at a glance:

* :func:`debug_relax_plot`, once per iteration of ``relax_to_barotrope()`` or
  ``relax_to_density()``: density, what changed with respect to the previous
  iteration, pressure with the deviation between the rules, and the
  convergence history. It is also drawn when the pressure comes out negative
  or non-monotonic, which is where Simpson's rule fails for a density jump
  (switch off the option ``simpson_check`` to see it).
* :func:`debug_AlgoToF_plot`, once per iteration of the Theory of Figures
  algorithm (``relax_to_shape()``, and twice per call from the relaxations
  above): the volume integrals S_2j (or the primed S'_2j, see the option
  ``debug_primed``) with Simpson's rule and the trapezoid rule side by side,
  the stability check of Simpson's rule, the shape functions that go into the
  step and the new figure functions.

Colors: two quantities that are expected to agree (Simpson's and the
trapezoid rule, the current and the previous iteration) share a color, in a
dark and a light shade of the tab20 colors, the dark one drawn dotted and on
top. In the Theory of Figures figure, black is used for what does not belong to
an index j, blue for the check on S_0 (which is j = 0), and the color gives
the index j otherwise.
Panels that show the same quantity on the x axis share it.

"show" displays the figures (inline in a notebook, in a window in a script),
"save" writes PNG files called debug_relax_<name>_<i>.png and
debug_AlgoToF_<i>.png into the current working directory, with the first
unused i so that nothing is overwritten.

This module is imported lazily (only when ``debug_plot`` is on), so
matplotlib is not needed for normal use. The plot style is matplotlib's
current one (set your own rcParams if you want a different look).
"""

import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

# Simpson's rule is drawn first, so that the dotted trapezoid line is on top.
# "shade" is the tab20 shade of a colored pair: 0 dark, 1 light. The dotted
# line is the dark one for readability.
_SIMPSON = {"label": "Simpson", "linestyle": "-", "color": "k", "shade": 1}
_TRAPEZOID = {
    "label": "trapezoid",
    "linestyle": ":",
    "color": "0.65",
    "shade": 0,
}

# For quantities that are shown with one rule only:
_PLAIN = {"linestyle": "-", "shade": 0}

_DPI = 150


def _color(k, shade):
    """The dark (shade 0) or light (shade 1) tab20 color of the color k."""
    return plt.get_cmap("tab20")(2 * k + shade)


def _normalized(y):
    """y divided by its largest absolute value (y itself if that is zero)."""
    scale = np.max(np.abs(y))
    return y / scale if scale > 0 else y


def _emit(fig, mode, base_name):
    """
    Show ``fig`` (mode "show") or save it as base_name_<i>.png with the first
    unused i (any other mode), then close it.
    """
    if mode == "show":
        plt.show()
    else:
        i = 0
        while os.path.exists(f"{base_name}_{i}.png"):
            i += 1
        fig.savefig(f"{base_name}_{i}.png", dpi=_DPI)
    plt.close(fig)


def _panels(nrows, ncols, title, size=(3.6, 3.0)):
    """A figure with a grid of panels, returned as a 2D array of axes."""
    fig, axs = plt.subplots(
        nrows,
        ncols,
        figsize=(ncols * size[0], nrows * size[1]),
        dpi=_DPI,
        layout="constrained",
        squeeze=False,
    )
    fig.suptitle(title, fontsize="large")
    for ax in axs.flat:
        ax.grid(alpha=0.3)
    return fig, axs


def _share_x(axs, columns, skip=()):
    """
    Let the panels of each column in ``columns`` share their x axis, which is
    the same quantity in the whole column. Only the lowest panel of a column
    keeps its tick labels (the caller sets its x label). ``skip`` lists
    (row, column) of panels that do not take part, e.g. a text panel. Call
    this before plotting.
    """
    for c in columns:
        column = [axs[r, c] for r in range(axs.shape[0]) if (r, c) not in skip]
        for ax in column[1:]:
            ax.sharex(column[0])
        for ax in column[:-1]:
            ax.tick_params(labelbottom=False)


def _set(ax, title, xlabel=None, ylabel=None, fontsize="medium"):
    ax.set_title(title, fontsize=fontsize)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)


def _note(ax, text, **kwargs):
    """Text in the middle of a panel (for what cannot be drawn)."""
    ax.text(
        0.5,
        0.5,
        text,
        transform=ax.transAxes,
        ha="center",
        va="center",
        color="gray",
        **kwargs,
    )


def _info_panel(ax, lines):
    """A panel that only holds text, one entry per line."""
    ax.axis("off")
    ax.grid(False)
    ax.text(0, 1, "\n".join(lines), va="top", family="monospace", fontsize=9)


def _mode(value):
    return "show" if value == "show" else "save"


# ------------------------------------------------------------ relax_to_*


def debug_relax_plot(tof, convergence=None, failure=False):
    """
    One figure with the state of a relax_to_barotrope() or relax_to_density()
    iteration, drawn after the iteration (``convergence``) or when the
    pressure is invalid (``failure``). The panels of a column share the x axis
    r/R.

    Parameters
    ----------
    tof:          PyToF object, see ClassToF.py.
    convergence:  (drot, dJ, drho) of the iteration just finished. It is added
                  to the convergence history. None if the figure is drawn in
                  the middle of an iteration.
    failure:      True if drawn because the pressure is invalid: the
                  densities have not been updated yet and both integration
                  rules are shown.
    """

    from PyToF.FunctionsToF import _pressure_profile

    name = getattr(tof, "_debug_name", "relax")
    iteration = getattr(tof, "_debug_iteration", 0)
    history = getattr(tof, "_debug_history", [])
    previous = history[-1] if history else None
    current = {"rho": tof.rhoi.copy(), "convergence": convergence}
    if convergence is not None:
        history.append(current)
    runs = [h for h in history if h["convergence"] is not None]

    r = tof.li / tof.li[0]
    rho = tof.rhoi / 1000  # g/cm^3
    P = tof.Pi / 1e5  # bar

    fig = plt.figure(figsize=(10.4, 8.4), dpi=_DPI, layout="constrained")
    fig.suptitle(f"relax_to_{name}, iteration {iteration}", fontsize="large")
    grid = fig.add_gridspec(3, 2, height_ratios=[3, 1, 2.2])
    axs = np.empty((3, 2), dtype=object)
    for row in range(2):
        for col in range(2):
            axs[row, col] = fig.add_subplot(
                grid[row, col], sharex=axs[0, col] if row else None
            )
            axs[row, col].grid(alpha=0.3)
    for col in range(2):
        axs[0, col].tick_params(labelbottom=False)
    axs[2, 0] = fig.add_subplot(grid[2, 0])
    axs[2, 0].grid(alpha=0.3)
    axs[2, 1] = fig.add_subplot(grid[2, 1])

    # Density, with the previous iteration for comparison:
    ax = axs[0, 0]
    if previous is not None:
        ax.plot(
            r, previous["rho"] / 1000, color=_color(0, 1), label="previous"
        )
    ax.plot(
        r,
        rho,
        color=_color(0, 0),
        linestyle=":",
        label="current",
    )
    _set(ax, "", None, r"density $\rho$ [g/cm$^3$]")
    ax.legend(frameon=False)

    # Where the density still changes:
    ax = axs[1, 0]
    if failure:
        _note(ax, "densities not yet updated")
    elif previous is not None:
        with np.errstate(divide="ignore", invalid="ignore"):
            change = np.abs(tof.rhoi[1:] / previous["rho"][1:] - 1)
        ax.semilogy(r[1:], np.maximum(change, 1e-17), color=_color(0, 0))
    else:
        _note(ax, "no previous iteration")
    _set(
        ax,
        "",
        "mean radius $r/R$",
        r"$|\rho/\rho_\mathrm{previous} - 1|$",
    )

    # Pressure from both integration rules:
    ax = axs[0, 1]
    profiles = {}
    for rule in (_SIMPSON, _TRAPEZOID):
        if rule is _SIMPSON and not tof.simpson_in_use and not failure:
            continue
        profiles[rule["label"]] = _pressure_profile(tof, rule is _SIMPSON)
        ax.plot(
            r,
            profiles[rule["label"]] / 1e5,
            color=_color(1, rule["shade"]),
            linestyle=rule["linestyle"],
            label=rule["label"],
        )
    scale = max(np.max(np.abs(P)), 1e-300)
    ax.set_yscale("log")
    ax.set_ylim(bottom=scale / 1e5 * 1e-6)
    _set(ax, "", None, r"pressure $P$ [bar]")
    ax.legend(frameon=False)

    # The two rules compared:
    ax = axs[1, 1]
    if len(profiles) == 2:
        with np.errstate(divide="ignore", invalid="ignore"):
            deviation = np.abs(profiles["Simpson"] / profiles["trapezoid"] - 1)
        ax.semilogy(r, np.maximum(deviation, 1e-17), color=_color(1, 0))
    else:
        _note(ax, "Simpson's rule not in use")
    _set(
        ax,
        "",
        "mean radius $r/R$",
        r"$|P_\mathrm{Simpson}/P_\mathrm{trapezoid} - 1|$",
    )

    # Convergence history:
    ax = axs[2, 0]
    if runs:
        k = np.arange(len(runs)) + 1
        for i, (label, marker) in enumerate(
            (("drot", "o"), ("dJ", "s"), ("drho", "^"))
        ):
            ax.semilogy(
                k,
                [max(h["convergence"][i], 1e-17) for h in runs],
                marker=marker,
                color=_color(2 + i, 0),
                label=label,
            )
        ax.legend(frameon=False)
        ax.set_xlim(0.5, len(runs) + 0.5)
    else:
        _note(ax, "no finished iteration")
    if len(runs) <= 10:
        ax.set_xticks(np.arange(len(runs)) + 1)
    else:
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    _set(
        ax,
        "",
        "iteration",
    )

    # Text, for what is not shown in the panels:
    opts = tof.opts
    lines = [
        f"in use: {'Simpson' if tof.simpson_in_use else 'trapezoid'}",
        f"N = {len(r)}, order = {opts['order']}",
        f"central density: {tof.rhoi[-1]:.1f} kg/m^3",
        f"central pressure: {P[-1]:.3e} bar",
        f"surface density: {tof.rhoi[0]:.1f} kg/m^3",
        f"surface pressure: {P[0]:.3e} bar",
    ]
    if convergence is not None:
        lines.append("")
        lines.append("finished:   value   tolerance")
        for label, value, tol in zip(
            ("drot", "dJ", "drho"),
            convergence,
            (opts["drot_tol"], opts["dJ_tol"], opts["drho_tol"]),
            strict=True,
        ):
            lines.append(f"{label:<5} {value:10.1e} {tol:10.1e}")
    _info_panel(axs[2, 1], lines)

    _emit(fig, _mode(tof.opts["debug_plot"]), f"debug_relax_{name}")


# --------------------------------------------------------------- AlgoToF


def _plot_rules(ax, x, curves_of_rule, rules):
    """
    The curves of every rule, one line per index j in the dark (Simpson) or
    light (trapezoid) shade of the color of j. Simpson's rule comes first, so
    the dotted trapezoid line is drawn on top of it. ``curves_of_rule(rule)``
    returns a 2D array with one row per j.
    """
    for rule in rules:
        for j, y in enumerate(curves_of_rule(rule)):
            ax.plot(
                x,
                y,
                color=_color(j, rule["shade"]),
                linestyle=rule["linestyle"],
                alpha=0.9,
            )


def debug_AlgoToF_plot(
    z,
    domain,
    fs,
    ss,
    Js,
    old_Js,
    order,
    iteration,
    use_simpson,
    mode,
    label="",
    primed=False,
):
    """
    One figure with the state of one iteration of AlgoToF.Algorithm. The
    volume integrals are calculated with both Simpson's rule (solid) and the
    trapezoid rule (dotted), whichever is in use, to see where they differ.
    The panels in the two left columns share the x axis z.

    Parameters
    ----------
    z:            Normalized radii, innermost first.
    domain:       Densities divided by the mean density, rho / rho_bar.
    fs:           Shape functions f_2j, followed by the primed f'_2j, see
                  arXiv:1708.06177v1.
    ss:           Figure functions s_2j after the iteration.
    Js:           Gravitational moments after the iteration.
    old_Js:       Gravitational moments of the previous iteration.
    order:        Order of the theory of figures.
    iteration:    Step of the algorithm (0, 1, ...), for the title.
    use_simpson:  Whether the algorithm uses Simpson's rule.
    mode:         "show" or "save", see the module docstring.
    label:        Added to the title, e.g. the relaxation it belongs to.
    primed:       If False, show the volume integrals S_2j, if True the
                  primed ones S'_2j (the stability check is always on S_0).
    """

    from PyToF.AlgoToF import _integrals, _volume_integrals

    n_j = order + 1
    j = np.arange(n_j)
    uniform = np.allclose(np.diff(z), np.diff(z)[0])
    key = "S_p" if primed else "S"

    # The volume integrals for both rules (Simpson's only on an equidistant
    # grid):
    rules = ([_SIMPSON] if uniform else []) + [_TRAPEZOID]
    results = {}
    for rule in rules:
        _, _, integral, integral_p = _integrals(
            fs, z, domain, order, rule is _SIMPSON
        )
        volume = _volume_integrals(fs, z, domain, integral, integral_p, order)
        results[rule["label"]] = {
            "S": np.array(volume[:n_j]),
            "S_p": np.array(volume[n_j:]),
        }
    trapezoid = results["trapezoid"]

    title = "AlgoToF"
    if label:
        title += f", {label}"
    title += f", step {iteration}"
    title += f" (using {'Simpson' if use_simpson else 'trapezoid rule'})"
    fig, axs = _panels(2, 3, title, size=(4.2, 3.4))
    _share_x(axs, (0, 1))

    # The density slope, which enters the integrands of Simpson's rule, and
    # the stability check of Simpson's rule (S_0 must not increase with z):
    axs[0, 0].plot(z, np.gradient(domain, z, edge_order=2), color="k")
    _set(
        axs[0, 0],
        "",
        None,
        r"d$(\rho/\bar\rho)$/d$z$",
    )

    ax = axs[1, 0]
    for rule in rules:
        S0 = results[rule["label"]]["S"][0]
        dS0 = np.diff(S0) / np.diff(z)
        ax.plot(
            z[:-1],
            dS0,
            color=_color(0, rule["shade"]),
            linestyle=rule["linestyle"],
            label=rule["label"],
        )
    _set(
        ax,
        r"$S_0(z)=\frac{m(z)}{M z^3}$, must satisfy"
        r" $\mathrm{d}S_0/\mathrm{d}z\leq 0$",
        "normalized radius $z$",
        r"d$S_0$/d$z$",
    )

    # The volume integrals of both rules:
    ax = axs[0, 1]
    _plot_rules(
        ax,
        z,
        lambda rule: [
            y / max(np.max(np.abs(t)), 1e-300)
            for y, t in zip(
                results[rule["label"]][key], trapezoid[key], strict=True
            )
        ],
        rules,
    )
    if primed:
        text = (
            r"$S^\prime_{2j}(z)=-\frac{\rho(z)}{\bar\rho} f^\prime_{2j}(z)"
            r"+\frac{1}{z^{2-2j}}\left(\frac{\rho(1)}{\bar\rho}"
            r"f^\prime_{2j}(1)-\int_z^1\frac{\mathrm{d}\rho}{\bar\rho}"
            r"z'^{2-2j}f^\prime_{2j}(z')\right)$"
        )
        ylabel = r"$S^\prime_{2j}$ / max |$S^\prime_{2j}$|"
    else:
        text = (
            r"$S_{2j}(z)=\frac{\rho(z)}{\bar\rho} f_{2j}(z)"
            r"-\frac{1}{z^{2j+3}}\int_0^z\frac{\mathrm{d}\rho}{\bar\rho}"
            r"z'^{2j+3}f_{2j}(z')$"
        )
        ylabel = r"$S_{2j}$ / max |$S_{2j}$|"
    _set(ax, text, None, ylabel, fontsize="small")

    # The new figure functions, the result of the step:
    _plot_rules(
        axs[1, 1],
        z,
        lambda rule: [_normalized(s) for s in ss],
        [_PLAIN],
    )
    _set(
        axs[1, 1],
        r"$r_z(\vartheta)=z\,(1+\Sigma)$, "
        r"$\Sigma=\sum_j s_{2j}(z)P_{2j}(\cos\vartheta)$",
        "normalized radius $z$",
        "$s_{2j}$ / max |$s_{2j}$|",
    )

    # The shape functions, the input of the step:
    offset = n_j if primed else 0
    _plot_rules(
        axs[0, 2],
        z,
        lambda rule: [_normalized(f) for f in fs[offset : offset + n_j]],
        [_PLAIN],
    )
    _set(
        axs[0, 2],
        (
            r"$f^\prime_{2j}(z)=\frac{3}{2(2-2j)}\int_{-1}^{1}\mathrm{d}\mu"
            r"\,P_{2j}(\mu)(1+\Sigma)^{2-2j}$"
            if primed
            else r"$f_{2j}(z)=\frac{3}{2(2j+3)}\int_{-1}^{1}\mathrm{d}\mu"
            r"\,P_{2j}(\mu)(1+\Sigma)^{2j+3}$"
        ),
        "normalized radius $z$",
        (
            r"$f^\prime_{2j}$ / max |$f^\prime_{2j}$|"
            if primed
            else r"$f_{2j}$ / max |$f_{2j}$|"
        ),
    )

    # Text and the legend of the colors:
    ax = axs[1, 2]
    lines = [
        f"step {iteration}",
        f"in use: {'Simpson' if use_simpson else 'trapezoid'}",
        f"max |dJ/J|: {_max_change(Js, old_Js):.1e}",
    ]
    lines += [f"J{2 * i} = {Js[i] * 1e6:.4f}e-6" for i in range(1, 4)]
    _info_panel(ax, lines)
    handles = [
        Line2D(
            [],
            [],
            color=_color(jj, rule["shade"]),
            linestyle=rule["linestyle"],
            label=f"j = {jj}, {rule['label']}",
        )
        for rule in rules
        for jj in j
    ]
    ax.legend(
        handles=handles,
        loc="lower left",
        ncol=len(rules),
        frameon=False,
        fontsize="small",
        title="light solid: Simpson, dark dotted: trapezoid",
        title_fontsize="small",
    )

    _emit(fig, _mode(mode), "debug_AlgoToF")


def _max_change(new, old):
    """Largest relative change of the moments (0 if nothing to compare)."""
    mask = new != 0
    if not mask.any():
        return 0.0
    return float(np.max(np.abs((old[mask] - new[mask]) / new[mask])))
