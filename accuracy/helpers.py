"""
Shared code of the accuracy notebooks: running PyToF (and, for comparison, the
ToF code of Movshovitz & Fortney 2022) on a grid of settings, plotting the
convergence, and writing the error tables that ``ToF.get_Js_errors()`` uses.

Usable as a script to regenerate the error tables (takes a while):

    python accuracy/helpers.py --write-accuracy-data
"""

import argparse
import multiprocessing
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import references  # noqa: E402

ORDERS = (4, 7, 10)
J_NAMES = ["R_eq_to_R_m"] + [f"J_{2 * i}" for i in range(1, 11)] + ["time"]
J_TITLES = (
    ["mean radius / equatorial radius"]
    + [rf"$J_{{{2 * i}}}$" for i in range(1, 11)]
    + ["runtime"]
)
JUPITER_RADIUS = references.JUPITER_RADIUS


# --------------------------------------------------------------------------
# Running the codes
# --------------------------------------------------------------------------
def run_pytof(N, n_bin, order, alphas=None, use_simpson=True):
    """
    One PyToF calculation of the Wisdom & Hubbard (2016) Jupiter polytrope.

    Returns (runtime per iteration [s], [R_eq/R_m, J2, ..., J_2*order],
    number of iterations). All nan if n_bin < 4 (too few bins).
    """
    from PyToF import ToF

    if n_bin < 4:
        return np.nan, np.full(order + 1, np.nan), np.nan

    if alphas is None:
        alphas = np.zeros(12)

    X = ToF(
        N=N,
        n_bin=n_bin,
        order=order,
        alphas=alphas,
        verbosity=0,
        R_ref=JUPITER_RADIUS,
        use_simpson=use_simpson,
    )
    X.set_barotrope(lambda P, param: np.sqrt(P))

    tic = time.time()
    it = X.relax_to_barotrope()
    toc = time.time()

    return (toc - tic) / it, np.hstack([X.R_eq_to_R_m, X.Js[1:]]), it


def run_movshovitz(N, order):
    """
    The same calculation with the code of Movshovitz & Fortney (2022) (the
    folder nmovshov_ToFPlanet of this repository), for comparison.
    """
    sys.path.insert(0, str(ROOT / "nmovshov_ToFPlanet"))
    import TOFPlanet

    res = TOFPlanet.TOFPlanet(
        N=N,
        nx=-1,
        toforder=order,
        dJtol=1e-10,
        drottol=1e-10,
        drhotol=1e-10,
        MaxIterHE=2,
        MaxIterBar=100,
        verbosity=0,
    )
    res.G = 6.6738480e-11
    res.GM = 126686536.1 * 1000**3
    res.mass = res.GM / res.G
    res.radius = JUPITER_RADIUS
    res.period = 2 * np.pi / np.sqrt(res.GM * 0.089195487 / res.radius**3)
    res.P0 = 0
    res.si = res.radius * np.linspace(1, 1 / N, N)
    res.rhoi = np.ones(N) * res.mass / (4 * np.pi / 3 * res.radius**3)
    res.set_barotrope(lambda P: np.sqrt(P))

    tic = time.time()
    it = res.relax_to_barotrope()
    toc = time.time()

    return (toc - tic) / it, np.hstack([res.aos, res.Js[1:]])


def _pytof_task(args):
    return run_pytof(*args)


def _movshovitz_task(args):
    return run_movshovitz(*args)


def _map(func, tasks, processes, progress, desc):
    ctx = multiprocessing.get_context("fork")
    with ctx.Pool(processes) as pool:
        it = pool.imap(func, tasks)
        if progress:
            from tqdm import tqdm

            it = tqdm(it, total=len(tasks), desc=desc)
        return list(it)


def sweep_pytof(
    Ns,
    steps=(1,),
    alphas=None,
    use_simpson=True,
    orders=ORDERS,
    processes=None,
    progress=True,
):
    """
    Run PyToF for every N in Ns, every binning N / n_bin in steps and every
    order. Returns a dict order -> dict(res=array (len(Ns), order + 1,
    len(steps)), time=array (len(Ns), len(steps))).
    """
    out = {}
    for order in orders:
        tasks = [
            (int(N), int(N / s), order, alphas, use_simpson)
            for N in Ns
            for s in steps
        ]
        raw = _map(
            _pytof_task, tasks, processes, progress, f"PyToF, order {order}"
        )
        res = np.array([r[1] for r in raw]).reshape(
            len(Ns), len(steps), order + 1
        )
        time_ = np.array([r[0] for r in raw]).reshape(len(Ns), len(steps))
        out[order] = dict(res=res.transpose(0, 2, 1), time=time_)
    return out


def sweep_movshovitz(Ns, orders=(4, 7), processes=None, progress=True):
    """Same as sweep_pytof for the comparison code (orders 4 and 7 only)."""
    out = {}
    for order in orders:
        tasks = [(int(N), order) for N in Ns]
        raw = _map(
            _movshovitz_task,
            tasks,
            processes,
            progress,
            f"Movshovitz, order {order}",
        )
        out[order] = dict(
            res=np.array([r[1] for r in raw]),
            time=np.array([r[0] for r in raw]),
        )
    return out


# --------------------------------------------------------------------------
# Plotting
# --------------------------------------------------------------------------
def plot_convergence(
    Ns,
    reference,
    ref_name,
    pytof,
    steps=(1,),
    movshovitz=None,
    which=None,
    outdir=None,
    prefix="",
):
    """
    One figure per quantity: relative difference of R_eq/R_m (no relative
    difference, plain ratio) and J_2 ... J_20 to the reference solution,
    and the runtime, as a function of the number of grid points N.

    pytof, movshovitz:  output of sweep_pytof / sweep_movshovitz.
    steps:              the binnings N / n_bin of pytof; if there is more
                        than one, the binning is shown by lightness.
    which:              indices into J_NAMES to plot (0: R_eq/R_m, 1..10:
                        J_2..J_20, 11: runtime).
    outdir, prefix:     if given, every figure is saved as
                        outdir/prefix + name + .png.

    Returns the list of figures.
    """
    import matplotlib
    import matplotlib.pyplot as plt

    which = range(12) if which is None else which
    binned = len(steps) > 1
    cmaps = [plt.cm.Blues_r, plt.cm.Oranges_r, plt.cm.Greens_r]
    norm = matplotlib.colors.Normalize(vmin=0, vmax=max(len(steps) - 1, 1))
    figs = []

    for i in which:
        fig, ax = plt.subplots(layout="constrained")
        for c, order in enumerate(pytof):
            n_values = order + 1  # R_eq/R_m and J_2 ... J_2*order
            if i >= n_values and i != 11:
                continue
            for j in range(len(steps)):
                color = cmaps[c](norm(j)) if binned else f"C{c}"
                if i == 11:
                    y = pytof[order]["time"][:, j]
                else:
                    res = pytof[order]["res"][:, i, j]
                    y = (
                        res
                        if i == 0 and np.isnan(reference[0])
                        else np.abs((res - reference[i]) / res)
                    )
                ax.plot(Ns, y, "o-", color=color)
            if movshovitz is not None and order in movshovitz:
                m = movshovitz[order]
                y = m["time"] if i == 11 else (
                    np.abs((m["res"][:, i] - reference[i]) / m["res"][:, i])
                    if i < m["res"].shape[1] else None
                )  # fmt: skip
                if y is not None:
                    ax.plot(Ns, y, "o--", color=f"C{c}")

        ax.set_title(J_TITLES[i], fontsize=20)
        ax.set_xlabel(r"Number of gridpoints $N$")
        ax.set_xscale("log", base=2)
        if i == 11:
            ax.set_yscale("log")
            ax.set_ylabel("runtime per iteration [s]")
        elif i == 0 and np.isnan(reference[0]):
            ax.set_ylabel(r"$R_\mathrm{eq}/R_\mathrm{m}$ (no reference)")
        elif i == 0:
            ax.set_ylabel(f"rel. diff. to {ref_name}")
            ax.set_yscale("log")
        else:
            ax.set_yscale("log")
            ax.set_ylabel(f"rel. diff. to {ref_name}")

        handles = [
            matplotlib.patches.Patch(color=f"C{c}", label=str(order))
            for c, order in enumerate(pytof)
            if i < order + 1 or i == 11
        ]
        ax.add_artist(
            ax.legend(
                handles=handles,
                title="ToF order",
                loc="lower left" if i not in (0, 11) else "lower right",
                handletextpad=0.3,
                labelspacing=0.3,
                handleheight=1.0,
                handlelength=1.0,
                frameon=False,
            )
        )
        if movshovitz is not None:
            ax.legend(
                handles=[
                    matplotlib.lines.Line2D(
                        [], [], color="k", linestyle=ls, label=lab
                    )
                    for ls, lab in (("-", "PyToF"), ("--", "M&F 22"))
                ],
                ncol=2,
                loc="upper right" if i != 11 else "upper left",
                handletextpad=0.3,
                columnspacing=1.0,
                frameon=False,
            )
        if binned:
            ticks = np.arange(len(steps))
            sm = matplotlib.cm.ScalarMappable(
                norm=matplotlib.colors.BoundaryNorm(
                    np.arange(len(steps) + 1), ncolors=plt.cm.Greys_r.N
                ),
                cmap=plt.cm.Greys_r,
            )
            cbar = fig.colorbar(sm, ax=ax, pad=0.02)
            cbar.set_label(r"$N / n_\mathrm{bin}$")
            cbar.ax.set_yticks(ticks[::2] + 0.5)
            cbar.ax.set_yticklabels([rf"$2^{{{t}}}$" for t in ticks[::2]])

        if outdir is not None:
            Path(outdir).mkdir(parents=True, exist_ok=True)
            fig.savefig(Path(outdir) / f"{prefix}{J_NAMES[i]}.png")
        figs.append(fig)
    return figs


# --------------------------------------------------------------------------
# Error tables used by ToF.get_Js_errors()
# --------------------------------------------------------------------------
def error_tables(Ns, steps, sweeps_pytof, reference=references.Bessel):
    """
    The relative errors of J_2 ... J_2*order with respect to the reference as
    stored in PyToF/accuracy_data_*.npz.
    """
    Ns_n_bins = np.array(
        [(int(N), int(N / s)) for N in Ns for s in steps]
    ).astype(int)
    data = {"Ns_n_bins": Ns_n_bins}
    for order, key in zip(ORDERS, ("04", "07", "10"), strict=True):
        res = sweeps_pytof[order]["res"][:, 1:, :]
        ref = reference[1 : order + 1][None, :, None]
        data[f"rel_error_{key}"] = np.abs((res - ref) / res)
    return data


def write_accuracy_data(directory=None, processes=None, progress=True):
    """
    Recompute the error tables (Simpson and trapezoid rule, N = 2^3 ... 2^15,
    all binnings) and save them as accuracy_data_{simpson,trapezoid}.npz in
    ``directory`` (default: the PyToF package folder, which OVERWRITES the
    tables PyToF uses).
    """
    directory = Path(directory) if directory else ROOT / "PyToF"
    Ns = 2 ** np.linspace(3, 15, 13)
    steps = 2 ** np.arange(int(np.log2(Ns[-2])))
    for name, simpson in (("simpson", True), ("trapezoid", False)):
        sw = sweep_pytof(
            Ns,
            steps,
            use_simpson=simpson,
            processes=processes,
            progress=progress,
        )
        np.savez(
            directory / f"accuracy_data_{name}.npz",
            **error_tables(Ns, steps, sw),
        )
        print(f"saved {directory / f'accuracy_data_{name}.npz'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-accuracy-data", action="store_true")
    parser.add_argument(
        "--directory", help="where to write (default: PyToF package folder)"
    )
    parser.add_argument("--processes", type=int, default=None)
    args = parser.parse_args()
    if args.write_accuracy_data:
        write_accuracy_data(args.directory, args.processes)
    else:
        parser.print_help()
