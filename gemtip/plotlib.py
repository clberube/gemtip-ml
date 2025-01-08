#
# Author: Charles L. Bérubé
# Created on: Fri Jun 02 2023
#
# Copyright (c) 2023 C.L. Bérubé & J.-L. Gagnon
#

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl


def restore_minor_ticks_log_plot(ax, n_subticks=9, axis="both"):
    """For axes with a logrithmic scale where the span (max-min) exceeds
    10 orders of magnitude, matplotlib will not set logarithmic minor ticks.
    If you don't like this, call this function to restore minor ticks.

    Args:
        ax:
        n_subticks: Number of Should be either 4 or 9.

    Returns:
        None
    """
    if ax is None:
        ax = plt.gca()
    # Method from SO user importanceofbeingernest at
    # https://stackoverflow.com/a/44079725/5972175
    locmaj = mpl.ticker.LogLocator(base=10, numticks=1000)
    locmin = mpl.ticker.LogLocator(
        base=10.0, subs=np.linspace(0, 1.0, n_subticks + 2)[1:-1], numticks=1000
    )

    if axis == "x" or axis == "both":
        ax.xaxis.set_major_locator(locmaj)
        ax.xaxis.set_minor_locator(locmin)
        ax.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())
    if axis == "y" or axis == "both":
        ax.yaxis.set_major_locator(locmaj)
        ax.yaxis.set_minor_locator(locmin)
        ax.yaxis.set_minor_formatter(mpl.ticker.NullFormatter())


def plot_conductivities(f, Z, **kwargs):
    Zxx = Z[:, 0, 0]
    Zyy = Z[:, 1, 1]
    Zzz = Z[:, 2, 2]

    fig, axs = plt.subplots(2, 1, sharex=True)
    ax = axs[0]
    ax.plot(
        f,
        1000 * torch.real(Zxx),
        linestyle="-",
        label=r"$\sigma_{x}$",
        **kwargs,
    )
    ax.plot(
        f,
        1000 * torch.real(Zyy),
        linestyle="--",
        label=r"$\sigma_{y}$",
        **kwargs,
    )
    ax.plot(
        f,
        1000 * torch.real(Zzz),
        linestyle=":",
        label=r"$\sigma_{z}$",
        **kwargs,
    )
    ax.set_ylabel(r"$\sigma'$ (mS/m)")
    ax.legend(loc=0)

    ax = axs[1]
    ax.plot(
        f,
        1000 * torch.imag(Zxx),
        linestyle="-",
        **kwargs,
    )
    ax.plot(
        f,
        1000 * torch.imag(Zyy),
        linestyle="--",
        **kwargs,
    )
    ax.plot(
        f,
        1000 * torch.imag(Zzz),
        linestyle=":",
        **kwargs,
    )
    ax.set_ylabel(r"$\sigma''$ (mS/m)")

    ax.set_xscale("log")
    ax.set_xlabel(r"$f$ (Hz)")
    restore_minor_ticks_log_plot(ax, axis="x")

    plt.tight_layout()
    return fig, axs
