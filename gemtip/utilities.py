#
# Author: Charles L. Bérubé
# Created on: Fri Jun 02 2023
#
# Copyright (c) 2023 C.L. Bérubé & J.-L. Gagnon
#

import math

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl


def truncate(n, decimals=0):
    multiplier = 10**decimals
    return int(n * multiplier) / multiplier


def to_latex_scientific_notation(mean, std, maxint=2):
    exponent_mean = int(np.floor(np.log10(abs(mean))))
    exponent_std = int(np.floor(np.log10(abs(std))))
    precision = abs(exponent_mean - exponent_std)
    coefficient_mean = round(mean / 10**exponent_mean, precision)
    coefficient_std = round(std / 10**exponent_mean, precision)
    if -maxint <= exponent_mean <= 0:
        return f"${truncate(mean, -exponent_std)} \\pm {truncate(std, -exponent_std)}$"
    elif 0 <= exponent_mean <= maxint and exponent_std >= 0:
        return f"${round(truncate(mean, -exponent_std))} \\pm {round(truncate(std, -exponent_std))}$"
    elif 0 <= exponent_mean <= maxint:
        return f"${truncate(mean, -exponent_std)} \\pm {truncate(std, -exponent_std)}$"
    else:
        if precision == 0:
            return (
                f"$({round(coefficient_mean)} \\pm {round(coefficient_std)}) \\cdot 10^{{{exponent_mean}}}$"
                if exponent_mean != 0
                else f"{round(coefficient_mean)} \\pm {round(coefficient_std)}$"
            )
        else:
            return (
                f"$({coefficient_mean} \\pm {coefficient_std}) \\cdot 10^{{{exponent_mean}}}$"
                if exponent_mean != 0
                else f"{coefficient_mean} \\pm {coefficient_std}$"
            )


def str_with_err(value, error):
    if error > 0:
        digits = -int(math.floor(math.log10(error)))
    else:
        digits = 0
    if digits < 0:
        digits = 0
    err10digits = math.floor(error * 10**digits)
    return "${0:.{2}f} \pm {1:.{2}f}$".format(value, error, digits)


def p_metric(y_pred, y_true, eps=1e-12):
    # Number of significant digits
    return np.log10(np.abs((y_pred - y_true) / y_true) + eps)


def r_metric(y_pred, y_true):
    # Should be centered around 0 and narrow
    return np.log10(np.abs(y_pred / y_true))


def mape(y_pred, y_true):
    return 100 * np.abs((y_pred - y_true) / y_true)


def normalize(x, xmin, xmax, ymin, ymax):
    # x mapped from xmin, xmax to ymin, ymax
    return (ymax - ymin) * (x - xmin) / (xmax - xmin) + ymin


def denormalize(x, xmin, xmax, ymin, ymax):
    # x mapped from ymin, ymax back to xmin, xmax
    return (x - ymin) * (xmax - xmin) / (ymax - ymin) + xmin


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
