from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import polars as pl
import seaborn as sns

from .. import config
from ..data import Source, Split, load_targets


def interp_vs_binary(targets_path: Path, column: str, shared: bool = False):
    targets = load_targets(
        targets_path,
        experiments=[Source.BINARY, Source.HYPERCUBE],
        split=Split.TRAIN,
    )
    if not shared:
        fig, ax = plt.subplots(2, 1, sharex=True, sharey=True)
        plot_binary(targets, ax[0], column)
        plot_interp(targets, ax[1], column)
    else:
        fig, ax = plt.subplots()
        plot_both(targets, ax, column)
    fig.suptitle("Interpolated vs. Binary Histogram")
    plt.close()
    return fig


def column_units(column) -> str:
    for o in config.data.outputs:
        output = config.data.outputs[o]
        if column == o:
            return output.units


def custom_formatter(x, pos) -> str:
    return f"{x*1e-3:.0f}K"


def plot_both(targets, ax, column) -> None:
    selected = targets.select(pl.col("experiment"), pl.col(column))
    title = " ".join(column.split("_")).title()
    units = column_units(column)
    ax.set_xlabel(f"{title} ({units})")
    # ax.yaxis.set_major_formatter(ticker.FuncFormatter(custom_formatter))
    ax.set_ylabel("Density")
    sns.histplot(
        selected,
        x=column,
        hue="experiment",
        kde=True,
        ax=ax,
        stat="density",
        common_norm=False,
    )


def plot_binary(targets, ax, column):
    data = targets.filter(pl.col("experiment") == "binary")
    filtered = filter_outliers(data, column)
    selected = filtered.select(pl.col(column))
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(custom_formatter))
    ax.set_ylabel("Binary (#)")
    sns.histplot(selected, x=column, kde=True, ax=ax)


def plot_interp(targets, ax, column):
    data = targets.filter(pl.col("experiment") == "interp_hypercube")
    filtered = filter_outliers(data, column)
    selected = filtered.select(pl.col(column))
    title = " ".join(column.split("_")).title()
    units = column_units(column)
    ax.set_xlabel(f"{title} ({units})")
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(custom_formatter))
    ax.set_ylabel("Interpolated (#)")
    sns.histplot(selected, x=column, kde=True, ax=ax)


def filter_outliers(targets, column_name):
    # calculate median and mad (median absolute deviation)
    median = targets[column_name].median()
    mad = (targets[column_name] - median).abs().median()

    # define a threshold for outliers, typically 2.5 or 3 times the mad
    threshold = 4 * mad

    # calculate deviations from the median
    deviations = (targets[column_name] - median).abs()

    # filter outliers based on the mad method
    return targets.filter(deviations <= threshold)
