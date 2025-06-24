from pathlib import Path
from .. import config, logger
from ..data import experiment_name_to_paper_label

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import polars as pl
import seaborn as sns


def plot_r2_vs_sample_size_cartesian(score_path: Path, ax: Axes):
    """Plot r2 vs. training size from previously generated data."""
    df = pl.scan_csv(score_path, separator="|")

    sns.boxplot(data=df.collect().to_pandas(), x="training_samples", y="r2", ax=ax, showfliers=False)
    ax.set_ylim([0,1.0])
    ax.set_xticklabels(df.collect()["training_samples"].unique(), rotation=80)


def plot_r2_vs_sample_size(train_source: str, dev_source: str, score_path: Path, aggregation: str = "mean"):
    """Plot r2 vs. training size from previously generated data."""
    df = pl.scan_csv(score_path, separator="|")
    if aggregation == "overall":
        fig, ax = plt.subplots(1, 4, sharey=True, figsize=(16, 4))
        ax[0].set_ylim(0, 1.0)
        plot_region(df, ax[0])
        plot_year(df, ax[1])
        plot_quantity(df, ax[2])
        plot_overall(df, ax[3])
        plt.tight_layout()
        plt.close()
        fig.suptitle(f"Training Samples vs. $R^2$ ({experiment_name_to_paper_label(str(train_source))}_vs_{experiment_name_to_paper_label(str(dev_source))})", fontsize=16, y=1.05)
        return fig
    elif aggregation == "boxplot":
        fig = plt.figure(figsize=(5, 5))
        sns.boxplot(data=df.collect().to_pandas(), x="training_samples", y="r2", showfliers=False)
        plt.ylim([0,1.0])
        plt.xticks(rotation=80)
        plt.tight_layout()
        plt.close()
        fig.suptitle("Training Samples vs. $R^2$", fontsize=16, y=1.05)
        return fig

    fig, ax = plt.subplots(1, 3, sharey=True, sharex=True, figsize=(12, 4))
    for i, metric in enumerate(["region", "year", "target"]):
        grouped = df.group_by(["training_samples", "train", metric])
        if aggregation == "above_0_9":
            label = "Fraction Above 0.9"
            agg = grouped.agg((pl.col("r2") > 0.9).mean().alias(label))
        elif aggregation == "above_0_95":
            label = "Fraction Above 0.95"
            agg = grouped.agg((pl.col("r2") > 0.95).mean().alias(label))
        elif aggregation == "median":
            label = "Median r2"
            agg = grouped.agg(pl.col("r2").median().alias(label))
        else:
            label = "Mean r2"
            agg = grouped.agg(pl.col("r2").mean().alias(label))
        row = agg.collect()
        sns.lineplot(
            row,
            x="training_samples",
            y=label,
            hue=metric,
            ax=ax[i],
            palette="tab10",
        )
        ax[i].legend().remove()
        ax[i].set_xlim([0, 3500])
        ax[i].set_ylim([0, 1.1])
        ax[i].set_xlabel("Training Samples")
        ax[i].set_title(metric.capitalize() + "s")
    ax[0].set_ylabel(label)
    plt.tight_layout()
    plt.close()
    return fig


def errbar(data):
    return data.min(), data.max()


def plot_region(df, ax):
    data = (
        df.group_by(pl.col("training_samples"), pl.col("region"))
        .agg(pl.col("r2").median().alias("r2"))
        .collect()
    )
    sns.lineplot(
        data,
        x="training_samples",
        y="r2",
        alpha=0.8,
        ax=ax,
        hue="region",
        palette="tab10",
    )
    ax.set_ylabel("Median $R^2$")
    ax.set_xlabel("Training Samples")
    ax.set_title("Region")
    ax.get_legend().remove()


def plot_year(df, ax):
    data = (
        df.group_by(pl.col("training_samples"), pl.col("year"))
        .agg(pl.col("r2").median().alias("r2"))
        .collect()
    )
    sns.lineplot(data, x="training_samples", y="r2", ax=ax, hue="year", palette="GnBu")
    ax.set_ylabel("Median $R^2$")
    ax.set_xlabel("Training Samples")
    ax.set_title("Year")
    ax.get_legend().remove()
    norm = plt.Normalize(2020, 2100)
    sm = plt.cm.ScalarMappable(cmap="GnBu", norm=norm)
    sm.set_array([])
    # cbar = plt.colorbar(sm, ax=ax, pad=0.1)
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes

    cax = inset_axes(
        ax,
        width="5%",
        height="50%",
        loc="lower right",
        # bbox_to_anchor=(0.8, 0.1, 1, 1),
        bbox_to_anchor=(-0.2, 0.05, 1, 1),
        bbox_transform=ax.transAxes,
        borderpad=0,
    )
    cbar = plt.colorbar(sm, cax=cax)
    cbar.set_label("Year")


def plot_quantity(df, ax):
    from .. import config
    import numpy as np
    import pandas as pd
    color_dict = {
            "energy": "orange",
            "land": "green",
            "water": "blue",
        }
    
    collect = []
    for key, value in config.data.outputs.items():
        row = value.copy()
        collect.append(row)
    targets = pl.DataFrame(collect)
    data = (
        df.collect()
        .join(targets, right_on=["key"], left_on=["target"])
        .group_by(pl.col("training_samples"), pl.col("resource"), pl.col("target"))
        .agg(pl.col("r2").median().alias("r2"))
    ).sort(by='resource')
    # # calculate the highest and lowest quantities by area under the curve
    # collect_sums = []
    # for output in config.data.output_keys:
    #     sum = 0
    #     for value in data.filter(pl.col('target') == output)['r2']:
    #         sum = sum + value if value > 0 else sum
    #     collect_sums.append((output, sum))
    # sum_df = pl.DataFrame(collect_sums).sort('column_1')
    # low=sum_df.head(3)["column_0"]
    # high=sum_df.tail(3)["column_0"]
    # logger.info(f"highest: [{high[0]}, {high[1]}, {high[2]}]")
    # logger.info(f"lowest: [{low[0]}, {low[1]}, {low[2]}]")
    sns.lineplot(
        data,
        x="training_samples",
        y="r2",
        alpha=0.7,
        ax=ax,
        hue="resource",
        units="target",
        palette=color_dict,
        estimator=None,
    )
    ax.get_legend().set_title(None)
    ax.set_ylabel("Median $R^2$")
    ax.set_xlabel("Training Samples")
    ax.set_title("Quantity")


def plot_overall(df, ax):
    data = df.collect()
    sns.lineplot(
        data,
        x="training_samples",
        y="r2",
        ax=ax,
        palette="tab10",
        estimator="median",
        errorbar=("pi", 50),
    )
    ax.set_ylabel("Median $R^2$")
    ax.set_xlabel("Training Samples")
    ax.set_title("Overall")
