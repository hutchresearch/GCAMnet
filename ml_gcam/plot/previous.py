from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns

from ..data import GcamDataset, Normalization, Split
from ..data import experiment_name_to_paper_label
from ..inference import Inference


def renewable_adoption(
    targets_path: Path,
    inference_source: str,
    denormalize_source: str,
    checkpoint_path: Path,
    evaluate_on_test: bool,
):
    train_set = GcamDataset.from_targets(
        save_path=targets_path,
        experiment=denormalize_source,
        split=Split.TRAIN,
    )
    normalization = Normalization(outputs=train_set.outputs)
    train_set.with_normalization(normalization)
    dev_set = GcamDataset.from_targets(
        save_path=targets_path,
        experiment=inference_source,
        split=Split.DEV if not evaluate_on_test else Split.TEST,
    )
    dev_set.with_normalization(normalization)
    inference = (
        Inference.from_checkpoint(checkpoint_path)
        .eval_with(dev_set)
        .denormalize_with(train_set.normalization)
    )
    top = top_regions(dev_set.targets)
    core = inference.y_true_df.filter(pl.col("region").is_in(top)).filter(
        pl.col("year").cast(pl.UInt16) <= 2050,
    )
    binary = inference.y_pred_df.filter(pl.col("region").is_in(top)).filter(
        pl.col("year").cast(pl.UInt16) <= 2050,
    )

    fig, ax = plt.subplots(2, 2, sharey=True, sharex=True, figsize=(10, 8))
    plt.subplots_adjust(bottom=0.2)
    fig.suptitle(
        f"Renewable Adoption - GCAM vs. Emulator",
        fontsize=16,
    )
    plot_core(core, ax[0, :])
    plot_binary(binary, ax[1, :], denormalize_source)
    handles, labels = ax[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4)
    for axes in ax.flat:
        axes.get_legend().remove()
    plt.tight_layout(rect=[0.05, 0.08, 0.95, 1])
    plt.close()
    return fig


def top_regions(targets):
    column = "energy_supply_electricity_solar"
    return (
        targets.group_by("region")
        .agg(pl.col(column).sum().alias("solar"))
        .sort("solar", descending=True)[:8, "region"]
        .to_list()
    )


def errbar(data):
    return data.min(), data.max()


def plot_solar(data, ax, title):
    column = "energy_supply_electricity_solar"
    data = data.select("region", "year", pl.col(column).alias("solar"))
    ax.set_ylabel("Electricity Generation (EJ)")
    sns.lineplot(
        data,
        x="year",
        y="solar",
        ax=ax,
        hue="region",
        palette="tab10",
        errorbar="sd",
    )
    ax.set_title(title)


def plot_wind(df, ax, title):
    column = "energy_supply_electricity_wind"
    data = df.select("region", "year", pl.col(column).alias("solar"))
    sns.lineplot(
        data,
        x="year",
        y="solar",
        ax=ax,
        hue="region",
        palette="tab10",
        errorbar="sd",
    )
    ax.set_title(title)


def plot_core(data, ax):
    plot_wind(data, ax[1], "Wind (GCAM)")
    plot_solar(data, ax[0], "Solar (GCAM)")


def plot_binary(data, ax, emulator):
    name = experiment_name_to_paper_label(str(emulator))
    plot_wind(data, ax[1], f"Wind ({name.capitalize()} Emulator)")
    plot_solar(data, ax[0], f"Solar ({name.capitalize()} Emulator)")
    ax[0].set_xlabel("Years")
    ax[1].set_xlabel("Years")
