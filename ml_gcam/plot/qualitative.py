import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns
from ..data import region_to_continent, quantity_to_sector

from pathlib import Path

import matplotlib.gridspec as gridspec

def qualitative(
    score_path: Path,
    train_source: str,
):
    scores = (
        pl.scan_csv(
            str(score_path), separator="|"
        ).filter(
            (pl.col("train_source") == str(train_source)) &
            (pl.col("dev_source") == str(train_source))
        ).drop(
            ["train_source", "dev_source"]
        ).collect()
    )

    real = (scores
        .filter(pl.col("r2") > 0.0)
        .sort("year").sort("target").sort("region")
    )
    sectors = [quantity_to_sector(quantity) for quantity in real["target"]]
    continents = [region_to_continent(region) for region in real["region"]]
    real = real.with_columns(pl.Series("sector", sectors))
    real = real.with_columns(pl.Series("continent", continents))

    fig = plt.figure(figsize=(20,15))
    gs = gridspec.GridSpec(2, 2, width_ratios=[1.5, 1], height_ratios=[3,2])

    ax1 = fig.add_subplot(gs[:, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 1])
    
    def plot_agg(agg, axes):
        order = real.group_by(agg).median()
        hue = None
        color_palette = None

        if agg == "target":
            sectors = [quantity_to_sector(quantity) for quantity in order["target"]]
            order = order.with_columns(pl.Series("sector", sectors))
            order = order.sort("r2").sort("sector")
            hue = "sector"
            color_palette = {
                "energy": "orange",
                "land": "green",
                "water": "blue",
            }
        elif agg == "region":
            continents = [region_to_continent(region) for region in order["region"]]
            order = order.with_columns(pl.Series("continent", continents))
            order = order.sort("r2").sort("continent")
            hue = "continent"
        elif agg == "year":
            order = order.sort("year")

        order = order[agg].to_list()

        sns.set(font_scale=1.5)

        sns.boxplot(
            x='r2',
            y=agg,
            data=real,
            showfliers=True,
            hue=hue,
            palette=color_palette,
            ax=axes,
            order=order,
            orient="h",
        )

        agg = "quantity" if agg == "target" else agg
        axes.yaxis.tick_right()
        axes.set_xlabel('R² Score', fontsize=12)
        axes.set_ylabel(agg.capitalize(), fontsize=14)
        axes.set_xlim(0.95, 1.0)
        axes.legend(fontsize=10)

    plot_agg("target", ax1)
    plot_agg("region", ax2)
    plot_agg("year", ax3)

    fig.tight_layout()
    return fig
