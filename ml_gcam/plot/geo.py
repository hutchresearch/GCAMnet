from pathlib import Path

import polars as pl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.cm import ScalarMappable

from .. import config
from ..data import GcamDataset, load_targets
from ..data.enums import NormStrat
from ..data.normalization import Normalization
from ..inference import Inference


def polygons():
    """Get shapefiles for regions."""
    import geopandas
    import pandas as pd

    shapefile_path = Path(config.paths.data) / "gcam-shapefiles" / "reg32_spart.shp"
    geom = geopandas.read_file(shapefile_path)
    region_path = Path(config.paths.data) / "gcam-shapefiles" / "region_geometries.csv"
    regions = pd.read_csv(region_path, sep="|")
    geom = pd.merge(geom, regions)
    del regions
    return geom


def geography(
    targets_path: Path,
    train_source: str,
    dev_source: str,
    checkpoint_path: Path,
    ax,
    strategy: NormStrat,
):
    """Plot r2 with a global map from region shapefiles."""
    train_targets = load_targets(
        save_path=targets_path,
        experiments=[train_source],
        split="train",
    )
    train_set = GcamDataset(train_targets)
    normalization = Normalization(outputs=train_set.outputs, strategy=strategy)
    train_set.with_normalization(normalization)

    dev_targets = load_targets(
        save_path=targets_path,
        experiments=[dev_source],
        split="dev",
    )
    dev_set = GcamDataset(dev_targets)
    dev_set.with_normalization(normalization)
    inference = (
        Inference.from_checkpoint(checkpoint_path)
        .eval_with(dev_set)
        .denormalize_with(train_set.normalization)
    )
    geom = polygons()

    # fig, ax = plt.subplots(1, 1, figsize=(12, 4))
    scores = (
        inference.scores.melt(
            id_vars=["region", "year"],
            value_vars=sorted(config.data.output_keys),
            value_name="r2",
            variable_name="target",
        )
        .group_by("region")
        .median()[["region", "r2"]]
    )
    data = geom.merge(scores.to_pandas(), on=["region"])
    colors = ['#E8AF30', '#269453', '#194361']
    positions = [0.0, 0.97, 1.0]
    cmap = LinearSegmentedColormap.from_list("custom_gradient", list(zip(positions, colors)))
    fig = ax.get_figure()
    _map = data.plot(
        "r2",
        cmap=cmap,
        # legend=True,
        ax=ax,
        vmin=0.0,
        vmax=1.0,
        # legend_kwds={"shrink": 0.3, "label": "Median $R^2$", "format": "{:.3f}".format},
    )
    # cbar = fig.get_axes()[1]
    # cbar.set_yticklabels(cbar.get_yticklabels(), fontsize=8)
    ax.set_axis_off()
    # ax.set_title(f"{train_source} vs. {dev_source}")
    plt.tight_layout()
    # plt.close()
    # return fig
    sm = ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=1))
    return sm



def single_map(
    targets_path: Path,
    train_source: str,
    dev_source: str,
    checkpoint_path: Path,
    strategy: NormStrat,
):
    """Plot r2 with a global map from region shapefiles."""
    train_targets = load_targets(
        save_path=targets_path,
        experiments=[train_source],
        split="train",
    )
    train_set = GcamDataset(train_targets)
    normalization = Normalization(outputs=train_set.outputs, strategy=strategy)
    train_set.with_normalization(normalization)

    dev_targets = load_targets(
        save_path=targets_path,
        experiments=[dev_source],
        split="dev",
    )
    dev_set = GcamDataset(dev_targets)
    dev_set.with_normalization(normalization)
    inference = (
        Inference.from_checkpoint(checkpoint_path)
        .eval_with(dev_set)
        .denormalize_with(train_set.normalization)
    )
    geom = polygons()

    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    scores = (
        inference.scores.melt(
            id_vars=["region", "year"],
            value_vars=sorted(config.data.output_keys),
            value_name="r2",
            variable_name="target",
        )
        .filter(pl.col("r2") > 0)
        .group_by("region")
        .median()[["region", "r2"]]
        # .mean()[["region", "r2"]]
        # .quantile(0.01)[["region", "r2"]]
    )

    data = geom.merge(scores.to_pandas(), on=["region"])
    colors = ['#E8AF30', '#269453', '#194361']
    positions = [0.0, 0.5, 1.0]
    cmap = LinearSegmentedColormap.from_list("custom_gradient", list(zip(positions, colors)))
    data.plot(
        "r2",
        cmap=cmap,
        legend=True,
        ax=ax,
        vmin=scores.min()["r2"],
        vmax=scores.max()["r2"],
        # vmin=0.985,
        # vmax=0.999,
        legend_kwds={"shrink": 0.6, "label": "Median $R^2$", "format": "{:.3f}".format},
    )
    cbar = fig.get_axes()[1]
    cbar.set_yticklabels(cbar.get_yticklabels(), fontsize=8)
    ax.set_axis_off()
    plt.tight_layout()
    plt.close()
    return fig
