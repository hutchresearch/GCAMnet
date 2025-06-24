from pathlib import Path

import click
import matplotlib

from ..data import NormStrat, Source
from .. import config, logger
from ..cli.options import (
    checkpoint_path,
    dev_source,
    force,
    save_path,
    targets_path,
    train_source,
    validate_force,
    evaluate_on_test,
)

matplotlib.use("Agg")


@click.group()
def cli():
    """Plot results."""


@click.option(
    "-n",
    "--normalization-strategy",
    "strategy",
    help="how to handle normalization of target before training",
    type=click.Choice(["z_score", "min_max", "robust"]),
    default="z_score",
    required=True,
)
@click.option(
    "-c",
    "--checkpoint_path",
    help="/path/to/checkpoints",
    required=True,
)
@targets_path()
@force()
@save_path(path_help="path to figure.png output", suffix=".png")
@cli.command("plot:map")
def plot_geography(
    strategy: str,
    checkpoint_path: Path,
    targets_path: Path,
    force: bool,
    save_path: Path,
):
    """Plot r2 with a global map from region shapefiles."""
    from ..plot.geo import geography
    import matplotlib.pyplot as plt
    from ..data import experiment_name_to_paper_label
    from matplotlib.gridspec import GridSpec
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    if save_path.exists() and not force:
        raise click.UsageError(f"{save_path} exists. use --force to overwrite")
    fig = plt.figure(figsize=(16, 8))
    sources = ['binary', 'interp_hypercube', 'mixed']
    gs = GridSpec(len(sources), len(sources)+1, width_ratios=[1, 1, 1, 0.05], wspace=0.02)
    for i, train_source in enumerate(sources):
        for j, dev_source in enumerate(sources):
            ax = fig.add_subplot(gs[i,j])
            sm = geography(
                targets_path,
                train_source,
                dev_source,
                Path(checkpoint_path) / f"{str(train_source)}_vs_{str(dev_source)}",
                ax,
                strategy=NormStrat.from_str(strategy),
            )
            if i == 0:
                ax.set_title(experiment_name_to_paper_label(dev_source).capitalize(), fontsize=16)
            if j == 0:
                ax.text(-0.1, 0.5, experiment_name_to_paper_label(train_source).capitalize(), va='center', ha='right', transform=ax.transAxes, fontsize=16)
    cax = fig.add_subplot(gs[:,3])
    cbar = fig.colorbar(sm, cax=cax)
    cax.set_position([0.91, 0.15, 0.015, 0.65])
    cbar.set_label("Median $R^2$", fontsize=17)
    plt.tight_layout()
    fig.savefig(save_path, dpi=300)
    logger.info(f"saved: {save_path}")


@click.option(
    "-s",
    "--score_path",
    type=click.Path(dir_okay=False, exists=True, readable=True, path_type=Path),
    help="/path/to//sample-size/scores.csv",
    required=True,
    default=Path(config.paths.data) / "processed/sample_size/r2_scores.csv",
)
@click.option(
    "-a",
    "--aggregation",
    type=click.Choice(["median", "mean", "above_0_9", "above_0_95", "overall", "boxplot"]),
    default="mean",
)
@force()
@train_source()
@dev_source()
@save_path(path_help="path to save figure.png", suffix=".png")
@cli.command("plot:sample-size")
def plot_sample_size(
    train_source: str,
    dev_source: str,
    score_path: Path,
    aggregation: str,
    force: bool,
    save_path: Path,
):
    """Plot r2 vs. training size from previously generated data."""
    from ..plot.sample_size import plot_r2_vs_sample_size

    if save_path.exists() and not force:
        raise click.UsageError(f"{save_path} exists. use --force to overwrite")
    fig = plot_r2_vs_sample_size(train_source, dev_source, score_path, aggregation)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    logger.info(f"saved: {save_path}")


@click.option(
    "-s",
    "--score_path",
    type=click.Path(dir_okay=True, exists=True, readable=True, path_type=Path),
    help="/path/to/sample-size/scores.csv",
    required=True,
)
@force()
@save_path(path_help="path to save figure.png", suffix=".png")
@cli.command("plot:sample-size-cartesian")
def plot_sample_size_cartesian(
    score_path: Path,
    force: bool,
    save_path: Path,
):
    """Plot r2 vs. training size from previously generated data."""
    from ..plot.sample_size import plot_r2_vs_sample_size_cartesian
    from ..data import experiment_name_to_paper_label
    import matplotlib.pyplot as plt

    if save_path.exists() and not force:
        raise click.UsageError(f"{save_path} exists. use --force to overwrite")

    sources = ['binary', 'interp_hypercube', 'mixed']
    fig, ax = plt.subplots(len(sources), len(sources), sharex=True, sharey=True, figsize=(9,9))
    for i, train_source in enumerate(sources):
        for j, dev_source in enumerate(sources):
            plot_r2_vs_sample_size_cartesian(score_path / f"{train_source}_vs_{dev_source}.csv", ax[i][j])
            if i == 0:
                ax[i][j].set_title(experiment_name_to_paper_label(dev_source))
            if j == 0:
                ax[i][j].set_ylabel(experiment_name_to_paper_label(train_source))
    plt.tight_layout()
    fig.savefig(save_path, dpi=300)
    logger.info(f"saved: {save_path}")


@click.option(
    "-s",
    "--score_path",
    type=click.Path(dir_okay=False, exists=True, readable=True, path_type=Path),
    help="/path/to/fraction_binary_r2.csv",
    required=True,
    default=Path(config.paths.data) / "processed/fraction_binary_r2.csv",
)
@click.option(
    "-a",
    "--aggregation",
    type=click.Choice(["median", "mean", "above_0_9", "above_0_95", "overall"]),
    default="mean",
)
@force()
@save_path(path_help="path to save figure.png", suffix=".png")
@cli.command("plot:fraction-binary")
def plot_fraction_binary(
    score_path: Path,
    aggregation: str,
    force: bool,
    save_path: Path,
):
    """Plot r2 vs. training size from previously generated data."""
    from ..plot.fraction_binary import plot_fraction_binary

    validate_force(save_path, force)
    fig = plot_fraction_binary(score_path, aggregation)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    logger.info(f"saved: {save_path}")


@train_source()
@dev_source()
@checkpoint_path(exists=True)
@targets_path()
@force()
@save_path(path_help="/path/to/figure.png output", suffix=".png")
@evaluate_on_test()
@cli.command("plot:previous-adoption")
def previous_adoptions(
    train_source: str,
    dev_source: str,
    checkpoint_path: Path,
    targets_path: Path,
    force: bool,
    save_path: Path,
    evaluate_on_test: bool,
):
    """Plot emulator vs binary experiment."""
    from ..plot import renewable_adoption

    validate_force(save_path, force)
    fig = renewable_adoption(
        targets_path=targets_path,
        denormalize_source=train_source,
        inference_source=dev_source,
        checkpoint_path=checkpoint_path,
        evaluate_on_test=evaluate_on_test,
    )
    fig.savefig(save_path, dpi=300)
    logger.info(f"saved: {save_path}")



@click.option(
    "-n",
    "--normalization-strategy",
    "strategy",
    help="how to handle normalization of target before training",
    type=click.Choice(["z_score", "min_max", "robust"]),
    default="z_score",
    required=True,
)
@click.option(
    "-c",
    "--checkpoint_path",
    help="/path/to/checkpoints",
    required=True,
)
@targets_path()
@force()
@save_path(path_help="path to figure.png output", suffix=".png")
@train_source()
@dev_source()
@cli.command("plot:one-map")
def plot_geography(
    strategy: str,
    checkpoint_path: Path,
    targets_path: Path,
    train_source: str,
    dev_source: str,
    force: bool,
    save_path: Path,
):
    """Plot r2 with a global map from region shapefiles."""
    from ..plot.geo import single_map
    from ..data import experiment_name_to_paper_label

    if save_path.exists() and not force:
        raise click.UsageError(f"{save_path} exists. use --force to overwrite")

    fig = single_map(
        targets_path,
        train_source,
        dev_source,
        Path(checkpoint_path) / f"{str(train_source)}_vs_{str(dev_source)}",
        strategy=NormStrat.from_str(strategy),
    )
    fig.savefig(save_path, dpi=300)
    logger.info(f"saved: {save_path}")


@cli.command("plot:qualitative")
@click.option(
    "-s",
    "--score_path",
    type=click.Path(dir_okay=False, exists=True, readable=True, path_type=Path),
    help="/path/to/cartesian/scores.csv",
    required=True,
)
@train_source()
@force()
@save_path(path_help="path to figure.png output", suffix=".png")
def qualitative_analysis(
    score_path: Path,
    train_source: str,
    force: bool,
    save_path: Path,
):
    from ..plot.qualitative import qualitative

    if save_path.exists() and not force:
        raise click.UsageError(f"{save_path} exists. use --force to overwrite")

    fig = qualitative(
        score_path=score_path,
        train_source=train_source,
    )
    fig.savefig(save_path, dpi=300)
    logger.info(f"saved: {save_path}")
