import os
from pathlib import Path
from typing import List

import click

from .. import config
from ..cli.options import checkpoint_path, dev_source, targets_path, train_source
from ..data import NormStrat, Source
from ..emulator.model import Arch


@click.group()
def cli():
    """Train the dang emulator!!!"""
    pass


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
    "-s",
    "--samples",
    help="number of samples to train on",
    type=int,
    required=False,
)
@click.option(
    "-a",
    "--arch",
    help="model architecture to use for training",
    type=click.Choice(["deep", "linear"]),
    default="deep",
    required=False,
)
@click.option(
    "--fraction",
    "fraction",
    type=float,
    help="fraction of binary samples to use for training [0.0, 1.0]",
    default=float(config.training.binary_fraction),
    required=False,
)
@train_source()
@dev_source()
@checkpoint_path()
@targets_path()
@cli.command("training:run")
def train_run(
    strategy: str,
    train_source: str,
    dev_source: str,
    fraction: float,
    checkpoint_path: Path,
    targets_path: Path,
    samples: int = None,
    arch: str = "deep",
) -> None:
    """Main training loop."""
    import os

    from ..training.train import train_main

    os.environ["ML_GCAM__TRAINING__TRAIN_SOURCE"] = str(train_source)
    os.environ["ML_GCAM__TRAINING__DEV_SOURCE"] = str(dev_source)
    os.environ["ML_GCAM__MODEL__ARCH"] = str(arch)
    if fraction is not None:
        os.environ["ML_GCAM__TRAINING__BINARY_FRACTION"] = str(fraction)
    config.reload()

    if checkpoint_path.parent.exists() and not checkpoint_path.exists():
        checkpoint_path.mkdir()
    train_main(
        train_source=train_source,
        dev_source=dev_source,
        targets_path=targets_path,
        checkpoint_path=checkpoint_path,
        samples=samples,
        arch=Arch.from_str(arch),
        strategy=NormStrat.from_str(strategy),
    )


@click.option(
    "-s",
    "--splits",
    multiple=True,
    type=float,
    help="samples to use for training, in percent of train_set",
    default=config.sample_size.splits,
    show_default=True,
    required=True,
)
@train_source()
@dev_source()
@checkpoint_path(exists=True, multiple=True)
@targets_path()
@cli.command("training:sample-size")
def train_sample_size(
    splits: List[float],
    train_source: str,
    dev_source: str,
    checkpoint_path: Path,
    targets_path: Path,
) -> None:
    """Sql db based training loop for r2."""
    from ..training.train import train_sample_size

    train_sample_size(
        train_source=train_source,
        dev_source=dev_source,
        targets_path=targets_path,
        splits=splits,
        checkpoint_path=checkpoint_path,
    )


@click.option(
    "--fraction",
    "fractions",
    multiple=True,
    type=float,
    help="fraction of binary samples to use for training [0.0, 1.0]",
    default=config.mixed_fraction.fractions,
    show_default=True,
    required=True,
)
@dev_source()
@checkpoint_path(exists=True, multiple=True)
@targets_path()
@cli.command("training:fraction-binary")
def train_fraction_binary(
    fractions: List[float],
    dev_source: str,
    checkpoint_path: Path,
    targets_path: Path,
) -> None:
    """Sql db based training loop for r2."""
    from ..training.train import train_mixed_fraction

    os.environ["ML_GCAM__TRAINING__TRAIN_SOURCE"] = str(Source.MIXED)
    os.environ["ML_GCAM__TRAINING__DEV_SOURCE"] = str(dev_source)
    config.reload()
    train_mixed_fraction(
        dev_source=dev_source,
        targets_path=targets_path,
        fractions=fractions,
        checkpoint_path=checkpoint_path,
    )


@checkpoint_path(exists=True, multiple=True)
@train_source(multiple=True)
@dev_source(multiple=True)
@targets_path()
@cli.command("training:cartesian")
def train_cartesian(
    checkpoint_path: Path,
    train_source: List[str],
    dev_source: List[str],
    targets_path: Path,
):
    """Configure a wandb hyperparameter sweep and make a sweep id."""
    from ..training.train import train_cartesian

    train_cartesian(
        train_sources=train_source,
        dev_sources=dev_source,
        targets_path=targets_path,
        checkpoint_path=checkpoint_path,
    )


@cli.command("training:sweep-init")
def sweep_init():
    """Run the sweep after init step and token/id generation."""
    from rich.console import Console

    console = Console()
    from ..training.sweep import init

    sweep_id = init()
    console.print(f"sweep_id = {sweep_id}")


@cli.command("training:sweep-run")
@train_source()
@dev_source()
@click.option(
    "-a",
    "--arch",
    help="model architecture to use for training",
    type=click.Choice(["deep", "linear"]),
    default="deep",
    required=False,
)
@click.option("-s", "--sweep_id", type=str, required=True)
@click.option("-r", "--runs", type=int, default=10, required=True)
def sweep_run(train_source, dev_source, arch: str, sweep_id: str, runs: int):
    """Run the sweep after init step and token/id generation."""
    from ..training.sweep import sweep

    os.environ["ML_GCAM__TRAINING__TRAIN_SOURCE"] = str(train_source)
    os.environ["ML_GCAM__TRAINING__DEV_SOURCE"] = str(dev_source)
    os.environ["ML_GCAM__MODEL__ARCH"] = str(arch)
    config.reload()
    sweep(sweep_id, runs)
