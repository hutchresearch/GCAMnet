import time
from pathlib import Path
from typing import Optional

import click

from .. import logger
from ..cli.options import (
    checkpoint_path,
    evaluate_on_test,
    force,
    save_path,
    targets_path,
    train_source,
    dev_source,
    validate_force,
)


@click.group()
def cli():
    """Generate r2 values given weights from pretrained emulator."""
    ...


@cli.command("evaluate:cartesian")
@click.option(
    "-c",
    "--checkpoint_path",
    type=click.Path(file_okay=False, readable=True, exists=True, path_type=Path),
    help="/path/to/model/checkpoints/",
    required=True,
)
@evaluate_on_test()
@force()
@save_path(suffix=".csv", path_help="path to save score.csv")
@targets_path()
def evaluate_cartesian(
    checkpoint_path: Path,
    evaluate_on_test,
    force: bool,
    save_path: Path,
    targets_path: Path,
):
    """Load and create r2 scores for cartesian sweep."""
    from ..evaluate import evaluate_cartesian

    validate_force(save_path, force)
    scores = evaluate_cartesian(
        targets_path,
        checkpoint_path,
        evaluate_on_test=evaluate_on_test,
    )
    scores.write_csv(save_path, separator="|", quote_style="non_numeric")
    logger.info(f"saved: {save_path}")


@cli.command("evaluate:sample-size")
@click.option(
    "-c",
    "--checkpoint_path",
    type=click.Path(file_okay=False, readable=True, exists=True, path_type=Path),
    help="/path/to/model/checkpoints/for-sweep/",
    required=True,
)
@train_source()
@dev_source()
@force()
@save_path(suffix=".csv", path_help="path to save score.csv")
@targets_path()
@evaluate_on_test()
def evaluate_sample_size(
    checkpoint_path: Path,
    train_source: str,
    dev_source: str,
    force: bool,
    save_path: Path,
    targets_path: Path,
    evaluate_on_test: bool,
):
    """Load and create r2 scores for training size sweep."""
    from ..evaluate import evaluate_sample_size

    validate_force(save_path, force)
    scores = evaluate_sample_size(
        targets_path,
        train_source,
        dev_source,
        checkpoint_path,
        evaluate_on_test=evaluate_on_test,
    )
    scores.write_csv(save_path, separator="|", quote_style="non_numeric")
    logger.info(f"saved: {save_path}")

@targets_path()
@checkpoint_path(exists=True, multiple=True)
@save_path(
    path_help="path to save fraction_binary_r2_table.csv",
    suffix=".csv",
    required=True,
)
@force()
@cli.command("evaluate:fraction-binary")
def evaluate_fraction_binary(
    targets_path: Path,
    checkpoint_path: Path,
    save_path: Optional[Path],
    force: bool,
):
    """Create latex table of cartesian experiments."""
    from ..evaluate import evaluate_fraction_binary

    validate_force(save_path, force)

    start_time = time.perf_counter()
    table = evaluate_fraction_binary(
        targets_path=targets_path,
        checkpoint_parent=checkpoint_path,
    )
    table.write_csv(save_path, separator="|")
    end_time = time.perf_counter()
    logger.info(f"saved {save_path} [{end_time - start_time:.2f} seconds]")

@cli.command("evaluate:previous-adoption")
@checkpoint_path()
@train_source()
@force()
@save_path(suffix=".csv", path_help="path to save score.csv")
@targets_path()
@evaluate_on_test()
def evaluate_previous_adoption(
    checkpoint_path: Path,
    train_source: str,
    force: bool,
    save_path: Path,
    targets_path: Path,
    evaluate_on_test: bool,
):
    """Load and create r2 scores for training size sweep."""
    from ..renewable_adoption.evaluate import (
        evaluate_renewable_adoption_vs_previous as evaluate,
    )

    validate_force(save_path, force)
    scores = evaluate(
        targets_path=targets_path,
        train_source=train_source,
        checkpoint_path=checkpoint_path,
        evaluate_on_test=evaluate_on_test,
    )
    scores.write_csv(save_path, separator="|", quote_style="non_numeric")
    logger.info(f"saved: {save_path}")
