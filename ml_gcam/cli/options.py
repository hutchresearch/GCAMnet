from pathlib import Path
from typing import Optional

import click
import pandas as pd
import polars as pl

from .. import config, logger
from ..data import Source


def save_or_print(table, table_format, save_path):
    from rich.console import Console

    console = Console()

    if isinstance(table, pd.DataFrame):
        table = pl.from_pandas(table)
    if table_format == "latex":
        output = table.to_pandas().to_latex(index=False, escape=True, float_format="%.3f")
    elif table_format == "markdown":
        output = table.to_pandas().to_markdown(index=False)
    else:
        raise click.UsageError(f"--table_format {table_format} unknown")
    if save_path is not None:
        with open(save_path, "w") as f:
            f.write(output)
    else:
        console.print(output)


def targets_path(func=None):
    def decorator(func):
        click.option(
            "--targets_path",
            type=click.Path(dir_okay=False, readable=True, exists=True, path_type=Path),
            help="/path/to/targets.parquet",
            default=Path(config.paths.targets),
            show_default=True,
            required=True,
        )(func)
        return func

    return decorator


def evaluate_on_test(func=None):
    def decorator(func):
        click.option(
            "--test/--no-test",
            "evaluate_on_test",
            is_flag=True,
            type=bool,
            default=bool(int(config.training.evaluate_on_test)),
            help="should evaluation happen on dev or test set",
            show_default=True,
            required=True,
        )(func)
        return func

    return decorator


def validate_force(save_path: Optional[Path], force: bool):
    if save_path is not None and (save_path.exists() and not force):
        raise click.UsageError(f"{save_path} exists. use --force to overwrite")


def table_format(func=None):
    def decorator(func):
        click.option(
            "-t",
            "--table_format",
            "table_format",
            type=click.Choice(["latex", "markdown"]),
            default="latex",
        )(func)
        return func

    return decorator


def train_source(func=None, multiple: bool = False):
    default = config.data.train_sources if multiple else config.training.train_source

    def callback(ctx, param, value):
        if multiple:
            value = [Source.from_str(v) for v in value]
        else:
            value = Source.from_str(value)
        logger.debug(f"train_source {value}")
        return value

    def decorator(func):
        click.option(
            "-t",
            "--train_source",
            help="name of experiment(s) to use as train_source",
            default=default,
            type=click.Choice(config.data.train_sources),
            required=True,
            multiple=bool(multiple),
            callback=callback,
            show_default=True,
        )(func)
        return func

    return decorator


def dev_source(func=None, multiple: bool = False):
    default = config.data.dev_sources if multiple else config.training.dev_source

    def callback(ctx, param, value):
        if multiple:
            value = [Source.from_str(v) for v in value]
        else:
            value = Source.from_str(value)
        logger.debug(f"dev_source {value}")
        return value

    def decorator(func):
        click.option(
            "-d",
            "--dev_source",
            help="name of experiment(s) to use as dev_srouce",
            default=default,
            type=click.Choice(config.data.dev_sources),
            required=True,
            multiple=bool(multiple),
            show_default=True,
            callback=callback,
        )(func)
        return func

    return decorator


def checkpoint_path(func=None, exists: bool = False, multiple=False):
    def validate_checkpoint_path(ctx, param, value):
        if exists and not multiple:
            contents = list(value.iterdir())
            if ".safetensors" not in [f.suffix for f in contents if f.is_file()]:
                raise click.BadParameter(
                    f"checkpoint path {value} does not contain .safetensor file",
                )
        logger.debug(f"checkpoint {value}")
        return value

    def decorator(func):
        click.option(
            "-c",
            "--checkpoint_path",
            type=click.Path(
                file_okay=False,
                readable=True,
                writable=bool(not exists),
                exists=bool(exists),
                path_type=Path,
            ),
            callback=validate_checkpoint_path,
            help="/path/to/model/checkpoint/name/",
            required=bool(exists),
            default=Path(config.paths.checkpoint),
        )(func)

        return func

    return decorator


def save_path(
    func=None,
    path_help="path to save the output to",
    path_default: Optional[Path] = None,
    suffix: Optional[str] = None,
    is_directory: bool = False,
    required: bool = True,
):
    def validate_save_path(ctx, param, value):
        if not required and value is None:
            return value
        if suffix is not None and value.suffix != suffix:
            raise click.BadParameter(f"{value} must end with {suffix}")
        if is_directory and not value.is_dir():
            raise click.BadParameter(f"{value} must be a directory")

        if not value.parent.exists():
            raise click.BadParameter(
                f"parent of {value} does not exist. please create it first.",
            )
        return value

    def decorator(func):
        click.option(
            "-o",
            "--save_path",
            type=click.Path(
                file_okay=not is_directory,
                dir_okay=bool(is_directory),
                writable=True,
                path_type=Path,
            ),
            callback=validate_save_path,
            default=path_default,
            help=path_help,
            required=bool(required),
        )(func)

        return func

    return decorator


def force(func=None):
    def decorator(func):
        click.option(
            "-f",
            "--force",
            is_flag=True,
            default=False,
            help="forces the --save_path to overwrite existing data",
        )(func)
        return func

    return decorator
