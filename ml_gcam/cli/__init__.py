import warnings

import click

from .. import config, logger
from .core import cli as core_cli
from .data import cli as data_cli
from .evaluate import cli as evaluate_cli
from .interpolate import cli as interpolate_cli
from .plot import cli as plot_cli
from .table import cli as table_cli
from .training import cli as training_cli
from .sensitivity import cli as sensitivity_cli

# suppress future warnings
warnings.simplefilter(action="ignore", category=FutureWarning)


@click.group()
def main():
    pass


class GcamCommand(click.CommandCollection):
    def __init__(self, *args, **kwargs):
        super(GcamCommand, self).__init__(*args, **kwargs)
        self.help = "Run ml_gcam commands"


def validate_options(ctx, param, value):
    import logging
    import os

    if value is None:
        return value
    if param.name == "debug":
        os.environ["ML_GCAM__DEBUG"] = "1" if value else "0"
        level = logging.DEBUG if value else logging.WARNING
        logger.setLevel(level)
    if param.name == "wandb":
        os.environ["ML_GCAM__WANDB__ENABLED"] = "1" if value else "0"
    if param.name == "group":
        os.environ["ML_GCAM__GROUP"] = value
    if param.name == "pretend":
        os.environ["ML_GCAM__PRETEND"] = "1" if value else "0"
    logger.debug(f"{param.name}: {value}")
    config.reload()
    return value


@click.command(
    cls=GcamCommand,
    sources=[
        core_cli,
        interpolate_cli,
        training_cli,
        evaluate_cli,
        plot_cli,
        table_cli,
        data_cli,
        main,
        sensitivity_cli,
    ],
)
@click.option(
    "--pretend/--no-pretend",
    type=bool,
    is_flag=True,
    default=bool(int(config.pretend)),
    help="avoid running expensive or destructive calculations.",
    callback=validate_options,
)
@click.option(
    "--debug/--no-debug",
    type=bool,
    is_flag=True,
    default=bool(int(config.debug)),
    help="enable debugging",
    callback=validate_options,
)
@click.option(
    "--wandb/--no-wandb",
    type=bool,
    is_flag=True,
    default=bool(int(config.wandb.enabled)),
    help="enable wandb",
    callback=validate_options,
)
@click.option(
    "--group",
    type=str,
    default=str(config.group),
    help="group name for wandb",
    callback=validate_options,
)
def cli(pretend: bool, debug: bool, wandb: bool, group: str):
    pass


if __name__ == "__main__":
    cli()
