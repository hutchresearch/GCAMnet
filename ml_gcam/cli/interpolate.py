from pathlib import Path
from typing import Optional

import click


@click.group()
def cli():
    """Explore gcam configs, inputs and outputs."""
    ...


@cli.command("interpolate:sample")
@click.option("--samples", default=4096, type=int)
@click.option(
    "--name",
    default="interp_hypercube",
    type=click.Choice(["interp_hypercube", "interp_dgsm", "binary"]),
)
def sample_bits(samples: int, name: str, save_to: Optional[Path] = None):
    """Creates a metadata.csv with [--sample] samples of input space."""
    from ..core.interpolate import sample_bits

    save_to = None

    sample_bits(samples, name, save_to)


@cli.command("interpolate:make-inputs")
@click.option(
    "-n",
    "--num_to_create",
    required=True,
    type=int,
    help="number of paths to create",
)
@click.option(
    "--name",
    default="interp_hypercube",
    type=click.Choice(["interp_hypercube", "interp_dgsm", "binary"]),
)
def make_inputs(num_to_create, name):
    """Create paths containing interpolated inputs."""
    from ..core.interpolate import make_inputs

    make_inputs(num_to_create, name)


@cli.command("interpolate:make-configs")
@click.option(
    "-n",
    "--name",
    "name",
    type=click.Choice(["interp_random", "interp_hypercube", "binary"]),
    required=True,
)
def make_configs(name):
    """Create configs from set of interpolated inputs."""
    from ..core.interpolate import make_configs

    make_configs(name)
