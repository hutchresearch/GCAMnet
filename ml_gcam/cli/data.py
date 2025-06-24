"""cli for data."""

import time
from pathlib import Path
from typing import List, Optional

import click

from .. import config, logger
from ..cli.options import force, save_path, targets_path, validate_force

QUERIES = [
    "agriculture_prices",
    "electricity_supply",
    "emissions_capture",
    "energy_demand_share_electricity",
    "energy_demand_share_primary",
    "energy_prices",
    "energy_supply_share_electricity",
    "energy_supply_share_primary",
    "land_demand",
    "land_prices",
    "land_supply_allocation",
    "land_supply_production",
    "water_demand",
    "water_consumption",
]


@click.group()
def cli():
    """Run data related tasks."""
    pass


@cli.command("data:create-extracts")
@click.option(
    "-e",
    "--experiment",
    type=click.Choice(config.data.extract_sources),
    default=config.data.extract_sources,
)
@click.option(
    "-q",
    "--queries",
    multiple=True,
    type=click.Choice(QUERIES),
    default=QUERIES,
)
@click.option(
    "-g",
    "--gcamreader-outputs",
    "outputs_path",
    envvar="ML_GCAM__PATHS__EXTRACT__GCAMREADER_OUTPUTS",
    required=True,
    type=click.Path(file_okay=False, exists=True, readable=True, path_type=Path),
    help="directory with the .csv outputs from gcamreader",
)
@save_path(
    path_help="/path/to/save/{experiment}_targets.csv",
    is_directory=False,
    suffix=".csv",
)
@force()
@click.option("--pretend/--no-pretend", is_flag=True, default=False)
def extract_run(
    experiment,
    queries,
    outputs_path: Path,
    save_path: Path,
    force: bool,
):
    """Aggregated raw extract csv from all experiments via sql templates."""
    from ..core.extract import extract_outputs_from_gcamreader

    save_path = save_path / f"{experiment}_targets.csv"
    df = extract_outputs_from_gcamreader(experiment, queries, outputs_path)
    start_time = time.perf_counter()
    df.write_csv(save_path, separator="|", quote_style="non_numeric")
    end_time = time.perf_counter()
    logger.info(f"saved {save_path} [{end_time - start_time:.2f} seconds]")


@cli.command("data:create-scenarios")
@save_path(
    path_help="/path/to/data/scenarios.csv",
    path_default=Path(config.paths.repo) / "data/raw/scenarios.csv",
    is_directory=False,
    suffix=".csv",
)
@force()
def create_scenarios(save_path: Path, force: bool = False):
    """Create meta.scenarios table."""
    from ..data import create_scenarios

    validate_force(save_path, force)

    df = create_scenarios()
    df.write_csv(save_path, separator="|", quote_style="non_numeric")
    logger.info(f"saved {len(df)} scenarios {save_path}")


@cli.command("data:create-targets")
@click.option(
    "-e",
    "--experiment",
    multiple=True,
    type=click.Choice(config.data.extract_sources),
    default=config.data.extract_sources,
)
@click.option(
    "--raw_path",
    type=click.Path(file_okay=False, readable=True, exists=True, path_type=Path),
    help="/path/to/targets/",
    default=Path(config.paths.repo) / "data/raw/targets/",
)
@click.option(
    "--scenarios_path",
    type=click.Path(dir_okay=False, readable=True, exists=True, path_type=Path),
    help="path to scenarios.csv",
    default=Path(config.paths.repo) / "data/raw/scenarios.csv",
)
@save_path(
    path_help="/path/to/targets/targets.parquet (output partitioned by experiment and split)",
    path_default=Path(config.paths.repo) / "data/processed/targets/targets.parquet",
    is_directory=False,
    suffix=".parquet",
)
@force()
def create_targets(
    experiment: List[str],
    raw_path: Path,
    scenarios_path: Path,
    save_path: Path,
    force: bool = False,
):
    """Create targets.parquet table from experiment extracts and scenarios."""
    from ..data.targets import create_targets

    df = create_targets(experiment, raw_path, scenarios_path)
    logger.info(f"writing targets {len(df)} rows to {save_path}")
    start_time = time.perf_counter()
    df.write_parquet(
        save_path,
        pyarrow_options={"partition_cols": ["experiment", "split"]},
    )
    end_time = time.perf_counter()
    logger.info(f"done writing [{end_time - start_time:.2f} seconds]")
