from pathlib import Path

import click

from .. import config
from ..cli.options import force, save_or_print, save_path, table_format


@click.group()
def cli():
    """Explore gcam configs, inputs and outputs."""
    ...


@cli.command("core:input-main")
@click.option("-o", "--output_file", type=str, default="scenarios.csv")
@click.option(
    "--output_dir",
    type=Path,
    show_default=True,
)
@click.option(
    "--config_dir",
    type=Path,
    show_default=True,
)
def input_main(output_dir, output_file, config_dir):
    """Parse scenario config -> inputs used."""
    from ..core.input import main

    main(output_dir, output_file, config_dir)


@cli.command("core:input-parse")
@click.option(
    "--batch_path",
    type=click.Path(dir_okay=False, exists=True, readable=True, path_type=Path),
    required=True,
)
@click.option("-f", "--filename", type=str, default="scenarios.csv")
@click.option("-d", "--filedir", type=str, default="scenarios.csv")
@table_format()
@save_path(
    path_help="/path/to/run_permutations.tex output",
    suffix=".tex",
    required=False,
)
@force()
def input_parse(
    batch_path: Path,
    filename,
    filedir,
    table_format: str,
    save_path: Path,
    force: bool,
):
    """Exploratory analysis of a single config file from the experiment."""
    from ..core.input import parse

    table = parse(batch_path, filename, filedir)
    save_or_print(table, table_format, save_path)


@click.option(
    "--config_path",
    type=click.Path(dir_okay=False, exists=True, readable=True, path_type=Path),
    required=True,
)
@table_format()
@save_path(
    path_help="/path/to/run_permutations.tex output",
    suffix=".tex",
    required=False,
)
@force()
@cli.command("core:input-defaults")
def input_defaults(config_path: Path, table_format: str, save_path: Path, force: bool):
    """Finds the common default values in the config file of a gcam run."""
    from ..core.input import defaults

    table = defaults(config_path)
    save_or_print(table, table_format, save_path)


######################################
#
# Outputs
#
######################################


@click.option(
    "--xml_path",
    type=click.Path(dir_okay=False, exists=True, readable=True, path_type=Path),
    default=Path(config.paths.repo) / "ml_gcam/core/queries/Main_queries.xml",
    required=True,
)
@click.option(
    "--data_path",
    type=click.Path(file_okay=False, exists=True, readable=True, path_type=Path),
    default=Path(config.paths.repo) / "data/raw/query_outputs/binary/sample",
    required=True,
)
@table_format()
@save_path(
    path_help="/path/to/run_permutations.tex output",
    suffix=".tex",
    required=False,
)
@force()
@cli.command("core:output-explore")
def output_main_queries(
    xml_path: Path,
    data_path: Path,
    table_format: str,
    save_path: Path,
    force: bool,
):
    """Expoloratory analysis on the available xpath queries to extract gcam databases."""
    from ..core.output import main_queries

    table = main_queries(xml_path, data_path)
    save_or_print(table, table_format, save_path)
