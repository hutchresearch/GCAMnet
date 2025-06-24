from pathlib import Path

import polars as pl

from .. import config
from ..data import experiment_name_to_label, load_targets, experiment_name_to_paper_label
from ..data.enums import Source, Split


def quantities():
    """Print all quantities."""
    df = pl.DataFrame([value for key, value in config.data.outputs.items()])
    table = (
        df.filter(pl.col("subset") == "new")
        .select(
            [
                pl.col("resource").alias("Resource"),
                pl.col("metric").alias("Metric"),
                pl.col("sector").alias("Sector"),
                pl.col("units").alias("Units"),
                pl.col("query").alias("Query"),
            ],
        )
        .sort(["Resource", "Metric", "Sector"])
    )
    return table


def regions():
    """Print all regions."""
    from ..config import config
    table = pl.DataFrame(config.data.region_keys)
    return table


def years():
    """Print all years."""
    from ..config import config
    table = pl.DataFrame(config.data.year_keys)
    return table


def inputs():
    """Print all inputs."""
    return (
        pl.DataFrame([value for key, value in config.data.inputs.items()])
        .select(
            pl.col("label").alias("Input"),
            pl.col("description").alias("Description"),
            pl.col("key").alias("Key"),
            pl.when(pl.col("interpolated"))
            .then(pl.lit("Yes"))
            .otherwise(pl.lit("No"))
            .alias("Interpolated?"),
        )
        .sort("Key")
    )


def datasets(targets_path: Path):
    """Print all inputs."""
    sources = [value for key, value in config.data.sources.items()]
    sources = pl.DataFrame(sources).filter(pl.col('enabled') == 'true')
    targets = load_targets(save_path=targets_path, split=[Split.TRAIN, Split.DEV, Split.TEST])
    datasets = sources.filter(pl.col("enabled")).select(
        pl.col("key").alias("experiment"),
        pl.col("name"),
        pl.col("new_samples"),
        pl.col("range"),
    )

    scenarios = targets.group_by("experiment").agg(
        pl.col("scenario_id")
        .filter(pl.col("split") == "train")
        .n_unique()
        .alias("train"),
        pl.col("scenario_id").filter(pl.col("split") == "dev").n_unique().alias("dev"),
        pl.col("scenario_id")
        .filter(pl.col("split") == "test")
        .n_unique()
        .alias("test"),
    )

    table = datasets.join(scenarios, on=["experiment"])

    return table.select(
        pl.col("name").str.to_titlecase().alias("Sampling"),
        pl.when(pl.col("range") == "binary")
        .then(pl.lit(r"x \in {0, 1}"))
        .otherwise(pl.lit(r"x \in [0, 1]"))
        .alias("Range"),
        pl.col("train").alias("Training Scenarios"),
        pl.col("dev").alias("Validation Scenarios"),
        pl.col("test").alias("Test Scenarios"),
        pl.when(pl.col("new_samples"))
        .then(pl.lit("Yes"))
        .otherwise(pl.lit("No"))
        .alias("New GCAM Samples?"),
    )


def cartesian(score_path: Path):
    """Print all outputs."""
    df = pl.scan_csv(score_path, separator="|")
    df = df.with_columns(
        pl.col("train_source")
        .map_elements(experiment_name_to_paper_label, return_dtype=pl.String)
        .alias("train"),
        pl.col("dev_source")
        .map_elements(experiment_name_to_paper_label, return_dtype=pl.String)
        .alias("test"),
    )
    grouped = (
        df.group_by("train", "test").agg(pl.median("r2").alias("overall")).collect()
    )

    for g in ["region", "year", "target"]:
        median = (
            df.group_by("train", "test", g)
            .agg(pl.col("r2").median())
            .group_by("train", "test")
            .agg(pl.median("r2").alias(g))
            .collect()
        )
        grouped = grouped.join(median, on=["train", "test"], how="outer_coalesce")

    ordered = [
        "binary",
        "interpolated",
        "mixed",
    ]
    order_dict = {k: i for i, k in enumerate(ordered)}

    return grouped.sort(
        by=[
            pl.col("train").map_elements(
                lambda x: order_dict.get(x, float("inf")),
                return_dtype=pl.Float32,
            ),
            pl.col("test").map_elements(
                lambda x: order_dict.get(x, float("inf")),
                return_dtype=pl.Float32,
            ),
        ],
        descending=[False, False],
    ).select(
        "train",
        "test",
        "region",
        "year",
        pl.col("target").alias("quantity"),
        "overall",
    )


def sample_size(score_path: Path, aggregation: str = "mean"):
    """Table of overall r2 vs. training size."""
    df = pl.scan_csv(score_path, separator="|")

    grouped = df.group_by(pl.col("training_samples").alias("Samples"))
    agg = grouped.agg(
        [
            (pl.col("r2") > 0.9).mean().alias("R2 > 0.9"),
            (pl.col("r2") > 0.95).mean().alias("R2 > 0.95"),
            pl.col("r2").median().alias("Median R2"),
            pl.col("r2").mean().alias("Mean R2"),
        ],
    )
    return agg.sort("Samples").collect()


def fraction_binary_vs_r2(score_path: Path, aggregation: str):
    from ..evaluate import calculate_r2_aggs

    table = pl.read_csv(score_path, separator="|")
    keys: pl.DataFrame = table.select(
        "fraction_binary",
        "dev_source",
        "training_samples",
    ).unique()

    aggs = []
    for frac, dev_source, training_samples in keys.iter_rows():
        agg = calculate_r2_aggs(
            table.filter(pl.col("fraction_binary") == round(frac, 2)).filter(
                pl.col("dev_source") == dev_source,
            ),
        )
        agg |= {
            "fraction_binary": frac,
            "dev_source": dev_source,
            "training_samples": training_samples,
        }
        aggs.append(agg)
    data = pl.DataFrame(aggs)
    table = data.select(
        "dev_source",
        "fraction_binary",
        f"r2_{aggregation}",
        (pl.col("training_samples") * pl.col("fraction_binary"))
        .cast(pl.UInt16)
        .alias("binary_samples"),
        (
            pl.col("training_samples")
            - (pl.col("training_samples") * pl.col("fraction_binary"))
        )
        .cast(pl.UInt16)
        .alias("random_samples"),
    ).sort("dev_source", "fraction_binary")
    return table


def fraction_binary_negative_values(score_path: Path):
    table = pl.read_csv(score_path, separator="|")
    # table.select(
    #     "fraction_binary",
    #     "dev_source",
    #     "training_samples",
    # ).unique()

    melted = table.melt(
        id_vars=["region", "year", "training_samples", "dev_source", "fraction_binary"],
        value_vars=config.data.output_keys,
        value_name="r2",
        variable_name="output",
    )

    melted.filter(pl.col("r2") < 0).group_by("region").count().sort("count")
    melted.filter(pl.col("r2") < 0).group_by("year").count().sort("count")
    melted.filter(pl.col("r2") < 0).group_by("output").count().sort("count")

    breakpoint()
    return table
