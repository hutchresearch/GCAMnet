from pathlib import Path

import polars as pl
import numpy as np

from .. import config, logger

INPUT_COLUMNS = [
    pl.col("back").alias("back").cast(pl.Float64),
    pl.col("bio").alias("bio").cast(pl.Float64),
    pl.col("ccs").alias("ccs").cast(pl.Float64),
    pl.col("elec").alias("elec").cast(pl.Float64),
    pl.col("emiss").alias("emiss").cast(pl.Float64),
    pl.col("energy").alias("energy").cast(pl.Float64),
    pl.col("ff").alias("ff").cast(pl.Float64),
    pl.col("nuc").alias("nuc").cast(pl.Float64),
    pl.col("solarS").alias("solarS").cast(pl.Float64),
    pl.col("solarT").alias("solarT").cast(pl.Float64),
    pl.col("windS").alias("windS").cast(pl.Float64),
    pl.col("windT").alias("windT").cast(pl.Float64),
]


def create_scenarios():
    """Create scenarios .csv file."""
    logger.debug("loading scenarios from raw data")
    binary = _load_binary()
    hypercube = _load_hypercube()
    dgsm = _load_dgsm()
    return pl.concat([binary, hypercube, dgsm])


def _load_hypercube():
    metadata_path = Path(config.paths.data) / "meta" / "interp_hypercube.csv"

    df = (
        pl.scan_csv(metadata_path, separator="|")
        .select(
            pl.lit("interp_hypercube").alias("experiment"),
            pl.col("scenario_id").alias("scenario_id"),
            (pl.lit("scenario_") + pl.col("scenario_id").cast(pl.String)).alias(
                "scenario",
            ),
            pl.lit(None).alias("split"),
            *INPUT_COLUMNS,
        )
        .collect()
    )
    logger.info(f"found {len(df)} hypercube scenarios {metadata_path}")
    train_samples = int(0.8 * len(df))
    dev_samples = int(0.1 * len(df))
    return df.sample(len(df), shuffle=True, seed=config.random_seed).with_columns(
        pl.when(pl.arange(0, pl.count()) <= train_samples)
        .then(pl.lit("train"))
        .when(pl.arange(0, pl.count()) <= train_samples + dev_samples)
        .then(pl.lit("dev"))
        .otherwise(pl.lit("test"))
        .alias("split"),
    )


def _load_dgsm():
    metadata_path = Path(config.paths.data) / "meta" / "interp_dgsm.csv"

    df = (
        pl.scan_csv(metadata_path, separator="|")
        .select(
            pl.lit("interp_dgsm").alias("experiment"),
            pl.col("scenario_id").alias("scenario_id"),
            (pl.lit("scenario_") + pl.col("scenario_id").cast(pl.String)).alias(
                "scenario",
            ),
            pl.lit("dev").alias("split"),
            *INPUT_COLUMNS,
        )
        .collect()
    )
    logger.info(f"found {len(df)} dgsm scenarios {metadata_path}")
    return df


def _load_binary():
    metadata_path = Path(config.paths.data) / "query_output/pnnl/binary"
    database = [
        "_" + f.stem.lstrip("out_database_basexdb") for f in metadata_path.iterdir()
    ]
    encodings = [_get_input_encoding(d) for d in database]

    df = pl.concat(
        [pl.DataFrame({"scenario": database}), pl.DataFrame(encodings)],
        how="horizontal",
    ).select(
        pl.lit("binary").alias("experiment"),
        pl.arange(1, pl.count() + 1).alias("scenario_id"),
        pl.col("scenario"),
        pl.lit(None).alias("split"),
        *INPUT_COLUMNS,
    )

    logger.info(f"found {len(df)} binary scenarios {metadata_path}")
    train_samples = int(0.8 * len(df))
    dev_samples = int(0.1 * len(df))
    return df.sample(len(df), shuffle=True, seed=config.random_seed).with_columns(
        pl.when(pl.arange(0, pl.count()) <= train_samples)
        .then(pl.lit("train"))
        .when(pl.arange(0, pl.count()) <= train_samples + dev_samples)
        .then(pl.lit("dev"))
        .otherwise(pl.lit("test"))
        .alias("split"),
    )


def _get_input_encoding(name: str) -> dict[str, int]:
    """Takes in a file name as an argument and determines the lo-hi encoding scheme and creates a dictionary with the corresponding binary values."""
    # All of the bits are separated by an underscore (do not include start of file name or end)
    fields = name.split("_")[1:]

    # Each file name looks like ff-lo_emiss-lo_bio-hi_nuc-lo
    # So split each pair up [ff-lo, emiss-lo, bio-hi]
    # And replace lo with 0 and hi with 1
    bits = {
        field.split("-")[0]: 1 if field.split("-")[1] == "hi" else 0 for field in fields
    }

    # Sort the inputs alphabetically for consistency and turn back into a dictionary
    sorted_bits = dict(sorted(bits.items(), key=lambda x: x[0]))
    return sorted_bits
    # return torch.tensor(list(sorted_bits.values()), dtype=torch.float32)
