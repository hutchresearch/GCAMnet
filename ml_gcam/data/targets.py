"""for creating and accessing the targets of the emulator."""

import time
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import polars as pl
from polars.exceptions import PolarsError

from .. import config, logger
from .enums import Source, Split

def create_targets(
    experiment: List[str],
    targets_path: Path,
    scenarios_path: Path,
):
    """Load raw data and exports joined targets and scenarios to parquet files."""
    logger.debug(f"scanning {targets_path} and {scenarios_path}")
    start_time = time.perf_counter()
    target_columns = []
    for attribute in config.data.output_keys:
        target_columns.append(pl.col(attribute))
    targets = []
    for target in targets_path.iterdir():
        if target.suffix == ".csv" and target.stem[: -len("_target") - 1] in experiment:
            df = (
                pl.scan_csv(target, separator="|")
                .select(
                    pl.col("experiment"),
                    pl.col("scenario"),
                    pl.col("region"),
                    pl.col("year"),
                    *config.data.output_keys,
                )
                .collect()
            )
            logger.debug(f"found {target.name} with shape {df.shape}")
            targets.append(df)
    targets = pl.concat(targets)
    scenarios = pl.scan_csv(scenarios_path, separator="|").collect()
    result = scenarios.join(
        targets,
        on=["experiment", "scenario"],
        how="left",
    ).fill_null(0)
    end_time = time.perf_counter()
    logger.debug(f"loading done [{end_time - start_time:.2f} seconds]")

    return result


SOURCES_WITH_DATA = Source.sampled()
ALL_SPLITS = Split.all()
ID_COLS = [
    pl.col("experiment").cast(pl.String),
    pl.col("scenario_id").cast(pl.Int16),
    pl.col("region").cast(pl.String),
    pl.col("year").cast(pl.Int16),
    pl.col("split").cast(pl.String),
]
INPUT_COLS = [
    pl.col(a).alias(a).cast(pl.Float32) for a in sorted(config.data.input_keys)
]
OUTPUT_COLS = [
    pl.col(a).alias(a).cast(pl.Float32) for a in sorted(config.data.output_keys)
]


def load_targets(
    save_path: Path = Path(config.paths.targets),
    experiments: List[Source] = SOURCES_WITH_DATA,
    split: Union[str, Split, List[Split]] = ALL_SPLITS,
    samples: Optional[int] = None,
    fraction_binary: float = float(config.training.binary_fraction),
):
    """Main way for targets.parquet to be loaded from disk."""
    experiments, splits = _backward_compatible(experiments, split)
    short_path = "/".join(save_path.parts[-3:])
    logger.debug(f"loading {experiments}|{splits} from {short_path}")
    start_time = time.perf_counter()

    targets = pl.scan_parquet(save_path)
    targets = _filter_experiments(targets, experiments)
    targets = _filter_splits(targets, splits)
    targets = _filter_years(targets)
    targets = _filter_samples(targets, experiments, splits, samples, fraction_binary)
    targets = (
        targets.select(
            *ID_COLS,
            *OUTPUT_COLS,
            *INPUT_COLS,
        )
        .sort(*ID_COLS)
        .collect()
    )
    end_time = time.perf_counter()
    logger.debug(f"loading done [{end_time - start_time:.2f} seconds]")
    return targets


def _filter_years(targets):
    return targets.filter(pl.col("year").is_in(config.data.year_keys))


def _filter_splits(targets, splits: List[Split]):
    keep = list(map(str, splits))
    return targets.filter(pl.col("split").is_in(keep))


def _filter_experiments(targets, experiments: List[Source]):
    if Source.MIXED in experiments:
        if len(experiments) != 1:
            logger.warn("loaded mixed with other experiments")
        keep = [Source.BINARY, Source.HYPERCUBE]
    elif len(experiments) == 0:
        keep = Source.sampled()
    else:
        keep = experiments
    logger.debug(f"filtered to {keep}")
    keep = list(map(str, keep))
    return targets.filter(pl.col("experiment").is_in(keep))


def _get_limit(
    experiments: List[Source],
    splits: List[Split],
    max_size: int,
) -> Optional[int]:
    if len(experiments) == 1 and len(splits) == 1:
        if Source.DGSM in experiments:
            if Split.TEST in splits:
                raise ValueError("bad sample limit")
            limit = 3650
        elif Split.TRAIN in splits:
            limit = config.training.max_train_samples
        elif (Split.DEV in splits) or (Split.TEST in splits):
            limit = config.training.max_dev_samples
        else:
            raise ValueError("bad sample limit")
    else:
        limit = max_size
    return limit


def _filter_samples(targets, experiments, splits, samples, fraction_binary=config.training.binary_fraction):
    keys = (
        targets.select(pl.col("experiment"), pl.col("scenario_id"))
        .unique()
        .sort("experiment", "scenario_id")
        .with_columns(
            pl.int_range(1, pl.len() + 1, dtype=pl.UInt32)
            .sort_by("scenario_id")
            .over("experiment")
            .alias("index"),
        )
        .collect()
    )

    samples = (
        _get_limit(experiments, splits, max_size=len(keys))
        if samples is None
        else samples
    )

    if Source.MIXED in experiments:
        interp_limit = int((samples * (1 - fraction_binary)))
        binary_limit = samples - interp_limit  # should be (samples * fraction_binary)
        logger.debug(f"splits [{binary_limit} binary | {interp_limit} hypercube]")
        keys = keys.filter(
            pl.when(pl.col("experiment") == "binary")
            .then(pl.col("index") <= binary_limit)
            .when(pl.col("experiment") == "interp_hypercube")
            .then(pl.col("index") <= interp_limit),
        )
        samples = len(keys)
    try:
        if samples <= len(keys):
            keys = keys.sort("index", "experiment").sample(
                n=samples if samples is not None else None,
                fraction=1 if samples is None else None,
                shuffle=True,
                seed=config.random_seed,
            )
        else:
            raise ValueError(
                f"samples requested ({samples}) larger than available ({len(keys)})",
            )
    except PolarsError:
        logger.error(
            f"loading targets {experiments} | {splits} | {samples}",
            exc_info=True,
        )
        return None
    logger.debug(f"{experiments}|{splits} returned {len(keys)} samples")
    targets = keys.lazy().join(targets, on=["experiment", "scenario_id"])
    return targets


def _backward_compatible(experiments, split):
    for source in experiments:
        if isinstance(source, str):
            logger.warning(
                "using a str for load_target experiments param is deprecated, use List[Source] instead",
            )
            experiments.remove(source)
            experiments.append(Source.from_str(source))
    if isinstance(split, str):
        logger.warning(
            "using a str for load_target experiments param is deprecated, use List[Source] instead",
        )
        splits = [Split.from_str(split)]
    elif isinstance(split, Split):
        splits = [split]
    else:
        splits = split
    for split in splits:
        if isinstance(split, str):
            logger.warning(
                "using a str for load_target split param is deprecated, use List[Split] instead",
            )
            splits.remove(split)
            splits.append(Split.from_str(split))
    return experiments, splits


def convert_targets_to_numpy(targets: pl.DataFrame) -> np.ndarray:
    from .. import config

    sort_columns = [
        pl.col("experiment").alias("experiment"),
        pl.col("scenario_id").alias("scenario_id"),
        pl.col("region").alias("region"),
        pl.col("year").alias("year"),
    ]
    from einops import rearrange

    outputs = (
        targets.sort(*sort_columns).select(sorted(config.data.output_keys)).to_numpy()
    )
    n_targets = len(config.data.output_keys)
    outputs = rearrange(
        outputs,
        "(samples dimensions) targets -> samples dimensions targets",
        dimensions=config.data.n_dimensions,
        targets=n_targets,
    )
    inputs = targets.select(sorted(config.data.input_keys)).to_numpy()[
        :: config.data.n_dimensions
    ]

    return inputs, outputs
