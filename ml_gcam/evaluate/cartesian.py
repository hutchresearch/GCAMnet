import time
from pathlib import Path

import polars as pl

from .. import config, logger
from ..data import GcamDataset, Normalization, Source, Split
from ..inference import Inference


def evaluate_cartesian(
    targets_path: Path,
    checkpoint_path: Path,
    evaluate_on_test: bool,
):
    """Load and create r2 scores for cartesian sweep."""
    logger.info("calculating cartesian")
    start_time = time.perf_counter()
    collect = []
    train_sources = [Source.from_str(source) for source in config.data.train_sources]
    dev_sources = [Source.from_str(source) for source in config.data.dev_sources]
    for train in train_sources:
        train_set = GcamDataset.from_targets(
            save_path=targets_path,
            experiment=train,
            split=Split.TRAIN,
        )
        normalization = Normalization(outputs=train_set.outputs)
        train_set.with_normalization(normalization)
        for dev in dev_sources:
            checkpoint = checkpoint_path / f"{train}_vs_{dev}"
            if not checkpoint.exists():
                logger.warning(f"could not find checkpoint: {checkpoint}")
                logger.warning(f"skipping {train} vs {dev}")
                continue
            if evaluate_on_test:
                split = Split.DEV if str(dev) == "interp_dgsm" else Split.TEST
            else:
                split = Split.DEV
            dev_set = GcamDataset.from_targets(
                save_path=targets_path,
                experiment=dev,
                split=split,
            )
            dev_set.with_normalization(normalization)
            inference = (
                Inference.from_checkpoint(checkpoint)
                .eval_with(dev_set)
                .denormalize_with(train_set.normalization)
            )
            scores = inference.scores.melt(
                id_vars=["region", "year"],
                value_vars=config.data.output_keys,
                variable_name="target",
                value_name="r2",
            )
            scores = scores.with_columns(
                pl.lit(str(train)).alias("train_source"),
                pl.lit(str(dev)).alias("dev_source"),
            )
            collect.append(scores)
    end_time = time.perf_counter()
    logger.info(f"cartesian done [{end_time - start_time:.2f} seconds]")
    return pl.concat(collect)
