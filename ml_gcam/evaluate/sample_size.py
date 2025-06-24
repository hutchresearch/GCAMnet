import time
from pathlib import Path

import polars as pl

from .. import config, logger
from ..data import GcamDataset, Normalization, Source, Split
from ..inference import Inference


def evaluate_sample_size(
    targets_path: Path,
    train_source: Source,
    dev_source: Source,
    checkpoint_path: Path,
    evaluate_on_test: bool = False,
):
    """Load and create r2 scores for training size sweep."""
    logger.info("calculating sample-size")
    start_time = time.perf_counter()
    train_set = GcamDataset.from_targets(
        save_path=targets_path,
        experiment=train_source,
        split=Split.TRAIN,
    )
    normalization = Normalization(outputs=train_set.outputs)
    train_set.with_normalization(normalization)
    dev_set = GcamDataset.from_targets(
        save_path=targets_path,
        experiment=dev_source,
        split=Split.DEV if not evaluate_on_test else Split.TEST,
    )
    dev_set.with_normalization(normalization)

    collect = []
    for checkpoint in checkpoint_path.iterdir():
        samples = int(checkpoint.stem.split("_")[0])
        inference = (
            Inference.from_checkpoint(checkpoint)
            .eval_with(dev_set)
            .denormalize_with(train_set.normalization)
        )
        melted = inference.scores.melt(
            id_vars=["region", "year"],
            value_vars=config.data.output_keys,
            variable_name="target",
            value_name="r2",
        )
        melted = melted.with_columns(
            pl.lit(str(train_source)).alias("train"),
            pl.lit(samples).alias("training_samples"),
        )
        collect.append(melted)

    end_time = time.perf_counter()
    logger.info(f"sample-size done [{end_time - start_time:.2f} seconds]")
    return pl.concat(collect)
