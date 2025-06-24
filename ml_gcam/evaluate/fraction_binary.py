import time
from pathlib import Path

import polars as pl

from .. import logger
from ..data import GcamDataset, Normalization, Source, Split
from ..inference import Inference


def evaluate_fraction_binary(targets_path: Path, checkpoint_parent: Path):
    """Load and create r2 scores for fraction of binary experiment."""
    start_time = time.perf_counter()
    logger.debug("starting fraction of binary evaluation")
    files = []
    for checkpoint in checkpoint_parent.iterdir():
        split = checkpoint.name.split("_")
        frac = float(f"{split[1]}.{split[2]}")
        files.append((checkpoint, frac))

    logger.debug(f"found {len(files)} files.")
    collect = []
    for checkpoint, frac in files:
        train_set = GcamDataset.from_targets(
            save_path=targets_path,
            experiment=Source.MIXED,
            split=Split.TRAIN,
            fraction_binary=frac,
        )
        normalization = Normalization(outputs=train_set.outputs)
        train_set.with_normalization(normalization)
        for dev_source in [Source.HYPERCUBE, Source.BINARY]:
            dev_set = GcamDataset.from_targets(
                save_path=targets_path,
                experiment=dev_source,
                split=Split.DEV,
            )
            dev_set.with_normalization(normalization)
            inference = (
                Inference.from_checkpoint(checkpoint)
                .eval_with(dev_set)
                .denormalize_with(train_set.normalization)
            )
            scores = inference.scores

            scores = scores.with_columns(
                [
                    pl.lit(str(dev_source)).alias("dev_source"),
                    pl.lit(frac).cast(pl.Float32).alias("fraction_binary"),
                    pl.lit(len(train_set)).cast(pl.UInt16).alias("training_samples"),
                ],
            )

            collect.append(scores)

    stack = pl.concat(collect)
    end_time = time.perf_counter()
    logger.info(f"sample-size done [{end_time - start_time:.2f} seconds]")
    return stack
