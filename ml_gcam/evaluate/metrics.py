from itertools import product
from typing import Dict

import numpy as np
import polars as pl
from sklearn.metrics import r2_score

from .. import config


def calculate_r2(y_pred: np.ndarray, y_true: np.ndarray):
    regions = sorted(config.data.region_keys)
    years = sorted(config.data.years)
    tuples = list(product(regions, years))
    keys = pl.DataFrame(tuples, schema=["region", "year"], orient="row").sort("region", "year")

    scores = []
    for dimension in range(config.data.n_dimensions):
        r2 = r2_score(
            y_pred[:, dimension, :],
            y_true[:, dimension, :],
            multioutput="raw_values",
        )
        scores.append(r2)
    arr = np.asarray(scores)

    df = pl.DataFrame(arr, schema=sorted(config.data.output_keys))
    return pl.concat([keys, df], how="horizontal")


def calculate_r2_aggs(scores) -> Dict[str, float]:
    stats = {}

    df = scores.melt(
        id_vars=["region", "year"],
        value_vars=config.data.output_keys,
        variable_name="target",
        value_name="r2",
    )

    if bool(int(config.training.detailed_logging)):
        for dimension in ["target", "region", "year"]:
            groups = (
                df.group_by(pl.col(dimension))
                .mean()
                .select(pl.col(dimension), pl.col("r2"))
            )
            for key, r2 in groups.iter_rows():
                stats |= {str(key): round(r2, 4)}

    arr = df.get_column("r2").to_numpy()
    total = len(arr)
    stats |= {"r2_mean": round(np.mean(arr), 4).item()}
    stats |= {"r2_median": round(np.median(arr), 4).item()}
    stats |= {"r2_std": round(np.std(arr), 4).item()}
    stats |= {"r2_min": round(np.min(arr), 4).item()}
    stats |= {"r2_max": round(np.max(arr), 4).item()}
    stats |= {"r2_above_0": round(len(arr[arr > 0]) / total, 4)}
    stats |= {"r2_above_0_5": round(len(arr[arr > 0.5]) / total, 4)}
    stats |= {"r2_above_0_8": round(len(arr[arr > 0.8]) / total, 4)}
    stats |= {"r2_above_0_9": round(len(arr[arr > 0.9]) / total, 4)}
    stats |= {"r2_above_0_95": round(len(arr[arr > 0.95]) / total, 4)}
    return stats


def dimensions_with_negative_scores(scores: pl.DataFrame) -> pl.DataFrame:
    df = scores.melt(
        id_vars=["region", "year"],
        variable_name="output",
        value_vars=config.data.output_keys,
        value_name="r2",
    )
    return df.filter(pl.col("r2") < 0).sort("r2", "output", "region", "year")
