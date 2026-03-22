import numpy as np
import polars as pl
from pathlib import Path
from sklearn.metrics import r2_score

from .. import config
from ..data import experiment_name_to_paper_label


def _finite_r2(y_true, y_pred):
    """Compute R² only over pairs where both values are finite."""
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if mask.sum() < 2:
        return float("nan")
    return r2_score(y_true[mask], y_pred[mask])

def sensitivity_table(dir_path: Path):
    sample_collect = []
    for source in ["binary", "interp_hypercube", "mixed"]:
        worlds = [world for world in sorted((dir_path / source).iterdir())]
        collect = []
        for file in worlds:
            df = pl.scan_csv(file, separator='|').collect()
            collect.append(df)
        df = pl.concat(collect)

        to_drop = ["bio", "elec", "emiss"]
        input_keys = [i for i in config.data.input_keys if i not in to_drop]

        emulator_df = df.filter(pl.col("source")=="emulator")
        core_df = df.filter(pl.col("source")=="core")

        overall_r2 = _finite_r2(core_df[input_keys].to_numpy().flatten(), emulator_df[input_keys].to_numpy().flatten())

        metric_collect = {}
        metric_collect["train"] = experiment_name_to_paper_label(source)
        metric_collect["test"] = "dgsm"

        for metric in ["region", "year", "quantity"]:
            grouped_core = core_df.group_by(metric).median().sort(metric)[input_keys]
            grouped_emulator = emulator_df.group_by(metric).median().sort(metric)[input_keys]
            r2 = _finite_r2(grouped_core.to_numpy().flatten(), grouped_emulator.to_numpy().flatten())
            metric_collect[metric] = r2
        metric_collect["overall"] = overall_r2
        sample_collect.append(pl.DataFrame(metric_collect))
    table = pl.concat(sample_collect)
    return table
