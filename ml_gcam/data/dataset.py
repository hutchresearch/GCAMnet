from pathlib import Path
from typing import List

import numpy as np
import polars as pl
import torch
from torch.utils.data import Dataset
from torchtyping import TensorType

from .. import config
from .enums import Source, Split
from .normalization import Normalization
from .targets import convert_targets_to_numpy

SMALL = 1e-3


class GcamDataset(Dataset):
    targets: pl.DataFrame
    keys: pl.DataFrame
    sources: List[str]
    splits: List[str]
    outputs: np.ndarray
    inputs: np.ndarray
    normalized: np.ndarray
    normalization: Normalization

    @staticmethod
    def from_targets(
        save_path: Path = Path(config.paths.targets),
        experiment=Source.BINARY,
        split=Split.TRAIN,
        samples=None,
        fraction_binary=float(config.training.binary_fraction),
    ):
        from .targets import load_targets

        targets = load_targets(
            save_path,
            experiments=[experiment],
            split=split,
            samples=samples,
            fraction_binary=fraction_binary,
        )
        return GcamDataset(targets)

    @staticmethod
    def from_dataframe(targets: pl.DataFrame) -> "GcamDataset":
        return GcamDataset(targets)

    def __init__(
        self,
        targets: pl.DataFrame,
    ):
        """Builds default pytorch dataset from targets dataframe."""
        self.targets = targets
        self.keys = targets.select(pl.col("experiment"), pl.col("scenario_id")).unique()
        self.sources = targets.select(pl.col("experiment")).unique()[:, 0].to_list()
        self.splits = targets.select(pl.col("split")).unique()[:, 0].to_list()

        self.inputs, self.outputs = convert_targets_to_numpy(self.targets)

    def with_normalization(self, normalization: Normalization):
        self.normalization = normalization
        self.normalized = self.normalization.normalize(self.outputs)

    def __getitem__(
        self,
        idx: int,
    ) -> tuple[
        TensorType[len(config.data.input_keys)],
        TensorType[config.data.n_dimensions, len(config.data.output_keys)],
    ]:
        outputs_tensor = torch.tensor(self.normalized[idx], dtype=torch.float32)
        inputs_tensor = torch.tensor(self.inputs[idx], dtype=torch.float32)
        return inputs_tensor, outputs_tensor

    def __len__(self) -> int:
        return len(self.outputs)
