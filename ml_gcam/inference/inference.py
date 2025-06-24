import time
from pathlib import Path
from typing import Union

import numpy as np
import polars as pl
import torch
from accelerate import Accelerator
from torch import nn
from torch.utils.data import DataLoader

from .. import config, logger
from ..data import GcamDataset, Normalization
from ..emulator.model import Arch, DNN


class Inference:
    """Runs inference on checkpoint."""

    accelerator: Accelerator
    model: Union[Path, nn.Module]
    sample_size: int
    y_pred_denorm: np.ndarray
    y_true_denorm: np.ndarray
    eval_dataset: GcamDataset
    denorm_dataset: GcamDataset
    dataloader: DataLoader
    x_true: torch.Tensor
    y_true: torch.Tensor
    y_pred: torch.Tensor
    y_pred_df: pl.DataFrame
    y_true_df: pl.DataFrame
    x_df: pl.DataFrame
    scores: pl.DataFrame

    def __init__(self, model: nn.Module):
        """Create inference from given model."""
        self.accelerator = Accelerator()
        self.model = model

    @staticmethod
    def from_model(model: nn.Module) -> "Inference":
        """Create from preloaded model weights."""
        return Inference(model)

    @staticmethod
    def from_checkpoint(checkpoint_path: Path) -> "Inference":
        """Create from checkpoint path."""
        short_path = "/".join(checkpoint_path.parts[-3:])
        logger.debug(f"loading checkpoint {short_path}")
        model = _model_from_checkpoint(checkpoint_path)
        return Inference(model)

    def eval_with(self, dataset: GcamDataset) -> "Inference":
        """
        Specify the dataset to use for inputs and outputs.

        pre: existing model
        pre: evalutation dataset
        """
        self.eval_dataset = dataset
        logger.debug(
            f"running inference: {self.eval_dataset.sources} | {self.eval_dataset.splits}",
        )
        start_time = time.perf_counter()
        dataloader = DataLoader(
            dataset,
            batch_size=int(config.training.batch_size),
            shuffle=False,
        )
        self.model, self.dataloader = self.accelerator.prepare(self.model, dataloader)
        self.model.eval()
        y_true, y_pred, x_true = [], [], []
        for x, y in self.dataloader:
            x_true.append(x)
            y_true.append(y)
            y_pred.append(self.model(x).detach())

        self.x_true = torch.cat(x_true).cpu()
        self.y_true = torch.cat(y_true).cpu()
        self.y_pred = torch.cat(y_pred).cpu()
        end_time = time.perf_counter()
        logger.debug(f"inference done [{end_time - start_time:.2f} seconds]")
        return self

    def denormalize_with(self, normalization: Normalization) -> "Inference":
        """Specify the dataset to use for denormalizing data."""
        self.sample_size = len(self.y_pred)
        self.y_pred_denorm = normalization.denormalize(self.y_pred)
        # train_dataset.denormalize(self.eval_dataset.normalized)
        self.y_true_denorm = self.eval_dataset.outputs
        self._to_df()
        self._r2()
        return self

    def use_inputs(self, inputs) -> "Inference":
        """
        Instead of a given dataset, run a set of given inputs through the model.

        R2 scores will be inaccurate since the dataset is using all zeros as outputs.
        """
        targets = pl.DataFrame(inputs, schema=sorted(config.data.input_keys))
        # create dummy outputs to work with gcamdataset
        outputs = pl.DataFrame(
            np.zeros((len(inputs), len(config.data.output_keys))),
            schema=sorted(config.data.output_keys),
        )
        targets = targets.concat([targets, outputs], how="horizonal")
        return self.eval_with(GcamDataset(targets))

    def use_samples(self, samples: int) -> "Inference":
        """Instead of a given dataset, run a set of random inputs through the model."""
        inputs = np.random.uniform(0, 1, (samples, 12))
        return self.use_inputs(inputs)

    def _to_df(self) -> None:
        from itertools import product

        regions = sorted(config.data.region_keys)
        years = sorted(config.data.years)
        tuples = list(product(regions, years)) * self.sample_size
        keys = pl.DataFrame(tuples, schema=["region", "year"], orient="row")
        y_pred = np.vstack(self.y_pred_denorm)
        y_true = np.vstack(self.y_true_denorm)
        
        y_pred_df = pl.DataFrame(y_pred, schema=sorted(config.data.output_keys))
        y_true_df = pl.DataFrame(y_true, schema=sorted(config.data.output_keys))

        self.y_pred_df = pl.concat([keys, y_pred_df], how="horizontal")
        self.y_true_df = pl.concat([keys, y_true_df], how="horizontal")
        self.x_df = pl.DataFrame(
            self.x_true.numpy(),
            schema=sorted(config.data.input_keys),
        )

    def _r2(self) -> None:
        from ..evaluate import calculate_r2

        logger.debug(
            f"validating: {self.eval_dataset.sources} | {self.eval_dataset.splits}",
        )
        start_time = time.perf_counter()
        self.scores = calculate_r2(self.y_pred, self.y_true)
        end_time = time.perf_counter()
        logger.debug(f"validating done [{end_time - start_time:.2f} seconds]")


def _model_from_checkpoint(checkpoint_path: Path):
    assert checkpoint_path.exists(), f"no weights found: {checkpoint_path.absolute()}"

    arch = Arch.from_str(config.model.arch)
    model = arch.init_model()

    try:
        accelerator = Accelerator()
        model = accelerator.prepare(model)
        accelerator.load_state(checkpoint_path)
    except RuntimeError as e:
        raise RuntimeError(
            f"""
            Make sure the following env/config variables match the loaded weights:
            ML_GCAM__MODEL__DEPTH=\t(current: {config.model.depth}),
            ML_GCAM__MODEL__HIDDEN_SIZE=\t(current: {config.model.hidden_size})
            """,
        ) from e
    return model
