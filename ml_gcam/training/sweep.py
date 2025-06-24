import os
from pathlib import Path

import wandb

from .. import config
from ..data import GcamDataset, NormStrat, load_targets, Split
from ..data.normalization import Normalization
from .train import run_training


def cli():
    """Run a wandb hyperparameter sweep."""
    ...


def init():
    sweep_configuration = {
        "method": "bayes",
        # "method": "random",
        "metric": {"goal": "minimize", "name": "r2_below_0_8"},
        # "metric": {"goal": "maximize", "name": "r2_mean"},
        "parameters": {
            "hidden_size": {
                "distribution": "categorical",
                "values": [32, 64, 128, 256],
            },
            # "batch_size": {
            #    "distribution": "categorical",
            #    "values": [16, 32, 64, 128, 256],
            # },
            "depth": {"distribution": "categorical", "values": [1, 2, 4, 8]},
            "epochs": {"distribution": "int_uniform", "min": 25, "max": 250},
            "learning_rate": {
                "distribution": "categorical",
                "values": [1e-3, 5e-3, 1e-2],
            },
            "norm_strat": {
                "distribution": "categorical",
                "values": ["z_score", "min_max", "robust"],
            },
        },
    }
    # 3: Start the sweep
    return wandb.sweep(sweep=sweep_configuration, project=config.project)


def sweep(sweep_id, runs):
    os.environ["ML_GCAM__WANDB__SWEEP_ID"] = str(sweep_id)
    config.reload()
    wandb.agent(sweep_id, function=train, count=runs, project=config.project)


def train() -> None:
    """Experimental training loop."""
    targets_path = Path(config.paths.targets)
    train_set = GcamDataset.from_targets(
        save_path=targets_path,
        experiment=config.training.train_set,
        split=Split.TRAIN,
    )
    normalization = Normalization(outputs=train_set.outputs, strategy=NormStrat.from_str(config.training.norm_strat))
    train_set.with_normalization(normalization)
    config.reload()

    dev_set = GcamDataset.from_targets(
        save_path=targets_path,
        experiment=config.training.dev_set,
        split=Split.DEV,
    )
    dev_set.with_normalization(normalization)

    run_training(train_set=train_set, dev_set=dev_set, checkpoint_path=None)


def set_sweep_params(wandb_config):
    if "hidden_size" in wandb_config:
        os.environ["ML_GCAM__MODEL__HIDDEN_SIZE"] = str(wandb_config.hidden_size)
    if "depth" in wandb_config:
        os.environ["ML_GCAM__MODEL__DEPTH"] = str(wandb_config.depth)
    if "epochs" in wandb_config:
        os.environ["ML_GCAM__TRAINING__EPOCHS"] = str(wandb_config.epochs)
    if "batch_size" in wandb_config:
        os.environ["ML_GCAM__TRAINING__BATCH_SIZE"] = str(wandb_config.batch_size)
    if "learning_rate" in wandb_config:
        os.environ["ML_GCAM__TRAINING__LEARNING_RATE"] = str(wandb_config.learning_rate)
    if "norm_strat" in wandb_config:
        os.environ["ML_GCAM__TRAINING__NORM_STRAT"] = str(wandb_config.norm_strat)
    config.reload()
