from pathlib import Path
from typing import Optional

import numpy as np
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from .. import config, logger
from ..inference import Inference
from .logger import ConsoleTracker, Logger


class Runner:
    """Runner class that is in charge of implementing routine training functions such as running epochs or doing inference time."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        dev_loader: DataLoader,
        optimizer: Optimizer,
    ):
        # Accelerator is in charge of auto casting tensors to the appropriate GPU device
        self.accelerator = self._init_accelerator()

        self.logger = Logger(self.accelerator)
        # Prepare all objects for multi-GPU training
        model, optimizer, train_loader, dev_loader = self.accelerator.prepare(
            model,
            optimizer,
            train_loader,
            dev_loader,
        )
        self.model = model
        self.train_loader = train_loader
        self.dev_loader = dev_loader
        self.optimizer = optimizer

        # Since data is for targets, use Mean Squared Error Loss
        self.criterion = nn.MSELoss()
        if "sweep_id" in config.wandb:
            import wandb

            from ..training.sweep import set_sweep_params

            set_sweep_params(wandb.config)
        self.log_debug()

    def log_debug(self):
        logger.debug(f"train source: {config.training.train_source}")
        logger.debug(f"dev source: {config.training.dev_source}")
        logger.debug(
            f"dev split: {'test' if bool(int(config.training.evaluate_on_test)) else 'dev'}",
        )
        logger.debug(f"normalization: {config.training.norm_strat}")
        logger.debug(f"batch size: {config.training.batch_size}")
        logger.debug(f"learning rate: {config.training.learning_rate}")
        logger.debug(f"epochs: {config.training.epochs}")
        logger.debug(f"arch: {config.model.arch}")
        logger.debug(f"hidden size: {config.model.hidden_size}")
        logger.debug(f"depth: {config.model.depth}")
        logger.debug(f"act_fn: {config.model.act_fn}")

    def _init_accelerator(self) -> Accelerator:
        from accelerate.utils import LoggerType

        console = ConsoleTracker("")
        loggers = [console]
        project_config = ProjectConfiguration(project_dir=".")
        if bool(int(config.wandb.enabled)):
            loggers.append(LoggerType.WANDB)
        if bool(int(config.tensorboard.enabled)):
            loggers.append(LoggerType.TENSORBOARD)
            # needed to play nice with tensorboard
            project_config.logging_dir = config.paths.tensorboard
        return Accelerator(log_with=loggers, project_config=project_config)

    def epoch(self, training=True) -> float:
        """Runs an epoch of training.

        Includes updating model weights and tracking training loss

        Returns:
            float: The loss averaged over the entire epoch
        """
        # Turn the model to training mode (affects batchnorm and dropout)
        if training:
            self.model.train()
            dataloader = self.train_loader
        else:
            self.model.eval()
            dataloader = self.dev_loader

        running_loss = 0.0

        for sample, target in dataloader:
            # Make sure there are no leftover gradients before starting training an epoch
            self.optimizer.zero_grad()
            prediction = self.model(sample)  # Forward pass through model
            loss = self.criterion(prediction, target)  # Error calculation
            running_loss += loss.item()  # Increment running loss

            # Only update model weights on training
            if training:
                self.accelerator.backward(
                    loss,
                )  # Increment gradients within model by sending loss backwards
                self.optimizer.step()  # Update model weights
            # yield round(loss.item() / len(sample), 4)
        return round(running_loss / len(dataloader), 4)

    def train(self, epochs: int = 50, checkpoint_path: Optional[Path] = None):
        validate_every = int(config.training.metric_freq)
        checkpoint_every = int(config.training.checkpoint_freq)
        for epoch in range(int(config.training.epochs)):
            # Run one loop of training and record the average loss
            loss = self.epoch(training=True)
            self.logger.log({"toss": loss})

            # Run a validation loop on the val set
            loss = self.epoch(training=False)
            self.logger.log({"voss": loss})

            if (epoch == 0) or ((epoch + 1) % validate_every) == 0:
                inference = (
                    Inference.from_model(self.model)
                    .eval_with(self.dev_loader.dataset)
                    .denormalize_with(self.train_loader.dataset.normalization)
                )
                self.logger.log_r2_scores(inference.scores)
            if ((epoch + 1) % checkpoint_every) == 0 and checkpoint_path is not None:
                logger.info(f"saving checkpoint: {checkpoint_path}")
                # Make a checkpoint
                self.accelerator.save_state(checkpoint_path)

            self.logger.flush(epoch + 1)
        if checkpoint_path is not None:
            logger.info(f"saving checkpoint: {checkpoint_path}")
            # Make a checkpoint
            self.accelerator.save_state(checkpoint_path)
        self.accelerator.end_training()
