from accelerate import Accelerator
from accelerate.tracking import GeneralTracker

import wandb

from .. import config, logger


class Logger:
    log_dict = {}
    accelerator: Accelerator = None

    def __init__(self, accelerator):
        # Variable Initialization
        self.accelerator = accelerator

        kwargs = self._init_kwargs()
        run_config = {
            "device": f"{self.accelerator.device}",
            "train_source": config.training.train_source,
            "dev_source": config.training.dev_source,
            "learning_rate": float(config.training.learning_rate),
            "epochs": int(config.training.epochs),
            "hidden_size": int(config.model.hidden_size),
            "batch_size": int(config.training.batch_size),
            "depth": int(config.model.depth),
            "model": config.model.arch,
            "act_fn": config.model.act_fn,
            "norm_strat": config.training.norm_strat,
            "training_samples": int(config.training.samples),
            "dev_split": "test"
            if bool(int(config.training.evaluate_on_test))
            else "dev",
        }
        if "binary_fraction" in config.training:
            run_config |= {"binary_fraction": float(config.training.binary_fraction)}
        self.accelerator.init_trackers(
            project_name=config.project,
            config=run_config,
            init_kwargs=kwargs,
        )

    def _init_kwargs(self):
        kwargs = {}
        if bool(int(config.wandb.enabled)):
            if isinstance(config.tags, str):
                tags = config.tags.split(",")
            else:
                tags = config.tags
            kwargs |= {
                "wandb": {
                    "group": config.group,
                    "tags": tags,
                },
            }
        if bool(int(config.tensorboard.enabled)):
            pass
        return kwargs

    def log(self, stats: dict[str, float | wandb.Image | wandb.Video]):
        self.log_dict |= stats

    def log_r2_scores(self, scores):
        from ..evaluate import calculate_r2_aggs

        stats = calculate_r2_aggs(scores)

        self.log(stats)

    def flush(self, epoch):
        """Log the dictionary to WandB and clear it."""
        self.log_dict |= {"epoch": epoch}
        self.accelerator.log(self.log_dict)
        self.log_dict = {}

    def finish(self):
        """Finishes the run by syncing the logging data if possible."""
        pass


class ConsoleTracker(GeneralTracker):
    name = "console"
    requires_logging_directory = False

    def __init__(self, run_name: str):
        self.run_name = run_name
        # Initialize logging
        self.logger = logger

    @property
    def tracker(self):
        return None  # No external tracking tool used

    def store_init_configuration(self, values: dict):
        self.logger.info(f"{values}")

    def log(self, values: dict, step=None):
        if step is not None:
            self.logger.info(f"{step}: {values}")
        else:
            self.logger.info(f"{values}")
