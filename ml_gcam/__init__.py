import polars as pl
import torch

from .config import config
from .logging import logger

if "random_seed" in config:
    pl.set_random_seed(config.random_seed)
    torch.manual_seed(config.random_seed)
else:
    pl.set_random_seed(42)
    torch.manual_seed(42)

__all__ = [logger, config]
