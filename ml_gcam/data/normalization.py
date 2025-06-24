
import numpy as np

from .. import config
from .enums import NormStrat

DEFAULT_NORM_STRAT = NormStrat.from_str(config.training.norm_strat)
SMALL = 1e-3


class Normalization:
    def __init__(self, outputs: np.ndarray, strategy: NormStrat = DEFAULT_NORM_STRAT):
        self.strategy = strategy
        self.outputs = outputs
        self._init_stats()

    def _init_stats(self):
        if self.strategy == NormStrat.ROBUST:
            self.median = np.median(self.outputs, axis=0)
            q1 = np.percentile(self.outputs, 25, axis=0)
            q3 = np.percentile(self.outputs, 75, axis=0)
            self.iqr = q3 - q1
            self.iqr[self.iqr == 0] = SMALL
        elif self.strategy == NormStrat.Z_SCORE:
            self.mean = np.mean(self.outputs, axis=0)
            self.std = np.std(self.outputs, axis=0)
            self.std[self.std == 0] = SMALL
        elif self.strategy == NormStrat.MIN_MAX:
            self.min = np.min(self.outputs, axis=0)
            self.max = np.max(self.outputs, axis=0)
        else:
            raise NotImplementedError(f"{self.strategy} not supported yet")

    def normalize(self, data: np.ndarray):
        if self.strategy == NormStrat.ROBUST:
            scaled_data = (data - self.median) / self.iqr
        elif self.strategy == NormStrat.Z_SCORE:
            scaled_data = (data - self.mean) / self.std
        elif self.strategy == NormStrat.MIN_MAX:
            scaled_data = (data - self.min) / (self.max - self.min)
        else:
            raise NotImplementedError(f"{self.strategy} not supported yet")
        return scaled_data

    def denormalize(self, data: np.ndarray):
        if self.strategy == NormStrat.ROBUST:
            self.iqr[self.iqr == 0] = 1  # handle divide by zero
            unscaled_data = data * self.iqr + self.median
        elif self.strategy == NormStrat.Z_SCORE:
            unscaled_data = data * self.std + self.mean
        elif self.strategy == NormStrat.MIN_MAX:
            unscaled_data = data * (self.max - self.min) + self.min
        else:
            raise NotImplementedError(f"{self.strategy} not supported yet")

        return unscaled_data
