from enum import Enum, auto
from typing import List

from .. import config


class NormStrat(Enum):
    Z_SCORE = auto()
    MIN_MAX = auto()
    MAX_ABS = auto()
    ROBUST = auto()

    @staticmethod
    def from_str(strategy: str) -> "NormStrat":
        if strategy == "min_max":
            norm = NormStrat.MIN_MAX
        elif strategy == "robust":
            norm = NormStrat.ROBUST
        else:
            norm = NormStrat.Z_SCORE
        return norm


class Split(Enum):
    """
    Training data splits.

    80:10:10 for train:dev:test
    """

    TRAIN = auto()
    DEV = auto()
    TEST = auto()

    def __eq__(self, other):
        """Helps in comparison to strings."""
        if isinstance(other, str):
            return other == self._key()
        return super().__eq__(other)

    def __str__(self):
        """Maps to the well known key of the splits."""
        return self.name.lower()

    def __repr__(self):
        """Maps to the well known key of the splits."""
        return f"<Split.{self.name}>"

    @staticmethod
    def from_str(value: str) -> "Split":
        """Create a Split enum from a string key."""
        if isinstance(value, str):
            value = value.upper()
        return Split[value]

    @staticmethod
    def all() -> List["Split"]:
        """Return all splits available."""
        return [Split.TRAIN, Split.DEV, Split.TEST]

    @staticmethod
    def contains(list1, list2):
        return all(item in list2 for item in list1)


class Source(Enum):
    """Data sources used throughout the codebase for different experiments."""

    BINARY = auto()
    HYPERCUBE = auto()
    DGSM = auto()
    MIXED = auto()

    def __eq__(self, other):
        """Helps in comparison to strings."""
        if isinstance(other, str):
            return other == self._key()
        return super().__eq__(other)

    def __str__(self):
        """Maps to the well known key of the data source."""
        for k in config.data.sources:
            source = config.data.sources[k]
            if self.name.lower() == source.name:
                return source.key
        raise ValueError("could not find key from config")

    def __repr__(self):
        """Maps to the well known key of the data source."""
        return f"<Source.{self.name}>"

    def _key(self):
        for k in config.data.sources:
            source = config.data.sources[k]
            if self.name.lower() == source.name:
                return source.key
        raise ValueError("could not find key from config")

    @staticmethod
    def _get(value):
        for k in config.data.sources:
            source = config.data.sources[k]
            if value == source.key:
                return source

    @staticmethod
    def enabled() -> List["Source"]:
        """List of enabled sources - uses config.data.source.{name}.enabled."""
        results = []
        for k in config.data.sources:
            source = config.data.sources[k]
            if source.enabled:
                results.append(Source[source.name.upper()])
        return results

    @staticmethod
    def from_str(value: str) -> "Source":
        """Create a Source enum from a string key."""
        value = value.lower()
        for k in config.data.sources:
            source = config.data.sources[k]
            if value == source.key:
                if source.enabled:
                    return Source[source.name.upper()]
                else:
                    raise ValueError(f"{value} is not an enabled datasource")
        raise ValueError(f"{value} is an unknown data source key")

    @staticmethod
    def sampled() -> List["Source"]:
        """Data sources that were sampled from GCAM core, not derived."""
        results = []
        for k in config.data.sources:
            source = config.data.sources[k]
            if source.enabled and source.new_samples:
                results.append(Source[source.name.upper()])
        return results
