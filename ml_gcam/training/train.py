import os
from pathlib import Path
from typing import List, Optional

from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

from .. import config
from ..data import GcamDataset, Source, Split, load_targets
from ..data.enums import NormStrat
from ..data.normalization import Normalization
from ..emulator import Arch
from .runner import Runner


def train_main(
    train_source: Source,
    dev_source: Source,
    targets_path: Path,
    strategy: NormStrat,
    checkpoint_path: Optional[Path] = None,
    samples: int = None,
    arch: Arch = Arch.DEEP,
) -> None:
    """Main emulator training loop."""
    dev_split = Split.TEST if bool(int(config.training.evaluate_on_test)) else Split.DEV

    if samples is not None:
        train_set = GcamDataset.from_targets(
            save_path=targets_path,
            experiment=train_source,
            split=Split.TRAIN,
            samples=samples,
        )
    else:
        train_set = GcamDataset.from_targets(
            save_path=targets_path,
            experiment=train_source,
            split=Split.TRAIN,
        )
    normalization = Normalization(outputs=train_set.outputs, strategy=strategy)
    train_set.with_normalization(normalization)
    os.environ["ML_GCAM__TRAINING__SAMPLES"] = str(len(train_set))
    config.reload()

    dev_set = GcamDataset.from_targets(
        save_path=targets_path,
        experiment=dev_source,
        split=dev_split,
    )
    dev_set.with_normalization(normalization)

    run_training(
        train_set=train_set,
        dev_set=dev_set,
        checkpoint_path=checkpoint_path,
        arch=arch,
    )


def run_training(
    train_set: GcamDataset,
    dev_set: GcamDataset,
    checkpoint_path: Optional[Path] = None,
    arch: Arch = Arch.DEEP,
) -> nn.Module:
    # Initialize the training set and a dataloader to iterate over the dataset
    train_loader = DataLoader(
        train_set,
        batch_size=int(config.training.batch_size),
        shuffle=True,
    )
    dev_loader = DataLoader(
        dev_set,
        batch_size=int(config.training.batch_size),
        shuffle=False,
    )
    arch = Arch.from_str(config.model.arch)
    emulator = arch.init_model()
    optimizer = AdamW(emulator.parameters(), lr=float(config.training.learning_rate))

    # Create a runner that will handle training and checkpointing
    runner = Runner(
        model=emulator,
        train_loader=train_loader,
        dev_loader=dev_loader,
        optimizer=optimizer,
    )
    runner.train(epochs=int(config.training.epochs), checkpoint_path=checkpoint_path)
    return emulator


def train_sample_size(
    train_source: Source,
    dev_source: Source,
    targets_path: Path,
    splits: List[float],
    checkpoint_path: Optional[Path] = None,
) -> None:
    """Training loop for sample size experiment."""
    os.environ["ML_GCAM__TRAINING__TRAIN_SOURCE"] = str(train_source)
    os.environ["ML_GCAM__TRAINING__DEV_SOURCE"] = str(dev_source)
    dev_targets = load_targets(
        save_path=targets_path,
        experiments=[dev_source],
        split=Split.DEV,
    )
    dev_set = GcamDataset(dev_targets)
    total_samples = config.sample_size.total_samples
    for split in splits:
        samples = int(total_samples * float(split))
        os.environ["ML_GCAM__TRAINING__SAMPLES"] = str(samples)
        config.reload()
        save_path = None
        if checkpoint_path is not None:
            save_path = checkpoint_path / f"{samples}_samples"
        train_targets = load_targets(
            save_path=targets_path,
            experiments=[train_source],
            split=Split.TRAIN,
            samples=samples,
        )
        train_set = GcamDataset(train_targets)
        normalization = Normalization(outputs=train_set.outputs)
        train_set.with_normalization(normalization)
        dev_set.with_normalization(normalization)
        run_training(train_set, dev_set, save_path)


def train_cartesian(
    train_sources: List[Source],
    dev_sources: List[Source],
    targets_path: Path,
    checkpoint_path: Optional[Path],
) -> None:
    """Compare all combinations of train and dev sets."""
    for train in train_sources:
        train_targets = load_targets(
            save_path=targets_path,
            experiments=[train],
            split=Split.TRAIN,
        )
        train_set = GcamDataset(train_targets)
        normalization = Normalization(outputs=train_set.outputs)
        train_set.with_normalization(normalization)
        for dev in dev_sources:
            os.environ["ML_GCAM__TRAINING__TRAIN_SOURCE"] = str(train)
            os.environ["ML_GCAM__TRAINING__DEV_SOURCE"] = str(dev)
            config.reload()
            save_path = None
            if checkpoint_path is not None:
                save_path = checkpoint_path / f"{train}_vs_{dev}"
            dev_targets = load_targets(
                save_path=targets_path,
                experiments=[dev],
                split=Split.DEV,
            )
            dev_set = GcamDataset(dev_targets)
            dev_set.with_normalization(normalization)
            run_training(train_set, dev_set, save_path)


def train_mixed_fraction(
    dev_source: Source,
    targets_path: Path,
    fractions: List[float],
    checkpoint_path: Optional[Path] = None,
) -> None:
    """Training loop for mixed fraction experiment."""
    dev_set = GcamDataset.from_targets(
        save_path=targets_path,
        experiment=dev_source,
        split=Split.DEV,
    )
    for fraction in fractions:
        os.environ["ML_GCAM__TRAINING__BINARY_FRACTION"] = str(fraction)
        config.reload()
        save_path = None
        if checkpoint_path is not None:
            name = str(fraction).replace(".", "_")
            save_path = checkpoint_path / f"binary_{name}_fraction_binary"
        train_set = GcamDataset.from_targets(
            save_path=targets_path,
            experiment=Source.MIXED,
            split=Split.TRAIN,
            fraction_binary=fraction,
        )
        normalization = Normalization(outputs=train_set.outputs)
        train_set.with_normalization(normalization)
        dev_set.with_normalization(normalization)
        run_training(train_set, dev_set, save_path)
