from enum import Enum, auto

import torch
from torch import nn
from torchtyping import TensorType

from .. import config


class Arch(Enum):
    DEEP = auto()
    LINEAR = auto()

    @staticmethod
    def from_str(arch: str) -> "Arch":
        if arch == "deep":
            arch = Arch.DEEP
        elif arch == "linear":
            arch = Arch.LINEAR
        else:
            raise NotImplentedError(f"Model architecture not implemented: {arch}")
        return arch

    def init_model(self):
        """Instantiate a model from given archtecture."""
        if self == Arch.DEEP:
            return DNN(
                in_size=len(config.data.input_keys),
                hidden_size=int(config.model.hidden_size),
                depth=int(config.model.depth),
                n_heads=config.data.n_dimensions,
                n_features=len(config.data.output_keys),
                act_fn=config.model.act_fn,
            )
        elif self == Arch.LINEAR:
            return LL(
                in_size=len(config.data.input_keys),
                n_heads=config.data.n_dimensions,
                n_features=len(config.data.output_keys),
            )



class DNN(nn.Module):
    """A simple deep neural network capable of processing GCAM data."""

    def __init__(
        self, 
        in_size: int, 
        hidden_size: int, 
        depth: int,
        n_heads: int, 
        n_features: int,
        act_fn="relu",
    ):
        super().__init__()

        # Define the activation function and the linear functions
        assert depth >= 1, f"should be positive ({depth=})"
        self.n_heads = n_heads
        self.n_features = n_features
        if act_fn == "leaky":
            stack = [nn.Linear(in_size, hidden_size), nn.LeakyReLU()]
        elif act_fn == "tanh":
            stack = [nn.Linear(in_size, hidden_size), nn.Tanh()]
        else:  # relu
            stack = [nn.Linear(in_size, hidden_size), nn.ReLU()]
        self.linear = nn.Sequential()
        for _ in range(depth - 1):
            if act_fn == "relu":
                hidden = [nn.Linear(hidden_size, hidden_size), nn.ReLU()]
            elif act_fn == "leaky":
                hidden = [nn.Linear(hidden_size, hidden_size), nn.LeakyReLU()]
            else:
                hidden = [nn.Linear(hidden_size, hidden_size), nn.Tanh()]
            stack.extend(hidden)
        output_layer = nn.Linear(hidden_size, (n_heads*n_features))
        stack.extend([output_layer])
        self.stack = nn.Sequential(*stack)


    def forward(
        self,
        x: TensorType["batch", "feature_size"],
    ) -> TensorType["batch", "out_size"]:
        batch, _ = x.shape

        x = self.stack(x)
        out = x.reshape(-1, self.n_heads, self.n_features)
        return out


class LL(nn.Module):
    """Linear regression model for processing GCAM data."""

    def __init__(self, in_size: int, n_heads: int, n_features: int):
        super().__init__()

        self.in_size = in_size
        self.n_heads = n_heads
        self.n_features = n_features

        self.flatten = nn.Flatten()
        self.out_linear = nn.Linear(in_size, n_heads * n_features)

    def forward(
        self,
        x: TensorType["batch", "feature_size"],
    ) -> TensorType["batch", "out_size"]:
        batch, _ = x.shape

        # flatten x
        x = self.flatten(x)

        # Send x through the output layer
        out = self.out_linear(x)

        # Reshape the output to be a tensor of size (batch, n_heads, n_features)
        out = out.reshape(batch, self.n_heads, self.n_features)

        return out
