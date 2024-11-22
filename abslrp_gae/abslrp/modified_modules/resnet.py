import torch
from torch import nn


class ResidualAddition(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x1: torch.tensor, x2: torch.tensor) -> torch.tensor:
        return x1 + x2
