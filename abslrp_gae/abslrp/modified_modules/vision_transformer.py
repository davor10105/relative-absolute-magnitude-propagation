import torch
from torch import nn


class LayerNormalization(nn.Module):
    def __init__(self, eps: float, weight: nn.Parameter, bias: nn.Parameter) -> None:
        super().__init__()
        self.eps = eps
        self.weight = weight
        self.bias = bias

    def forward(self, x: torch.tensor) -> torch.tensor:
        x_mean = x.mean(-1, keepdim=True).detach()
        x_var = x.var(-1, keepdim=True, unbiased=False).detach()

        xm = x - x_mean
        h1 = xm / (x_var + self.eps).pow(0.5)
        h2 = h1 * self.weight + self.bias

        return h2


class CLSTokenPool(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.tensor) -> torch.tensor:
        return x[:, 0]


class QKVLayer(nn.Module):
    def __init__(self, qkv: nn.Module, num_heads: int, head_dim: int) -> None:
        super().__init__()
        self.qkv = qkv
        self.num_heads = num_heads
        self.head_dim = head_dim

    def forward(self, x: torch.tensor) -> torch.tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        return qkv


class QKMultiply(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, q: torch.tensor, k: torch.tensor) -> torch.tensor:
        return q @ k.transpose(-2, -1)


class SoftmaxAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.tensor) -> torch.tensor:
        x_max = x.max(-1, keepdim=True)[0].detach()
        xm = x - x_max
        xme = xm.exp()
        h = xme / xme.sum(-1, keepdim=True).detach()

        return h


class AttentionVMultiply(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, attn: torch.tensor, v: torch.tensor) -> torch.tensor:
        return attn @ v


class PositionEmbed(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.tensor, pos_embed: torch.tensor) -> torch.tensor:
        return x + pos_embed
