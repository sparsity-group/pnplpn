"""Input-convex LPN denoiser for 320 x 320 magnitude MR images."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class HuberizedReLU(nn.Module):
    def __init__(self, delta: float = 0.1):
        super().__init__()
        if delta <= 0:
            raise ValueError("delta must be positive")
        self.delta = float(delta)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.where(
            x <= 0,
            torch.zeros_like(x),
            torch.where(x < self.delta, x.square() / (2 * self.delta), x - self.delta / 2),
        )


class LPN(nn.Module):
    def __init__(
        self,
        in_dim: int,
        alpha: float,
        hidden: int,
        activation: str = "huberizedrelu",
        beta: float | None = None,
        huber_delta: float = 0.1,
    ):
        super().__init__()
        self.alpha = float(alpha)
        self.lin = nn.ModuleList([
            nn.Conv2d(in_dim, hidden, 4, stride=2, padding=1),
            nn.Conv2d(hidden, hidden, 3, padding=1, bias=False),
            nn.Conv2d(hidden, hidden, 4, stride=2, padding=1, bias=False),
            nn.Conv2d(hidden, hidden, 3, padding=1, bias=False),
            nn.Conv2d(hidden, hidden, 4, stride=2, padding=1, bias=False),
            nn.Conv2d(hidden, hidden, 4, stride=2, padding=1, bias=False),
            nn.Conv2d(hidden, hidden, 3, padding=1, bias=False),
        ])
        # Input skips for all internal layers except the last, as in the CT model.
        self.res = nn.ModuleList([
            nn.Conv2d(in_dim, hidden, 3, stride=1, padding=1)
            for _ in range(5)
        ])
        self.lin_final = nn.Linear(hidden, 1)

        name = activation.lower()
        if name == "softplus":
            if beta is None:
                raise ValueError("beta is required for softplus")
            self.act = nn.Softplus(beta=beta)
        elif name == "relu":
            self.act = nn.ReLU()
        elif name == "leakyrelu":
            self.act = nn.LeakyReLU()
        elif name in {"huberizedrelu", "huberized_relu"}:
            self.act = HuberizedReLU(huber_delta)
        else:
            raise ValueError(f"Unsupported activation: {activation}")

    def scalar(self, x: torch.Tensor) -> torch.Tensor:
        y = self.act(self.lin[0](x))
        residual_index = 0
        for layer_index, core in enumerate(self.lin[1:], start=1):
            y = core(y)
            if layer_index != len(self.lin) - 1:
                x_scaled = F.interpolate(x, y.shape[-2:], mode="bilinear", align_corners=False)
                y = y + self.res[residual_index](x_scaled)
                residual_index += 1
            y = self.act(y)
        return self.lin_final(y.mean(dim=(-2, -1)))

    def wclip(self) -> None:
        with torch.no_grad():
            for core in self.lin[1:]:
                core.weight.clamp_(min=0.0)
            self.lin_final.weight.clamp_(min=0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.enable_grad():
            if not x.requires_grad:
                x.requires_grad_(True)
            gradient = torch.autograd.grad(
                self.scalar(x).sum(), x, retain_graph=True, create_graph=True
            )[0]
        return gradient + self.alpha * x
