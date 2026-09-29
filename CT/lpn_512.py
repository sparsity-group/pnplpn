"""Input-convex learned proximal network for 512-by-512 CT images."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn


class HuberizedReLU(nn.Module):
    """Continuously differentiable ReLU with a quadratic transition at zero."""

    def __init__(self, delta: float = 0.1) -> None:
        super().__init__()
        if delta <= 0:
            raise ValueError("delta must be positive")
        self.delta = float(delta)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.where(
            x <= 0,
            torch.zeros_like(x),
            torch.where(
                x < self.delta,
                x.square() / (2 * self.delta),
                x - self.delta / 2,
            ),
        )


class LPN(nn.Module):
    """Learn the gradient of a convex scalar potential over CT images."""

    def __init__(
        self,
        in_dim: int,
        alpha: float,
        hidden: int,
        activation: str = "softplus",
        beta: float | None = None,
        huber_delta: float = 0.1,
    ) -> None:
        super().__init__()
        self.alpha = alpha

        self.lin = nn.ModuleList(
            [
                nn.Conv2d(in_dim, hidden, 4, bias=True, stride=2, padding=1),
                nn.Conv2d(hidden, hidden, 3, bias=False, stride=1, padding=1),
                nn.Conv2d(hidden, hidden, 4, bias=False, stride=2, padding=1),
                nn.Conv2d(hidden, hidden, 3, bias=False, stride=1, padding=1),
                nn.Conv2d(hidden, hidden, 4, bias=False, stride=2, padding=1),
                nn.Conv2d(hidden, hidden, 4, bias=False, stride=2, padding=1),
                nn.Conv2d(hidden, hidden, 3, bias=False, stride=1, padding=1),
            ]
        )

        self.lin_final = nn.Linear(hidden, 1)

        self.res = nn.ModuleList(
            [
                nn.Conv2d(in_dim, hidden, 3, stride=1, padding=1),
                nn.Conv2d(in_dim, hidden, 4, stride=2, padding=1),
                nn.Conv2d(in_dim, hidden, 3, stride=1, padding=1),
                nn.Conv2d(in_dim, hidden, 4, stride=2, padding=1),
                nn.Conv2d(in_dim, hidden, 4, stride=2, padding=1),
            ]
        )

        activation = activation.lower()
        if activation == "softplus":
            if beta is None:
                raise ValueError("beta is required when activation='softplus'")
            self.act = nn.Softplus(beta=beta)
        elif activation == "relu":
            self.act = nn.ReLU()
        elif activation == "leakyrelu":
            self.act = nn.LeakyReLU()
        elif activation in {"huberizedrelu", "huberized_relu"}:
            self.act = HuberizedReLU(delta=huber_delta)
        else:
            raise ValueError(
                f"Unsupported activation {activation!r}; expected 'softplus', "
                "'relu', 'leakyrelu', or 'huberizedrelu'"
            )

    def scalar(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluate the learned scalar potential before the quadratic term."""
        if x.ndim != 4 or x.shape[-2:] != (512, 512):
            raise ValueError(
                f"expected input shape (batch, channels, 512, 512), got {x.shape}"
            )

        batch_size = x.shape[0]
        y = self.act(self.lin[0](x))
        sizes = (256, 256, 128, 128, 64, 32, 32)

        residual_index = 0
        for layer_index, (core, sz) in enumerate(
            zip(self.lin[1:], sizes[:-1]), start=1
        ):
            y = core(y)
            # Every internal layer has an input skip except the final one;
            # forward() already supplies the alpha * I output connection.
            if layer_index != len(self.lin) - 1:
                x_scaled = nn.functional.interpolate(
                    x,
                    (sz, sz),
                    mode="bilinear",
                    align_corners=False,
                )
                y = y + self.res[residual_index](x_scaled)
                residual_index += 1
            y = self.act(y)

        if y.shape[2:] != (sizes[-1], sizes[-1]):
            raise RuntimeError(f"unexpected feature-map shape: {y.shape}")
        y = y.mean(dim=(2, 3))
        return self.lin_final(y.reshape(batch_size, -1))

    def init_weights(self, mean: float, std: float) -> None:
        """Initialize the nonnegative constrained weights."""
        with torch.no_grad():
            for core in self.lin[1:]:
                for layer in core.modules():
                    if isinstance(layer, (nn.Conv2d, nn.Linear)):
                        layer.weight.data.normal_(mean, std).exp_()
            self.lin_final.weight.data.normal_(mean, std).exp_()

    def wclip(self) -> None:
        """Project constrained weights onto the nonnegative orthant."""
        with torch.no_grad():
            # lin[0] acts directly on the input and is intentionally unconstrained.
            # All later convolutions and the final linear layer must have
            # nonnegative weights to preserve convexity.
            for core in self.lin[1:]:
                for layer in core.modules():
                    if isinstance(layer, (nn.Conv2d, nn.Linear)):
                        layer.weight.clamp_(min=0.0)
            self.lin_final.weight.clamp_(min=0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return the gradient of the strongly convex learned potential."""
        with torch.enable_grad():
            if not x.requires_grad:
                x.requires_grad_(True)
            scalar = self.scalar(x)
            gradient = torch.autograd.grad(
                scalar.sum(), x, retain_graph=True, create_graph=True
            )[0]
        return gradient + self.alpha * x

    def apply_numpy(self, image: np.ndarray) -> np.ndarray:
        """Apply the network to one NumPy image."""
        if image.shape[:2] != (512, 512):
            raise ValueError(f"expected a 512-by-512 image, got {image.shape}")

        device = next(self.parameters()).device
        input_dimensions = image.ndim
        if input_dimensions == 2:
            image = image[:, :, np.newaxis]
        image = np.transpose(image, (2, 0, 1))
        image_tensor = torch.as_tensor(
            image, dtype=torch.float32, device=device
        ).unsqueeze(0)
        with torch.no_grad():
            output = self(image_tensor)
        output_array = output[0].detach().cpu().numpy().transpose(1, 2, 0)
        if input_dimensions == 2:
            output_array = np.squeeze(output_array, axis=2)
        return output_array
