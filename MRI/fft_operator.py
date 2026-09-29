"""Centered Cartesian MRI operators implemented with ``torch.fft``."""

from __future__ import annotations

import torch


def fft2c(image: torch.Tensor) -> torch.Tensor:
    """Centered orthonormal FFT over the last two dimensions."""
    image = torch.fft.ifftshift(image, dim=(-2, -1))
    return torch.fft.fftshift(torch.fft.fft2(image, norm="ortho"), dim=(-2, -1))


def ifft2c(kspace: torch.Tensor) -> torch.Tensor:
    """Centered orthonormal inverse FFT over the last two dimensions."""
    kspace = torch.fft.ifftshift(kspace, dim=(-2, -1))
    return torch.fft.fftshift(torch.fft.ifft2(kspace, norm="ortho"), dim=(-2, -1))


def cartesian_mask(
    shape: tuple[int, int],
    acceleration: int = 4,
    center_fraction: float = 0.08,
    seed: int = 0,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Create a reproducible 1-D random Cartesian phase-encode mask."""
    height, width = shape
    if acceleration < 1:
        raise ValueError("acceleration must be at least one")
    if not 0 < center_fraction < 1:
        raise ValueError("center_fraction must lie between zero and one")
    if acceleration == 1:
        return torch.ones((height, width), dtype=torch.bool, device=device)
    num_low = max(1, round(width * center_fraction))
    target = max(num_low, round(width / acceleration))
    generator = torch.Generator().manual_seed(seed)
    center_start = (width - num_low) // 2
    selected = torch.zeros(width, dtype=torch.bool)
    selected[center_start:center_start + num_low] = True
    candidates = torch.arange(width)[~selected]
    extra = target - num_low
    if extra > 0:
        chosen = candidates[torch.randperm(len(candidates), generator=generator)[:extra]]
        selected[chosen] = True
    return selected[None, :].expand(height, width).to(device=device)


class MaskedFFT:
    """Single-coil masked Fourier operator A and Hermitian adjoint A*."""

    def __init__(self, mask: torch.Tensor):
        if mask.ndim != 2:
            raise ValueError("mask must be two-dimensional")
        self.mask = mask.bool()

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return fft2c(image) * self.mask

    def adjoint(self, kspace: torch.Tensor) -> torch.Tensor:
        return ifft2c(kspace * self.mask)

    def zero_filled(self, kspace: torch.Tensor) -> torch.Tensor:
        return self.adjoint(kspace).real


def estimate_sensitivities(
    kspace: torch.Tensor,
    output_shape: tuple[int, int] | None = None,
    calibration_lines: int = 24,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Estimate RSS-normalized coil maps from central calibration lines."""
    if kspace.ndim != 3:
        raise ValueError("multicoil kspace must have shape (coils, height, width)")
    width = kspace.shape[-1]
    if not 1 <= calibration_lines <= width:
        raise ValueError("calibration_lines must fit within k-space width")
    center_start = (width - calibration_lines) // 2
    calibration = torch.zeros_like(kspace)
    calibration[..., center_start:center_start + calibration_lines] = (
        kspace[..., center_start:center_start + calibration_lines]
    )
    low_resolution = ifft2c(calibration)
    if output_shape is not None:
        target_height, target_width = output_shape
        height, image_width = low_resolution.shape[-2:]
        top = (height - target_height) // 2
        left = (image_width - target_width) // 2
        low_resolution = low_resolution[
            ..., top:top + target_height, left:left + target_width
        ]
    rss = low_resolution.abs().square().sum(dim=0).sqrt()
    sensitivities = low_resolution / rss.clamp_min(eps)
    norm = sensitivities.abs().square().sum(dim=0).sqrt()
    return sensitivities / norm.clamp_min(eps)


class MultiCoilMaskedFFT:
    """Sensitivity-encoded Cartesian MRI operator and its Hermitian adjoint."""

    def __init__(self, mask: torch.Tensor, sensitivities: torch.Tensor):
        if mask.ndim != 2:
            raise ValueError("mask must be two-dimensional")
        if sensitivities.ndim != 3 or sensitivities.shape[-2:] != mask.shape:
            raise ValueError(
                "sensitivities must have shape (coils, height, width) matching mask"
            )
        self.mask = mask.bool()
        self.sensitivities = sensitivities

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return fft2c(self.sensitivities * image) * self.mask

    def adjoint(self, kspace: torch.Tensor) -> torch.Tensor:
        coil_images = ifft2c(kspace * self.mask)
        return (self.sensitivities.conj() * coil_images).sum(dim=0)

    def zero_filled(self, kspace: torch.Tensor) -> torch.Tensor:
        return self.adjoint(kspace).real
