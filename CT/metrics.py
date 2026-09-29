"""Image-quality metrics used by the CT experiments."""

from __future__ import annotations

import numpy as np
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity


def compute_image_metrics(
    reference: np.ndarray, estimate: np.ndarray
) -> tuple[float, float]:
    """Return PSNR and SSIM for a 2-D image using the reference image range."""
    if reference.shape != estimate.shape:
        raise ValueError(
            "reference and estimate must have matching shapes, got "
            f"{reference.shape} and {estimate.shape}"
        )

    data_range = float(reference.max() - reference.min())
    if data_range <= 0.0:
        raise ValueError("The reference image must have a non-zero data range.")
    psnr = peak_signal_noise_ratio(reference, estimate, data_range=data_range)
    ssim = structural_similarity(reference, estimate, data_range=data_range)
    return float(psnr), float(ssim)
