"""Reconstruct fastMRI knee slices with multicoil SENSE + the trained LPN."""

from __future__ import annotations

import argparse
import ast
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from tqdm import tqdm

from fastmri import load_dataset
from fft_operator import (
    MultiCoilMaskedFFT,
    cartesian_mask,
    estimate_sensitivities,
    fft2c,
    ifft2c,
)
from lpn_320 import LPN


DATA_ROOT = Path("../../shared/datasets/fastmri/multicoil_knee")
GPU_INDEX = 0  # choose GPU 0 or GPU 1
MODEL_LOCATION = Path("results/20260904_144820_PDT")
IMAGE_INDEX = 10
NUM_IMAGES = 1
IMAGE_INDICES: list[int] = [10, 11, 12, 16]  # e.g. [2, 10, 47]; overrides the settings above
ACCELERATION = 2
CENTER_FRACTION = 0.1

CALIBRATION_LINES = 30
ITERATIONS = 100
STEP_SIZE = 0.99
DENOISER_STRENGTH = 1
NOISE_STD = 0.1 # relative to the RMS of acquired complex k-space samples
SEED = 0
DEVICE = f"cuda:{GPU_INDEX}"

# GAMMA = 1 uses the standard LPN. Values strictly between 0 and 1 use the
# relaxed proximal mapping from the CT experiment.
GAMMA = 1
RELAXED_LBFGS_ITERATIONS = 5000
INNER_TOLERANCE_C = 1.0
INNER_TOLERANCE_P = 1.6
INNER_TOLERANCE_OVERRIDE: float | None = 1e-2


def load_model(model_location: Path, device: torch.device) -> LPN:
    checkpoint_path = model_location / "model.pt" if model_location.is_dir() else model_location
    parameter_path = checkpoint_path.parent / "training_parameters.txt"
    if not checkpoint_path.is_file() or not parameter_path.is_file():
        raise FileNotFoundError(f"Expected model.pt and training_parameters.txt at {checkpoint_path.parent}")
    parameters = {}
    for line in parameter_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            key, value = line.split("=", 1)
            parameters[key.strip()] = ast.literal_eval(value.strip())
    options = dict(
        in_dim=int(parameters.get("IN_DIM", 1)),
        alpha=float(parameters["ALPHA"]),
        hidden=int(parameters["HIDDEN"]),
        activation=str(parameters["ACTIVATION"]),
    )
    if options["activation"].lower() == "softplus":
        options["beta"] = float(parameters["BETA"])
    elif options["activation"].lower() in {"huberizedrelu", "huberized_relu"}:
        options["huber_delta"] = float(parameters["HUBER_DELTA"])
    model = LPN(**options)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint.get("model_state_dict", checkpoint))
    return model.eval().to(device)


def metrics(clean: torch.Tensor, estimate: torch.Tensor) -> tuple[float, float]:
    reference = clean.detach().cpu().numpy()
    prediction = estimate.clamp(0, 1).detach().cpu().numpy()
    return (
        float(peak_signal_noise_ratio(reference, prediction, data_range=1.0)),
        float(structural_similarity(reference, prediction, data_range=1.0)),
    )


def center_crop(image: torch.Tensor, shape: tuple[int, int]) -> torch.Tensor:
    """Crop the final two dimensions around their center."""
    target_height, target_width = shape
    height, width = image.shape[-2:]
    if target_height > height or target_width > width:
        raise ValueError(f"cannot crop {height} x {width} data to {shape}")
    top = (height - target_height) // 2
    left = (width - target_width) // 2
    return image[..., top:top + target_height, left:left + target_width]


def load_multicoil_kspace(
    path: Path,
    slice_index: int,
    device: torch.device,
) -> torch.Tensor:
    """Load one fully sampled raw fastMRI multicoil slice."""
    with h5py.File(path, "r") as handle:
        if "kspace" not in handle:
            raise KeyError(f"multicoil k-space is missing from {path}")
        kspace = torch.from_numpy(np.asarray(handle["kspace"][slice_index]))
    return kspace.to(device)


def apply_lpn(
    model: LPN,
    image: torch.Tensor,
    gamma: float,
    outer_iteration: int,
    tolerance_c: float = INNER_TOLERANCE_C,
    tolerance_p: float = INNER_TOLERANCE_P,
    tolerance_override: float | None = INNER_TOLERANCE_OVERRIDE,
) -> torch.Tensor:
    """Apply the standard or gamma-relaxed LPN proximal mapping.

    For ``gamma < 1``, solve

        min_w gamma/2 ||w||^2 + (1-gamma) Psi(w) - <image, w>

    where ``grad Psi`` is the standard LPN, and then return

        (image - gamma*w) / (1-gamma).
    """
    if not 0.0 < gamma <= 1.0:
        raise ValueError("gamma must lie in (0, 1]")
    if outer_iteration < 0:
        raise ValueError("outer iteration must be non-negative")
    if tolerance_c <= 0.0:
        raise ValueError("inner tolerance constant c must be positive")
    if tolerance_p <= 1.5:
        raise ValueError("inner tolerance exponent p must be greater than 3/2")
    if tolerance_override is not None and tolerance_override <= 0.0:
        raise ValueError("inner tolerance override must be positive")
    image_dtype = image.dtype
    model_dtype = next(model.parameters()).dtype
    image_batch = image[None, None].to(dtype=model_dtype)
    if gamma == 1.0:
        output = model(image_batch).detach().squeeze(0).squeeze(0)
        return output.to(dtype=image_dtype)

    w = image_batch.clone().detach().requires_grad_(True)
    inner_tolerance = (
        tolerance_override
        if tolerance_override is not None
        else tolerance_c / (1.0 + outer_iteration) ** tolerance_p
    )
    optimizer = torch.optim.LBFGS(
        [w],
        max_iter=(
            RELAXED_LBFGS_ITERATIONS
            if tolerance_override is not None
            else min(20, RELAXED_LBFGS_ITERATIONS)
        ),
        tolerance_grad=inner_tolerance if tolerance_override is not None else 0.0,
        tolerance_change=1e-9 if tolerance_override is not None else 0.0,
        history_size=10,
        line_search_fn="strong_wolfe",
    )

    def closure() -> torch.Tensor:
        optimizer.zero_grad(set_to_none=True)
        # Psi has gradient grad(psi_theta) + alpha*w, which is model.forward(w).
        psi = model.scalar(w).sum() + 0.5 * model.alpha * w.square().sum()
        objective = (
            0.5 * gamma * w.square().sum()
            + (1.0 - gamma) * psi
            - (image_batch * w).sum()
        )
        gradient = torch.autograd.grad(objective, w)[0]
        w.grad = gradient
        return objective

    if tolerance_override is not None:
        optimizer.step(closure)
    else:
        completed_inner_iterations = 0
        while True:
            closure()
            inner_gradient_norm = float(
                torch.linalg.vector_norm(w.grad).detach().cpu()
            )
            if inner_gradient_norm < inner_tolerance:
                break
            if completed_inner_iterations >= RELAXED_LBFGS_ITERATIONS:
                raise RuntimeError(
                    "relaxed LPN inner solve did not converge: "
                    f"||grad Q||={inner_gradient_norm:.6g}, "
                    f"required < {inner_tolerance:.6g} at outer iteration "
                    f"K={outer_iteration}"
                )
            chunk_size = min(
                20, RELAXED_LBFGS_ITERATIONS - completed_inner_iterations
            )
            optimizer.param_groups[0]["max_iter"] = chunk_size
            optimizer.param_groups[0]["max_eval"] = chunk_size * 5 // 4
            optimizer.step(closure)
            completed_inner_iterations += chunk_size
    with torch.no_grad():
        relaxed = (image_batch - gamma * w) / (1.0 - gamma)
    return relaxed.squeeze(0).squeeze(0).to(dtype=image_dtype)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_location", nargs="?", type=Path, default=MODEL_LOCATION)
    parser.add_argument("--image-index", type=int, default=IMAGE_INDEX)
    parser.add_argument(
        "--num-images", type=int, default=NUM_IMAGES,
        help="number of consecutive validation images to reconstruct",
    )
    parser.add_argument(
        "--image-indices", type=int, nargs="+", default=IMAGE_INDICES,
        help="specific validation-image indices (overrides --image-index and --num-images)",
    )
    parser.add_argument("--iterations", type=int, default=ITERATIONS)
    parser.add_argument("--acceleration", type=int, default=ACCELERATION)
    parser.add_argument("--center-fraction", type=float, default=CENTER_FRACTION)
    parser.add_argument("--step-size", type=float, default=STEP_SIZE)
    parser.add_argument(
        "--denoiser-strength",
        type=float,
        default=DENOISER_STRENGTH,
        help=(
            "weight of the LPN output in the denoising step; 0 uses the "
            "identity and 1 uses the full denoiser"
        ),
    )
    parser.add_argument("--noise-std", type=float, default=NOISE_STD)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--gamma",
        type=float,
        default=GAMMA,
        help="1 for standard PnP; a value in (0,1) for the relaxed proximal map",
    )
    return parser.parse_args()


def main(
    args: argparse.Namespace | None = None,
    *,
    use_data_fidelity: bool = True,
) -> None:
    args = parse_args() if args is None else args
    if not 0.0 < args.gamma <= 1.0:
        raise ValueError("gamma must lie in (0, 1]")
    if args.num_images < 1:
        raise ValueError("num_images must be positive")
    requested_indices = getattr(args, "image_indices", None)
    if requested_indices is not None and any(index < 0 for index in requested_indices):
        raise ValueError("image indices must be non-negative")
    if not 0.0 < args.step_size <= 1.0:
        raise ValueError("step size must lie in (0, 1]")
    if not 0.0 <= args.denoiser_strength <= 1.0:
        raise ValueError("denoiser strength must lie in [0, 1]")
    if args.noise_std < 0.0:
        raise ValueError("noise standard deviation cannot be negative")

    if GPU_INDEX not in (0, 1):
        raise ValueError(f"GPU_INDEX must be 0 or 1, got {GPU_INDEX}")
    if not torch.cuda.is_available() or GPU_INDEX >= torch.cuda.device_count():
        raise RuntimeError(
            f"GPU_INDEX={GPU_INDEX} was requested, but only "
            f"{torch.cuda.device_count()} CUDA device(s) are available"
        )
    torch.cuda.set_device(GPU_INDEX)
    device = torch.device(DEVICE)
    dataset = load_dataset(DATA_ROOT, "val")
    model = load_model(args.model_location, device)

    method_name = "LPN" if args.gamma == 1.0 else f"Relaxed LPN (gamma={args.gamma:g})"
    if args.denoiser_strength != 1.0:
        method_name += f" (strength={args.denoiser_strength:g})"
    if not use_data_fidelity:
        method_name += " only"
    output_dir = (args.model_location if args.model_location.is_dir() else args.model_location.parent)
    output_dir.mkdir(parents=True, exist_ok=True)
    image_indices = requested_indices or list(
        range(args.image_index, args.image_index + args.num_images)
    )
    results = []
    for instance_number, image_index in enumerate(image_indices):
        clean = dataset[image_index]["image"].squeeze(0).to(device)
        source_path, slice_index = dataset.slices[image_index]
        full_kspace = load_multicoil_kspace(source_path, slice_index, device)
        mask = cartesian_mask(
            clean.shape, args.acceleration, args.center_fraction,
            args.seed + instance_number, device=device,
        )
        sensitivities = estimate_sensitivities(
            full_kspace, output_shape=tuple(clean.shape),
            calibration_lines=CALIBRATION_LINES,
        )
        operator = MultiCoilMaskedFFT(mask, sensitivities)
        y = operator.forward(clean)
        if args.noise_std:
            generator = torch.Generator(device=device).manual_seed(
                args.seed + instance_number
            )
            acquired_rms = y[:, mask].abs().square().mean().sqrt()
            noise = torch.complex(
                torch.randn(y.shape, generator=generator, device=device),
                torch.randn(y.shape, generator=generator, device=device),
            ) * (args.noise_std * acquired_rms / np.sqrt(2.0))
            y = y + noise * mask
        zero_filled = operator.zero_filled(y)
        x = zero_filled.clone()
        # RSS-normalized coil maps give ||MFS|| <= 1, so STEP_SIZE <= 1 is safe.
        for outer_iteration in tqdm(
            range(args.iterations),
            desc=f"Reconstructing image {image_index}",
        ):
            data_gradient = (
                operator.adjoint(operator.forward(x) - y).real
                if use_data_fidelity else torch.zeros_like(x)
            )
            data_step = x - args.step_size * data_gradient
            denoised = apply_lpn(
                model, data_step, args.gamma, outer_iteration
            )
            x = torch.lerp(data_step, denoised, args.denoiser_strength)
        zero_metrics = metrics(clean, zero_filled)
        recon_metrics = metrics(clean, x)
        print(
            f"Image {image_index} zero-filled: "
            f"PSNR={zero_metrics[0]:.2f} dB, SSIM={zero_metrics[1]:.4f}"
        )
        print(
            f"Image {image_index} {method_name}: "
            f"PSNR={recon_metrics[0]:.2f} dB, SSIM={recon_metrics[1]:.4f}"
        )
        results.append(
            (clean, zero_filled, x.clamp(0, 1), zero_metrics, recon_metrics)
        )

    num_images = len(image_indices)
    fig, axes = plt.subplots(3, num_images, figsize=(3.5 * num_images, 10), squeeze=False)
    display_font_size = 14
    row_titles = ["Ground truth", "Zero-filled", "Reconstructed"]
    for column, result in enumerate(results):
        clean, zero_filled, reconstruction, zero_metrics, recon_metrics = result
        panels = (clean, zero_filled, reconstruction)
        panel_metrics = (None, zero_metrics, recon_metrics)
        for row, (panel, image_metrics) in enumerate(zip(panels, panel_metrics)):
            axis = axes[row, column]
            axis.imshow(panel.detach().cpu(), cmap="gray", vmin=0, vmax=1)
            if column == 0:
                axis.set_ylabel(row_titles[row], fontsize=display_font_size)
            if image_metrics is not None:
                psnr, ssim = image_metrics
                axis.text(0.02, 0.98, f"PSNR {psnr:.2f} dB\nSSIM {ssim:.3f}",
                          transform=axis.transAxes, fontsize=display_font_size, va="top", ha="left",
                          bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "none", "alpha": 0.75})
            axis.set_xticks([])
            axis.set_yticks([])
    fig.tight_layout(h_pad=0.8, w_pad=1.2)
    fig.savefig(
        output_dir / "mri_reconstruction.png",
        dpi=200,
        bbox_inches="tight",
    )
    fig.savefig(
        output_dir / "mri_reconstruction.pdf",
        bbox_inches="tight",
    )
    plt.close(fig)


if __name__ == "__main__":
    main()
