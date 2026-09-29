"""Reconstruct Mayo CT slices with parallel-beam tomography and a trained LPN.

Set ``GAMMA = 1`` for standard plug-and-play reconstruction or a gamma strictly
between zero and one for the relaxed proximal mapping with an inner LBFGS solve.
"""

from __future__ import annotations

import ast
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from ct_dataset import DEFAULT_DATA_ROOT, load_dataset
from inverse_mayoct_tomo import LeapCTOperator
from lpn_512 import LPN
from metrics import compute_image_metrics


# Configuration
DATA_ROOT = DEFAULT_DATA_ROOT
GPU_INDEX = 0  # choose GPU 0 or GPU 1
MODEL_LOCATION = Path("results/20260829_141445_PDT")

IMG_SIZE = 512
SPACE_RANGE = 128.0
NUM_ANGLES = 180
DET_SHAPE = 400

IMAGE_INDICES = [11]
ITERATIONS = 100
NOISE_STD = 1
SEED = 0

DENOISER_STRENGTH = 1.0
DENOISER_EVERY_N_ITERATIONS = 1

CREATE_GRID = False
GRID_INDICES = [0]
GRID_ITERATIONS = 2
GRID_NOISE_STD = 0.2

NORM_ITERATIONS = 10
# Additional step applied after normalizing the data-fidelity gradient to be
# 1-Lipschitz. Reduce this below 1.0 to make the reconstruction updates smaller.
STEP_SIZE = 0.99

# GAMMA = 1 uses standard PnP. Values strictly between 0 and 1 use the
# relaxed proximal mapping and an inner LBFGS solve at every denoising step.
GAMMA = 1.0
RELAXED_LBFGS_ITERATIONS = 1000
RELAXED_LBFGS_TOLERANCE = 1e-6


def compute_display_metrics(
    clean: np.ndarray, estimate: np.ndarray
) -> tuple[float, float]:
    """Compute metrics after clipping to the range used for image display."""
    display_min = float(clean.min())
    display_max = float(clean.max())
    clipped_estimate = np.clip(estimate, display_min, display_max)
    return compute_image_metrics(clean, clipped_estimate)


def load_model_configuration(model_location: Path) -> dict[str, object]:
    """Load a checkpoint and its architecture settings from a saved run."""
    model_location = model_location.expanduser()
    checkpoint = (
        model_location / "model.pt" if model_location.is_dir() else model_location
    )
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Model checkpoint not found: {checkpoint}")

    parameters_path = checkpoint.parent / "training_parameters.txt"
    if not parameters_path.is_file():
        raise FileNotFoundError(
            f"Training parameters not found next to the model: {parameters_path}"
        )

    parameters: dict[str, object] = {}
    for line_number, raw_line in enumerate(
        parameters_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise ValueError(f"Malformed line {line_number} in {parameters_path}")
        name, value = line.split("=", 1)
        try:
            parameters[name.strip()] = ast.literal_eval(value.strip())
        except (SyntaxError, ValueError) as error:
            raise ValueError(
                f"Invalid value on line {line_number} in {parameters_path}"
            ) from error

    required = {"ACTIVATION", "ALPHA", "HIDDEN"}
    missing = required - parameters.keys()
    if missing:
        raise ValueError(
            f"Missing required parameter(s) in {parameters_path}: "
            f"{', '.join(sorted(missing))}"
        )
    activation = str(parameters["ACTIVATION"]).lower()
    if activation == "softplus" and "BETA" not in parameters:
        raise ValueError(f"BETA is required for softplus models in {parameters_path}")
    if (
        activation in {"huberizedrelu", "huberized_relu"}
        and "HUBER_DELTA" not in parameters
    ):
        raise ValueError(
            f"HUBER_DELTA is required for huberized ReLU models in {parameters_path}"
        )

    return {"CHECKPOINT": checkpoint, **parameters}


def load_lpn(
    checkpoint_path: Path,
    device: torch.device,
    activation: str,
    alpha: float,
    hidden: int,
    beta: float | None = None,
    huber_delta: float | None = None,
) -> LPN:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model_options: dict[str, object] = {
        "in_dim": 1,
        "activation": activation,
        "alpha": alpha,
        "hidden": hidden,
    }
    activation_name = activation.lower()
    if activation_name == "softplus":
        model_options["beta"] = beta
    elif activation_name in {"huberizedrelu", "huberized_relu"}:
        model_options["huber_delta"] = huber_delta
    model = LPN(**model_options)
    model.load_state_dict(state_dict)
    return model.eval().to(device)


def apply_lpn(
    model: LPN,
    image: np.ndarray,
    device: torch.device,
    denoiser_strength: float = 1.0,
) -> np.ndarray:
    """Apply the learned denoiser to one image."""
    if not 0.0 <= denoiser_strength <= 1.0:
        raise ValueError("denoiser_strength must lie in [0, 1].")
    image_tensor = torch.as_tensor(
        np.asarray(image), dtype=torch.float32, device=device
    )[None, None]
    # LPN.forward computes the gradient of its scalar network internally.
    result = (
        denoiser_strength * model(image_tensor)
        + (1.0 - denoiser_strength) * image_tensor
    )
    return result.squeeze(0).squeeze(0).detach().cpu().numpy()


def apply_lpn_mapping(
    model: LPN,
    image: np.ndarray,
    device: torch.device,
    gamma: float = 1.0,
    denoiser_strength: float = 1.0,
) -> np.ndarray:
    """Apply standard PnP or the gamma-relaxed proximal mapping."""
    if not 0.0 < gamma <= 1.0:
        raise ValueError("gamma must lie in (0, 1].")
    if gamma == 1.0:
        return apply_lpn(model, image, device, denoiser_strength)

    image_tensor = torch.as_tensor(
        np.asarray(image), dtype=torch.float32, device=device
    )[None, None]
    w = image_tensor.clone().detach().requires_grad_(True)
    optimizer = torch.optim.LBFGS(
        [w],
        max_iter=RELAXED_LBFGS_ITERATIONS,
        tolerance_grad=RELAXED_LBFGS_TOLERANCE,
        tolerance_change=1e-9,
        history_size=10,
        line_search_fn="strong_wolfe",
    )

    def closure() -> torch.Tensor:
        optimizer.zero_grad(set_to_none=True)
        # Psi is the potential whose gradient is the standard LPN.
        psi = model.scalar(w).sum() + 0.5 * model.alpha * w.square().sum()
        value = (
            0.5 * gamma * w.square().sum()
            + (1.0 - gamma) * psi
            - (image_tensor * w).sum()
        )
        gradient = torch.autograd.grad(value, w)[0]
        w.grad = gradient
        return value

    optimizer.step(closure)
    with torch.no_grad():
        output = (image_tensor - gamma * w) / (1.0 - gamma)
        output = torch.lerp(image_tensor, output, denoiser_strength)
    return output.squeeze(0).squeeze(0).cpu().numpy()


def reconstruct(
    y: np.ndarray,
    operator: LeapCTOperator,
    model: LPN,
    device: torch.device,
    gradient_scale: float,
    step_size: float,
    iterations: int,
    gamma: float = 1.0,
    denoiser_strength: float = 1.0,
) -> np.ndarray:
    """Reconstruct one image from its noisy sinogram."""
    if iterations < 1:
        raise ValueError("iterations must be positive.")
    if step_size <= 0:
        raise ValueError("step_size must be positive.")
    if not 0.0 < gamma <= 1.0:
        raise ValueError("gamma must lie in (0, 1].")
    if not 0.0 <= denoiser_strength <= 1.0:
        raise ValueError("denoiser_strength must lie in [0, 1].")
    xk = operator.fbp(y)

    for iteration in range(iterations):
        data_gradient = gradient_scale * operator.adjoint(
            operator.forward(xk) - y
        )
        data_step = xk - step_size * data_gradient
        if (iteration + 1) % DENOISER_EVERY_N_ITERATIONS == 0:
            xk = apply_lpn_mapping(
                model,
                data_step,
                device,
                gamma,
                denoiser_strength,
            )
        else:
            xk = data_step

    return np.asarray(xk)


def save_reconstruction_plot(
    clean_images: list[np.ndarray],
    fbp_images: list[np.ndarray],
    reconstructions: list[np.ndarray],
    image_indices: list[int],
    path: Path,
) -> None:
    """Save horizontal ground-truth, FBP, and reconstruction comparisons."""
    num_images = len(image_indices)
    if not (
        len(clean_images) == len(fbp_images) == len(reconstructions) == num_images
    ):
        raise ValueError("images and image_indices must have matching lengths")
    if num_images == 0:
        raise ValueError("at least one image is required")

    fig, axes = plt.subplots(
        num_images, 3, figsize=(10, 3.5 * num_images), squeeze=False
    )
    display_font_size = 14
    column_titles = ["Ground truth", "Noisy FBP", "Reconstructed"]
    for row, (clean, fbp, reconstruction) in enumerate(
        zip(clean_images, fbp_images, reconstructions)
    ):
        panel_metrics: tuple[None | tuple[float, float], ...] = (
            None,
            compute_display_metrics(clean, fbp),
            compute_display_metrics(clean, reconstruction),
        )
        for column, (panel, image_metrics) in enumerate(
            zip((clean, fbp, reconstruction), panel_metrics)
        ):
            axis = axes[row, column]
            axis.imshow(panel, cmap="gray", vmin=0, vmax=1)
            if row == 0:
                axis.set_title(column_titles[column], fontsize=display_font_size)
            if image_metrics is not None:
                psnr, ssim = image_metrics
                axis.text(
                    0.02,
                    0.98,
                    f"PSNR {psnr:.2f} dB\nSSIM {ssim:.3f}",
                    transform=axis.transAxes,
                    fontsize=display_font_size,
                    va="top",
                    ha="left",
                    bbox={
                        "boxstyle": "round,pad=0.25",
                        "facecolor": "white",
                        "edgecolor": "none",
                        "alpha": 0.75,
                    },
                )
            axis.set_xticks([])
            axis.set_yticks([])
    fig.tight_layout(h_pad=0.8, w_pad=1.2)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    warnings.filterwarnings(
        "ignore",
        message=(
            "Attempting to run cuBLAS, but there was no current CUDA context!.*"
        ),
        category=UserWarning,
    )
    if ITERATIONS < 1 or GRID_ITERATIONS < 1:
        raise ValueError("iteration counts must be positive")
    if not IMAGE_INDICES:
        raise ValueError("IMAGE_INDICES must contain at least one image index")
    image_indices = list(IMAGE_INDICES)
    if any(index < 0 for index in image_indices):
        raise ValueError("image indices must be non-negative")
    if not 0.0 < STEP_SIZE <= 1.0:
        raise ValueError("step size must lie in (0, 1]")
    if not 0.0 <= DENOISER_STRENGTH <= 1.0:
        raise ValueError("denoiser strength must lie in [0, 1]")
    if NOISE_STD < 0.0:
        raise ValueError("noise standard deviation cannot be negative")
    if SEED < 0:
        raise ValueError("seed must be non-negative")
    if NUM_ANGLES < 1 or DET_SHAPE < 1:
        raise ValueError("num_angles and det_shape must be positive")
    if DENOISER_EVERY_N_ITERATIONS < 1:
        raise ValueError("DENOISER_EVERY_N_ITERATIONS must be positive")
    if not 0.0 < GAMMA <= 1.0:
        raise ValueError("gamma must lie in (0, 1]")

    if GPU_INDEX < 0:
        raise ValueError(f"GPU_INDEX must be non-negative, got {GPU_INDEX}")
    if not torch.cuda.is_available() or GPU_INDEX >= torch.cuda.device_count():
        raise RuntimeError(
            f"GPU_INDEX={GPU_INDEX} was requested, but only "
            f"{torch.cuda.device_count()} CUDA device(s) are available"
        )
    torch.cuda.set_device(GPU_INDEX)
    device = torch.device("cuda", GPU_INDEX)

    model_configuration = load_model_configuration(MODEL_LOCATION)
    checkpoint_path = Path(model_configuration["CHECKPOINT"])
    output_dir = checkpoint_path.parent
    activation = str(model_configuration["ACTIVATION"])
    alpha = float(model_configuration["ALPHA"])
    hidden = int(model_configuration["HIDDEN"])
    model_settings = [
        f"checkpoint={checkpoint_path}",
        f"activation={activation}",
        f"alpha={alpha}",
        f"hidden={hidden}",
    ]
    activation_name = activation.lower()
    if activation_name == "softplus":
        model_settings.append(f"beta={model_configuration['BETA']}")
    elif activation_name in {"huberizedrelu", "huberized_relu"}:
        model_settings.append(
            f"huber_delta={model_configuration['HUBER_DELTA']}"
        )
    print("model configuration: " + "; ".join(model_settings))
    method = "standard PnP" if GAMMA == 1.0 else "relaxed PnP"
    print(
        f"reconstruction mode: {method}; gamma={GAMMA:g}; "
        f"denoiser strength={DENOISER_STRENGTH:g}"
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    dataset = load_dataset(DATA_ROOT, "test")
    model = load_lpn(
        checkpoint_path,
        device,
        activation,
        alpha,
        hidden,
        beta=model_configuration.get("BETA"),
        huber_delta=model_configuration.get("HUBER_DELTA"),
    )
    operator = LeapCTOperator(
        SPACE_RANGE,
        IMG_SIZE,
        NUM_ANGLES,
        DET_SHAPE,
    )
    operator_norm = operator.operator_norm(NORM_ITERATIONS, SEED)
    gradient_scale = 1.0 / (operator_norm + 1.0) ** 2
    print(
        f"device={device}; operator norm={operator_norm:.6g}; "
        f"gradient scale={gradient_scale:.6g}; step={STEP_SIZE:.6g}"
    )

    clean_images, noisy_images, reconstructions = [], [], []
    for image_index in image_indices:
        clean = dataset[image_index]["image"][0].numpy()
        sinogram = operator.forward(clean)
        noise = rng.normal(0.0, NOISE_STD, sinogram.shape).astype(np.float32)
        noisy_sinogram = sinogram + noise
        reconstruction = reconstruct(
            noisy_sinogram,
            operator,
            model,
            device,
            gradient_scale,
            STEP_SIZE,
            ITERATIONS,
            gamma=GAMMA,
            denoiser_strength=DENOISER_STRENGTH,
        )
        noisy_fbp = operator.fbp(noisy_sinogram)
        fbp_metrics = compute_display_metrics(clean, noisy_fbp)
        reconstruction_metrics = compute_display_metrics(clean, reconstruction)
        print(
            f"Image {image_index} noisy FBP: PSNR={fbp_metrics[0]:.2f} dB, "
            f"SSIM={fbp_metrics[1]:.4f}"
        )
        print(
            f"Image {image_index} reconstructed: "
            f"PSNR={reconstruction_metrics[0]:.2f} dB, "
            f"SSIM={reconstruction_metrics[1]:.4f}"
        )
        clean_images.append(clean)
        noisy_images.append(noisy_fbp)
        reconstructions.append(reconstruction)
    reconstruction_output = (
        np.stack(reconstructions)
        if len(image_indices) > 1
        else reconstructions[0]
    )
    np.save(output_dir / "ct_reconstruction.npy", reconstruction_output)
    save_reconstruction_plot(
        clean_images,
        noisy_images,
        reconstructions,
        image_indices,
        output_dir / "ct_reconstruction.png",
    )

    if CREATE_GRID:
        clean_images = [dataset[i]["image"][0].numpy() for i in GRID_INDICES]
        noisy_sinograms = [
            operator.forward(image)
            + rng.normal(
                0.0, GRID_NOISE_STD, (NUM_ANGLES, DET_SHAPE)
            ).astype(np.float32)
            for image in clean_images
        ]
        denoised = [
            reconstruct(
                y,
                operator,
                model,
                device,
                gradient_scale,
                STEP_SIZE,
                GRID_ITERATIONS,
                gamma=GAMMA,
                denoiser_strength=DENOISER_STRENGTH,
            )
            for y in noisy_sinograms
        ]
        noisy_images = [operator.fbp(y) for y in noisy_sinograms]
        save_reconstruction_plot(
            clean_images,
            noisy_images,
            denoised,
            GRID_INDICES,
            output_dir / "ct_denoise_light_noise.png",
        )


if __name__ == "__main__":
    if len(sys.argv) != 1:
        raise SystemExit(
            "inverse_ct_test.py takes no command-line arguments; "
            "edit its configuration block instead"
        )
    main()
