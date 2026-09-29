"""Train a learned proximal network on noisy CT slices."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from ct_dataset import DEFAULT_DATA_ROOT, HU_MAX, HU_MIN, load_dataset
from lpn_512 import LPN


# Training configuration
DATA_ROOT = DEFAULT_DATA_ROOT
GPU_INDEX = 0
SIGMA = 0.05
IMG_SIZE = 512
TRAIN_BATCH_SIZE = 64
NUM_STEPS = 40_000
LR = 1e-4
LR_MIN = 1e-6
SAVE_EVERY = 5_000
VALIDATE_EVERY = 2_500
VAL_NUM_IMAGES = 16
VAL_BATCH_SIZE = 4
SEED = 0
RESUME_FROM: str | Path | None = None

IN_DIM = 1
ACTIVATION = "huberizedrelu"
BETA = 5.0
HUBER_DELTA = 0.1
ALPHA = 0.8
HIDDEN = 256


def training_parameters() -> dict[str, object]:
    """Return the serializable configuration stored beside each model."""
    parameters: dict[str, object] = {
        "DATA_ROOT": str(DATA_ROOT),
        "GPU_INDEX": GPU_INDEX,
        "HU_MIN": HU_MIN,
        "HU_MAX": HU_MAX,
        "SIGMA": SIGMA,
        "IMG_SIZE": IMG_SIZE,
        "TRAIN_BATCH_SIZE": TRAIN_BATCH_SIZE,
        "NUM_STEPS": NUM_STEPS,
        "LR": LR,
        "LR_MIN": LR_MIN,
        "SAVE_EVERY": SAVE_EVERY,
        "VALIDATE_EVERY": VALIDATE_EVERY,
        "VAL_NUM_IMAGES": VAL_NUM_IMAGES,
        "VAL_BATCH_SIZE": VAL_BATCH_SIZE,
        "SEED": SEED,
        "RESUME_FROM": str(RESUME_FROM) if RESUME_FROM is not None else None,
        "IN_DIM": IN_DIM,
        "ACTIVATION": ACTIVATION,
        "ALPHA": ALPHA,
        "HIDDEN": HIDDEN,
    }
    activation_name = ACTIVATION.lower()
    if activation_name == "softplus":
        parameters["BETA"] = BETA
    elif activation_name in {"huberizedrelu", "huberized_relu"}:
        parameters["HUBER_DELTA"] = HUBER_DELTA
    return parameters


def validate_configuration() -> str:
    """Validate module-level settings and return the normalized activation."""
    activation_name = ACTIVATION.lower()
    supported_activations = {
        "softplus",
        "relu",
        "leakyrelu",
        "huberizedrelu",
        "huberized_relu",
    }
    if activation_name not in supported_activations:
        raise ValueError(
            f"ACTIVATION must be one of {sorted(supported_activations)}, "
            f"got {ACTIVATION!r}"
        )
    if NUM_STEPS < 1:
        raise ValueError("NUM_STEPS must be positive")
    if TRAIN_BATCH_SIZE < 1:
        raise ValueError("TRAIN_BATCH_SIZE must be positive")
    if VAL_NUM_IMAGES < 1 or VAL_BATCH_SIZE < 1:
        raise ValueError("VAL_NUM_IMAGES and VAL_BATCH_SIZE must be positive")
    if SAVE_EVERY < 1 or VALIDATE_EVERY < 1:
        raise ValueError("SAVE_EVERY and VALIDATE_EVERY must be positive")
    if SAVE_EVERY % VALIDATE_EVERY != 0:
        raise ValueError("SAVE_EVERY must be a multiple of VALIDATE_EVERY")
    return activation_name


def add_gaussian_noise(
    clean_images: torch.Tensor,
    sigma: float,
    generator: torch.Generator,
) -> torch.Tensor:
    """Add independent, zero-mean Gaussian noise to an image batch."""
    expected_shape = (1, IMG_SIZE, IMG_SIZE)
    if clean_images.ndim != 4 or clean_images.shape[1:] != expected_shape:
        raise ValueError(
            f"expected images with shape (batch, {expected_shape}), "
            f"got {tuple(clean_images.shape)}"
        )
    noise = torch.randn(
        clean_images.shape,
        dtype=clean_images.dtype,
        device=clean_images.device,
        generator=generator,
    )
    return clean_images + sigma * noise


def select_cuda_device() -> torch.device:
    """Select the configured CUDA device."""
    if GPU_INDEX < 0:
        raise ValueError(f"GPU_INDEX must be non-negative, got {GPU_INDEX}")
    if not torch.cuda.is_available() or GPU_INDEX >= torch.cuda.device_count():
        raise RuntimeError(
            f"GPU_INDEX={GPU_INDEX} was requested, but only "
            f"{torch.cuda.device_count()} CUDA device(s) are available"
        )
    torch.cuda.set_device(GPU_INDEX)
    return torch.device("cuda", GPU_INDEX)


def create_model(activation_name: str, device: torch.device) -> LPN:
    """Build the configured network on the selected device."""
    model_options: dict[str, object] = {
        "hidden": HIDDEN,
        "in_dim": IN_DIM,
        "activation": ACTIVATION,
        "alpha": ALPHA,
    }
    if activation_name == "softplus":
        model_options["beta"] = BETA
    elif activation_name in {"huberizedrelu", "huberized_relu"}:
        model_options["huber_delta"] = HUBER_DELTA
    return LPN(**model_options).to(device)


def evaluate_fixed_validation(
    model: LPN,
    clean_images: torch.Tensor,
    noisy_images: torch.Tensor,
    device: torch.device,
) -> tuple[float, float]:
    """Return mean per-image MSE and PSNR on a fixed validation set."""
    model.eval()
    mse_values = []
    for start in range(0, len(clean_images), VAL_BATCH_SIZE):
        stop = min(start + VAL_BATCH_SIZE, len(clean_images))
        clean_batch = clean_images[start:stop].to(device, non_blocking=True)
        noisy_batch = noisy_images[start:stop].to(device, non_blocking=True)
        prediction = model(noisy_batch).detach()
        batch_mse = (prediction - clean_batch).square().flatten(1).mean(1)
        mse_values.append(batch_mse.cpu())

    per_image_mse = torch.cat(mse_values)
    mean_mse = per_image_mse.mean().item()
    # Images use a [0, 1] intensity range, so MAX_I squared is 1.
    mean_psnr = (
        -10.0 * torch.log10(per_image_mse.clamp_min(1e-12))
    ).mean().item()
    return mean_mse, mean_psnr


def resolve_run_paths() -> tuple[Path, Path | None]:
    """Resolve the output directory and optional checkpoint to resume."""
    if RESUME_FROM is not None:
        resume_path = Path(RESUME_FROM).expanduser()
        if not resume_path.is_absolute():
            resume_path = Path(__file__).resolve().parent / resume_path
        if not resume_path.is_file():
            raise FileNotFoundError(f"checkpoint not found: {resume_path}")
        return resume_path.parent.parent, resume_path

    timestamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime(
        "%Y%m%d_%H%M%S_%Z"
    )
    run_dir = Path(__file__).resolve().parent / "results" / timestamp
    return run_dir, None


def save_checkpoint(
    path: Path,
    model: LPN,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    step: int,
    train_loss: float,
    val_mse: float | None = None,
    val_psnr: float | None = None,
) -> None:
    """Save model and optimizer state for resuming training."""
    model.wclip()
    checkpoint: dict[str, object] = {
        "iteration": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "loss": train_loss,
    }
    if val_mse is not None and val_psnr is not None:
        checkpoint["val_loss"] = val_mse
        checkpoint["val_psnr"] = val_psnr
    torch.save(checkpoint, path)


def main() -> None:
    """Train and periodically validate and checkpoint the configured model."""
    activation_name = validate_configuration()
    torch.manual_seed(SEED)
    device = select_cuda_device()
    run_dir, resume_path = resolve_run_paths()
    checkpoint_dir = run_dir / "checkpoints"
    tensorboard_dir = run_dir / "tensorboard"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    print(f"Run outputs: {run_dir}")
    print(
        f"Selected {device}: {torch.cuda.get_device_name(device)} "
        f"({free_bytes / 2**30:.1f}/{total_bytes / 2**30:.1f} GiB free)"
    )
    print(f"Batch size: {TRAIN_BATCH_SIZE} images on one GPU")

    model = create_model(activation_name, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=NUM_STEPS,
        eta_min=LR_MIN,
    )
    global_step = 0
    if resume_path is not None:
        checkpoint = torch.load(resume_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        global_step = checkpoint["iteration"]
        if "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        else:
            scheduler.step(global_step)
        if global_step >= NUM_STEPS:
            raise ValueError(
                f"checkpoint is already at step {global_step}; increase "
                f"NUM_STEPS above {NUM_STEPS} to continue training"
            )
        print(f"Resumed from {resume_path} at step {global_step}")

    train_dataset = load_dataset(DATA_ROOT, "train")
    validation_dataset = load_dataset(DATA_ROOT, "val")
    if len(train_dataset) < TRAIN_BATCH_SIZE:
        raise ValueError(
            f"training split has {len(train_dataset)} images, fewer than the "
            f"batch size of {TRAIN_BATCH_SIZE}"
        )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=TRAIN_BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )
    validation_count = min(VAL_NUM_IMAGES, len(validation_dataset))
    if validation_count == 0:
        raise ValueError("validation split is empty")
    validation_clean = torch.stack(
        [validation_dataset[index]["image"] for index in range(validation_count)]
    )
    validation_generator = torch.Generator().manual_seed(SEED)
    validation_noisy = add_gaussian_noise(
        validation_clean,
        SIGMA,
        validation_generator,
    )
    training_generator = torch.Generator().manual_seed(SEED)

    loss_function = torch.nn.MSELoss()
    optimizer.zero_grad(set_to_none=True)
    writer = SummaryWriter(log_dir=tensorboard_dir)
    progress = tqdm(total=NUM_STEPS, initial=global_step, desc="Train")
    latest_val_mse: float | None = None
    latest_val_psnr: float | None = None

    while global_step < NUM_STEPS:
        for batch in train_loader:
            model.train()
            clean_cpu = batch["image"]
            noisy = add_gaussian_noise(
                clean_cpu,
                SIGMA,
                training_generator,
            ).to(device, non_blocking=True)
            clean = clean_cpu.to(device, non_blocking=True)

            prediction = model(noisy)
            loss = loss_function(prediction, clean)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            model.wclip()

            train_loss = loss.detach().item()
            writer.add_scalar("Loss/train", train_loss, global_step)
            writer.add_scalar("LR", optimizer.param_groups[0]["lr"], global_step)
            global_step += 1
            progress.update(1)
            progress.set_postfix(
                loss=f"{train_loss:.4f}",
                lr=f"{optimizer.param_groups[0]['lr']:.2e}",
            )

            if global_step % VALIDATE_EVERY == 0:
                latest_val_mse, latest_val_psnr = evaluate_fixed_validation(
                    model,
                    validation_clean,
                    validation_noisy,
                    device,
                )
                writer.add_scalar("Loss/val", latest_val_mse, global_step)
                writer.add_scalar("PSNR/val", latest_val_psnr, global_step)
                progress.write(
                    f"Validation step {global_step}: "
                    f"MSE={latest_val_mse:.6g}, "
                    f"PSNR={latest_val_psnr:.2f} dB"
                )

            if global_step % SAVE_EVERY == 0:
                save_checkpoint(
                    checkpoint_dir / f"checkpoint_{global_step:08d}.pt",
                    model,
                    optimizer,
                    scheduler,
                    global_step,
                    train_loss,
                    latest_val_mse,
                    latest_val_psnr,
                )

            if global_step >= NUM_STEPS:
                break

    save_checkpoint(
        run_dir / "model.pt",
        model,
        optimizer,
        scheduler,
        global_step,
        train_loss,
        latest_val_mse,
        latest_val_psnr,
    )
    parameters_path = run_dir / "training_parameters.txt"
    with parameters_path.open("w", encoding="utf-8") as parameter_file:
        for name, value in training_parameters().items():
            parameter_file.write(f"{name} = {value!r}\n")
    progress.close()
    writer.close()


if __name__ == "__main__":
    main()
