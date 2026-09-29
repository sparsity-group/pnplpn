"""Train the LPN as a Gaussian denoiser on fastMRI knee magnitude images."""

from __future__ import annotations

import os
from collections import Counter
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from fastmri import load_dataset
from lpn_320 import LPN


# Configuration (kept at the top to match the CT experiment).
DATA_ROOT = Path("../../shared/datasets/fastmri/multicoil_knee")
IMG_SIZE = 320
SIGMA = 0.1
TRAIN_BATCH_SIZE = 64
NUM_STEPS = 40_000
LR = 1e-4
LR_MIN = 1e-6
SAVE_EVERY = 5_000
VALIDATE_EVERY = 2_500
VAL_NUM_IMAGES = 16
VAL_BATCH_SIZE = 4
NUM_WORKERS = 4
SEED = 0
RESUME_FROM = None
GPU_INDEX = 1  # choose GPU 0 or GPU 1

IN_DIM = 1
ACTIVATION = "huberizedrelu"
HUBER_DELTA = 0.1
BETA = 5.0
ALPHA = 0.8
HIDDEN = 256


def add_gaussian_noise(
    clean: torch.Tensor, sigma: float, generator: torch.Generator
) -> torch.Tensor:
    return clean + sigma * torch.randn(
        clean.shape, dtype=clean.dtype, device=clean.device, generator=generator
    )


def main() -> None:
    distributed = False
    local_rank, rank, world_size = 0, 0, 1
    if GPU_INDEX not in (0, 1):
        raise ValueError(f"GPU_INDEX must be 0 or 1, got {GPU_INDEX}")
    if not torch.cuda.is_available() or GPU_INDEX >= torch.cuda.device_count():
        raise RuntimeError(
            f"GPU_INDEX={GPU_INDEX} was requested, but only "
            f"{torch.cuda.device_count()} CUDA device(s) are available"
        )
    torch.cuda.set_device(GPU_INDEX)
    device = torch.device("cuda", GPU_INDEX)
    is_main = rank == 0

    script_dir = Path(__file__).resolve().parent
    resume_path = Path(RESUME_FROM).expanduser().resolve() if RESUME_FROM else None
    if resume_path:
        run_dir = resume_path.parent.parent
    else:
        name = (
            datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y%m%d_%H%M%S_%Z")
            if is_main else None
        )
        if distributed:
            holder = [name]
            dist.broadcast_object_list(holder, src=0)
            name = holder[0]
        run_dir = script_dir / "results" / str(name)
    checkpoint_dir = run_dir / "checkpoints"
    if is_main:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        print(f"Run outputs: {run_dir}")
        print(f"Device: {device}; effective batch size: {TRAIN_BATCH_SIZE}")

    options = dict(in_dim=IN_DIM, alpha=ALPHA, hidden=HIDDEN, activation=ACTIVATION)
    activation = ACTIVATION.lower()
    if activation == "softplus":
        options["beta"] = BETA
    elif activation in {"huberizedrelu", "huberized_relu"}:
        options["huber_delta"] = HUBER_DELTA
    model = LPN(**options).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=NUM_STEPS, eta_min=LR_MIN
    )
    global_step = 0
    if resume_path:
        checkpoint = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        global_step = int(checkpoint["iteration"])

    if distributed:
        model = DDP(
            model, device_ids=[local_rank], output_device=local_rank,
            find_unused_parameters=True,
        )

    train_dataset = load_dataset(DATA_ROOT, "train")
    val_dataset = load_dataset(DATA_ROOT, "val")
    if is_main:
        train_contrasts = Counter(
            str(values["acquisition"]) for values in train_dataset.metadata.values()
        )
        val_contrasts = Counter(
            str(values["acquisition"]) for values in val_dataset.metadata.values()
        )
        print(f"Train volumes by acquisition: {dict(train_contrasts)}")
        print(f"Validation volumes by acquisition: {dict(val_contrasts)}")
    sampler = DistributedSampler(train_dataset, shuffle=True, seed=SEED) if distributed else None
    loader = DataLoader(
        train_dataset,
        batch_size=TRAIN_BATCH_SIZE,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=NUM_WORKERS,
        pin_memory=device.type == "cuda",
        persistent_workers=NUM_WORKERS > 0,
        drop_last=True,
    )
    fixed_count = min(VAL_NUM_IMAGES, len(val_dataset))
    fixed_clean = torch.stack([val_dataset[i]["image"] for i in range(fixed_count)]) if is_main else None
    validation_generator = torch.Generator().manual_seed(SEED)
    fixed_noisy = add_gaussian_noise(fixed_clean, SIGMA, validation_generator) if is_main else None
    writer = SummaryWriter(str(run_dir / "tensorboard")) if is_main else None
    noise_generator = torch.Generator().manual_seed(SEED + rank)
    loss_function = torch.nn.MSELoss()

    def validate(model_to_evaluate: LPN) -> tuple[float, float]:
        model_to_evaluate.eval()
        losses = []
        for start in range(0, fixed_count, VAL_BATCH_SIZE):
            clean = fixed_clean[start:start + VAL_BATCH_SIZE].to(device)
            noisy = fixed_noisy[start:start + VAL_BATCH_SIZE].to(device)
            prediction = model_to_evaluate(noisy).detach()
            losses.append((prediction - clean).square().flatten(1).mean(1).cpu())
        values = torch.cat(losses)
        return values.mean().item(), (-10 * torch.log10(values.clamp_min(1e-12))).mean().item()

    progress = tqdm(total=NUM_STEPS, initial=global_step, disable=not is_main, desc="Train")
    epoch = 0
    optimizer.zero_grad(set_to_none=True)
    last_val = (float("nan"), float("nan"))
    while global_step < NUM_STEPS:
        if sampler is not None:
            sampler.set_epoch(epoch)
        for batch in loader:
            model.train()
            clean_cpu = batch["image"]
            noisy = add_gaussian_noise(clean_cpu, SIGMA, noise_generator).to(device, non_blocking=True)
            clean = clean_cpu.to(device, non_blocking=True)
            loss = loss_function(model(noisy), clean)
            loss.backward()

            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            bare_model = model.module if distributed else model
            bare_model.wclip()
            train_loss = loss.detach().item()
            global_step += 1
            if is_main:
                writer.add_scalar("Loss/train", train_loss, global_step)
                writer.add_scalar("LR", scheduler.get_last_lr()[0], global_step)
                progress.update(1)
                progress.set_postfix(loss=f"{train_loss:.4g}")

            if global_step % VALIDATE_EVERY == 0:
                if is_main:
                    last_val = validate(bare_model)
                    writer.add_scalar("Loss/val", last_val[0], global_step)
                    writer.add_scalar("PSNR/val", last_val[1], global_step)
                    progress.write(f"Validation {global_step}: MSE={last_val[0]:.6g}, PSNR={last_val[1]:.2f} dB")
                if distributed:
                    dist.barrier()
            if global_step % SAVE_EVERY == 0:
                if is_main:
                    torch.save({
                        "iteration": global_step,
                        "model_state_dict": bare_model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "loss": train_loss,
                        "val_loss": last_val[0],
                        "val_psnr": last_val[1],
                    }, checkpoint_dir / f"checkpoint_{global_step:08d}.pt")
                if distributed:
                    dist.barrier()
            if global_step >= NUM_STEPS:
                break
        epoch += 1

    bare_model = model.module if distributed else model
    bare_model.wclip()
    if is_main:
        torch.save({"iteration": global_step, "model_state_dict": bare_model.state_dict()}, run_dir / "model.pt")
        parameters = {
            "DATA_ROOT": str(DATA_ROOT), "IMG_SIZE": IMG_SIZE, "SIGMA": SIGMA,
            "TRAIN_BATCH_SIZE": TRAIN_BATCH_SIZE,
            "NUM_STEPS": NUM_STEPS, "LR": LR, "LR_MIN": LR_MIN,
            "IN_DIM": IN_DIM, "ACTIVATION": ACTIVATION, "ALPHA": ALPHA,
            "HIDDEN": HIDDEN, "HUBER_DELTA": HUBER_DELTA, "BETA": BETA,
            "GPU_INDEX": GPU_INDEX,
        }
        (run_dir / "training_parameters.txt").write_text(
            "".join(f"{key} = {value!r}\n" for key, value in parameters.items()), encoding="utf-8"
        )
        progress.close()
        writer.close()
    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
