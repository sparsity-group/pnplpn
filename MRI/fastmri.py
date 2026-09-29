"""Lazy access to 320 x 320 fastMRI knee magnitude reconstructions."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class FastMRIKneeDataset(Dataset):
    """Expose every slice in ``reconstruction_rss`` as a normalized image.

    Normalization is done independently per volume using the 99th percentile.
    This preserves relative contrast between slices while limiting rare bright
    pixels; the result is clipped to [0, 1].
    """

    SPLIT_DIRECTORIES = {
        "train": "multicoil_train",
        "val": "multicoil_val",
        "test": "multicoil_test",
    }

    def __init__(
        self,
        root: str | Path,
        split: str,
        image_size: int = 320,
        acquisitions: set[str] | None = None,
    ):
        if split not in self.SPLIT_DIRECTORIES:
            raise ValueError(f"Unknown split {split!r}; use train, val, or test")
        self.root = Path(root).expanduser()
        self.data_dir = self.root / self.SPLIT_DIRECTORIES[split]
        self.image_size = int(image_size)
        if not self.data_dir.is_dir():
            raise FileNotFoundError(f"fastMRI split directory not found: {self.data_dir}")

        self.files = sorted(self.data_dir.glob("*.h5"))
        if not self.files:
            raise FileNotFoundError(f"No .h5 files found in {self.data_dir}")

        self.slices: list[tuple[Path, int]] = []
        self.metadata: dict[Path, dict[str, object]] = {}
        self.scales: dict[Path, float] = {}
        for path in self.files:
            with h5py.File(path, "r") as handle:
                if "reconstruction_rss" not in handle:
                    # The official challenge test set has no ground truth.
                    continue
                acquisition = str(handle.attrs.get("acquisition", "unknown"))
                if acquisitions is not None and acquisition not in acquisitions:
                    continue
                shape = handle["reconstruction_rss"].shape
                if len(shape) != 3 or tuple(shape[1:]) != (image_size, image_size):
                    raise ValueError(
                        f"Expected reconstruction_rss (*, {image_size}, {image_size}) "
                        f"in {path}, got {shape}"
                    )
                self.slices.extend((path, index) for index in range(shape[0]))
                self.metadata[path] = {
                    "acquisition": acquisition,
                    "patient_id": str(handle.attrs.get("patient_id", "unknown")),
                }
        if not self.slices:
            raise ValueError(f"No reconstruction_rss targets found in {self.data_dir}")

    def __len__(self) -> int:
        return len(self.slices)

    def _volume_scale(self, path: Path) -> float:
        scale = self.scales.get(path)
        if scale is None:
            with h5py.File(path, "r") as handle:
                volume = np.asarray(handle["reconstruction_rss"], dtype=np.float32)
            scale = float(np.percentile(volume, 99.0))
            if not np.isfinite(scale) or scale <= 0:
                raise ValueError(f"Invalid intensity scale {scale} in {path}")
            self.scales[path] = scale
        return scale

    def __getitem__(self, index: int) -> dict[str, object]:
        path, slice_index = self.slices[index]
        with h5py.File(path, "r") as handle:
            image = np.asarray(
                handle["reconstruction_rss"][slice_index], dtype=np.float32
            )
        image = np.clip(image / self._volume_scale(path), 0.0, 1.0)
        return {
            "image": torch.from_numpy(image).unsqueeze(0),
            "filename": path.name,
            "slice": slice_index,
            **self.metadata[path],
        }


def load_dataset(root: str | Path, split: str) -> FastMRIKneeDataset:
    return FastMRIKneeDataset(root, split)
