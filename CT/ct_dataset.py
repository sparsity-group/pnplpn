"""Dataset loader for uncompressed, full-dose liver CT DICOM slices."""

import struct
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


# This is the same broad CT window used by the former preprocessed MayoCT data:
# air maps to 0, water maps near 0.5, and dense bone clips at 1.
HU_MIN = -1000.0
HU_MAX = 1000.0
DEFAULT_DATA_ROOT = (
    Path(__file__).resolve().parents[2] / "shared/datasets/full_dose_ct_liver"
)
SUPPORTED_SPLITS = frozenset({"train", "val", "test"})
_LONG_VALUE_REPRESENTATIONS = frozenset(
    {b"OB", b"OD", b"OF", b"OL", b"OW", b"SQ", b"UC", b"UR", b"UT", b"UN"}
)


def _element_value(
    data: bytes,
    group: int,
    element: int,
    *,
    stop: int | None = None,
) -> bytes:
    """Read one top-level Explicit-VR Little-Endian DICOM element."""
    tag = struct.pack("<HH", group, element)
    position = data.find(tag, 0, len(data) if stop is None else stop)
    if position < 0:
        raise ValueError(f"Missing DICOM tag ({group:04x},{element:04x})")

    vr = data[position + 4 : position + 6]
    if vr in _LONG_VALUE_REPRESENTATIONS:
        length = struct.unpack_from("<I", data, position + 8)[0]
        value_start = position + 12
    else:
        length = struct.unpack_from("<H", data, position + 6)[0]
        value_start = position + 8
    return data[value_start : value_start + length]


def _unsigned_short(data: bytes, group: int, element: int, *, stop: int) -> int:
    value = _element_value(data, group, element, stop=stop)
    if len(value) != 2:
        raise ValueError(f"Expected a 2-byte value for ({group:04x},{element:04x})")
    return struct.unpack("<H", value)[0]


def _decimal_string(
    data: bytes, group: int, element: int, *, stop: int, default: float
) -> float:
    try:
        value = _element_value(data, group, element, stop=stop)
    except ValueError:
        return default
    return float(value.decode("ascii").strip(" \0"))


def read_ct_dicom(path: str | Path) -> np.ndarray:
    """Decode one dataset slice and return HU values as a float32 array."""
    path = Path(path)
    data = path.read_bytes()
    if len(data) < 132 or data[128:132] != b"DICM":
        raise ValueError(f"{path} is not a Part-10 DICOM file")

    transfer_syntax = _element_value(data, 0x0002, 0x0010).rstrip(b"\0 ")
    if transfer_syntax != b"1.2.840.10008.1.2.1":
        raise ValueError(
            f"Unsupported transfer syntax {transfer_syntax!r} in {path}; "
            "this loader expects uncompressed Explicit VR Little Endian DICOM"
        )

    pixel_tag = struct.pack("<HH", 0x7FE0, 0x0010)
    pixel_position = data.find(pixel_tag)
    if pixel_position < 0:
        raise ValueError(f"Missing Pixel Data in {path}")

    rows = _unsigned_short(data, 0x0028, 0x0010, stop=pixel_position)
    columns = _unsigned_short(data, 0x0028, 0x0011, stop=pixel_position)
    samples = _unsigned_short(data, 0x0028, 0x0002, stop=pixel_position)
    bits_allocated = _unsigned_short(data, 0x0028, 0x0100, stop=pixel_position)
    bits_stored = _unsigned_short(data, 0x0028, 0x0101, stop=pixel_position)
    pixel_representation = _unsigned_short(data, 0x0028, 0x0103, stop=pixel_position)
    photometric = _element_value(
        data, 0x0028, 0x0004, stop=pixel_position
    ).strip(b" \0")
    pixel_format = (samples, bits_allocated, bits_stored, photometric)
    if pixel_format != (1, 16, 16, b"MONOCHROME2"):
        raise ValueError(
            f"Unsupported pixel format in {path}: samples={samples}, "
            f"bits={bits_stored}/{bits_allocated}, photometric={photometric!r}"
        )
    if pixel_representation not in (0, 1):
        raise ValueError(
            f"Invalid Pixel Representation in {path}: {pixel_representation}"
        )

    pixel_bytes = _element_value(data, 0x7FE0, 0x0010)
    expected_pixels = rows * columns
    dtype = "<i2" if pixel_representation == 1 else "<u2"
    stored = np.frombuffer(pixel_bytes, dtype=dtype, count=expected_pixels)
    if stored.size != expected_pixels:
        raise ValueError(
            f"Pixel Data in {path} has {stored.size} samples; "
            f"expected {expected_pixels}"
        )

    slope = _decimal_string(data, 0x0028, 0x1053, stop=pixel_position, default=1.0)
    intercept = _decimal_string(data, 0x0028, 0x1052, stop=pixel_position, default=0.0)
    return stored.reshape(rows, columns).astype(np.float32) * slope + intercept


class MayoCTDataset(Dataset):
    """Load a flat ``train``/``val``/``test`` directory of DICOM slices."""

    def __init__(self, root: str | Path, split: str):
        if split not in SUPPORTED_SPLITS:
            raise ValueError(
                f"split must be one of {sorted(SUPPORTED_SPLITS)}, got {split!r}"
            )
        self.data_dir = Path(root).expanduser() / split
        if not self.data_dir.is_dir():
            raise FileNotFoundError(
                f"CT split directory does not exist: {self.data_dir}"
            )
        self.files = sorted(self.data_dir.glob("*.dcm"))
        if not self.files:
            raise FileNotFoundError(f"No .dcm files found in {self.data_dir}")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        hu = read_ct_dicom(self.files[idx])
        image = np.clip(hu, HU_MIN, HU_MAX)
        image = ((image - HU_MIN) / (HU_MAX - HU_MIN)).astype(np.float32)
        return {"image": torch.from_numpy(image).unsqueeze(0)}


def load_dataset(root: str | Path, split: str) -> MayoCTDataset:
    """Create a CT dataset for one of the supported splits."""
    return MayoCTDataset(root=root, split=split)
