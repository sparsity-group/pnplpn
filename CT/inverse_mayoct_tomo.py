"""Parallel-beam CT operators backed by LEAP."""

from __future__ import annotations

import numpy as np
from leapctype import tomographicModels


class LeapCTOperator:
    """Expose LEAP's 3D projector as a convenient single-slice 2D operator."""

    def __init__(
        self,
        space_range: float,
        img_size: int,
        num_angles: int,
        det_shape: int,
    ) -> None:
        if space_range <= 0:
            raise ValueError("space_range must be positive")
        if min(img_size, num_angles, det_shape) < 1:
            raise ValueError("img_size, num_angles, and det_shape must be positive")

        self.img_size = img_size
        self.num_angles = num_angles
        self.det_shape = det_shape
        self.leapct = tomographicModels()

        voxel_size = 2.0 * space_range / img_size
        # Cover the diagonal of the square reconstruction domain.
        detector_width = 2.0 * np.sqrt(2.0) * space_range
        detector_pixel_size = detector_width / det_shape
        angles = self.leapct.setAngleArray(num_angles, 180.0)

        geometry_ok = self.leapct.set_parallelbeam(
            numAngles=num_angles,
            numRows=1,
            numCols=det_shape,
            pixelHeight=voxel_size,
            pixelWidth=detector_pixel_size,
            centerRow=0.0,
            centerCol=0.5 * (det_shape - 1),
            phis=angles,
        )
        volume_ok = self.leapct.set_volume(
            numX=img_size,
            numY=img_size,
            numZ=1,
            voxelWidth=voxel_size,
            voxelHeight=voxel_size,
        )
        if not geometry_ok or not volume_ok:
            raise ValueError("LEAP rejected the CT geometry or volume parameters.")

    @staticmethod
    def _float32_c(array: np.ndarray) -> np.ndarray:
        return np.ascontiguousarray(array, dtype=np.float32)

    def forward(self, image: np.ndarray) -> np.ndarray:
        expected_shape = (self.img_size, self.img_size)
        if np.shape(image) != expected_shape:
            raise ValueError(
                f"expected image shape {expected_shape}, got {np.shape(image)}"
            )
        volume = self._float32_c(np.asarray(image)[None])
        projections = np.zeros(
            (self.num_angles, 1, self.det_shape), dtype=np.float32
        )
        self.leapct.project(projections, volume)
        return projections[:, 0, :]

    def adjoint(self, sinogram: np.ndarray) -> np.ndarray:
        self._validate_sinogram(sinogram)
        projections = self._float32_c(
            np.asarray(sinogram).reshape(self.num_angles, 1, self.det_shape)
        )
        volume = np.zeros((1, self.img_size, self.img_size), dtype=np.float32)
        self.leapct.backproject(projections, volume)
        return volume[0]

    def fbp(self, sinogram: np.ndarray) -> np.ndarray:
        self._validate_sinogram(sinogram)
        projections = self._float32_c(
            np.asarray(sinogram).reshape(self.num_angles, 1, self.det_shape)
        )
        volume = np.zeros((1, self.img_size, self.img_size), dtype=np.float32)
        self.leapct.FBP(projections, volume)
        return volume[0]

    def operator_norm(self, iterations: int = 10, seed: int = 0) -> float:
        """Estimate ||A||_2 using power iteration on A^T A."""
        if iterations < 1:
            raise ValueError("iterations must be positive")

        rng = np.random.default_rng(seed)
        image = rng.standard_normal((self.img_size, self.img_size)).astype(np.float32)
        image /= np.linalg.norm(image)

        for _ in range(iterations):
            image = self.adjoint(self.forward(image))
            image_norm = np.linalg.norm(image)
            if image_norm == 0.0:
                return 0.0
            image /= image_norm

        return float(np.linalg.norm(self.forward(image)))

    def _validate_sinogram(self, sinogram: np.ndarray) -> None:
        expected_shape = (self.num_angles, self.det_shape)
        if np.shape(sinogram) != expected_shape:
            raise ValueError(
                f"expected sinogram shape {expected_shape}, got {np.shape(sinogram)}"
            )
