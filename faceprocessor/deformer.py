"""Torch-based face deformer used by the FaceProcessor pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F


@dataclass
class TorchDeformer:
    """Barycentric triangle-based image deformer implemented with PyTorch."""

    device: Optional[torch.device] = None
    dtype: torch.dtype = torch.float32
    padding_mode: str = "zeros"
    eps: float = 1e-4

    def warp(
        self,
        src_image: torch.Tensor,
        src_landmarks: torch.Tensor,
        dst_landmarks: torch.Tensor,
        triangles: torch.Tensor,
    ) -> torch.Tensor:
        """Warp ``src_image`` using triangle mappings from ``src_landmarks`` to ``dst_landmarks``."""

        if src_image.ndim != 4 or src_image.shape[0] != 1:
            raise ValueError("src_image must be shaped as [1, H, W, C]")

        device = self.device or src_image.device
        dtype = self.dtype

        image = src_image.to(device=device, dtype=dtype)
        src = src_landmarks.to(device=device, dtype=dtype)
        dst = dst_landmarks.to(device=device, dtype=dtype)
        tris = triangles.to(device=device, dtype=torch.long)

        _, height, width, _ = image.shape

        grid = torch.full((1, height, width, 2), -2.0, dtype=dtype, device=device)

        arange_x = torch.arange(width, device=device, dtype=dtype)
        arange_y = torch.arange(height, device=device, dtype=dtype)

        for tri_indices in tris:
            dst_tri = dst[tri_indices]
            src_tri = src[tri_indices]

            min_x = int(torch.floor(dst_tri[:, 0].min()).clamp(0, width - 1).item())
            max_x = int(torch.ceil(dst_tri[:, 0].max()).clamp(0, width - 1).item())
            min_y = int(torch.floor(dst_tri[:, 1].min()).clamp(0, height - 1).item())
            max_y = int(torch.ceil(dst_tri[:, 1].max()).clamp(0, height - 1).item())

            if min_x >= max_x or min_y >= max_y:
                continue

            dst_matrix = torch.stack(
                (dst_tri[1] - dst_tri[0], dst_tri[2] - dst_tri[0]), dim=1
            )

            if torch.abs(torch.linalg.det(dst_matrix)) < self.eps:
                continue

            dst_inv = torch.inverse(dst_matrix)

            xs = arange_x[min_x : max_x + 1]
            ys = arange_y[min_y : max_y + 1]
            yy, xx = torch.meshgrid(ys, xs, indexing="ij")

            coords = torch.stack((xx, yy), dim=-1).reshape(-1, 2)
            rel = coords - dst_tri[0]
            uv = rel @ dst_inv

            u = uv[:, 0]
            v = uv[:, 1]
            w = 1.0 - u - v

            inside = (
                (u >= -self.eps)
                & (v >= -self.eps)
                & (w >= -self.eps)
                & (u <= 1 + self.eps)
                & (v <= 1 + self.eps)
                & (w <= 1 + self.eps)
            )

            if not torch.any(inside):
                continue

            src_coords = (
                src_tri[0] * w[:, None]
                + src_tri[1] * u[:, None]
                + src_tri[2] * v[:, None]
            )

            selected = coords[inside]
            src_selected = src_coords[inside]

            grid_x = (2.0 * src_selected[:, 0] / max(width - 1, 1)) - 1.0
            grid_y = (2.0 * src_selected[:, 1] / max(height - 1, 1)) - 1.0

            xs_sel = selected[:, 0].long()
            ys_sel = selected[:, 1].long()

            grid[0, ys_sel, xs_sel, 0] = grid_x
            grid[0, ys_sel, xs_sel, 1] = grid_y

        image_chw = image.permute(0, 3, 1, 2)
        warped = F.grid_sample(
            image_chw,
            grid,
            mode="bilinear",
            align_corners=True,
            padding_mode=self.padding_mode,
        )

        result = warped.permute(0, 2, 3, 1)
        return result
