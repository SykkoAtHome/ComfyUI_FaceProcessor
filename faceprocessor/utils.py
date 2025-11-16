"""Utility helpers for the FaceProcessor pipeline."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Optional, Sequence

import numpy as np
from PIL import Image

try:  # pragma: no cover - optional torch dependency for imports
    import torch
except ImportError:  # pragma: no cover - optional torch dependency for imports
    torch = None  # type: ignore[assignment]

from ..core.base_mesh import MediapipeBaseLandmarks

if TYPE_CHECKING:
    from .pipe import FacePipe


def _require_torch() -> None:
    if torch is None:  # pragma: no cover - executed only when torch is unavailable
        raise RuntimeError(
            "PyTorch is required for this operation but is not installed in the current environment",
        )


def tensor_to_numpy(image: torch.Tensor) -> np.ndarray:
    """Convert a batched ComfyUI tensor ``[B, H, W, C]`` to ``[H, W, C]`` numpy image."""

    _require_torch()
    if image.ndim != 4:
        raise ValueError(
            f"Expected a 4D tensor shaped [B, H, W, C], got tensor with shape {tuple(image.shape)}",
        )

    batch = image.shape[0]
    if batch != 1:
        raise ValueError("tensor_to_numpy currently supports only a single image (batch size of 1).")

    tensor = image[0].detach().cpu().clamp(0.0, 1.0)
    np_img = (tensor.numpy() * 255.0).round().astype(np.uint8)
    return np_img


def numpy_to_tensor(np_img: np.ndarray) -> torch.Tensor:
    """Convert a numpy image ``[H, W, C]`` to a ComfyUI tensor ``[1, H, W, C]``."""

    _require_torch()
    if np_img.ndim != 3:
        raise ValueError(
            f"Expected a numpy array shaped [H, W, C], got array with shape {tuple(np_img.shape)}",
        )

    tensor = torch.from_numpy(np_img).float() / 255.0
    tensor = tensor.clamp(0.0, 1.0)
    tensor = tensor.unsqueeze(0)  # Add batch dimension.
    return tensor


def get_logger(name: str = "FaceProcessor", level: int = logging.INFO) -> logging.Logger:
    """Return a module-wide configured logger instance."""

    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("[%(levelname)s] %(name)s: %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.propagate = False
    logger.setLevel(level)
    return logger


def _ensure_tensor_4d(image: torch.Tensor) -> torch.Tensor:
    """Return ``image`` reshaped to ``[1, H, W, C]`` if possible."""

    _require_torch()
    if image.ndim == 3:
        return image.unsqueeze(0)
    if image.ndim == 4:
        return image
    raise ValueError(
        "normalize_image_input expected tensor shaped [H, W, C] or [B, H, W, C], "
        f"got shape {tuple(image.shape)}",
    )


def _tensor_to_batch_list(tensor: torch.Tensor) -> list[torch.Tensor]:
    """Split a ``[B, H, W, C]`` tensor into a list of ``[1, H, W, C]`` tensors."""

    _require_torch()
    tensor = tensor.detach().clone()
    return [tensor[i : i + 1] for i in range(tensor.shape[0])]


def _load_image_tensor(path: str) -> torch.Tensor:
    """Load an image from ``path`` and convert it to ``[1, H, W, 3]`` tensor."""

    _require_torch()
    img = Image.open(path).convert("RGB")
    np_img = np.array(img)
    return numpy_to_tensor(np_img)


def normalize_image_input(image_or_paths) -> tuple[str, list[torch.Tensor], list[str]]:
    """Normalize different image entrypoints into a common representation."""

    if torch is not None and isinstance(image_or_paths, torch.Tensor):
        tensor = _ensure_tensor_4d(image_or_paths)
        tensors = _tensor_to_batch_list(tensor)
        frame_ids = [f"frame_{idx:04d}" for idx in range(len(tensors))]
        return "tensor_batch", tensors, frame_ids

    if isinstance(image_or_paths, (str, Path)):
        image_or_paths = [image_or_paths]

    if isinstance(image_or_paths, Sequence):
        tensors: list[torch.Tensor] = []
        frame_ids: list[str] = []
        for path in image_or_paths:
            if not isinstance(path, (str, Path)):
                raise TypeError(
                    "normalize_image_input expected a sequence of paths when not given a tensor",
                )
            path = Path(path)
            tensors.append(_load_image_tensor(str(path)))
            frame_ids.append(path.stem or path.name)
        return "paths", tensors, frame_ids

    raise TypeError(
        "normalize_image_input supports torch.Tensor or a sequence of file paths. "
        f"Got value of type {type(image_or_paths)!r}.",
    )


def ensure_face_model(pipe: "FacePipe", size: int, bucket: str = "canonical_models") -> Dict[str, list]:
    """Ensure triangulation data for ``size`` exists in ``pipe.meta`` and return it."""

    models: Dict[str, Dict[str, list]] = pipe.meta.setdefault(bucket, {})
    key = str(size)
    model = models.get(key)
    if model is None:
        triangles, landmarks = MediapipeBaseLandmarks.get_face_triangles((size, size))
        model = {
            "size": size,
            "triangles": triangles.astype(np.int64).tolist(),
            "landmarks": landmarks.astype(np.float32).tolist(),
        }
        models[key] = model
    return model


