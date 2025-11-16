"""Typed data model for the FaceProcessor v2 pipeline."""
from __future__ import annotations

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, MutableMapping, Optional, Tuple, Union

import numpy as np
try:  # pragma: no cover - optional torch dependency
    import torch
except ImportError:  # pragma: no cover - optional torch dependency
    torch = None  # type: ignore[assignment]

LandmarkArray = Union[np.ndarray, "torch.Tensor"]


@dataclass
class FrameData:
    """Container describing a single processed frame."""

    frame_id: str
    orig_size: Tuple[int, int]
    bbox: Tuple[int, int, int, int]
    target_lm: Optional[LandmarkArray] = None
    detected_lm: Optional[LandmarkArray] = None
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FacePipe:
    """High level pipeline contract shared between Fit/Unwrap/Wrap/Restore."""

    frames: Dict[str, FrameData] = field(default_factory=dict)
    meta: Dict[str, Any] = field(default_factory=dict)


def get_or_create_frame(pipe: FacePipe, frame_id: str) -> FrameData:
    """Return ``frame_id`` from the pipe, creating a placeholder entry if missing."""

    if frame_id not in pipe.frames:
        pipe.frames[frame_id] = FrameData(
            frame_id=frame_id,
            orig_size=(0, 0),
            bbox=(0, 0, 0, 0),
        )
    return pipe.frames[frame_id]


def update_frame(pipe: FacePipe, frame: FrameData) -> None:
    """Store an updated :class:`FrameData` instance in the pipe."""

    pipe.frames[frame.frame_id] = frame


def face_pipe_from_legacy(fp_pipe: Optional[MutableMapping[str, Any]]) -> FacePipe:
    """Adapt the current dictionary-based ``fp_pipe`` into :class:`FacePipe`."""

    if fp_pipe is None:
        return FacePipe()

    frames: Dict[str, FrameData] = {}
    for frame_id, frame_info in fp_pipe.get("frames", {}).items():
        frames[frame_id] = _frame_from_legacy(frame_id, frame_info)

    meta = {k: v for k, v in fp_pipe.items() if k not in {"frames", "target_lm"}}

    target_lm = _landmarks_from_legacy(fp_pipe.get("target_lm"))
    if target_lm is not None:
        meta["target_lm"] = target_lm

    return FacePipe(frames=frames, meta=meta)


def face_pipe_to_legacy(pipe: FacePipe) -> Dict[str, Any]:
    """Convert :class:`FacePipe` back to the dict contract used by v1 nodes."""

    legacy: Dict[str, Any] = {k: v for k, v in pipe.meta.items() if k != "target_lm"}

    if (target_lm := pipe.meta.get("target_lm")) is not None:
        legacy["target_lm"] = _landmarks_to_legacy(target_lm)

    legacy_frames: Dict[str, Any] = {}
    for frame_id, frame in pipe.frames.items():
        legacy_frames[frame_id] = _frame_to_legacy(frame)
    legacy["frames"] = legacy_frames
    return legacy


def _frame_from_legacy(frame_id: str, frame_info: MutableMapping[str, Any]) -> FrameData:
    orig_size = _normalize_hw(frame_info.get("orig_size"))
    if orig_size == (0, 0):
        orig_size = _normalize_hw(frame_info.get("original_image_shape"))

    bbox = _normalize_bbox(frame_info.get("bbox"))
    if bbox == (0, 0, 0, 0):
        bbox = _normalize_bbox(frame_info.get("crop_bbox"))

    detected_lm = _landmarks_from_legacy(frame_info.get("detected_lm"))
    target_lm = _landmarks_from_legacy(frame_info.get("target_lm"))

    extra = {
        k: v
        for k, v in frame_info.items()
        if k not in {"orig_size", "original_image_shape", "bbox", "crop_bbox", "detected_lm"}
    }

    return FrameData(
        frame_id=frame_id,
        orig_size=orig_size,
        bbox=bbox,
        detected_lm=detected_lm,
        target_lm=target_lm,
        extra=extra,
    )


def _frame_to_legacy(frame: FrameData) -> Dict[str, Any]:
    legacy_frame: Dict[str, Any] = dict(frame.extra)

    legacy_frame["orig_size"] = list(frame.orig_size)
    legacy_frame["bbox"] = list(frame.bbox)

    if frame.detected_lm is not None:
        legacy_frame["detected_lm"] = _landmarks_to_legacy(frame.detected_lm)

    if frame.target_lm is not None:
        legacy_frame["target_lm"] = _landmarks_to_legacy(frame.target_lm)

    return legacy_frame


def _landmarks_from_legacy(data: Optional[MutableMapping[str, Any]]) -> Optional[np.ndarray]:
    if not data:
        return None

    x = np.asarray(data.get("x", []), dtype=np.float32)
    y = np.asarray(data.get("y", []), dtype=np.float32)
    if x.size == 0 or y.size == 0 or x.shape != y.shape:
        return None

    return np.stack([x, y], axis=-1)


def _landmarks_to_legacy(data: LandmarkArray) -> Dict[str, Any]:
    if torch is not None and isinstance(data, torch.Tensor):
        arr = data.detach().cpu().numpy()
    else:
        arr = np.asarray(data)

    if arr.ndim != 2 or arr.shape[1] < 2:
        raise ValueError("Landmark tensor must be shaped [N, 2]")

    arr = arr[:, :2]
    return {"x": arr[:, 0].tolist(), "y": arr[:, 1].tolist(), "indices": list(range(arr.shape[0]))}


def _normalize_hw(value: Any) -> Tuple[int, int]:
    if value is None:
        return (0, 0)

    if isinstance(value, (list, tuple)):
        if len(value) >= 2:
            return (int(value[0]), int(value[1]))
    return (0, 0)


def _normalize_bbox(value: Any) -> Tuple[int, int, int, int]:
    if value is None:
        return (0, 0, 0, 0)
    if isinstance(value, (list, tuple)) and len(value) >= 4:
        return (int(value[0]), int(value[1]), int(value[2]), int(value[3]))
    return (0, 0, 0, 0)
