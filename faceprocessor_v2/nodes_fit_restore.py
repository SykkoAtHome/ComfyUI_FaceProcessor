"""FaceFitAndRestore v2 node implementation."""

from __future__ import annotations

from dataclasses import replace
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np
import pandas as pd
import torch

from core.image_processor import ImageProcessor

from .landmarks import DlibRefiner, MediaPipeLandmarks
from .pipe import FacePipe, FrameData, face_pipe_from_legacy, face_pipe_to_legacy, get_or_create_frame, update_frame
from .frame_ids import resolve_frame_ids
from .utils import ensure_face_model, get_logger, normalize_image_input, numpy_to_tensor, tensor_to_numpy


class FaceFitAndRestoreV2:
    """Detects faces, prepares canonical crops, and restores frames via FacePipe."""

    CATEGORY = "Face Processor"
    FUNCTION = "process"
    RETURN_TYPES = ("IMAGE", "MASK", "DICT")
    RETURN_NAMES = ("image", "mask", "fp_pipe")

    def __init__(self) -> None:
        self.image_processor = ImageProcessor()
        self.landmark_detector = MediaPipeLandmarks()
        self.dlib_refiner = DlibRefiner()
        self.logger = get_logger("FaceFitAndRestoreV2")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mode": (["Fit", "Restore"], {"default": "Fit"}),
                "bbox_size": (["512", "1024", "2048"], {"default": "1024"}),
                "padding_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 0.5, "step": 0.01}),
            },
            "optional": {
                "image": ("IMAGE",),
                "image_paths": ("STRING", {"forceInput": True}),
                "fp_pipe": ("DICT", {"default": None}),
            },
        }

    def process(
        self,
        mode: str,
        bbox_size: str,
        padding_percent: float = 0.0,
        image: Optional[torch.Tensor] = None,
        image_paths: Optional[Sequence[str]] = None,
        fp_pipe: Optional[dict] = None,
    ):
        """Execute the selected Fit/Restore branch on the provided batch of frames."""
        if image is None and image_paths is None:
            raise ValueError("Either 'image' or 'image_paths' input must be provided")

        source_payload = image if image is not None else image_paths
        _, tensors, normalized_frame_ids = normalize_image_input(source_payload)
        if not tensors:
            raise ValueError("normalize_image_input returned no frames to process")
        source_paths: Optional[List[Optional[str]]] = None

        pipe = face_pipe_from_legacy(fp_pipe)
        bbox_size_int = int(bbox_size)
        if mode == "Fit":
            frame_ids = normalized_frame_ids
        else:
            frame_ids = resolve_frame_ids(pipe, len(tensors), fallback=normalized_frame_ids)

        if len(frame_ids) != len(tensors):
            raise ValueError("Frame id list length mismatch for FaceFitAndRestoreV2 batch")

        if mode == "Fit":
            source_paths = self._build_source_path_list(source_payload, len(tensors))
            images, masks = self._run_fit(
                tensors=tensors,
                frame_ids=frame_ids,
                source_paths=source_paths,
                pipe=pipe,
                padding_percent=padding_percent,
                bbox_size=bbox_size_int,
            )
        elif mode == "Restore":
            images, masks = self._run_restore(tensors=tensors, frame_ids=frame_ids, pipe=pipe)
        else:
            raise ValueError(f"Unsupported mode: {mode}")

        legacy_pipe = face_pipe_to_legacy(pipe)
        return images, masks, legacy_pipe

    # ------------------------------------------------------------------
    # Fit mode helpers
    # ------------------------------------------------------------------
    def _run_fit(
        self,
        tensors: List[torch.Tensor],
        frame_ids: List[str],
        source_paths: Optional[List[Optional[str]]],
        pipe: FacePipe,
        padding_percent: float,
        bbox_size: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        face_batches: List[torch.Tensor] = []
        mask_batches: List[torch.Tensor] = []

        if source_paths is None:
            raise ValueError("Fit mode requires source paths to be resolved")

        if not (len(tensors) == len(frame_ids) == len(source_paths)):
            raise ValueError("Input batches must share the same length in Fit mode")

        for tensor, frame_id, source_path in zip(tensors, frame_ids, source_paths):
            face_tensor, mask_tensor, frame_data = self._fit_single(
                tensor,
                frame_id,
                pipe,
                padding_percent,
                bbox_size,
                source_path,
            )

            face_batches.append(face_tensor)
            mask_batches.append(mask_tensor)

            if frame_data is not None:
                update_frame(pipe, frame_data)

        pipe.meta["frame_order"] = list(frame_ids)

        return torch.cat(face_batches, dim=0), torch.cat(mask_batches, dim=0)

    def _fit_single(
        self,
        tensor: torch.Tensor,
        frame_id: str,
        pipe: FacePipe,
        padding_percent: float,
        bbox_size: int,
        source_path: Optional[str],
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[FrameData]]:
        image_np = self.image_processor.convert_to_numpy(tensor)
        if image_np is None:
            self.logger.warning("Failed to convert tensor to numpy for frame %s", frame_id)
            return tensor, self._solid_mask(tensor.shape[1:3]), None

        landmarks = self.landmark_detector.detect(image_np)
        if landmarks is None:
            self.logger.warning("No landmarks detected for frame %s", frame_id)
            fallback = self.image_processor.resize_image(image_np, bbox_size)
            if fallback is not None:
                return (
                    numpy_to_tensor(fallback),
                    self._solid_mask((bbox_size, bbox_size)),
                    self._build_placeholder_frame(frame_id, image_np.shape, source_path, pipe),
                )
            return tensor, self._solid_mask(tensor.shape[1:3]), None

        refined_landmarks = self.dlib_refiner.refine(image_np, landmarks)
        if refined_landmarks is not None:
            landmarks = refined_landmarks

        landmarks_df = _landmarks_array_to_df(landmarks)

        rotation_angle = self.image_processor.calculate_rotation_angle(landmarks_df)
        rotated_image, rotated_landmarks = self.image_processor.rotate_image(image_np, landmarks_df)
        if rotated_image is None or rotated_landmarks is None:
            self.logger.warning("Rotation failed for frame %s", frame_id)
            return tensor, self._solid_mask(tensor.shape[1:3]), None

        cropped_face, crop_bbox = self.image_processor.crop_face_to_square(
            rotated_image, rotated_landmarks, padding_percent
        )
        if cropped_face is None or crop_bbox is None:
            self.logger.warning("Cropping failed for frame %s", frame_id)
            return tensor, self._solid_mask(tensor.shape[1:3]), None

        final_image = self.image_processor.resize_image(cropped_face, bbox_size)
        if final_image is None:
            self.logger.warning("Resizing failed for frame %s", frame_id)
            return tensor, self._solid_mask(tensor.shape[1:3]), None

        canonical_model = ensure_face_model(pipe, bbox_size, "canonical_models")
        canonical_landmarks = np.asarray(canonical_model["landmarks"], dtype=np.float32)

        face_tensor = numpy_to_tensor(final_image)
        mask_tensor = self._solid_mask((bbox_size, bbox_size))

        frame_data = replace(
            get_or_create_frame(pipe, frame_id),
            frame_id=frame_id,
            orig_size=(image_np.shape[0], image_np.shape[1]),
            bbox=crop_bbox,
            detected_lm=landmarks.astype(np.float32),
            target_lm=canonical_landmarks,
            extra={
                "rotation_angle": float(rotation_angle),
                "bbox_size": bbox_size,
                "padding_percent": float(padding_percent),
                "source_path": source_path,
            },
        )

        return face_tensor, mask_tensor, frame_data

    # ------------------------------------------------------------------
    # Restore mode helpers
    # ------------------------------------------------------------------
    def _run_restore(
        self,
        tensors: List[torch.Tensor],
        frame_ids: List[str],
        pipe: FacePipe,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        restored_batches: List[torch.Tensor] = []
        mask_batches: List[torch.Tensor] = []

        if len(tensors) != len(frame_ids):
            raise ValueError("Restore mode tensors/frame_ids length mismatch")

        for tensor, frame_id in zip(tensors, frame_ids):
            frame_data = pipe.frames.get(frame_id)
            if frame_data is None:
                self.logger.warning("Missing FrameData for frame %s", frame_id)
                restored_batches.append(tensor)
                mask_batches.append(self._solid_mask(tensor.shape[1:3]))
                continue

            restored_tensor, mask_tensor = self._restore_single(tensor, frame_data)
            restored_batches.append(restored_tensor)
            mask_batches.append(mask_tensor)

        return torch.cat(restored_batches, dim=0), torch.cat(mask_batches, dim=0)

    def _restore_single(self, tensor: torch.Tensor, frame_data: FrameData) -> Tuple[torch.Tensor, torch.Tensor]:
        if frame_data.bbox == (0, 0, 0, 0):
            self.logger.warning("Invalid bbox for frame %s", frame_data.frame_id)
            return tensor, self._solid_mask(tensor.shape[1:3])

        face_np = tensor_to_numpy(tensor)
        x, y, w, h = frame_data.bbox
        orig_h, orig_w = frame_data.orig_size

        restored_image = np.zeros((orig_h, orig_w, 3), dtype=np.uint8)
        resized_face = cv2.resize(face_np, (w, h), interpolation=cv2.INTER_LANCZOS4)
        restored_image[y : y + h, x : x + w] = resized_face

        rotation_angle = float(frame_data.extra.get("rotation_angle", 0.0))
        if rotation_angle:
            center = (orig_w // 2, orig_h // 2)
            rotation_matrix = cv2.getRotationMatrix2D(center, -rotation_angle, 1.0)
            restored_image = cv2.warpAffine(
                restored_image,
                rotation_matrix,
                (orig_w, orig_h),
                flags=cv2.INTER_LANCZOS4,
            )

        mask_tensor = self._restore_mask(frame_data, rotation_angle)
        restored_tensor = numpy_to_tensor(restored_image)
        return restored_tensor, mask_tensor

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------
    def _solid_mask(self, spatial: Tuple[int, int]) -> torch.Tensor:
        h, w = spatial
        return torch.ones((1, h, w), dtype=torch.float32)

    def _restore_mask(self, frame_data: FrameData, rotation_angle: float) -> torch.Tensor:
        h, w = frame_data.orig_size
        mask = np.zeros((h, w), dtype=np.float32)
        x, y, bw, bh = frame_data.bbox
        mask[y : y + bh, x : x + bw] = 1.0

        if rotation_angle:
            center = (w // 2, h // 2)
            rotation_matrix = cv2.getRotationMatrix2D(center, -rotation_angle, 1.0)
            mask = cv2.warpAffine(mask, rotation_matrix, (w, h), flags=cv2.INTER_LINEAR)

        return torch.from_numpy(mask).unsqueeze(0)

    def _build_placeholder_frame(
        self,
        frame_id: str,
        image_shape: Tuple[int, int, int],
        source_path: Optional[str],
        pipe: FacePipe,
    ) -> FrameData:
        placeholder = replace(
            get_or_create_frame(pipe, frame_id),
            frame_id=frame_id,
            orig_size=(image_shape[0], image_shape[1]),
            bbox=(0, 0, 0, 0),
            detected_lm=None,
            extra={"error": "no_face_detected", "source_path": source_path},
        )
        return placeholder

    def _build_source_path_list(self, payload, expected_len: int) -> List[Optional[str]]:
        paths: List[Optional[str]] = [None] * expected_len
        if isinstance(payload, str):
            paths = [payload]
        elif isinstance(payload, Sequence) and not isinstance(payload, torch.Tensor):
            paths = [str(p) if p is not None else None for p in payload]
        if len(paths) >= expected_len:
            return paths[:expected_len]
        return paths + [None] * max(0, expected_len - len(paths))


def _landmarks_array_to_df(landmarks: np.ndarray) -> pd.DataFrame:
    indices = np.arange(len(landmarks), dtype=np.int32)
    return pd.DataFrame({"x": landmarks[:, 0], "y": landmarks[:, 1], "index": indices})
