"""Facial landmark detectors and refiners for FaceProcessor."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

import cv2
import numpy as np

from core.image_processor import ImageProcessor
from core.lm_mapping import LandmarkMappings
from core.resources.model_loader import ModelMediaPipe

from .utils import get_logger

try:  # Optional dependency used only when dlib is available.
    from core.resources.model_loader import ModelDlib  # type: ignore
except Exception:  # pragma: no cover - exercised only when dlib missing.
    ModelDlib = None  # type: ignore


class FaceLandmarksBase(ABC):
    """Base class shared between landmark detectors/refiners."""

    def __init__(self, name: str = "FaceLandmarks") -> None:
        self.logger = get_logger(name)
        self._image_processor = ImageProcessor()

    def _ensure_numpy_image(self, image) -> Optional[np.ndarray]:
        if image is None:
            return None
        if isinstance(image, np.ndarray):
            return image
        try:
            return self._image_processor.convert_to_numpy(image)
        except Exception as exc:  # pragma: no cover - defensive logging
            self.logger.error("Failed to convert image to numpy: %s", exc)
            return None

    @abstractmethod
    def detect(self, image) -> Optional[np.ndarray]:
        """Return pixel-space facial landmarks shaped ``[N, 2]``."""


class MediaPipeLandmarks(FaceLandmarksBase):
    """MediaPipe Tasks backed detector that always yields canonical 468-point meshes."""

    CANONICAL_LANDMARKS = 468
    TASKS_LANDMARKS = 478
    _CANONICAL_INDICES = np.arange(CANONICAL_LANDMARKS, dtype=np.int64)

    def __init__(self) -> None:
        super().__init__("MediaPipeLandmarks")
        self._model: Optional[ModelMediaPipe] = None

    def detect(self, image) -> Optional[np.ndarray]:  # type: ignore[override]
        """Return ``[468, 2]`` pixel-space landmarks for ``image`` when a face is found."""
        np_img = self._ensure_numpy_image(image)
        if np_img is None:
            return None

        model = self._get_model()
        try:
            results = model.face_mesh.process(np_img)
        except RuntimeError as exc:
            # MediaPipe Tasks sometimes closes the runner; reinitialize on demand.
            self.logger.warning("MediaPipe detection failed: %s", exc)
            self._model = None
            model = self._get_model()
            results = model.face_mesh.process(np_img)

        face_landmarks_list = getattr(results, "multi_face_landmarks", None)
        if not face_landmarks_list:
            face_landmarks_list = getattr(results, "face_landmarks", None)
        if not face_landmarks_list:
            self.logger.info("MediaPipe did not detect a face")
            return None

        face_landmarks = face_landmarks_list[0]
        image_height, image_width = np_img.shape[:2]
        coords: List[Tuple[float, float]] = []
        landmarks_iter = getattr(face_landmarks, "landmark", face_landmarks)

        for landmark in landmarks_iter:
            coords.append((landmark.x * image_width, landmark.y * image_height))

        coords_array = np.asarray(coords, dtype=np.float32)
        normalized = self._normalize_landmarks(coords_array)
        return normalized

    def _normalize_landmarks(self, coords: np.ndarray) -> Optional[np.ndarray]:
        count = coords.shape[0]
        if count == self.CANONICAL_LANDMARKS:
            return coords
        if count == self.TASKS_LANDMARKS:
            # MediaPipe Tasks exposes iris landmarks appended at the end. Drop them
            # to maintain a canonical 468-point representation.
            return coords[self._CANONICAL_INDICES]

        if count < self.CANONICAL_LANDMARKS:
            self.logger.warning(
                "Unexpected landmark count %s (expected %s)",
                count,
                self.CANONICAL_LANDMARKS,
            )
            return None

        self.logger.info(
            "Landmark count %s larger than expected – truncating to %s",
            count,
            self.CANONICAL_LANDMARKS,
        )
        return coords[: self.CANONICAL_LANDMARKS]

    def _get_model(self) -> ModelMediaPipe:
        if self._model is None:
            self._model = ModelMediaPipe()
        return self._model


class DlibRefiner(FaceLandmarksBase):
    """Optional Dlib-based refiner that overwrites specific landmark groups."""

    _DIRECT_MAPPINGS: Tuple[Tuple[int, int], ...] = tuple(
        pair for pairs in LandmarkMappings.LANDMARKS_PAIRS.values() for pair in pairs
    )

    def __init__(self) -> None:
        super().__init__("DlibRefiner")
        self._model: Optional[ModelDlib] = None
        self._available: bool = self._try_initialize()
        self._warned_unavailable = False

    def detect(self, image) -> Optional[np.ndarray]:  # pragma: no cover - unused
        return None

    def refine(self, image, landmarks: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if landmarks is None:
            return None
        if not self._available:
            if not self._warned_unavailable:
                self.logger.info("dlib not available, skipping refinement")
                self._warned_unavailable = True
            return landmarks

        np_img = self._ensure_numpy_image(image)
        if np_img is None:
            return landmarks

        dlib_points = self._detect_dlib_landmarks(np_img)
        if dlib_points is None:
            return landmarks

        refined = landmarks.copy()
        for mp_idx, dlib_idx in self._DIRECT_MAPPINGS:
            if mp_idx >= refined.shape[0]:
                continue
            zero_based = dlib_idx - 1
            if zero_based < 0 or zero_based >= dlib_points.shape[0]:
                continue
            refined[mp_idx] = dlib_points[zero_based]
        return refined

    def _try_initialize(self) -> bool:
        if ModelDlib is None:
            return False
        try:
            self._model = ModelDlib()
            return True
        except Exception as exc:  # pragma: no cover - exercised when dlib absent
            self.logger.warning("Failed to initialize Dlib models: %s", exc)
            self._model = None
            return False

    def _detect_dlib_landmarks(self, image: np.ndarray) -> Optional[np.ndarray]:
        if self._model is None:
            return None
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        except Exception as exc:
            self.logger.warning("Failed to convert image for dlib refinement: %s", exc)
            return None

        try:
            faces = self._model.face_detector(gray)
        except Exception as exc:
            self.logger.warning("Dlib face detector error: %s", exc)
            return None

        if not faces:
            return None

        try:
            shape = self._model.shape_predictor(gray, faces[0])
        except Exception as exc:
            self.logger.warning("Dlib shape predictor error: %s", exc)
            return None

        coords = np.zeros((shape.num_parts, 2), dtype=np.float32)
        for i in range(shape.num_parts):
            part = shape.part(i)
            coords[i] = (float(part.x), float(part.y))
        return coords

