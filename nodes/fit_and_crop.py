import numpy as np
import torch
import cv2
import pandas as pd
from typing import Union, Optional, Tuple

from core.face_detector import FaceDetector
from core.image_processor import ImageProcessor


class FitAndCrop:
    """ComfyUI node for detecting, cropping, and fitting faces to a square bbox."""

    def __init__(self):
        self.face_detector = FaceDetector()

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "padding_percent": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05
                }),
                "bbox_size": (["512", "1024", "2048"], {
                    "default": "1024"
                })
            }
        }

    RETURN_TYPES = ("IMAGE", "FACE_SETTINGS")
    RETURN_NAMES = ("image", "face_settings")
    FUNCTION = "process_image"
    CATEGORY = "Face Processor"

    def process_image(self, image, padding_percent=0.0, bbox_size="1024"):
        # Convert tensor to numpy
        image_np = self._convert_to_numpy(image)
        if image_np is None:
            return (image, {})

        # Detect facial landmarks
        landmarks_df = self.face_detector.detect_landmarks(image_np)
        if landmarks_df is None:
            print("No face detected, returning original image")
            return (image, {})

        # Calculate rotation angle and rotate the image
        rotation_angle = ImageProcessor.calculate_rotation_angle(landmarks_df)
        rotated_image, updated_landmarks = ImageProcessor.rotate_image(image_np, landmarks_df)
        if rotated_image is None:
            print("Failed to rotate image, returning original")
            return (image, {})

        # Crop face region to a square (1:1)
        cropped_face, crop_bbox = ImageProcessor.crop_face_to_square(rotated_image, updated_landmarks, padding_percent)
        if cropped_face is None:
            print("Failed to crop face, returning original image")
            return (image, {})

        # Resize to target size
        target_size = int(bbox_size)
        final_image = ImageProcessor.resize_image(cropped_face, target_size)
        if final_image is None:
            print("Failed to resize image, returning original")
            return (image, {})

        # Convert back to float32 format expected by ComfyUI
        final_image = final_image.astype(np.float32) / 255.0
        final_image = torch.from_numpy(final_image).unsqueeze(0)

        # Save face settings for restoration
        face_settings = {
            "original_image_shape": image_np.shape,
            "rotation_angle": rotation_angle,
            "crop_bbox": crop_bbox,  # (x, y, w, h)
            "padding_percent": padding_percent,
            "bbox_size": target_size,
        }

        return (final_image, face_settings)

    def _convert_to_numpy(self, image: torch.Tensor) -> Optional[np.ndarray]:
        """Convert tensor to numpy array."""
        if torch.is_tensor(image):
            image = image.detach().cpu().numpy()
            if len(image.shape) == 4:
                image = image[0]
            image = (image * 255).astype(np.uint8)
            return image
        return None