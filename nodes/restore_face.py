import numpy as np
import torch
import cv2
from typing import Union, Optional

class RestoreFace:
    """ComfyUI node to restore a processed face back to its original position in the image."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "processed_face": ("IMAGE",),
                "face_settings": ("FACE_SETTINGS",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "restore_face"
    CATEGORY = "Face Processor"

    def restore_face(self, processed_face, face_settings):
        # Convert processed face to numpy
        processed_face_np = self._convert_to_numpy(processed_face)
        if processed_face_np is None:
            return (processed_face,)

        # Extract face settings
        original_image_shape = face_settings.get("original_image_shape")
        rotation_angle = face_settings.get("rotation_angle")
        crop_bbox = face_settings.get("crop_bbox")
        padding_percent = face_settings.get("padding_percent")
        bbox_size = face_settings.get("bbox_size")

        if not all([original_image_shape, crop_bbox]):
            print("Invalid face settings, returning processed face")
            return (processed_face,)

        # Resize the processed face back to the original crop size
        x1, y1, w, h = crop_bbox
        resized_face = cv2.resize(processed_face_np, (w, h), interpolation=cv2.INTER_LANCZOS4)

        # Create a blank image with the original size
        restored_image = np.zeros(original_image_shape, dtype=np.uint8)

        # Place the resized face back into the rotated image
        restored_image[y1:y1 + h, x1:x1 + w] = resized_face

        # Rotate the image back to the original orientation (reverse the rotation)
        if rotation_angle != 0:
            height, width = original_image_shape[:2]
            center = (width // 2, height // 2)
            rotation_matrix = cv2.getRotationMatrix2D(center, -rotation_angle, 1.0)  # Reverse the rotation
            restored_image = cv2.warpAffine(restored_image, rotation_matrix, (width, height), flags=cv2.INTER_LANCZOS4)

        # Convert back to float32 format expected by ComfyUI
        restored_image = restored_image.astype(np.float32) / 255.0
        restored_image = torch.from_numpy(restored_image).unsqueeze(0)

        return (restored_image,)

    def _convert_to_numpy(self, image: torch.Tensor) -> Optional[np.ndarray]:
        """Convert tensor to numpy array."""
        if torch.is_tensor(image):
            image = image.detach().cpu().numpy()
            if len(image.shape) == 4:
                image = image[0]
            image = (image * 255).astype(np.uint8)
            return image
        return None