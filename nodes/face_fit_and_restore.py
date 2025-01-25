import numpy as np
import torch
import cv2
import pandas as pd
from typing import Union, Optional, Tuple

from core.face_detector import FaceDetector


class FaceFitAndRestore:
    """ComfyUI node for fitting and restoring faces with a mode selection."""

    def __init__(self):
        self.face_detector = FaceDetector()

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mode": (["Fit", "Restore"], {
                    "default": "Fit"
                }),
                "image": ("IMAGE",),
                "padding_percent": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05
                }),
                "bbox_size": (["512", "1024", "2048"], {
                    "default": "1024"
                }),
            },
            "optional": {
                "face_settings": ("FACE_SETTINGS", {
                    "default": None
                }),
            }
        }

    RETURN_TYPES = ("IMAGE", "FACE_SETTINGS", "MASK")
    RETURN_NAMES = ("image", "face_settings", "mask")
    FUNCTION = "process_image"
    CATEGORY = "Face Processor"

    def process_image(self, mode, image, padding_percent=0.0, bbox_size="1024", face_settings=None):
        if mode == "Fit":
            return self._fit(image, padding_percent, bbox_size)
        elif mode == "Restore":
            if face_settings is None:
                print("Face settings are required in Restore mode, returning original image")
                return (image, {}, self._create_empty_mask(image))
            return self._restore(image, face_settings)
        else:
            print(f"Invalid mode: {mode}, returning original image")
            return (image, {}, self._create_empty_mask(image))

    def _fit(self, image, padding_percent, bbox_size):
        """Fit mode: Crop and process the face."""
        # Convert tensor to numpy
        image_np = self._convert_to_numpy(image)
        if image_np is None:
            return (image, {}, self._create_empty_mask(image))

        # Detect facial landmarks
        landmarks_df = self.face_detector.detect_landmarks(image_np)
        if landmarks_df is None:
            print("No face detected, returning original image")
            return (image, {}, self._create_empty_mask(image))

        # Calculate rotation angle and rotate the image
        rotation_angle = self._calculate_rotation_angle(landmarks_df)
        rotated_image, updated_landmarks = self._rotate_image(image_np, landmarks_df)
        if rotated_image is None:
            print("Failed to rotate image, returning original")
            return (image, {}, self._create_empty_mask(image))

        # Crop face region to a square (1:1)
        cropped_face, crop_bbox = self._crop_face_to_square(rotated_image, updated_landmarks, padding_percent)
        if cropped_face is None:
            print("Failed to crop face, returning original image")
            return (image, {}, self._create_empty_mask(image))

        # Resize to target size
        target_size = int(bbox_size)
        final_image = self._resize_image(cropped_face, target_size)
        if final_image is None:
            print("Failed to resize image, returning original")
            return (image, {}, self._create_empty_mask(image))

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

        # Create a mask for the face area (1:1)
        # Initialize mask with ones (all pixels are part of the face)
        mask = np.ones((target_size, target_size), dtype=np.float32)

        # If there was a rotation, exclude areas that are outside the original image
        if rotation_angle != 0:
            # Create a blank image with the original size
            original_mask = np.ones(image_np.shape[:2], dtype=np.float32)

            # Rotate the original mask to match the rotated image
            height, width = image_np.shape[:2]
            center = (width // 2, height // 2)
            rotation_matrix = cv2.getRotationMatrix2D(center, rotation_angle, 1.0)
            rotated_mask = cv2.warpAffine(original_mask, rotation_matrix, (width, height), flags=cv2.INTER_LINEAR)

            # Crop the rotated mask to the face region
            x1, y1, w, h = crop_bbox
            cropped_mask = rotated_mask[y1:y1 + h, x1:x1 + w]

            # Resize the cropped mask to the target size
            resized_mask = cv2.resize(cropped_mask, (target_size, target_size), interpolation=cv2.INTER_LINEAR)

            # Update the final mask to exclude areas outside the original image
            mask = resized_mask

        # Convert mask to torch tensor
        mask = torch.from_numpy(mask).unsqueeze(0)

        return (final_image, face_settings, mask)

    def _restore(self, image, face_settings):
        """Restore mode: Restore the face to the original image."""
        # Convert tensor to numpy
        processed_face_np = self._convert_to_numpy(image)
        if processed_face_np is None:
            return (image, {}, self._create_empty_mask(image))

        # Extract face settings
        original_image_shape = face_settings.get("original_image_shape")
        rotation_angle = face_settings.get("rotation_angle")
        crop_bbox = face_settings.get("crop_bbox")
        padding_percent = face_settings.get("padding_percent")
        bbox_size = face_settings.get("bbox_size")

        if not all([original_image_shape, crop_bbox]):
            print("Invalid face settings, returning processed face")
            return (image, {}, self._create_empty_mask(image))

        # Create a blank image with the original size
        restored_image = np.zeros(original_image_shape, dtype=np.uint8)

        # Resize the processed face back to the original crop size
        x1, y1, w, h = crop_bbox
        resized_face = cv2.resize(processed_face_np, (w, h), interpolation=cv2.INTER_LANCZOS4)

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

        # Create a mask for the face area (original image size)
        mask = self._create_mask(original_image_shape, crop_bbox, rotation_angle)

        return (restored_image, {}, mask)

    def _create_mask(self, image_shape, crop_bbox, rotation_angle):
        """Create a mask for the face area."""
        mask = np.zeros(image_shape[:2], dtype=np.float32)
        x1, y1, w, h = crop_bbox

        # Draw a white rectangle for the face area
        mask[y1:y1 + h, x1:x1 + w] = 1.0

        # Rotate the mask back to the original orientation (reverse the rotation)
        if rotation_angle != 0:
            height, width = image_shape[:2]
            center = (width // 2, height // 2)
            rotation_matrix = cv2.getRotationMatrix2D(center, -rotation_angle, 1.0)
            mask = cv2.warpAffine(mask, rotation_matrix, (width, height), flags=cv2.INTER_LINEAR)

        # Convert to torch tensor
        mask = torch.from_numpy(mask).unsqueeze(0)
        return mask

    def _create_empty_mask(self, image):
        """Create an empty mask with the same shape as the input image."""
        if torch.is_tensor(image):
            image_shape = image.shape[1:3]  # (H, W)
        else:
            image_shape = image.shape[:2]  # (H, W)
        mask = torch.zeros((1, *image_shape), dtype=torch.float32)
        return mask

    def _convert_to_numpy(self, image: torch.Tensor) -> Optional[np.ndarray]:
        """Convert tensor to numpy array."""
        if torch.is_tensor(image):
            image = image.detach().cpu().numpy()
            if len(image.shape) == 4:
                image = image[0]
            image = (image * 255).astype(np.uint8)
            return image
        return None

    def _calculate_rotation_angle(self, landmarks_df: pd.DataFrame) -> float:
        """Calculate rotation angle based on eyes position."""
        if landmarks_df is None or landmarks_df.empty:
            return 0.0

        LEFT_EYE = 33  # Center of the left eye
        RIGHT_EYE = 263  # Center of the right eye

        left_eye = landmarks_df[landmarks_df['index'] == LEFT_EYE].iloc[0]
        right_eye = landmarks_df[landmarks_df['index'] == RIGHT_EYE].iloc[0]

        dx = right_eye['x'] - left_eye['x']
        dy = right_eye['y'] - left_eye['y']
        return np.degrees(np.arctan2(dy, dx))

    def _rotate_image(self, image: np.ndarray, landmarks_df: pd.DataFrame) -> Tuple[Optional[np.ndarray], Optional[pd.DataFrame]]:
        """Rotate image based on facial landmarks."""
        if image is None or landmarks_df is None:
            return None, None

        angle = self._calculate_rotation_angle(landmarks_df)
        height, width = image.shape[:2]
        center = (width // 2, height // 2)

        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height), flags=cv2.INTER_LANCZOS4)

        # Transform landmarks
        ones = np.ones(shape=(len(landmarks_df), 1))
        points = np.hstack([landmarks_df[['x', 'y']].values, ones])
        transformed_points = rotation_matrix.dot(points.T).T

        updated_landmarks = landmarks_df.copy()
        updated_landmarks['x'] = transformed_points[:, 0]
        updated_landmarks['y'] = transformed_points[:, 1]

        return rotated_image, updated_landmarks

    def _crop_face_to_square(self, image: np.ndarray, landmarks_df: pd.DataFrame, padding_percent: float = 0.0) -> Tuple[Optional[np.ndarray], Optional[Tuple[int, int, int, int]]]:
        """Crop face region to a square (1:1) based on landmarks."""
        bbox = self._calculate_face_bbox(landmarks_df, padding_percent)
        if bbox is None:
            return None, None

        x, y, w, h = bbox

        # Calculate the center of the bounding box
        center_x = x + w // 2
        center_y = y + h // 2

        # Determine the size of the square crop
        crop_size = max(w, h)
        half_size = crop_size // 2

        # Calculate the crop coordinates
        x1 = max(0, center_x - half_size)
        y1 = max(0, center_y - half_size)
        x2 = min(image.shape[1], center_x + half_size)
        y2 = min(image.shape[0], center_y + half_size)

        # Adjust if the crop goes out of bounds
        if x2 - x1 < crop_size:
            x1 = max(0, x2 - crop_size)
        if y2 - y1 < crop_size:
            y1 = max(0, y2 - crop_size)

        # Crop the image
        cropped_face = image[y1:y2, x1:x2]

        # Return the cropped face and the crop bounding box
        return cropped_face, (x1, y1, x2 - x1, y2 - y1)

    def _calculate_face_bbox(self, landmarks_df: pd.DataFrame, padding_percent: float = 0.0) -> Optional[Tuple[int, int, int, int]]:
        """Calculate bounding box for face based on landmarks."""
        if landmarks_df is None or landmarks_df.empty:
            return None

        min_x = landmarks_df['x'].min()
        max_x = landmarks_df['x'].max()
        min_y = landmarks_df['y'].min()
        max_y = landmarks_df['y'].max()

        width = max_x - min_x
        height = max_y - min_y

        pad_x = width * padding_percent
        pad_y = height * padding_percent

        x1 = max(0, min_x - pad_x)
        y1 = max(0, min_y - pad_y)
        x2 = max_x + pad_x
        y2 = max_y + pad_y

        return (int(x1), int(y1), int(x2 - x1), int(y2 - y1))

    def _resize_image(self, image: np.ndarray, target_size: int) -> Optional[np.ndarray]:
        """Resize image to target size while maintaining aspect ratio."""
        if image is None:
            return None

        h, w = image.shape[:2]
        scale = min(target_size / h, target_size / w)
        new_h = int(h * scale)
        new_w = int(w * scale)

        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

        square = np.zeros((target_size, target_size, image.shape[2]), dtype=resized.dtype)
        y_offset = (target_size - new_h) // 2
        x_offset = (target_size - new_w) // 2

        square[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized

        return square