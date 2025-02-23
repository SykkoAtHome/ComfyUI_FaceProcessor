import cv2
import numpy as np
import torch
from PIL import Image

from ..core.face_detector import FaceDetector
from ..core.image_processor import ImageProcessor


class FaceFitAndRestore:
    def __init__(self):
        self.face_detector = FaceDetector()
        self.image_processor = ImageProcessor()

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "workflow": (["single_image", "image_sequence"], {
                    "default": "single_image"
                }),
                "mode": (["Fit", "Restore"], {
                    "default": "Fit"
                }),
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
                "image": ("IMAGE",),
                "fp_pipe": ("DICT", {
                    "default": None
                }),
                "frames_data": ("DICT", {
                    "default": None
                }),
            }
        }

    RETURN_TYPES = ("IMAGE", "DICT", "MASK", "INT")
    RETURN_NAMES = ("image", "fp_pipe", "mask", "bbox_size")
    FUNCTION = "process_image"
    CATEGORY = "Face Processor"

    def process_image(self, mode, workflow, padding_percent=0.0, bbox_size="1024",
                      image=None, fp_pipe=None, frames_data=None):

        if mode == "Restore" and fp_pipe and "workflow" in fp_pipe:
            workflow = fp_pipe["workflow"]
            print(f"Using workflow from pipe: {workflow}")

        if workflow == "single_image":
            if image is None:
                print("Error: Image is required for single_image workflow")
                return None, {}, None, int(bbox_size)

            if mode == "Fit":
                result = self._fit(image, padding_percent, bbox_size)
            else:  # Restore
                if fp_pipe is None:
                    print("Error: fp_pipe is required in Restore mode")
                    return None, {}, None, int(bbox_size)
                result = self._restore(image, fp_pipe)

            # Add workflow info to settings
            result_settings = result[1]
            result_settings["workflow"] = "single_image"
            return result

        elif workflow == "image_sequence":
            if frames_data is None or not frames_data.get("frames"):
                print("Error: Valid frames_data is required for image_sequence workflow")
                return None, {}, None, int(bbox_size)

            sequence_settings = {
                "workflow": "image_sequence",
                "frames": {}
            }

            first_result = None
            total_frames = len(frames_data["frames"])
            print(f"Processing sequence of {total_frames} frames...")

            for frame_idx, frame_path in frames_data["frames"].items():
                print(f"Processing frame {frame_idx + 1}/{total_frames}")

                try:
                    # Load image from path
                    frame_tensor = self._load_image_from_path(frame_path)
                    if frame_tensor is None:
                        continue

                    if mode == "Fit":
                        result = self._fit(frame_tensor, padding_percent, bbox_size)
                        result[1]["original_image_path"] = frame_path
                    elif mode == "Restore":
                        frame_settings = fp_pipe.get("frames", {}).get(f"frame_{frame_idx}")
                        if frame_settings is None:
                            print(f"Warning: No processor settings for frame {frame_idx}")
                            continue

                        print(f"Processing restore for frame {frame_idx} with settings: {frame_settings}")
                        result = self._restore(frame_tensor, frame_settings)

                    if first_result is None:
                        first_result = result

                    sequence_settings["frames"][f"frame_{frame_idx}"] = result[1]

                except Exception as e:
                    print(f"Error processing frame {frame_idx}: {str(e)}")
                    continue

            if first_result is None:
                return None, sequence_settings, None, int(bbox_size)

            return first_result[0], sequence_settings, first_result[2], int(bbox_size)

    def _fit(self, image, padding_percent, bbox_size):
        """Fit mode: Crop and process the face."""
        image_np = self.image_processor.convert_to_numpy(image)
        if image_np is None:
            return image, {}, self._create_empty_mask(image), int(bbox_size)

        # Detect facial landmarks using mediapipe
        landmarks_df = self.face_detector.detect_landmarks_mp(image_np)
        if landmarks_df is None:
            print("No face detected, returning original image")
            return image, {}, self._create_empty_mask(image), int(bbox_size)

        # Calculate rotation angle and rotate the image
        rotation_angle = self.image_processor.calculate_rotation_angle(landmarks_df)
        rotated_image, updated_landmarks = self.image_processor.rotate_image(image_np, landmarks_df)
        if rotated_image is None:
            print("Failed to rotate image, returning original")
            return image, {}, self._create_empty_mask(image), int(bbox_size)

        # Crop face region to a square (1:1)
        cropped_face, crop_bbox = self.image_processor.crop_face_to_square(rotated_image, updated_landmarks,
                                                                           padding_percent)
        if cropped_face is None:
            print("Failed to crop face, returning original image")
            return image, {}, self._create_empty_mask(image), int(bbox_size)

        # Resize to target size
        target_size = int(bbox_size)
        final_image = self.image_processor.resize_image(cropped_face, target_size)
        if final_image is None:
            print("Failed to resize image, returning original")
            return image, {}, self._create_empty_mask(image), int(bbox_size)

        # Convert to tensor format
        final_image = self.image_processor.numpy_to_tensor(final_image)

        # Save processor settings for restoration
        fp_pipe = {
            "original_image_shape": image_np.shape,
            "rotation_angle": rotation_angle,
            "crop_bbox": crop_bbox,  # (x, y, w, h)
            "padding_percent": padding_percent,
            "bbox_size": target_size,
            "original_image_path": None
        }

        # Create a mask for the face area
        mask = np.ones((target_size, target_size), dtype=np.float32)

        if rotation_angle != 0:
            original_mask = np.ones(image_np.shape[:2], dtype=np.float32)
            height, width = image_np.shape[:2]
            center = (width // 2, height // 2)
            rotation_matrix = cv2.getRotationMatrix2D(center, rotation_angle, 1.0)
            rotated_mask = cv2.warpAffine(original_mask, rotation_matrix, (width, height), flags=cv2.INTER_LINEAR)

            x1, y1, w, h = crop_bbox
            cropped_mask = rotated_mask[y1:y1 + h, x1:x1 + w]
            resized_mask = cv2.resize(cropped_mask, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
            mask = resized_mask

        mask = self.image_processor.convert_mask_to_tensor(mask)
        return final_image, fp_pipe, mask, int(bbox_size)

    def _restore(self, image, fp_pipe):
        """Restore mode: Restore the face to the original image."""

        if fp_pipe is None:
            print("Error in restore: fp_pipe is None")
            return image, {}, self._create_empty_mask(image)

        processed_face_np = self.image_processor.convert_to_numpy(image)
        if processed_face_np is None:
            print("Error in restore: Failed to convert image to numpy array")
            return image, {}, self._create_empty_mask(image)

        original_image_shape = fp_pipe.get("original_image_shape")
        rotation_angle = fp_pipe.get("rotation_angle")
        crop_bbox = fp_pipe.get("crop_bbox")
        padding_percent = fp_pipe.get("padding_percent")
        bbox_size = fp_pipe.get("bbox_size")

        # Debug info
        print(f"Restore settings: shape={original_image_shape}, rotation={rotation_angle}, bbox={crop_bbox}")

        if not all([original_image_shape, crop_bbox]):
            missing = []
            if not original_image_shape:
                missing.append("original_image_shape")
            if not crop_bbox:
                missing.append("crop_bbox")
            print(f"Error in restore: Missing required settings: {', '.join(missing)}")
            return image, {}, self._create_empty_mask(image)

        try:
            restored_image = np.zeros(original_image_shape, dtype=np.uint8)

            x1, y1, w, h = crop_bbox
            resized_face = cv2.resize(processed_face_np, (w, h), interpolation=cv2.INTER_LANCZOS4)

            restored_image[y1:y1 + h, x1:x1 + w] = resized_face

            if rotation_angle != 0:
                height, width = original_image_shape[:2]
                center = (width // 2, height // 2)
                rotation_matrix = cv2.getRotationMatrix2D(center, -rotation_angle, 1.0)
                restored_image = cv2.warpAffine(restored_image, rotation_matrix, (width, height),
                                                flags=cv2.INTER_LANCZOS4)

            # Convert to tensor format
            restored_image = self.image_processor.numpy_to_tensor(restored_image)
            mask = self._create_mask(original_image_shape, crop_bbox, rotation_angle)

            return restored_image, fp_pipe, mask

        except Exception as e:
            print(f"Error in restore: {str(e)}")
            return image, fp_pipe, self._create_empty_mask(image)

    def _create_mask(self, image_shape, crop_bbox, rotation_angle):
        """Create a mask for the face area."""
        mask = np.zeros(image_shape[:2], dtype=np.float32)
        x1, y1, w, h = crop_bbox

        # Draw a white rectangle for the face area
        mask[y1:y1 + h, x1:x1 + w] = 1.0

        # Rotate the mask back to the original orientation
        if rotation_angle != 0:
            height, width = image_shape[:2]
            center = (width // 2, height // 2)
            rotation_matrix = cv2.getRotationMatrix2D(center, -rotation_angle, 1.0)
            mask = cv2.warpAffine(mask, rotation_matrix, (width, height), flags=cv2.INTER_LINEAR)

        return self.image_processor.convert_mask_to_tensor(mask)

    def _create_empty_mask(self, image):
        """Create an empty mask with the same shape as the input image."""
        if torch.is_tensor(image):
            image_shape = image.shape[1:3]  # (H, W)
        else:
            image_shape = image.shape[:2]  # (H, W)
        mask = torch.zeros((1, *image_shape), dtype=torch.float32)
        return mask

    def _load_image_from_path(self, image_path):
        """Load and convert image from file path to tensor format."""
        try:
            image = Image.open(image_path)
            if image.mode != 'RGB':
                image = image.convert('RGB')
            return self.image_processor.pil_to_tensor(image)
        except Exception as e:
            print(f"Error loading image from {image_path}: {str(e)}")
            return None