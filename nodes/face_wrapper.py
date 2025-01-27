import torch
import numpy as np
import pandas as pd
from PIL import Image
import cv2

from ..core.face_detector import FaceDetector
from ..core.image_processor import ImageProcessor
from ..core.base_mesh import MediapipeBaseLandmarks
from ..core.cpu_deformer import CPUDeformer
from ..core.gpu_deformer import GPUDeformer

class FaceWrapper:
    """ComfyUI node for detecting facial landmarks with optional visualization and warping."""

    def __init__(self):
        self.face_detector = FaceDetector()
        self.image_processor = ImageProcessor()

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mode": (["Debug", "Un-Wrap", "Wrap"], {"default": "Debug"}),
                "device": (["CPU", "CUDA"], {"default": "CPU"}),
                "show_detection": ("BOOLEAN", {"default": False}),
                "show_target": ("BOOLEAN", {"default": False}),
                "landmark_size": ("INT", {"default": 4, "min": 1, "max": 10, "step": 1}),
                "show_labels": ("BOOLEAN", {"default": False}),
                "x_scale": ("FLOAT", {"default": 1.0, "min": 0.5, "max": 1.0, "step": 0.01}),
                "y_transform": ("FLOAT", {"default": 0.0, "min": -0.5, "max": 0.5, "step": 0.01})
            },
            "optional": {
                "processor_settings": ("DICT", {"default": None}),
                "mask": ("MASK", {"default": None})
            }
        }

    RETURN_TYPES = ("IMAGE", "DICT", "MASK")
    RETURN_NAMES = ("image", "processor_settings", "mask")
    FUNCTION = "detect_face"
    CATEGORY = "Face Processor"

    def detect_face(self, image, mode, device, show_detection, show_target, landmark_size,
                    show_labels, x_scale, y_transform, processor_settings=None, mask=None):
        # Convert input image to numpy with proper RGB format
        image_np = self._convert_to_numpy(image)
        height, width = image_np.shape[:2]

        # Convert mask if provided
        mask_np = None
        if mask is not None:
            mask_np = self._convert_mask_to_numpy(mask)

        if mode == "Wrap":
            return self._wrap_mode(image_np, None, width, height,
                                 device, x_scale, y_transform, processor_settings, mask_np)

        # Detect facial landmarks
        landmarks_df = self.face_detector.detect_landmarks(image_np)
        if landmarks_df is None:
            print("No face detected")
            empty_mask = torch.zeros((1, height, width), dtype=torch.float32) if mask is not None else None
            return (image, processor_settings or {}, empty_mask)

        # Handle different modes
        if mode == "Debug":
            return self._debug_mode(image_np, landmarks_df, width, height,
                                  show_detection, show_target, landmark_size,
                                  show_labels, x_scale, y_transform, processor_settings, mask_np)
        elif mode == "Un-Wrap":
            return self._unwrap_mode(image_np, landmarks_df, width, height,
                                   device, x_scale, y_transform, processor_settings, mask_np)

    def _convert_to_numpy(self, image):
        """Improved image conversion with channel handling"""
        if torch.is_tensor(image):
            image = image.detach().cpu().numpy()
            if len(image.shape) == 4:  # Batch dimension
                image = image[0]

            # Handle different channel formats
            if image.shape[0] == 1:  # Grayscale
                image = np.stack((image[0],) * 3, axis=-1)
            elif image.shape[0] == 3:  # RGB
                image = image.transpose(1, 2, 0)
            elif image.shape[0] == 4:  # RGBA
                image = image[:3].transpose(1, 2, 0)

            image = (image * 255).astype(np.uint8)

        # Ensure 3 channels for numpy arrays
        if isinstance(image, np.ndarray):
            if len(image.shape) == 2:  # Grayscale
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            elif image.shape[2] == 4:  # RGBA
                image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
            elif image.shape[2] == 1:  # Single channel
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        return image

    def _convert_mask_to_numpy(self, mask):
        """Convert mask tensor to numpy array"""
        if torch.is_tensor(mask):
            mask = mask.detach().cpu().numpy()
            if len(mask.shape) == 3:
                mask = mask[0]  # Remove batch dimension
            # Scale to 0-255 range
            mask = (mask * 255).astype(np.uint8)
        return mask

    def _convert_mask_to_tensor(self, mask_np):
        """Convert numpy mask back to tensor format"""
        if mask_np is not None:
            mask = mask_np.astype(np.float32) / 255.0
            return torch.from_numpy(mask).unsqueeze(0)
        return None

    def _debug_mode(self, image_np, landmarks_df, width, height, show_detection,
                    show_target, landmark_size, show_labels, x_scale, y_transform,
                    processor_settings, mask_np):
        result_image = image_np.astype(np.float32) / 255.0
        overlays = []

        if show_detection:
            det_overlay = ImageProcessor.draw_landmarks(
                (width, height), landmarks_df,
                transparency=0.4, color=(0, 255, 0),
                radius=landmark_size, label=show_labels
            )
            if det_overlay is not None:
                overlays.append(det_overlay)

        base_landmarks = MediapipeBaseLandmarks.get_base_landmarks(
            (width, height), x_scale=x_scale, y_translation=y_transform
        )

        if show_target:
            base_df = pd.DataFrame({
                'x': base_landmarks[:, 0],
                'y': base_landmarks[:, 1],
                'z': np.zeros(len(base_landmarks)),
                'index': range(len(base_landmarks))
            })
            target_overlay = ImageProcessor.draw_landmarks(
                (width, height), base_df,
                transparency=0.4, color=(255, 0, 0),
                radius=landmark_size, label=show_labels
            )
            if target_overlay is not None:
                overlays.append(target_overlay)

        for overlay in overlays:
            overlay = overlay.astype(np.float32) / 255.0
            alpha = overlay[:, :, 3:]
            rgb = overlay[:, :, :3]
            result_image = result_image * (1 - alpha) + rgb * alpha

        output_image = torch.from_numpy(result_image).unsqueeze(0)
        output_mask = self._convert_mask_to_tensor(mask_np)
        landmarks_data = self._prepare_landmarks_data(landmarks_df, base_landmarks)

        return (output_image, self._update_settings(processor_settings, landmarks_data), output_mask)

    def _unwrap_mode(self, image_np, landmarks_df, width, height, device,
                     x_scale, y_transform, processor_settings, mask_np):
        base_landmarks = MediapipeBaseLandmarks.get_base_landmarks(
            (width, height), x_scale=x_scale, y_translation=y_transform
        )
        source_landmarks = landmarks_df.iloc[:468][['x', 'y']].values.astype(np.float32)

        # Process image
        pil_image = Image.fromarray(image_np)
        warped_image = self._apply_warping(pil_image, source_landmarks, base_landmarks, device)

        # Process mask if provided
        warped_mask = None
        if mask_np is not None:
            pil_mask = Image.fromarray(mask_np)
            warped_mask = self._apply_warping(pil_mask, source_landmarks, base_landmarks, device)
            warped_mask = self._convert_mask_to_tensor(np.array(warped_mask))

        output_image = np.array(warped_image).astype(np.float32) / 255.0
        output_image = torch.from_numpy(output_image).unsqueeze(0)

        landmarks_data = self._prepare_landmarks_data(landmarks_df, base_landmarks)
        return (output_image, self._update_settings(processor_settings, landmarks_data), warped_mask)

    def _wrap_mode(self, image_np, landmarks_df, width, height, device,
                   x_scale, y_transform, processor_settings, mask_np):
        if not processor_settings or 'target_lm' not in processor_settings:
            print("No landmarks found in processor settings")
            output_image = torch.from_numpy(image_np.astype(np.float32) / 255.0).unsqueeze(0)
            output_mask = self._convert_mask_to_tensor(mask_np)
            return (output_image, {}, output_mask)

        source_x = processor_settings['target_lm']['x']
        source_y = processor_settings['target_lm']['y']
        source_landmarks = np.column_stack((source_x, source_y))[:468]

        detected_x = processor_settings['detected_lm']['x']
        detected_y = processor_settings['detected_lm']['y']
        target_landmarks = np.column_stack((detected_x, detected_y))[:468]

        # Process image
        pil_image = Image.fromarray(image_np)
        warped_image = self._apply_warping(pil_image, source_landmarks, target_landmarks, device)

        # Process mask if provided
        warped_mask = None
        if mask_np is not None:
            pil_mask = Image.fromarray(mask_np)
            warped_mask = self._apply_warping(pil_mask, source_landmarks, target_landmarks, device)
            warped_mask = self._convert_mask_to_tensor(np.array(warped_mask))

        output_image = np.array(warped_image).astype(np.float32) / 255.0
        output_image = torch.from_numpy(output_image).unsqueeze(0)

        return (output_image, processor_settings, warped_mask)

    def _apply_warping(self, image, source_landmarks, target_landmarks, device):
        """Apply warping to image using selected device"""
        if device == "CUDA" and torch.cuda.is_available():
            return GPUDeformer.warp_face(image, source_landmarks, target_landmarks)
        else:
            return CPUDeformer.warp_face(image, source_landmarks, target_landmarks)

    def _prepare_landmarks_data(self, detected_df, target_lm):
        return {
            'detected_lm': {
                'x': detected_df['x'].tolist(),
                'y': detected_df['y'].tolist(),
                'indices': detected_df['index'].tolist()
            },
            'target_lm': {
                'x': target_lm[:, 0].tolist(),
                'y': target_lm[:, 1].tolist(),
                'indices': list(range(len(target_lm)))
            }
        }

    def _update_settings(self, settings, new_data):
        return {**(settings or {}), **new_data}