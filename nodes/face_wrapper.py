import numpy as np
import pandas as pd
import torch

from ..core.base_mesh import MediapipeBaseLandmarks
from ..core.cpu_deformer import CPUDeformer
from ..core.face_detector import FaceDetector
from ..core.gpu_deformer import GPUDeformer
from ..core.image_processor import ImageProcessor


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
                "device": (["CPU", "CUDA"], {"default": "CUDA"}),
                "show_detection": ("BOOLEAN", {"default": False}),
                "show_target": ("BOOLEAN", {"default": False}),
                "refiner": (["None", "Dlib"], {"default": "None"}),
                "landmark_size": ("INT", {"default": 4, "min": 1, "max": 10, "step": 1}),
                "show_labels": ("BOOLEAN", {"default": False}),
                "x_scale": ("FLOAT", {"default": 1.0, "min": 0.5, "max": 1.0, "step": 0.01}),
                "y_transform": ("FLOAT", {"default": 0.0, "min": -0.5, "max": 0.5, "step": 0.01})
            },
            "optional": {
                "fp_pipe": ("DICT", {"default": None}),
                "mask": ("MASK", {"default": None})
            }
        }

    RETURN_TYPES = ("IMAGE", "DICT", "MASK")
    RETURN_NAMES = ("image", "fp_pipe", "mask")
    FUNCTION = "detect_face"
    CATEGORY = "Face Processor"

    def detect_face(self, image, mode, device, show_detection, show_target, refiner, landmark_size,
                    show_labels, x_scale, y_transform, fp_pipe=None, mask=None):
        # Convert input image to numpy with proper RGB format
        image_np = self.image_processor.convert_to_numpy(image)
        height, width = image_np.shape[:2]

        # Convert mask if provided
        mask_np = None
        if mask is not None:
            mask_np = self.image_processor.convert_mask_to_numpy(mask)

        if mode == "Wrap":
            return self._wrap_mode(image_np, None, width, height,
                                   device, x_scale, y_transform, fp_pipe, mask_np)

        # Detect facial landmarks
        landmarks_df = self.face_detector.detect_landmarks(image_np, refiner=(None if refiner == "None" else refiner))

        if landmarks_df is None:
            print("No face detected")
            empty_mask = torch.zeros((1, height, width), dtype=torch.float32) if mask is not None else None
            return image, fp_pipe or {}, empty_mask

        # Handle different modes
        if mode == "Debug":
            return self._debug_mode(image_np, landmarks_df, width, height,
                                    show_detection, show_target, landmark_size,
                                    show_labels, x_scale, y_transform, fp_pipe, mask_np)
        elif mode == "Un-Wrap":
            return self._unwrap_mode(image_np, landmarks_df, width, height,
                                     device, x_scale, y_transform, fp_pipe, mask_np)


    def _debug_mode(self, image_np, landmarks_df, width, height, show_detection,
                    show_target, landmark_size, show_labels, x_scale, y_transform,
                    fp_pipe, mask_np):
        result_image = image_np.astype(np.float32) / 255.0
        overlays = []

        if show_detection:
            det_overlay = self.image_processor.draw_landmarks(
                (width, height), landmarks_df,
                transparency=0.4, color=(0, 255, 0),
                radius=landmark_size, label=show_labels
            )
            if det_overlay is not None:
                overlays.append(det_overlay)

        # Get base landmarks
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
            target_overlay = self.image_processor.draw_landmarks(
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

        output_image = self.image_processor.numpy_to_tensor(result_image)
        output_mask = self.image_processor.convert_mask_to_tensor(mask_np)

        # Prepare and update landmarks data
        landmarks_data = self._prepare_landmarks_data(landmarks_df, base_landmarks)
        updated_pipe = self._update_pipe(fp_pipe, landmarks_data)

        return output_image, updated_pipe, output_mask

    def _unwrap_mode(self, image_np, landmarks_df, width, height, device,
                     x_scale, y_transform, fp_pipe, mask_np):
        # Get base landmarks
        base_landmarks = MediapipeBaseLandmarks.get_base_landmarks(
            (width, height), x_scale=x_scale, y_translation=y_transform
        )
        source_landmarks = landmarks_df.iloc[:468][['x', 'y']].values.astype(np.float32)

        # Process image
        pil_image = self.image_processor.numpy_to_pil(image_np)
        warped_image = self._apply_warping(pil_image, source_landmarks, base_landmarks, device)

        # Process mask if provided
        warped_mask = None
        if mask_np is not None:
            pil_mask = self.image_processor.numpy_to_pil(mask_np)
            warped_mask = self._apply_warping(pil_mask, source_landmarks, base_landmarks, device)
            warped_mask = self.image_processor.convert_mask_to_tensor(np.array(warped_mask))

        output_image = self.image_processor.pil_to_tensor(warped_image)

        # Prepare and update landmarks data
        landmarks_data = self._prepare_landmarks_data(landmarks_df, base_landmarks)
        updated_pipe = self._update_pipe(fp_pipe, landmarks_data)

        return output_image, updated_pipe, warped_mask

    def _wrap_mode(self, image_np, landmarks_df, width, height, device,
                   x_scale, y_transform, fp_pipe, mask_np):
        if not fp_pipe or 'target_lm' not in fp_pipe:
            print("No landmarks found in face processor pipe")
            output_image = self.image_processor.numpy_to_tensor(image_np)
            output_mask = self.image_processor.convert_mask_to_tensor(mask_np)
            return output_image, {}, output_mask

        current_frame = fp_pipe.get("current_frame", 0)
        frame_key = f"frame_{current_frame}"

        # Check if we have required data
        if frame_key not in fp_pipe["frames"] or "detected_lm" not in fp_pipe["frames"][frame_key]:
            print(f"No detected landmarks found for frame {current_frame}")
            output_image = self.image_processor.numpy_to_tensor(image_np)
            output_mask = self.image_processor.convert_mask_to_tensor(mask_np)
            return output_image, fp_pipe, output_mask

        # Get source landmarks from target_lm at root level
        source_x = fp_pipe['target_lm']['x']
        source_y = fp_pipe['target_lm']['y']
        source_landmarks = np.column_stack((source_x, source_y))[:468]

        # Get target landmarks from detected_lm in current frame
        frame_data = fp_pipe["frames"][frame_key]
        detected_x = frame_data['detected_lm']['x']
        detected_y = frame_data['detected_lm']['y']
        target_landmarks = np.column_stack((detected_x, detected_y))[:468]

        # Process image
        pil_image = self.image_processor.numpy_to_pil(image_np)
        warped_image = self._apply_warping(pil_image, source_landmarks, target_landmarks, device)

        # Process mask if provided
        warped_mask = None
        if mask_np is not None:
            pil_mask = self.image_processor.numpy_to_pil(mask_np)
            warped_mask = self._apply_warping(pil_mask, source_landmarks, target_landmarks, device)
            warped_mask = self.image_processor.convert_mask_to_tensor(np.array(warped_mask))

        output_image = self.image_processor.pil_to_tensor(warped_image)

        return output_image, fp_pipe, warped_mask

    def _apply_warping(self, image, source_landmarks, target_landmarks, device):
        """Apply warping to image using selected device"""
        if device == "CUDA" and torch.cuda.is_available():
            return GPUDeformer.warp_face(image, source_landmarks, target_landmarks)
        else:
            return CPUDeformer.warp_face(image, source_landmarks, target_landmarks)

    def _prepare_landmarks_data(self, detected_df, target_lm):
        """
        Prepare landmarks data in the correct format for fp_pipe

        Args:
            detected_df: DataFrame with detected landmarks
            target_lm: Target landmarks array

        Returns:
            dict: Landmarks data in the format compatible with fp_pipe
        """
        # Prepare detected landmarks data
        detected_data = {
            'x': detected_df['x'].tolist(),
            'y': detected_df['y'].tolist(),
            'indices': detected_df['index'].tolist()
        }

        # Prepare target landmarks data
        target_data = {
            'x': target_lm[:, 0].tolist(),
            'y': target_lm[:, 1].tolist(),
            'indices': list(range(len(target_lm)))
        }

        return detected_data, target_data

    def _update_pipe(self, pipe, landmarks_data):
        """
        Update fp_pipe with new landmarks data

        Args:
            pipe: Existing fp_pipe dictionary or None
            landmarks_data: Tuple of (detected_landmarks, target_landmarks)

        Returns:
            dict: Updated fp_pipe structure
        """
        if pipe is None:
            pipe = {
                "workflow": "image",
                "current_frame": 0,
                "frames": {}
            }

        detected_lm, target_lm = landmarks_data

        # Update target landmarks at the root level
        pipe["target_lm"] = target_lm

        # Update detected landmarks for the current frame
        current_frame = pipe.get("current_frame", 0)
        frame_key = f"frame_{current_frame}"

        # Ensure the frame exists in the structure
        if frame_key not in pipe["frames"]:
            pipe["frames"][frame_key] = {}

        # Update detected landmarks for the current frame
        pipe["frames"][frame_key]["detected_lm"] = detected_lm

        return pipe