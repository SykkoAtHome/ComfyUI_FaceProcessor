import torch
import numpy as np
import pandas as pd
from core.face_detector import FaceDetector
from core.image_processor import ImageProcessor
from core.base_landmarks import MediapipeBaseLandmarks


class FaceWrapper:
    """ComfyUI node for detecting facial landmarks with optional visualization."""

    def __init__(self):
        self.face_detector = FaceDetector()
        self.image_processor = ImageProcessor()

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "show_detection": ("BOOLEAN", {
                    "default": False,
                }),
                "show_target": ("BOOLEAN", {
                    "default": False,
                }),
                "landmark_size": ("INT", {
                    "default": 2,
                    "min": 1,
                    "max": 10,
                    "step": 1
                }),
                "show_labels": ("BOOLEAN", {
                    "default": False,
                }),
                "x_scale": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.5,
                    "max": 1.0,
                    "step": 0.05
                }),
                "y_transform": ("FLOAT", {
                    "default": 0.0,
                    "min": -0.5,
                    "max": 0.5,
                    "step": 0.05
                })
            }
        }

    RETURN_TYPES = ("IMAGE", "DICT")
    RETURN_NAMES = ("image", "landmarks")
    FUNCTION = "detect_face"
    CATEGORY = "Face Processor"

    def detect_face(self, image, show_detection, show_target, landmark_size=2,
                    show_labels=False, x_scale=1.0, y_transform=0.0):
        """Detect facial landmarks and optionally visualize them."""

        # Convert tensor to numpy for detection
        if torch.is_tensor(image):
            image_np = image.detach().cpu().numpy()
            if len(image_np.shape) == 4:
                image_np = image_np[0]
            image_np = (image_np * 255).astype(np.uint8)

        height, width = image_np.shape[:2]

        # Detect landmarks
        landmarks_df = self.face_detector.detect_landmarks(image_np)

        if landmarks_df is None:
            print("No face detected in the image")
            return (image, {})

        # Start with original image
        result_image = image_np.astype(np.float32) / 255.0
        overlays = []

        # Create detected landmarks overlay if requested
        if show_detection:
            landmark_overlay = ImageProcessor.draw_landmarks(
                (width, height),
                landmarks_df,
                transparency=0.4,
                color=(0, 255, 0),
                radius=landmark_size,
                label=show_labels
            )
            if landmark_overlay is not None:
                overlays.append(landmark_overlay)

        # Get base landmarks with transformations
        base_landmarks = MediapipeBaseLandmarks.get_base_landmarks(
            (width, height),
            x_scale=x_scale,
            y_translation=y_transform
        )

        # Create target landmarks overlay if requested
        if show_target:
            base_df = pd.DataFrame({
                'x': base_landmarks[:, 0],
                'y': base_landmarks[:, 1],
                'z': np.zeros(len(base_landmarks)),
                'index': range(len(base_landmarks))
            })

            target_overlay = ImageProcessor.draw_landmarks(
                (width, height),
                base_df,
                transparency=0.4,
                color=(255, 0, 0),
                radius=landmark_size,
                label=show_labels
            )
            if target_overlay is not None:
                overlays.append(target_overlay)

        # Blend all overlays with the image
        for overlay in overlays:
            overlay = overlay.astype(np.float32) / 255.0
            alpha = overlay[:, :, 3:]
            rgb = overlay[:, :, :3]
            result_image = result_image * (1 - alpha) + rgb * alpha

        # Convert final image back to torch tensor
        image = torch.from_numpy(result_image).unsqueeze(0)

        # Create landmarks dictionary
        landmarks = {
            'detected_lm': {
                'x': landmarks_df['x'].tolist(),
                'y': landmarks_df['y'].tolist(),
                'indices': landmarks_df['index'].tolist()
            },
            'target_lm': {
                'x': base_landmarks[:, 0].tolist(),
                'y': base_landmarks[:, 1].tolist(),
                'indices': list(range(len(base_landmarks)))
            }
        }

        return (image, landmarks)

