import mediapipe as mp
import numpy as np
import pandas as pd
import cv2
import torch
from PIL import Image
from typing import Union, Optional

class FaceDetector:
    def __init__(self):
        """Initialize the FaceDetector with MediaPipe Face Mesh."""
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5
        )

    def detect_landmarks(self, image: Union[torch.Tensor, np.ndarray, Image.Image]) -> Optional[pd.DataFrame]:
        """Detect facial landmarks in the given image."""
        image_np = self._convert_to_numpy(image)
        if image_np is None:
            return None

        results = self.face_mesh.process(image_np)
        if not results.multi_face_landmarks:
            print("No face detected in the image")
            return None

        face_landmarks = results.multi_face_landmarks[0]
        landmarks_data = {'x': [], 'y': [], 'z': [], 'index': []}
        image_height, image_width = image_np.shape[:2]

        for idx, landmark in enumerate(face_landmarks.landmark):
            x = landmark.x * image_width
            y = landmark.y * image_height
            z = landmark.z

            landmarks_data['x'].append(x)
            landmarks_data['y'].append(y)
            landmarks_data['z'].append(z)
            landmarks_data['index'].append(idx)

        return pd.DataFrame(landmarks_data)

    def _convert_to_numpy(self, image: Union[torch.Tensor, np.ndarray, Image.Image]) -> Optional[np.ndarray]:
        """Convert different image types to numpy array."""
        if torch.is_tensor(image):
            image = image.detach().cpu().numpy()
            if len(image.shape) == 4:
                image = image[0]
            image = (image * 255).astype(np.uint8)
        elif isinstance(image, Image.Image):
            image = np.array(image)
        elif isinstance(image, np.ndarray):
            if image.dtype == np.float32 and image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
            if len(image.shape) == 4:
                image = image[0]

        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)

        return image

    def __del__(self):
        """Clean up resources."""
        if hasattr(self, 'face_mesh'):
            self.face_mesh.close()