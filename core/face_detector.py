import mediapipe as mp
import numpy as np
import pandas as pd
import cv2
import torch
import os
from PIL import Image
from pandas import DataFrame
from typing import Union, Optional
from core.lm_mapping import LandmarkMappings
import dlib


class FaceDetector:
    _model_loaded = False  # Class variable to track if model was loaded

    def __init__(self):
        """Initialize the FaceDetector with MediaPipe Face Mesh and Dlib."""
        # MediaPipe initialization
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5
        )

        # Dlib initialization
        self.face_detector = dlib.get_frontal_face_detector()

        # Get path to the model file
        current_dir = os.path.dirname(os.path.realpath(__file__))
        model_path = os.path.join(current_dir, "resources", "shape_predictor_68_face_landmarks.dat")

        if not os.path.exists(model_path):
            print(f"Dlib landmarks model not found at: {model_path}")
            print("Please ensure the model file is placed in the core/resources directory")
            raise FileNotFoundError(f"Missing model file at: {model_path}")

        if not FaceDetector._model_loaded:
            print(f"Loading dlib model from: {model_path}")
            FaceDetector._model_loaded = True

        self.shape_predictor = dlib.shape_predictor(model_path)

    def detect_landmarks(self, image: Union[torch.Tensor, np.ndarray, Image.Image],
                         refine: bool = False) -> Union[DataFrame, None, str]:
        mp_landmarks = self.detect_landmarks_mp(image)
        if mp_landmarks is None:
            return None

        if refine:
            print("Detect with dlib")
            dlib_landmarks = self.detect_landmarks_dlib(image)
            if dlib_landmarks is not None:
                return self.landmarks_interpolation(mp_landmarks, dlib_landmarks, LandmarkMappings.landmarks_pairs)

        return mp_landmarks

    def detect_landmarks_mp(self, image: Union[torch.Tensor, np.ndarray, Image.Image]) -> Optional[pd.DataFrame]:
        """
        Detect facial landmarks in the given image using MediaPipe.

        Args:
            image: Input image in various formats

        Returns:
            Optional[pd.DataFrame]: DataFrame containing landmark coordinates (x, y) and indices or None if no face detected
        """
        image_np = self._convert_to_numpy(image)
        if image_np is None:
            return None

        results = self.face_mesh.process(image_np)
        if not results.multi_face_landmarks:
            print("No face detected in the image")
            return None

        face_landmarks = results.multi_face_landmarks[0]
        landmarks_data = {'x': [], 'y': [], 'index': []}
        image_height, image_width = image_np.shape[:2]

        for idx, landmark in enumerate(face_landmarks.landmark):
            x = landmark.x * image_width
            y = landmark.y * image_height

            landmarks_data['x'].append(x)
            landmarks_data['y'].append(y)
            landmarks_data['index'].append(idx)

        return pd.DataFrame(landmarks_data)

    def detect_landmarks_dlib(self, image: Union[torch.Tensor, np.ndarray, Image.Image]) -> Optional[pd.DataFrame]:
        """
        Detect facial landmarks using dlib's 68 point predictor.

        Args:
            image: Input image in various formats

        Returns:
            Optional[pd.DataFrame]: DataFrame containing landmark coordinates (x, y) and indices or None if no face detected
        """
        try:
            # Convert image to numpy format suitable for dlib
            image_np = self._convert_to_numpy(image)
            if image_np is None:
                print("Failed to convert image for dlib processing")
                return None

            # Convert to grayscale for dlib
            gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)

            # Detect faces
            faces = self.face_detector(gray)
            if not faces:
                print("No face detected by dlib")
                return None

            # Get first face
            face = faces[0]

            # Detect landmarks
            shape = self.shape_predictor(gray, face)

            # Create landmark data structure
            landmarks_data = {
                'x': [],
                'y': [],
                'index': []
            }

            # Extract landmarks (adding 1 to index to match dlib's 1-based indexing)
            for i in range(68):
                point = shape.part(i)
                landmarks_data['x'].append(float(point.x))
                landmarks_data['y'].append(float(point.y))
                landmarks_data['index'].append(i + 1)  # Adding 1 to match dlib's indexing

            return pd.DataFrame(landmarks_data)

        except Exception as e:
            print(f"Error in dlib landmark detection: {str(e)}")
            return None

    def landmarks_interpolation(self, mp_landmarks: pd.DataFrame, dlib_landmarks: pd.DataFrame,
                                mapping: dict) -> pd.DataFrame:
        """
        Interpolates all MediaPipe landmarks based on dlib reference points using RBF interpolation.

        Args:
            mp_landmarks: DataFrame with columns [x, y, index]
            dlib_landmarks: DataFrame with columns [x, y, index]
            mapping: Dictionary of corresponding landmark pairs between MediaPipe and dlib

        Returns:
            pd.DataFrame: DataFrame with interpolated landmark positions
        """
        from scipy.interpolate import RBFInterpolator
        # Validate input DataFrames
        required_columns = ['x', 'y', 'index']
        if not all(col in mp_landmarks.columns for col in required_columns):
            print("Error: MediaPipe landmarks missing required columns")
            return mp_landmarks

        if not all(col in dlib_landmarks.columns for col in required_columns):
            print("Error: Dlib landmarks missing required columns")
            return mp_landmarks

        try:
            # Collect control points
            control_src = []  # MediaPipe points
            control_dst = []  # Dlib points

            for feature, pairs in mapping.items():
                for mp_idx, dlib_idx in pairs:
                    mp_point = mp_landmarks[mp_landmarks['index'] == mp_idx].iloc[0]
                    dlib_point = dlib_landmarks[dlib_landmarks['index'] == dlib_idx].iloc[0]

                    control_src.append([mp_point['x'], mp_point['y']])
                    control_dst.append([dlib_point['x'], dlib_point['y']])

            control_src = np.array(control_src)
            control_dst = np.array(control_dst)

            print(f"Using {len(control_src)} control points for RBF interpolation")

            # Create and fit RBF interpolation for both x and y coordinates
            kernel = 'thin_plate_spline'
            rbf_x = RBFInterpolator(control_src, control_dst[:, 0], kernel=kernel)
            rbf_y = RBFInterpolator(control_src, control_dst[:, 1], kernel=kernel)

            # Transform all MediaPipe points
            all_points = mp_landmarks[['x', 'y']].values

            # Apply transformation
            transformed_x = rbf_x(all_points)
            transformed_y = rbf_y(all_points)

            # Create result DataFrame
            result = mp_landmarks.copy()
            result['x'] = transformed_x
            result['y'] = transformed_y

            print("Successfully interpolated all landmarks using RBF")
            return result

        except Exception as e:
            print(f"Error during landmark interpolation: {str(e)}")
            return mp_landmarks

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
