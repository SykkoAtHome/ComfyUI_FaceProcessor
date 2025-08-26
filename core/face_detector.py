from typing import Union, Optional

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from pandas import DataFrame

from ..core.lm_mapping import LandmarkMappings
from ..core.resources.model_loader import ModelDlib, ModelMediaPipe
from ..core.image_processor import ImageProcessor


class FaceDetector:
    def __init__(self):
        """Initialize the FaceDetector with MediaPipe Face Mesh."""
        # Models will be initialized on demand
        self.mediapipe_model = None
        self.dlib_model = None
        self.image_processor = ImageProcessor()

    def detect_landmarks(self,
                        image: Union[torch.Tensor, np.ndarray, Image.Image],
                        refiner: Optional[str] = None
                        ) -> Union[DataFrame, None]:
        """
        Detect facial landmarks using MediaPipe with optional refinement.

        Args:
            image: Input image in various formats
            refiner: Optional refinement method ('Dlib', 'InsightFace', or None)

        Returns:
            DataFrame with landmark coordinates (x, y) and indices or None if no face detected
        """
        # Always detect with MediaPipe first
        mp_landmarks = self.detect_landmarks_mp(image)
        if mp_landmarks is None:
            return None

        # Apply refinement if requested
        if refiner is not None:
            if refiner.lower() == 'dlib':
                dlib_landmarks = self.detect_landmarks_dlib(image)
                if dlib_landmarks is not None:
                    return self.landmarks_interpolation(mp_landmarks, dlib_landmarks)

        return mp_landmarks

    def __del__(self):
        """Clean up resources"""
        if hasattr(self, 'mediapipe_model') and self.mediapipe_model:
            self.mediapipe_model.face_mesh.close()

    def detect_landmarks_mp(self, image: Union[torch.Tensor, np.ndarray, Image.Image]) -> Optional[pd.DataFrame]:
        """
        Detect facial landmarks in the given image using MediaPipe.

        Args:
            image: Input image in various formats

        Returns:
            Optional[pd.DataFrame]: DataFrame containing landmark coordinates (x, y) and indices or None if no face detected
        """
        try:
            # Initialize MediaPipe model if not already done
            if self.mediapipe_model is None:
                self.mediapipe_model = ModelMediaPipe()

            image_np = self.image_processor.convert_to_numpy(image)
            if image_np is None:
                return None

            results = self.mediapipe_model.face_mesh.process(image_np)

            # Results may come from MediaPipe Solutions or Tasks. Try both field names.
            face_landmarks_list = getattr(results, 'multi_face_landmarks', None)
            if not face_landmarks_list:
                face_landmarks_list = getattr(results, 'face_landmarks', None)
            if not face_landmarks_list:
                print("No face detected in the image")
                return None

            # Take first detected face
            face_landmarks = face_landmarks_list[0]

            # Prepare data structure for landmarks
            landmarks_data = {
                'x': [],
                'y': [],
                'index': []
            }

            # Get image dimensions for coordinate conversion
            image_height, image_width = image_np.shape[:2]

            # Extract landmark coordinates
            for idx, landmark in enumerate(face_landmarks.landmark):
                # Convert normalized coordinates to pixel coordinates
                x = landmark.x * image_width
                y = landmark.y * image_height

                landmarks_data['x'].append(x)
                landmarks_data['y'].append(y)
                landmarks_data['index'].append(idx)

            return pd.DataFrame(landmarks_data)

        except Exception as e:
            print(f"Error in MediaPipe landmark detection: {str(e)}")
            return None

    def detect_landmarks_dlib(self, image: Union[torch.Tensor, np.ndarray, Image.Image]) -> Optional[pd.DataFrame]:
        """
        Detect facial landmarks using dlib's 68 point predictor.

        Args:
            image: Input image in various formats

        Returns:
            Optional[pd.DataFrame]: DataFrame containing landmark coordinates (x, y) and indices or None if no face detected
        """
        try:
            # Initialize Dlib model if not already done
            if self.dlib_model is None:
                self.dlib_model = ModelDlib()

            # Convert image to numpy format suitable for dlib
            image_np = self.image_processor.convert_to_numpy(image)
            if image_np is None:
                print("Failed to convert image for dlib processing")
                return None

            # Convert to grayscale for dlib
            gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)

            # Detect faces using Dlib face detector
            faces = self.dlib_model.face_detector(gray)
            if not faces:
                print("No face detected by dlib")
                return None

            # Get first face
            face = faces[0]

            # Detect landmarks using shape predictor
            shape = self.dlib_model.shape_predictor(gray, face)

            # Create landmark data structure
            landmarks_data = {
                'x': [],
                'y': [],
                'index': []
            }

            # Extract landmarks
            for i in range(68):
                point = shape.part(i)
                landmarks_data['x'].append(float(point.x))
                landmarks_data['y'].append(float(point.y))
                landmarks_data['index'].append(i + 1)  # Adding 1 to match dlib's indexing

            return pd.DataFrame(landmarks_data)

        except Exception as e:
            print(f"Error in dlib landmark detection: {str(e)}")
            return None


    def landmarks_interpolation(self, mp_landmarks: pd.DataFrame, dlib_landmarks: pd.DataFrame) -> pd.DataFrame:
        """
        Interpolates all MediaPipe landmarks based on dlib reference points using RBF interpolation.

        Args:
            mp_landmarks: DataFrame with columns [x, y, index]
            dlib_landmarks: DataFrame with columns [x, y, index]

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
            # Get control points from Dlib landmarks
            control_points_df = LandmarkMappings.get_control_points(dlib_landmarks)
            if control_points_df is None:
                print("Failed to generate control points, returning original landmarks")
                return mp_landmarks

            # Prepare source (MediaPipe) and destination (Dlib) points for RBF
            control_src = []  # MediaPipe points
            control_dst = []  # Dlib-based points

            for _, control_point in control_points_df.iterrows():
                mp_idx = control_point['index']
                mp_point = mp_landmarks[mp_landmarks['index'] == mp_idx]

                if not mp_point.empty:
                    control_src.append([mp_point['x'].iloc[0], mp_point['y'].iloc[0]])
                    control_dst.append([control_point['x'], control_point['y']])

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
