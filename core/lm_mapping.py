import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional


class LandmarkMappings:
    """
    Class for mapping and transforming facial landmarks between MediaPipe and Dlib coordinate spaces.
    Supports interpolated control points derived from Dlib landmark pairs.
    """

    # Base landmark pairs mapping MediaPipe to Dlib indices
    # Format: (mediapipe_idx, dlib_idx)
    LANDMARKS_PAIRS = {
        "left_eye": [
            (133, 40),  # Left eye outer corner
            (33, 37)  # Left eye inner corner
        ],
        "right_eye": [
            (362, 43),  # Right eye outer corner
            (263, 46)  # Right eye inner corner
        ],
        "mouth": [
            (0, 52),  # Upper lip top
            (17, 58),  # Bottom lip bottom
            (61, 49),  # Left mouth corner
            (291, 55),  # Right mouth corner
            (13, 63),  # Upper lip bottom
            (14, 67),  # Bottom lip top
            (78, 61),  # Left mouth inner corner
            (308, 65)  # Right mouth inner corner
        ],
        "nose_edge": [
            (6, 28),  # Nose bridge top
            (4, 31)  # Nose tip
        ],
        "face_oval": [
            (152, 9),  # Left jawline
            (127, 1),  # Chin left
            (356, 17),  # Right jawline
            (172, 5),  # Mid jawline left
            (397, 13)  # Mid jawline right
        ]
    }

    # Virtual points interpolated from Dlib landmark pairs
    # Format: mediapipe_idx: (dlib_idx1, dlib_idx2)
    VIRTUAL_PAIRS = {
        "9": (22, 23)
    }

    @classmethod
    def interpolate_point(cls, dlib_landmarks: pd.DataFrame, dlib_idx1: int, dlib_idx2: int) -> Optional[
        Tuple[float, float]]:
        """
        Interpolate a point between two Dlib landmarks.

        Args:
            dlib_landmarks: DataFrame with Dlib landmarks
            dlib_idx1: First Dlib landmark index
            dlib_idx2: Second Dlib landmark index

        Returns:
            Tuple[float, float]: Interpolated (x, y) coordinates or None if error
        """
        try:
            # Get the two Dlib points
            point1 = dlib_landmarks[dlib_landmarks['index'] == dlib_idx1]
            point2 = dlib_landmarks[dlib_landmarks['index'] == dlib_idx2]

            if point1.empty or point2.empty:
                print(f"Warning: Could not find Dlib landmarks {dlib_idx1} and/or {dlib_idx2}")
                return None

            # Calculate midpoint
            x = (point1['x'].iloc[0] + point2['x'].iloc[0]) / 2
            y = (point1['y'].iloc[0] + point2['y'].iloc[0]) / 2

            return float(x), float(y)

        except Exception as e:
            print(f"Error interpolating point between Dlib landmarks {dlib_idx1} and {dlib_idx2}: {str(e)}")
            return None

    @classmethod
    def get_control_points(cls, dlib_landmarks: pd.DataFrame) -> Optional[pd.DataFrame]:
        """
        Generate control points DataFrame based on Dlib landmarks.
        Includes both direct mappings and interpolated points.

        Args:
            dlib_landmarks: DataFrame containing Dlib landmark coordinates with columns [x, y, index]

        Returns:
            pd.DataFrame: DataFrame containing mapped MediaPipe landmark positions with columns [x, y, index]
                         or None if input data is invalid
        """
        try:
            # Validate input DataFrame
            required_columns = {'x', 'y', 'index'}
            if not all(col in dlib_landmarks.columns for col in required_columns):
                print("Error: Input DataFrame missing required columns [x, y, index]")
                return None

            # Prepare data structures for control points
            control_points = {
                'x': [],
                'y': [],
                'index': []
            }

            # Process regular landmark pairs
            print("Processing direct landmark mappings...")
            for feature_group, pairs in cls.LANDMARKS_PAIRS.items():
                for mp_idx, dlib_idx in pairs:
                    dlib_point = dlib_landmarks[dlib_landmarks['index'] == dlib_idx]

                    if dlib_point.empty:
                        print(f"Warning: Dlib landmark {dlib_idx} not found for MediaPipe index {mp_idx}")
                        continue

                    control_points['x'].append(float(dlib_point['x'].iloc[0]))
                    control_points['y'].append(float(dlib_point['y'].iloc[0]))
                    control_points['index'].append(mp_idx)

            # Process interpolated virtual points
            print("Processing interpolated control points...")
            for mp_idx, (dlib_idx1, dlib_idx2) in cls.VIRTUAL_PAIRS.items():
                interpolated = cls.interpolate_point(dlib_landmarks, dlib_idx1, dlib_idx2)
                if interpolated is not None:
                    x, y = interpolated
                    control_points['x'].append(x)
                    control_points['y'].append(y)
                    control_points['index'].append(int(mp_idx))
                    print(f"Added interpolated point for MediaPipe {mp_idx} at ({x:.1f}, {y:.1f})")

            # Create DataFrame with all control points
            control_df = pd.DataFrame(control_points)

            # Sort by MediaPipe landmark index for consistency
            control_df = control_df.sort_values('index').reset_index(drop=True)

            print(f"Successfully mapped {len(control_df)} control points "
                  f"({len(control_df) - len(cls.VIRTUAL_PAIRS)} regular, "
                  f"{len(cls.VIRTUAL_PAIRS)} interpolated)")

            return control_df

        except Exception as e:
            print(f"Error generating control points: {str(e)}")
            return None

    # TODO: filter_trackers()
    def filter_trackers(self, distance):
        """
        return list of base mediapipe landmarks ids as a list of trackers to use for cv2.goodFeaturesToTrack
        based on given distance
        """
        trackers = []
        return trackers
