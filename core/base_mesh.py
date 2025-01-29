import numpy as np
from typing import Optional, Tuple, List, Union

from core.resources.model_loader import ModelOBJ


class MediapipeBaseLandmarks:

    """
    Class for handling base landmark positions and face topology.
    Coordinates are stored in normalized 0-1 range.
    """
    _instance: Optional[ModelOBJ] = None
    _boundary_faces: Optional[np.ndarray] = None
    _boundary_landmarks: Optional[np.ndarray] = None
    _current_size: Optional[Union[int, Tuple[int, int]]] = None


    @classmethod
    def _get_model_instance(cls) -> ModelOBJ:
        """
        Get or create ModelOBJ instance using singleton pattern

        Returns:
            ModelOBJ: Instance of model loader
        """
        if cls._instance is None:
            cls._instance = ModelOBJ()
        return cls._instance

    @classmethod
    def create_boundary_triangles(cls, size) -> np.ndarray:
        """
        Create additional triangles between face oval and image boundaries

        Args:
            size: Target size (int or tuple of width, height)

        Returns:
            numpy.ndarray: Array of boundary face indices
        """
        if isinstance(size, int):
            width = height = size
        else:
            width, height = size

        print(f"Creating boundary triangles for image size: {width}x{height}")

        boundary_points = []
        boundary_triangles = []

        edges = {
            'top': {
                'landmarks': [54, 103, 67, 109, 10, 338, 297, 332, 284],
                'start': (0, 0),
                'middle': (0.5, 0),
                'end': (1, 0),
                'segments': 8
            },
            'right': {
                'landmarks': [284, 251, 389, 356, 454, 323, 361, 288, 397, 365],
                'start': (1, 0),
                'middle': (1, 0.5),
                'end': (1, 1),
                'segments': 9
            },
            'bottom': {
                'landmarks': [365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136],
                'start': (1, 1),
                'middle': (0.5, 1),
                'end': (0, 1),
                'segments': 10
            },
            'left': {
                'landmarks': [136, 172, 58, 132, 93, 234, 127, 162, 21, 54],
                'start': (0, 1),
                'middle': (0, 0.5),
                'end': (0, 0),
                'segments': 9
            }
        }

        current_point_idx = cls._get_model_instance().num_landmarks

        for edge_name, edge_info in edges.items():
            print(f"Processing {edge_name} edge")

            landmarks = edge_info['landmarks']
            n_segments = edge_info['segments']
            print(f"Creating {n_segments} segments for {len(landmarks)} landmarks")

            # Generate normalized edge points
            edge_points = cls._interpolate_edge_points(
                edge_info['start'],
                edge_info['end'],
                n_segments
            )

            # Create quads and convert to triangles
            for i in range(n_segments):
                # Define quad vertices
                quad = [
                    landmarks[i],  # Current landmark
                    landmarks[i + 1],  # Next landmark
                    current_point_idx + i + 1,  # Next edge point
                    current_point_idx + i  # Current edge point
                ]

                # Convert quad to two triangles
                boundary_triangles.extend([
                    [quad[0], quad[1], quad[2]],  # First triangle
                    [quad[0], quad[2], quad[3]]  # Second triangle
                ])

            # Add edge points to boundary points list
            boundary_points.extend(edge_points)
            current_point_idx += len(edge_points)

        print(f"Created {len(boundary_triangles)} triangles from {len(boundary_points)} boundary points")

        cls._boundary_landmarks = np.array(boundary_points)
        cls._boundary_faces = np.array(boundary_triangles)

        return cls._boundary_faces

    @staticmethod
    def _interpolate_edge_points(start_pos: Tuple[float, float],
                                 end_pos: Tuple[float, float],
                                 n_segments: int) -> List[Tuple[float, float]]:
        """
        Interpolate points along an edge

        Args:
            start_pos: Start position (x, y)
            end_pos: End position (x, y)
            n_segments: Number of segments to create

        Returns:
            List[Tuple[float, float]]: List of interpolated points
        """
        points = []
        for i in range(n_segments + 1):
            t = i / n_segments
            x = start_pos[0] + t * (end_pos[0] - start_pos[0])
            y = start_pos[1] + t * (end_pos[1] - start_pos[1])
            points.append((x, y))
        return points

    @classmethod
    def get_face_triangles(cls, size=None, x_scale: float = 1.0, y_translation: float = 0.0) -> Tuple[
        np.ndarray, np.ndarray]:
        """
        Get face triangulation including boundary triangles with optional transformations

        Args:
            size: Target size (int or tuple)
            x_scale: Horizontal scaling factor (0.5 to 1.0)
            y_translation: Vertical translation (-0.5 to 0.5)

        Returns:
            Tuple[np.ndarray, np.ndarray]: (triangles array, landmarks array)
        """
        model = cls._get_model_instance()
        landmarks = model.get_transformed_landmarks(x_scale, y_translation)
        faces = model.get_faces()

        # Create boundary triangles if needed (using normalized coordinates)
        if cls._boundary_faces is None:
            cls.create_boundary_triangles(size)

        if size is not None:
            if isinstance(size, tuple):
                scaled_landmarks = landmarks * np.array(size)
                if cls._boundary_landmarks is not None:
                    boundary_landmarks = cls._boundary_landmarks * np.array(size)
                    if y_translation != 0:
                        boundary_landmarks = boundary_landmarks.copy()
                        boundary_landmarks[:, 1] += y_translation * size[1]
            else:
                scaled_landmarks = landmarks * size
                if cls._boundary_landmarks is not None:
                    boundary_landmarks = cls._boundary_landmarks * size
                    if y_translation != 0:
                        boundary_landmarks = boundary_landmarks.copy()
                        boundary_landmarks[:, 1] += y_translation * size

            if cls._boundary_faces is not None:
                all_points = np.vstack([scaled_landmarks, boundary_landmarks])
                all_triangles = np.vstack([faces, cls._boundary_faces])
                return all_triangles, all_points

        return faces, landmarks

    @classmethod
    def get_base_landmarks(cls, size=None, x_scale: float = 1.0, y_translation: float = 0.0) -> np.ndarray:
        """
        Get base landmark positions with optional transformations.

        Args:
            size: Target size (int or tuple)
            x_scale: Horizontal scaling factor (0.5 to 1.0)
            y_translation: Vertical translation (-0.5 to 0.5)

        Returns:
            numpy.ndarray: Base landmarks with applied transformations
        """
        model = cls._get_model_instance()
        landmarks = model.get_transformed_landmarks(x_scale, y_translation)

        if size is not None:
            if isinstance(size, tuple):
                landmarks = landmarks * np.array(size)
            else:
                landmarks = landmarks * size

        return landmarks