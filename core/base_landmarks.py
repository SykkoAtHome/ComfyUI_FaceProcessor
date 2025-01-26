import numpy as np
import os


class MediapipeBaseLandmarks:
    """
    Class for handling base landmark positions and face topology loaded from OBJ file.
    Coordinates are stored in normalized 0-1 range.
    """
    _base_landmarks = None
    _faces = None
    _boundary_faces = None
    _boundary_landmarks = None

    # Face oval landmarks in clockwise order
    OVAL_LANDMARKS = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
                      397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
                      172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]

    @classmethod
    def _transform_landmarks(cls, landmarks, x_scale=1.0, y_translation=0.0):
        """
        Apply transformations to landmarks:
        - x_scale: horizontal scaling (0.5 to 1.0)
        - y_translation: vertical translation (-0.5 to 0.5)

        Args:
            landmarks: numpy array of landmarks
            x_scale: horizontal scaling factor (default: 1.0)
            y_translation: vertical translation value (default: 0.0)

        Returns:
            numpy array: Transformed landmarks
        """
        transformed = landmarks.copy()

        # Calculate center point for scaling
        center_x = (landmarks[:, 0].min() + landmarks[:, 0].max()) / 2

        # Apply horizontal scaling relative to center
        transformed[:, 0] = center_x + (transformed[:, 0] - center_x) * x_scale

        # Apply vertical translation
        transformed[:, 1] += y_translation

        return transformed

    @classmethod
    def _load_obj_data(cls):
        """
        Load UV coordinates, vertices and face indices from OBJ file.
        Returns array of normalized [u,v] coordinates mapped to correct vertex indices
        and array of face indices.
        """
        if cls._base_landmarks is not None and cls._faces is not None:
            return cls._base_landmarks, cls._faces

        current_dir = os.path.dirname(os.path.realpath(__file__))
        obj_path = os.path.join(current_dir, 'mediapipe_landmarks_468.obj')

        vertices = []
        uvs = []
        faces = []
        vertex_to_uv = {}

        print(f"Loading 3D face model from: {obj_path}")

        with open(obj_path, 'r') as f:
            for line in f:
                if line.startswith('v '):
                    parts = line.strip().split()
                    vertices.append([float(p) for p in parts[1:4]])
                elif line.startswith('vt '):
                    parts = line.strip().split()
                    u = float(parts[1])
                    v = 1.0 - float(parts[2])
                    uvs.append([u, v])
                elif line.startswith('f '):
                    parts = line.strip().split()[1:]
                    face_verts = []
                    for part in parts:
                        indices = part.split('/')
                        if len(indices) >= 2:
                            vertex_idx = int(indices[0]) - 1
                            uv_idx = int(indices[1]) - 1
                            vertex_to_uv[vertex_idx] = uv_idx
                            face_verts.append(vertex_idx)
                    if len(face_verts) == 3:
                        faces.append(face_verts)

        print(f"Loaded {len(vertices)} vertices, {len(uvs)} UVs, {len(faces)} faces")

        uvs = np.array(uvs, dtype=np.float32)
        faces = np.array(faces, dtype=np.int32)
        landmarks = np.zeros((468, 2), dtype=np.float32)

        for vertex_idx, uv_idx in vertex_to_uv.items():
            if vertex_idx < len(landmarks):
                landmarks[vertex_idx] = uvs[uv_idx]

        cls._base_landmarks = landmarks
        cls._faces = faces
        return cls._base_landmarks, cls._faces

    @staticmethod
    def interpolate_edge_points(start_pos, end_pos, n_segments):
        """
        Interpolate points along an edge
        """
        points = []
        for i in range(n_segments + 1):
            t = i / n_segments
            x = round(start_pos[0] + t * (end_pos[0] - start_pos[0]))
            y = round(start_pos[1] + t * (end_pos[1] - start_pos[1]))
            points.append((x, y))
        return points

    @classmethod
    def create_boundary_triangles(cls, size):
        """
        Create additional triangles between face oval and image boundaries
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
                'middle': (width / 2, 0),
                'end': (width, 0),
                'segments': 8
            },
            'right': {
                'landmarks': [284, 251, 389, 356, 454, 323, 361, 288, 397, 365],
                'start': (width, 0),
                'middle': (width, height / 2),
                'end': (width, height),
                'segments': 9
            },
            'bottom': {
                'landmarks': [365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136],
                'start': (width, height),
                'middle': (width / 2, height),
                'end': (0, height),
                'segments': 10
            },
            'left': {
                'landmarks': [136, 172, 58, 132, 93, 234, 127, 162, 21, 54],
                'start': (0, height),
                'middle': (0, height / 2),
                'end': (0, 0),
                'segments': 9
            }
        }

        current_point_idx = len(cls._base_landmarks)

        for edge_name, edge_info in edges.items():
            print(f"Processing {edge_name} edge")

            landmarks = edge_info['landmarks']
            n_segments = edge_info['segments']
            print(f"Creating {n_segments} segments for {len(landmarks)} landmarks")

            # Generate edge points
            edge_points = cls.interpolate_edge_points(
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

    @classmethod
    def get_base_landmarks(cls, size=None, x_scale=1.0, y_translation=0.0):
        """
        Get base landmark positions with optional transformations.

        Args:
            size: Target size (int or tuple)
            x_scale: Horizontal scaling factor (0.5 to 1.0)
            y_translation: Vertical translation (-0.5 to 0.5)

        Returns:
            numpy array: Base landmarks with applied transformations
        """
        landmarks, _ = cls._load_obj_data()

        # Apply transformations before scaling to size
        landmarks = cls._transform_landmarks(landmarks, x_scale, y_translation)

        if size is not None:
            if isinstance(size, tuple):
                landmarks = landmarks * np.array(size)
            else:
                landmarks = landmarks * size

        return landmarks

    @classmethod
    def get_face_triangles(cls, size=None, x_scale=1.0, y_translation=0.0):
        """
        Get face triangulation including boundary triangles with optional transformations

        Args:
            size: Target size (int or tuple)
            x_scale: Horizontal scaling factor (0.5 to 1.0)
            y_translation: Vertical translation (-0.5 to 0.5)

        Returns:
            tuple: (triangles array, landmarks array)
        """
        landmarks, faces = cls._load_obj_data()

        # Apply transformations
        landmarks = cls._transform_landmarks(landmarks, x_scale, y_translation)

        if size is not None:
            if isinstance(size, tuple):
                scaled_landmarks = landmarks * np.array(size)
            else:
                scaled_landmarks = landmarks * size

            if cls._boundary_faces is None:
                cls.create_boundary_triangles(size)

            if cls._boundary_faces is not None:
                # Transform boundary landmarks if they exist
                if cls._boundary_landmarks is not None:
                    boundary_landmarks = cls._boundary_landmarks
                    # Note: We don't apply x_scale to boundary landmarks as they should
                    # stay at the image edges, but we do apply y_translation
                    if y_translation != 0:
                        boundary_landmarks = boundary_landmarks.copy()
                        boundary_landmarks[:, 1] += y_translation * (size if isinstance(size, int) else size[1])

                    all_points = np.vstack([scaled_landmarks, boundary_landmarks])
                    all_triangles = np.vstack([faces, cls._boundary_faces])
                    return all_triangles, all_points

        return faces, landmarks

    @staticmethod
    def validate_size(size):
        """
        Validate size parameter
        """
        if isinstance(size, (int, tuple)):
            if isinstance(size, int) and size <= 0:
                raise ValueError("Size must be a positive integer")
            if isinstance(size, tuple) and (len(size) != 2 or size[0] <= 0 or size[1] <= 0):
                raise ValueError("Size tuple must contain two positive integers")
            return size
        raise ValueError("Size must be an integer or tuple of two integers")