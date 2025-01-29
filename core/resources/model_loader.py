import os
from typing import Optional, Dict, List

import dlib
import mediapipe as mp
import numpy as np
import requests
from tqdm import tqdm


class ModelDownload:
    """Class for managing model downloads and verification"""

    MODEL_URLS = {
        # Mediapipe 468 landmark model
        "canonical_face_model": "https://raw.githubusercontent.com/google-ai-edge/mediapipe/master/mediapipe/modules/face_geometry/data/canonical_face_model.obj",

        # Dlib 68 landmark model
        "shape_predictor_68_face_landmarks": "https://huggingface.co/spaces/asdasdasdasd/Face-forgery-detection/resolve/ccfc24642e0210d4d885bc7b3dbc9a68ed948ad6/shape_predictor_68_face_landmarks.dat"
    }

    def __init__(self):
        """Initialize model downloader"""
        self.chunk_size = 8192
        self.models_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "models")

    def download_model(self, model_name: str, target_dir: Optional[str] = None) -> Optional[str]:
        """
        Download model from predefined URL.

        Args:
            model_name: Name of the model to download
            target_dir: Optional target directory (defaults to models/{model_name})

        Returns:
            str: Path to downloaded model or None if failed
        """
        try:
            if model_name not in self.MODEL_URLS:
                print(f"Unknown model: {model_name}")
                return None

            # Set target directory
            if target_dir is None:
                target_dir = os.path.join(self.models_dir, os.path.splitext(model_name)[0])

            # Create target directory if it doesn't exist
            os.makedirs(target_dir, exist_ok=True)

            # Set target file path
            url = self.MODEL_URLS[model_name]
            if not url:  # Skip if URL is empty
                print(f"No URL defined for model: {model_name}")
                return None

            file_name = os.path.basename(url.split('?')[0])  # Remove URL parameters if any
            target_path = os.path.join(target_dir, file_name)

            # Check if file already exists
            if os.path.exists(target_path):
                print(f"Model {model_name} already exists at: {target_path}")
                return target_path

            print(f"Downloading {model_name} from: {url}")
            print(f"Target path: {target_path}")

            # Download file with progress bar
            response = requests.get(url, stream=True)
            total_size = int(response.headers.get('content-length', 0))

            with open(target_path, 'wb') as f:
                with tqdm(total=total_size, unit='B', unit_scale=True, desc=model_name) as pbar:
                    for data in response.iter_content(chunk_size=self.chunk_size):
                        f.write(data)
                        pbar.update(len(data))

            print(f"Successfully downloaded {model_name} to: {target_path}")
            return target_path

        except Exception as e:
            print(f"Error downloading model {model_name}: {str(e)}")
            return None

    def get_model_path(self, model_name: str, target_dir: Optional[str] = None) -> Optional[str]:
        """
        Get path to model, download if not exists.

        Args:
            model_name: Name of the model
            target_dir: Optional target directory

        Returns:
            str: Path to model or None if failed
        """
        if target_dir is None:
            target_dir = os.path.join(self.models_dir, os.path.splitext(model_name)[0])

        # Check if model exists
        if os.path.exists(target_dir):
            expected_file = os.path.join(target_dir, os.path.basename(self.MODEL_URLS[model_name].split('?')[0]))
            if os.path.exists(expected_file):
                return expected_file

        # Download if not exists
        return self.download_model(model_name, target_dir)


class ModelOBJ:
    """Class for loading and parsing OBJ files with landmark transformation support"""

    _instance: Optional['ModelOBJ'] = None
    _vertices: List[List[float]] = []
    _uvs: List[List[float]] = []
    _faces: List[List[int]] = []
    _vertex_to_uv: Dict[int, int] = {}
    _landmarks: Optional[np.ndarray] = None
    _num_landmarks: int = 468
    _model_loaded: bool = False

    def __new__(cls):
        """Implement singleton pattern"""
        if cls._instance is None:
            cls._instance = super(ModelOBJ, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        """Initialize ModelOBJ and load face model"""
        if not self._model_loaded:
            self.downloader = ModelDownload()
            self._load_obj()

    def _load_obj(self) -> None:
        """Load and parse MediaPipe face model OBJ file"""
        try:
            # Get model path, download if needed
            model_path = self.downloader.get_model_path("canonical_face_model")
            if not model_path:
                raise FileNotFoundError("Failed to get canonical face model")

            print(f"Loading 3D model from: {model_path}")

            with open(model_path, 'r') as f:
                for line in f:
                    if line.startswith('v '):
                        # Parse vertex coordinates
                        parts = line.strip().split()
                        self._vertices.append([float(p) for p in parts[1:4]])

                    elif line.startswith('vt '):
                        # Parse UV coordinates (flip V coordinate)
                        parts = line.strip().split()
                        u = float(parts[1])
                        v = 1.0 - float(parts[2])  # Flip V coordinate
                        self._uvs.append([u, v])

                    elif line.startswith('f '):
                        # Parse face indices
                        parts = line.strip().split()[1:]
                        face_verts = []
                        for part in parts:
                            indices = part.split('/')
                            if len(indices) >= 2:
                                vertex_idx = int(indices[0]) - 1  # OBJ indices are 1-based
                                uv_idx = int(indices[1]) - 1
                                self._vertex_to_uv[vertex_idx] = uv_idx
                                face_verts.append(vertex_idx)
                        if len(face_verts) == 3:  # Only store triangular faces
                            self._faces.append(face_verts)

            # Initialize landmarks
            self._landmarks = self._initialize_landmarks()
            self._model_loaded = True

            print(f"Successfully loaded model data:")
            print(f"- Vertices: {len(self._vertices)}")
            print(f"- UVs: {len(self._uvs)}")
            print(f"- Faces: {len(self._faces)}")
            print(f"- Vertex to UV mappings: {len(self._vertex_to_uv)}")

        except Exception as e:
            print(f"Error loading OBJ file: {str(e)}")
            raise

    def _initialize_landmarks(self) -> np.ndarray:
        """
        Initialize landmarks array from UV coordinates

        Returns:
            numpy.ndarray: Array of landmark coordinates in UV space
        """
        landmarks = np.zeros((self._num_landmarks, 2), dtype=np.float32)
        uvs = np.array(self._uvs, dtype=np.float32)

        for vertex_idx, uv_idx in self._vertex_to_uv.items():
            if vertex_idx < self._num_landmarks:
                landmarks[vertex_idx] = uvs[uv_idx]

        return landmarks

    def get_faces(self) -> np.ndarray:
        """
        Get face indices as numpy array

        Returns:
            numpy.ndarray: Array of face indices
        """
        return np.array(self._faces, dtype=np.int32)

    def get_transformed_landmarks(self, x_scale: float = 1.0, y_translation: float = 0.0) -> np.ndarray:
        """
        Get landmarks with applied transformations

        Args:
            x_scale: Horizontal scaling factor (0.5 to 1.0)
            y_translation: Vertical translation (-0.5 to 0.5)

        Returns:
            numpy.ndarray: Transformed landmarks
        """
        if self._landmarks is None:
            raise RuntimeError("Landmarks not initialized")

        transformed = self._landmarks.copy()

        # Calculate center point
        center_x = (self._landmarks[:, 0].min() + self._landmarks[:, 0].max()) / 2

        # Calculate horizontal distance from center (normalized to 0-1)
        dx = np.abs(transformed[:, 0] - center_x)

        # Avoid division by zero
        if np.abs(center_x) < 1e-6:  # Use small epsilon value
            influence = np.zeros_like(dx)
        else:
            influence = np.clip(dx / center_x, 0, 1)

        # Calculate scaled positions with horizontal influence
        scale_factor = 1.0 + (x_scale - 1.0) * influence
        transformed[:, 0] = center_x + (transformed[:, 0] - center_x) * scale_factor

        # Apply vertical translation
        transformed[:, 1] += y_translation

        return transformed

    @property
    def num_landmarks(self) -> int:
        """
        Get the number of landmarks

        Returns:
            int: Number of landmarks
        """
        return self._num_landmarks

    def _initialize_landmarks(self) -> np.ndarray:
        """
        Initialize landmarks array from UV coordinates

        Returns:
            numpy.ndarray: Array of landmark coordinates in UV space
        """
        landmarks = np.zeros((self._num_landmarks, 2), dtype=np.float32)
        uvs = np.array(self._uvs, dtype=np.float32)

        for vertex_idx, uv_idx in self._vertex_to_uv.items():
            if vertex_idx < self._num_landmarks:
                landmarks[vertex_idx] = uvs[uv_idx]

        return landmarks

    def get_faces(self) -> np.ndarray:
        """
        Get face indices as numpy array

        Returns:
            numpy.ndarray: Array of face indices
        """
        return np.array(self._faces, dtype=np.int32)

    def get_transformed_landmarks(self, x_scale: float = 1.0, y_translation: float = 0.0) -> np.ndarray:
        """
        Get landmarks with applied transformations

        Args:
            x_scale: Horizontal scaling factor (0.5 to 1.0)
            y_translation: Vertical translation (-0.5 to 0.5)

        Returns:
            numpy.ndarray: Transformed landmarks
        """
        if self._landmarks is None:
            raise RuntimeError("Landmarks not initialized")

        transformed = self._landmarks.copy()

        # Calculate center point
        center_x = (self._landmarks[:, 0].min() + self._landmarks[:, 0].max()) / 2

        # Calculate horizontal distance from center (normalized to 0-1)
        dx = np.abs(transformed[:, 0] - center_x)

        # Avoid division by zero
        if np.abs(center_x) < 1e-6:  # Use small epsilon value
            influence = np.zeros_like(dx)
        else:
            influence = np.clip(dx / center_x, 0, 1)

        # Calculate scaled positions with horizontal influence
        scale_factor = 1.0 + (x_scale - 1.0) * influence
        transformed[:, 0] = center_x + (transformed[:, 0] - center_x) * scale_factor

        # Apply vertical translation
        transformed[:, 1] += y_translation

        return transformed


class ModelDlib:
    """Class for loading and managing Dlib face detection models using singleton pattern"""

    _instance: Optional['ModelDlib'] = None
    _face_detector: Optional[dlib.fhog_object_detector] = None
    _shape_predictor: Optional[dlib.shape_predictor] = None
    _model_loaded: bool = False

    def __new__(cls):
        """Implement singleton pattern"""
        if cls._instance is None:
            cls._instance = super(ModelDlib, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        """Initialize Dlib model loader"""
        if not self._model_loaded:
            self.downloader = ModelDownload()
            self._initialize_models()

    def _initialize_models(self) -> None:
        """Initialize Dlib face detector and shape predictor"""
        try:
            print("Initializing Dlib face detector...")
            self._face_detector = dlib.get_frontal_face_detector()

            # Get model path, download if needed
            model_path = self.downloader.get_model_path("shape_predictor_68_face_landmarks")
            if not model_path:
                raise FileNotFoundError("Failed to get shape predictor model")

            print(f"Loading Dlib shape predictor from: {model_path}")
            self._shape_predictor = dlib.shape_predictor(model_path)
            self._model_loaded = True
            print("Dlib models initialized successfully")

        except Exception as e:
            print(f"Error initializing Dlib models: {str(e)}")
            raise

    @property
    def face_detector(self) -> dlib.fhog_object_detector:
        """Get Dlib face detector instance"""
        if not self._face_detector:
            self._initialize_models()
        return self._face_detector

    @property
    def shape_predictor(self) -> dlib.shape_predictor:
        """Get Dlib shape predictor instance"""
        if not self._shape_predictor:
            self._initialize_models()
        return self._shape_predictor


class ModelMediaPipe:
    """Class for loading and managing MediaPipe face mesh models using singleton pattern"""

    _instance: Optional['ModelMediaPipe'] = None
    _face_mesh: Optional[mp.solutions.face_mesh.FaceMesh] = None
    _model_loaded: bool = False

    def __new__(cls):
        """Implement singleton pattern"""
        if cls._instance is None:
            cls._instance = super(ModelMediaPipe, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        """Initialize MediaPipe model loader"""
        if not self._model_loaded:
            self._initialize_models()

    def _initialize_models(self) -> None:
        """Initialize MediaPipe face mesh model"""
        try:
            print("Initializing MediaPipe Face Mesh...")

            self._face_mesh = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5
            )

            self._model_loaded = True
            print("MediaPipe Face Mesh initialized successfully")

        except Exception as e:
            print(f"Error initializing MediaPipe models: {str(e)}")
            raise

    @property
    def face_mesh(self) -> mp.solutions.face_mesh.FaceMesh:
        """Get MediaPipe Face Mesh instance"""
        if not self._face_mesh:
            self._initialize_models()
        return self._face_mesh

    def __del__(self):
        """Clean up resources"""
        if self._face_mesh:
            self._face_mesh.close()