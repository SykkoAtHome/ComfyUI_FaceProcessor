import os
from typing import Optional, Dict, List

import dlib
import numpy as np
from insightface.app import FaceAnalysis
import mediapipe as mp



class ModelDownload:
    """
    placeholder for model download
    """
    MODEL_URLS = {
        "canonical_face_model": "https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/modules/face_geometry/data/canonical_face_model.obj", # Mediapipe 468 obj model
        "shape_predictor_68_face_landmarks": "https://huggingface.co/spaces/asdasdasdasd/Face-forgery-detection/resolve/ccfc24642e0210d4d885bc7b3dbc9a68ed948ad6/shape_predictor_68_face_landmarks.dat" # dlib model
    }
    def download_model(self, model_name):
        pass


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

class ModelOBJ:
    """Class for loading and parsing OBJ files with landmark transformation support"""

    def __init__(self):
        """Initialize ModelOBJ and load face model"""
        self.vertices: List[List[float]] = []
        self.uvs: List[List[float]] = []
        self.faces: List[List[int]] = []
        self.vertex_to_uv: Dict[int, int] = {}
        self.landmarks: Optional[np.ndarray] = None
        self.num_landmarks: int = 468

        # Load the model during initialization
        self._load_obj()

    def _load_obj(self) -> None:
        """Load and parse MediaPipe face model OBJ file"""
        current_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
        obj_path = os.path.join(current_dir, 'resources', 'models', 'canonical_face_model.obj')

        print(f"Loading 3D model from: {obj_path}")

        if not os.path.exists(obj_path):
            raise FileNotFoundError(f"OBJ file not found at: {obj_path}")

        try:
            with open(obj_path, 'r') as f:
                for line in f:
                    if line.startswith('v '):
                        # Parse vertex coordinates
                        parts = line.strip().split()
                        self.vertices.append([float(p) for p in parts[1:4]])

                    elif line.startswith('vt '):
                        # Parse UV coordinates (flip V coordinate)
                        parts = line.strip().split()
                        u = float(parts[1])
                        v = 1.0 - float(parts[2])  # Flip V coordinate
                        self.uvs.append([u, v])

                    elif line.startswith('f '):
                        # Parse face indices
                        parts = line.strip().split()[1:]
                        face_verts = []
                        for part in parts:
                            indices = part.split('/')
                            if len(indices) >= 2:
                                vertex_idx = int(indices[0]) - 1  # OBJ indices are 1-based
                                uv_idx = int(indices[1]) - 1
                                self.vertex_to_uv[vertex_idx] = uv_idx
                                face_verts.append(vertex_idx)
                        if len(face_verts) == 3:  # Only store triangular faces
                            self.faces.append(face_verts)

            # Initialize landmarks
            self.landmarks = self._initialize_landmarks()

            print(f"Successfully loaded model data:")
            print(f"- Vertices: {len(self.vertices)}")
            print(f"- UVs: {len(self.uvs)}")
            print(f"- Faces: {len(self.faces)}")
            print(f"- Vertex to UV mappings: {len(self.vertex_to_uv)}")

        except Exception as e:
            print(f"Error loading OBJ file: {str(e)}")
            raise

    def _initialize_landmarks(self) -> np.ndarray:
        """
        Initialize landmarks array from UV coordinates

        Returns:
            numpy.ndarray: Array of landmark coordinates in UV space
        """
        landmarks = np.zeros((self.num_landmarks, 2), dtype=np.float32)
        uvs = np.array(self.uvs, dtype=np.float32)

        for vertex_idx, uv_idx in self.vertex_to_uv.items():
            if vertex_idx < self.num_landmarks:
                landmarks[vertex_idx] = uvs[uv_idx]

        return landmarks

    def get_faces(self) -> np.ndarray:
        """
        Get face indices as numpy array

        Returns:
            numpy.ndarray: Array of face indices
        """
        return np.array(self.faces, dtype=np.int32)

    def get_transformed_landmarks(self, x_scale: float = 1.0, y_translation: float = 0.0) -> np.ndarray:
        """
        Get landmarks with applied transformations

        Args:
            x_scale: Horizontal scaling factor (0.5 to 1.0)
            y_translation: Vertical translation (-0.5 to 0.5)

        Returns:
            numpy.ndarray: Transformed landmarks
        """
        if self.landmarks is None:
            raise RuntimeError("Landmarks not initialized")

        transformed = self.landmarks.copy()

        # Calculate center point
        center_x = (self.landmarks[:, 0].min() + self.landmarks[:, 0].max()) / 2

        # Calculate horizontal distance from center (normalized to 0-1)
        dx = np.abs(transformed[:, 0] - center_x)
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
            self._initialize_models()

    def _initialize_models(self) -> None:
        """Initialize Dlib face detector and shape predictor"""
        try:
            # Initialize face detector
            print("Initializing Dlib face detector...")
            self._face_detector = dlib.get_frontal_face_detector()

            # Get path to the model file
            current_dir = os.path.dirname(os.path.realpath(__file__))
            model_path = os.path.join(current_dir, "models", "shape_predictor_68_face_landmarks.dat")

            if not os.path.exists(model_path):
                print(f"Dlib landmarks model not found at: {model_path}")
                print("Please ensure the model file is placed in the core/resources/models directory")
                raise FileNotFoundError(f"Missing model file at: {model_path}")

            # Load shape predictor
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

class ModelInsightFace:
    """Class for loading and managing InsightFace models using singleton pattern"""

    _instance: Optional['ModelInsightFace'] = None
    _app: Optional[FaceAnalysis] = None
    _model_loaded: bool = False

    def __new__(cls):
        """Implement singleton pattern"""
        if cls._instance is None:
            cls._instance = super(ModelInsightFace, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        """Initialize InsightFace model loader"""
        if not self._model_loaded:
            self._initialize_models()

    def _initialize_models(self) -> None:
        """Initialize InsightFace models"""
        try:
            print("Initializing InsightFace detector...")

            # Initialize FaceAnalysis with required modules
            self._app = FaceAnalysis(
                allowed_modules=['detection', 'landmark_2d_106'],
                providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
            )

            # Prepare the model
            self._app.prepare(ctx_id=0, det_size=(640, 640))

            self._model_loaded = True
            print("InsightFace models initialized successfully")

        except Exception as e:
            print(f"Error initializing InsightFace models: {str(e)}")
            raise

    @property
    def app(self) -> FaceAnalysis:
        """Get InsightFace FaceAnalysis instance"""
        if not self._app:
            self._initialize_models()
        return self._app