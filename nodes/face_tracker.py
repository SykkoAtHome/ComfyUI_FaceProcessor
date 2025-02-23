from ..core.face_detector import FaceDetector
from ..core.image_processor import ImageProcessor


class FaceTracker:
    """ComfyUI node for tracking facial features across image sequences."""

    def __init__(self):
        self.face_detector = FaceDetector()
        self.image_processor = ImageProcessor()

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "fp_pipe": ("DICT",),
                "debug": ("BOOLEAN", {
                    "default": False
                }),
                "use_motion_vectors": ("BOOLEAN", {
                    "default": False
                }),
                "reset_tracker_interval": ("INT", {
                    "default": 10,
                    "min": 3,
                    "max": 50,
                    "step": 1
                }),
                "tracker_region_size": (["32", "48", "64", "96", "128"], {
                    "default": "64"
                }),
                "proxy_scale": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.2,
                    "max": 1.0,
                    "step": 0.05
                }),
                "show_detection": ("BOOLEAN", {
                    "default": False
                }),
                "show_region": ("BOOLEAN", {
                    "default": False
                }),
            }
        }

    RETURN_TYPES = ("IMAGE", "DICT")
    RETURN_NAMES = ("image", "fp_pipe")
    FUNCTION = "track_face"
    CATEGORY = "Face Processor"

    def track_face(self, image, fp_pipe, debug, use_motion_vectors, reset_points_interval,
                  tracker_region_size, proxy_scale, show_detection, show_region):
        return image, fp_pipe