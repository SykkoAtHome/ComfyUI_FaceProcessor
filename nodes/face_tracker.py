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
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "track_face"
    CATEGORY = "Face Processor"

    def track_face(self, image):
        return (image,)