import os

from .nodes.image_feeder import ImageFeeder
from .nodes.image_filters import HighPassFilter
from .nodes.face_wrapper import FaceWrapper
from .nodes.face_fit_and_restore import FaceFitAndRestore

# Get the path to the current directory
NODE_PATH = os.path.dirname(os.path.realpath(__file__))

NODE_CLASS_MAPPINGS = {
    "FaceFitAndRestore": FaceFitAndRestore,
    "FaceWrapper": FaceWrapper,
    "HighPassFilter": HighPassFilter,
    "ImageFeeder": ImageFeeder
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FaceFitAndRestore": "Face Fit or Restore",
    "FaceWrapper": "Face Wrapper",
    "HighPassFilter": "High Pass Filter (HPF)",
    "ImageFeeder": "Image Feeder"
}

__version__ = "1.1.0"
