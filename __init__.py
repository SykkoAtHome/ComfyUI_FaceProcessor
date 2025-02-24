import os

from .nodes.image_feeder import ImageFeeder
from .nodes.image_filters import HighPassFilter
from .nodes.face_wrapper import FaceWrapper
from .nodes.face_fit_and_restore import FaceFitAndRestore
from .nodes.face_tracker import FaceTracker

# Get the path to the current directory
NODE_PATH = os.path.dirname(os.path.realpath(__file__))

NODE_CLASS_MAPPINGS = {
    "FaceFitAndRestore": FaceFitAndRestore,
    "FaceWrapper": FaceWrapper,
    "HighPassFilter": HighPassFilter,
    "ImageFeeder": ImageFeeder,
    "FaceTracker": FaceTracker
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FaceFitAndRestore": "Face Fit or Restore",
    "FaceWrapper": "Face Wrapper",
    "HighPassFilter": "High Pass Filter (HPF)",
    "ImageFeeder": "Image Feeder",
    "FaceTracker": "Face Tracker"
}

__version__ = "1.2.0"
