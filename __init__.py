import os

_SKIP_IMPORTS = os.environ.get("FACEPROCESSOR_SKIP_IMPORTS") == "1"

if not _SKIP_IMPORTS:
    from .nodes.image_feeder import ImageFeeder
    from .nodes.image_filters import HighPassFilter
    from .nodes.face_wrapper import FaceWrapper
    from .nodes.face_fit_and_restore import FaceFitAndRestore
    from .faceprocessor_v2.nodes_fit_restore import FaceFitAndRestoreV2
    from .faceprocessor_v2.nodes_wrapper import FaceWrapperV2
    from .nodes.face_tracker import FaceTracker
else:  # pragma: no cover - used only when optional deps are missing
    ImageFeeder = None
    HighPassFilter = None
    FaceWrapper = None
    FaceFitAndRestore = None
    FaceFitAndRestoreV2 = None
    FaceWrapperV2 = None
    FaceTracker = None

# Get the path to the current directory
NODE_PATH = os.path.dirname(os.path.realpath(__file__))

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

if FaceFitAndRestore is not None:
    NODE_CLASS_MAPPINGS["FaceFitAndRestore"] = FaceFitAndRestore
    NODE_DISPLAY_NAME_MAPPINGS["FaceFitAndRestore"] = "Face Fit or Restore"

if FaceFitAndRestoreV2 is not None:
    NODE_CLASS_MAPPINGS["FaceFitAndRestoreV2"] = FaceFitAndRestoreV2
    NODE_DISPLAY_NAME_MAPPINGS["FaceFitAndRestoreV2"] = "Face Fit or Restore (v2)"

if FaceWrapper is not None:
    NODE_CLASS_MAPPINGS["FaceWrapper"] = FaceWrapper
    NODE_DISPLAY_NAME_MAPPINGS["FaceWrapper"] = "Face Wrapper"

if FaceWrapperV2 is not None:
    NODE_CLASS_MAPPINGS["FaceWrapperV2"] = FaceWrapperV2
    NODE_DISPLAY_NAME_MAPPINGS["FaceWrapperV2"] = "Face Wrapper (v2)"

if HighPassFilter is not None:
    NODE_CLASS_MAPPINGS["HighPassFilter"] = HighPassFilter
    NODE_DISPLAY_NAME_MAPPINGS["HighPassFilter"] = "High Pass Filter (HPF)"

if ImageFeeder is not None:
    NODE_CLASS_MAPPINGS["ImageFeeder"] = ImageFeeder
    NODE_DISPLAY_NAME_MAPPINGS["ImageFeeder"] = "Image Feeder"

if FaceTracker is not None:
    NODE_CLASS_MAPPINGS["FaceTracker[EXPERIMENTAL]"] = FaceTracker
    NODE_DISPLAY_NAME_MAPPINGS["FaceTracker[EXPERIMENTAL]"] = "Face Tracker (Experimental)"

__version__ = "1.2.0"
