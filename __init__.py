import os
import warnings
from importlib import import_module

_SKIP_IMPORTS = os.environ.get("FACEPROCESSOR_SKIP_IMPORTS") == "1"

if not _SKIP_IMPORTS:
    _pkg_name = __name__.split(".")[0]

    def _optional_import(importer):  # pragma: no cover - exercised during import
        try:
            return importer()
        except (ModuleNotFoundError, ImportError) as exc:  # missing optional deps (torch, cv2, etc.)
            missing = getattr(exc, "name", None)
            if missing and missing.startswith(_pkg_name):
                raise
            warnings.warn(
                f"Skipping FaceProcessor node import due to missing dependency: {exc}",
                RuntimeWarning,
            )
            return None

    ImageFeeder = _optional_import(lambda: getattr(import_module(".nodes.image_feeder", __name__), "ImageFeeder"))
    HighPassFilter = _optional_import(lambda: getattr(import_module(".nodes.image_filters", __name__), "HighPassFilter"))
    FaceFitAndRestore = _optional_import(lambda: getattr(import_module(".faceprocessor.nodes_fit_restore", __name__), "FaceFitAndRestore"))
    FaceWrapper = _optional_import(lambda: getattr(import_module(".faceprocessor.nodes_wrapper", __name__), "FaceWrapper"))
    FaceTracker = _optional_import(lambda: getattr(import_module(".nodes.face_tracker", __name__), "FaceTracker"))
else:  # pragma: no cover - used only when optional deps are missing
    ImageFeeder = None
    HighPassFilter = None
    FaceFitAndRestore = None
    FaceWrapper = None
    FaceTracker = None

# Get the path to the current directory
NODE_PATH = os.path.dirname(os.path.realpath(__file__))

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

if FaceFitAndRestore is not None:
    NODE_CLASS_MAPPINGS["FaceFitAndRestore"] = FaceFitAndRestore
    NODE_DISPLAY_NAME_MAPPINGS["FaceFitAndRestore"] = "Face Fit or Restore"

if FaceWrapper is not None:
    NODE_CLASS_MAPPINGS["FaceWrapper"] = FaceWrapper
    NODE_DISPLAY_NAME_MAPPINGS["FaceWrapper"] = "Face Wrapper"

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
