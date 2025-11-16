import os
import warnings
from importlib import import_module

_SKIP_IMPORTS = os.environ.get("FACEPROCESSOR_SKIP_IMPORTS") == "1"

if not _SKIP_IMPORTS:
    _pkg_name = __name__.split(".")[0]

    def _missing_dependency_stub(node_name: str, display_name: str, exc: Exception, *,
                                 category: str, input_types: dict, return_types: tuple,
                                 return_names: tuple, function_name: str = "process"):
        """Build a lightweight ComfyUI node that surfaces missing dependency errors."""

        missing = getattr(exc, "name", None) or exc.__class__.__name__
        details = str(exc)

        class _MissingNode:
            CATEGORY = category
            FUNCTION = function_name
            RETURN_TYPES = return_types
            RETURN_NAMES = return_names

            @classmethod
            def INPUT_TYPES(cls):  # pragma: no cover - simple accessors
                return input_types

            def process(self, *_, **__):  # pragma: no cover - executed only when deps missing
                raise RuntimeError(
                    f"{display_name} is unavailable because the '{missing}' dependency could not be imported. "
                    "Install the packages listed in requirements.txt and restart ComfyUI. "
                    f"Original error: {details}"
                )

        _MissingNode.__name__ = f"{node_name}Unavailable"
        return _MissingNode

    def _optional_import(importer, *, node_name: str, display_name: str, stub_factory=None):
        """Import a node while gracefully handling missing optional dependencies."""

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
            if stub_factory is not None:
                return stub_factory(exc)
            return None

    ImageFeeder = _optional_import(
        lambda: getattr(import_module(".nodes.image_feeder", __name__), "ImageFeeder"),
        node_name="ImageFeeder",
        display_name="Image Feeder",
    )
    HighPassFilter = _optional_import(
        lambda: getattr(import_module(".nodes.image_filters", __name__), "HighPassFilter"),
        node_name="HighPassFilter",
        display_name="High Pass Filter (HPF)",
    )
    FaceFitAndRestore = _optional_import(
        lambda: getattr(import_module(".faceprocessor.nodes_fit_restore", __name__), "FaceFitAndRestore"),
        node_name="FaceFitAndRestore",
        display_name="Face Fit or Restore",
        stub_factory=lambda exc: _missing_dependency_stub(
            node_name="FaceFitAndRestore",
            display_name="Face Fit or Restore",
            exc=exc,
            category="Face Processor",
            input_types={
                "required": {
                    "mode": (["Fit", "Restore"], {"default": "Fit"}),
                    "bbox_size": (["512", "1024", "2048"], {"default": "1024"}),
                    "padding_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 0.5, "step": 0.01}),
                },
                "optional": {
                    "image": ("IMAGE",),
                    "image_paths": ("STRING", {"forceInput": True}),
                    "fp_pipe": ("DICT", {"default": None}),
                },
            },
            return_types=("IMAGE", "MASK", "DICT"),
            return_names=("image", "mask", "fp_pipe"),
        ),
    )
    FaceWrapper = _optional_import(
        lambda: getattr(import_module(".faceprocessor.nodes_wrapper", __name__), "FaceWrapper"),
        node_name="FaceWrapper",
        display_name="Face Wrapper",
        stub_factory=lambda exc: _missing_dependency_stub(
            node_name="FaceWrapper",
            display_name="Face Wrapper",
            exc=exc,
            category="Face Processor",
            input_types={
                "required": {
                    "mode": (["UNWRAP", "WRAP"], {"default": "UNWRAP"}),
                    "image": ("IMAGE",),
                    "unwrap_size": (["512", "768", "1024"], {"default": "1024"}),
                },
                "optional": {
                    "mask": ("MASK", {"default": None}),
                    "fp_pipe": ("DICT", {"default": None}),
                },
            },
            return_types=("IMAGE", "MASK", "DICT"),
            return_names=("image", "mask", "fp_pipe"),
        ),
    )
    FaceTracker = _optional_import(
        lambda: getattr(import_module(".nodes.face_tracker", __name__), "FaceTracker"),
        node_name="FaceTracker",
        display_name="Face Tracker (Experimental)",
    )
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
    # Backwards compatibility with saved workflows referencing the legacy v2 name
    NODE_CLASS_MAPPINGS["FaceFitAndRestoreV2"] = FaceFitAndRestore
    NODE_DISPLAY_NAME_MAPPINGS["FaceFitAndRestoreV2"] = "Face Fit or Restore"

if FaceWrapper is not None:
    NODE_CLASS_MAPPINGS["FaceWrapper"] = FaceWrapper
    NODE_DISPLAY_NAME_MAPPINGS["FaceWrapper"] = "Face Wrapper"
    # Backwards compatibility with saved workflows referencing the legacy v2 name
    NODE_CLASS_MAPPINGS["FaceWrapperV2"] = FaceWrapper
    NODE_DISPLAY_NAME_MAPPINGS["FaceWrapperV2"] = "Face Wrapper"

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
