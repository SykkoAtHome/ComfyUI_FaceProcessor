# __init__.py
import os
import sys
from .nodes.face_fit_and_restore import FaceFitAndRestore

# Add current directory to system path
sys.path.append(os.path.dirname(os.path.realpath(__file__)))

# Get the path to the current directory
NODE_PATH = os.path.dirname(os.path.realpath(__file__))


NODE_CLASS_MAPPINGS = {
    "FaceFitAndRestore": FaceFitAndRestore
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FaceFitAndRestore": "Face Fit or Restore"
}

__version__ = "1.0.0"