import importlib.util
from pathlib import Path
import sys
import types

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PKG_NAME = "faceprocessor_v2"
if PKG_NAME not in sys.modules:
    pkg = types.ModuleType(PKG_NAME)
    pkg.__path__ = [str(ROOT / PKG_NAME)]
    sys.modules[PKG_NAME] = pkg

pipe_spec = importlib.util.spec_from_file_location(
    f"{PKG_NAME}.pipe", ROOT / PKG_NAME / "pipe.py"
)
fp2_pipe = importlib.util.module_from_spec(pipe_spec)
assert pipe_spec and pipe_spec.loader
sys.modules[pipe_spec.name] = fp2_pipe
pipe_spec.loader.exec_module(fp2_pipe)
FacePipe = fp2_pipe.FacePipe

frame_ids_spec = importlib.util.spec_from_file_location(
    f"{PKG_NAME}.frame_ids", ROOT / PKG_NAME / "frame_ids.py"
)
fp2_frame_ids = importlib.util.module_from_spec(frame_ids_spec)
assert frame_ids_spec and frame_ids_spec.loader
sys.modules[frame_ids_spec.name] = fp2_frame_ids
frame_ids_spec.loader.exec_module(fp2_frame_ids)
resolve_frame_ids = fp2_frame_ids.resolve_frame_ids


def test_resolve_frame_ids_uses_meta_order():
    pipe = FacePipe(meta={"frame_order": ["frame_a", "frame_b", "frame_c"]})
    assert resolve_frame_ids(pipe, 2) == ["frame_a", "frame_b"]


def test_resolve_frame_ids_requires_enough_entries():
    pipe = FacePipe(meta={"frame_order": ["frame_a"]})
    with pytest.raises(ValueError):
        resolve_frame_ids(pipe, 2)


def test_resolve_frame_ids_falls_back_when_missing():
    pipe = FacePipe()
    assert resolve_frame_ids(pipe, 1, fallback=["fallback_0"]) == ["fallback_0"]
