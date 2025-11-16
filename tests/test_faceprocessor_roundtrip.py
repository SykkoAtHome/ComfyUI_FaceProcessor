"""Integration test covering the Fit→Unwrap→Wrap→Restore pipeline."""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

torch = pytest.importorskip("torch")
import torch.nn.functional as F  # noqa: E402  # pylint: disable=wrong-import-position

from faceprocessor.nodes_fit_restore import FaceFitAndRestore
from faceprocessor.nodes_wrapper import FaceWrapper
from faceprocessor.pipe import get_or_create_frame
from faceprocessor.utils import ensure_face_model


def _gradient_image(size: int = 512) -> torch.Tensor:
    coords = torch.linspace(0.0, 1.0, size, dtype=torch.float32)
    grid_x = coords.repeat(size, 1)
    grid_y = coords.view(-1, 1).repeat(1, size)
    grid_z = torch.flip(grid_x, dims=[1])
    image = torch.stack((grid_x, grid_y, grid_z), dim=-1)
    return image.unsqueeze(0)


def test_faceprocessor_roundtrip_pipeline(monkeypatch):
    node = FaceFitAndRestore()
    wrapper = FaceWrapper()
    input_image = _gradient_image()
    bbox_size = "512"

    def fake_fit_single(self, tensor, frame_id, pipe, padding_percent, bbox_size, source_path):
        canonical_model = ensure_face_model(pipe, bbox_size, "canonical_models")
        canonical_landmarks = np.asarray(canonical_model["landmarks"], dtype=np.float32)

        resized = tensor
        if tensor.shape[1] != bbox_size:
            resized = (
                F.interpolate(
                    tensor.permute(0, 3, 1, 2),
                    size=(bbox_size, bbox_size),
                    mode="bilinear",
                    align_corners=False,
                ).permute(0, 2, 3, 1)
            )

        mask = torch.ones((1, bbox_size, bbox_size), dtype=torch.float32)

        frame_data = replace(
            get_or_create_frame(pipe, frame_id),
            frame_id=frame_id,
            orig_size=(tensor.shape[1], tensor.shape[2]),
            bbox=(0, 0, tensor.shape[2], tensor.shape[1]),
            detected_lm=canonical_landmarks.copy(),
            target_lm=canonical_landmarks.copy(),
            extra={"bbox_size": bbox_size, "rotation_angle": 0.0},
        )
        return resized.clone(), mask, frame_data

    monkeypatch.setattr(FaceFitAndRestore, "_fit_single", fake_fit_single)

    fit_image, fit_mask, pipe = node.process(
        mode="Fit",
        bbox_size=bbox_size,
        padding_percent=0.0,
        image=input_image,
    )

    unwrap_image, unwrap_mask, pipe = wrapper.process(
        mode="UNWRAP",
        image=fit_image,
        unwrap_size=bbox_size,
        mask=fit_mask,
        fp_pipe=pipe,
    )

    rewrapped_image, rewrapped_mask, pipe = wrapper.process(
        mode="WRAP",
        image=unwrap_image,
        unwrap_size=bbox_size,
        mask=unwrap_mask,
        fp_pipe=pipe,
    )

    restored_image, restored_mask, _ = node.process(
        mode="Restore",
        bbox_size=bbox_size,
        padding_percent=0.0,
        image=rewrapped_image,
        fp_pipe=pipe,
    )

    assert restored_image.shape == input_image.shape
    assert restored_mask.shape[1:] == input_image.shape[1:3]

    mse = torch.mean((restored_image - input_image) ** 2).item()
    assert mse < 1e-4
    assert torch.mean(rewrapped_mask).item() > 0.9
