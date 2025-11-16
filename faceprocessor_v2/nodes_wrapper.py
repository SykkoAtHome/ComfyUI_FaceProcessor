"""FaceWrapper v2 node built on top of the TorchDeformer."""
from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from .deformer import TorchDeformer
from .pipe import FacePipe, FrameData, face_pipe_from_legacy, face_pipe_to_legacy, update_frame
from .frame_ids import resolve_frame_ids
from .utils import ensure_face_model, get_logger


class FaceWrapperV2:
    """Wraps/unwraps canonical face crops using the shared v2 Torch deformer."""

    CATEGORY = "Face Processor"
    FUNCTION = "process"
    RETURN_TYPES = ("IMAGE", "MASK", "DICT")
    RETURN_NAMES = ("image", "mask", "fp_pipe")

    def __init__(self) -> None:
        self.deformer = TorchDeformer()
        self.logger = get_logger("FaceWrapperV2")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mode": (["UNWRAP", "WRAP"], {"default": "UNWRAP"}),
                "image": ("IMAGE",),
                "unwrap_size": (["512", "768", "1024"], {"default": "1024"}),
            },
            "optional": {
                "mask": ("MASK", {"default": None}),
                "fp_pipe": ("DICT", {"default": None}),
            },
        }

    def process(
        self,
        mode: str,
        image: torch.Tensor,
        unwrap_size: str,
        mask: Optional[torch.Tensor] = None,
        fp_pipe: Optional[dict] = None,
    ):
        """Run the UNWRAP or WRAP stage for the supplied batch of canonical faces."""
        if image is None:
            raise ValueError("FaceWrapperV2 requires an input image tensor")

        pipe = face_pipe_from_legacy(fp_pipe)
        tensors = self._split_image_batch(image)
        masks = self._split_mask_batch(mask, len(tensors))
        try:
            frame_ids = resolve_frame_ids(pipe, len(tensors))
        except ValueError as exc:
            raise ValueError(
                "FaceWrapperV2 requires frame_order metadata from FaceFitAndRestoreV2"
            ) from exc
        unwrap_size_int = int(unwrap_size)

        if mode == "UNWRAP":
            pipe.meta["unwrap_size"] = unwrap_size_int
            images, mask_tensors = self._run_unwrap(
                tensors=tensors,
                masks=masks,
                frame_ids=frame_ids,
                pipe=pipe,
                unwrap_size=unwrap_size_int,
            )
        elif mode == "WRAP":
            active_unwrap_size = int(pipe.meta.get("unwrap_size", unwrap_size_int))
            images, mask_tensors = self._run_wrap(
                tensors=tensors,
                masks=masks,
                frame_ids=frame_ids,
                pipe=pipe,
                unwrap_size=active_unwrap_size,
            )
        else:
            raise ValueError(f"Unsupported FaceWrapperV2 mode: {mode}")

        legacy_pipe = face_pipe_to_legacy(pipe)
        return torch.cat(images, dim=0), torch.cat(mask_tensors, dim=0), legacy_pipe

    # ------------------------------------------------------------------
    # UNWRAP / WRAP loops
    # ------------------------------------------------------------------
    def _run_unwrap(
        self,
        tensors: List[torch.Tensor],
        masks: List[Optional[torch.Tensor]],
        frame_ids: List[str],
        pipe: FacePipe,
        unwrap_size: int,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        images: List[torch.Tensor] = []
        mask_tensors: List[torch.Tensor] = []

        for tensor, mask_tensor, frame_id in zip(tensors, masks, frame_ids):
            frame_data = pipe.frames.get(frame_id)
            image_out, mask_out = self._unwrap_single(
                tensor=tensor,
                mask_tensor=mask_tensor,
                frame_data=frame_data,
                pipe=pipe,
                unwrap_size=unwrap_size,
            )
            images.append(image_out)
            mask_tensors.append(mask_out)

        return images, mask_tensors

    def _run_wrap(
        self,
        tensors: List[torch.Tensor],
        masks: List[Optional[torch.Tensor]],
        frame_ids: List[str],
        pipe: FacePipe,
        unwrap_size: int,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        images: List[torch.Tensor] = []
        mask_tensors: List[torch.Tensor] = []

        for tensor, mask_tensor, frame_id in zip(tensors, masks, frame_ids):
            frame_data = pipe.frames.get(frame_id)
            image_out, mask_out = self._wrap_single(
                tensor=tensor,
                mask_tensor=mask_tensor,
                frame_data=frame_data,
                pipe=pipe,
                unwrap_size=unwrap_size,
            )
            images.append(image_out)
            mask_tensors.append(mask_out)

        return images, mask_tensors

    # ------------------------------------------------------------------
    # Per-frame helpers
    # ------------------------------------------------------------------
    def _unwrap_single(
        self,
        tensor: torch.Tensor,
        mask_tensor: Optional[torch.Tensor],
        frame_data: Optional[FrameData],
        pipe: FacePipe,
        unwrap_size: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if frame_data is None:
            self.logger.warning("Missing FrameData entry for unwrap request")
            return tensor, self._default_mask(tensor)

        src_landmarks = self._landmarks_to_tensor(frame_data.target_lm)
        if src_landmarks is None:
            self.logger.warning("Missing target landmarks for frame %s", frame_data.frame_id)
            return tensor, self._default_mask(tensor)

        canonical_size = int(frame_data.extra.get("bbox_size", tensor.shape[1]))
        ensure_face_model(pipe, canonical_size, "canonical_models")
        unwrap_model = ensure_face_model(pipe, unwrap_size, "unwrap_models")

        dst_landmarks = torch.tensor(unwrap_model["landmarks"], dtype=torch.float32)
        triangles = torch.tensor(unwrap_model["triangles"], dtype=torch.long)

        image_out = self.deformer.warp(tensor, src_landmarks, dst_landmarks, triangles)
        image_out = self._resize_image(image_out, (unwrap_size, unwrap_size))
        mask_image = self._mask_image(mask_tensor, (tensor.shape[1], tensor.shape[2]), tensor.device)
        mask_warped = self.deformer.warp(mask_image, src_landmarks, dst_landmarks, triangles)
        mask_out = self._image_to_mask(mask_warped, (unwrap_size, unwrap_size))

        frame_data.extra["unwrap_size"] = unwrap_size
        update_frame(pipe, frame_data)
        return image_out, mask_out

    def _wrap_single(
        self,
        tensor: torch.Tensor,
        mask_tensor: Optional[torch.Tensor],
        frame_data: Optional[FrameData],
        pipe: FacePipe,
        unwrap_size: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if frame_data is None:
            self.logger.warning("Missing FrameData entry for wrap request")
            return tensor, self._default_mask(tensor)

        src_landmarks = self._unwrap_landmarks(pipe, unwrap_size)
        dst_landmarks = self._landmarks_to_tensor(frame_data.target_lm)
        if dst_landmarks is None or src_landmarks is None:
            self.logger.warning("Incomplete landmark data for frame %s", frame_data.frame_id)
            return tensor, self._default_mask(tensor)

        canonical_size = int(frame_data.extra.get("bbox_size", tensor.shape[1]))
        canonical_model = ensure_face_model(pipe, canonical_size, "canonical_models")
        triangles = torch.tensor(canonical_model["triangles"], dtype=torch.long)

        image_out = self.deformer.warp(tensor, src_landmarks, dst_landmarks, triangles)
        image_out = self._resize_image(image_out, (canonical_size, canonical_size))
        mask_image = self._mask_image(mask_tensor, (tensor.shape[1], tensor.shape[2]), tensor.device)
        mask_warped = self.deformer.warp(mask_image, src_landmarks, dst_landmarks, triangles)
        mask_out = self._image_to_mask(mask_warped, (canonical_size, canonical_size))
        return image_out, mask_out

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------
    def _split_image_batch(self, image: torch.Tensor) -> List[torch.Tensor]:
        if image.ndim != 4:
            raise ValueError("FaceWrapperV2 expects image tensors shaped [B, H, W, 3]")
        return [image[i : i + 1].detach().clone() for i in range(image.shape[0])]

    def _split_mask_batch(self, mask: Optional[torch.Tensor], batch_len: int) -> List[Optional[torch.Tensor]]:
        if mask is None:
            return [None] * batch_len

        if mask.ndim == 2:
            mask = mask.unsqueeze(0)
        if mask.ndim != 3:
            raise ValueError("FaceWrapperV2 expects masks shaped [B, H, W]")

        if mask.shape[0] == 1 and batch_len > 1:
            return [mask.clone() for _ in range(batch_len)]
        if mask.shape[0] != batch_len:
            raise ValueError("Mask batch does not match image batch size")
        return [mask[i : i + 1].clone() for i in range(batch_len)]

    def _mask_image(
        self, mask_tensor: Optional[torch.Tensor], spatial: Tuple[int, int], device: torch.device
    ) -> torch.Tensor:
        if mask_tensor is None:
            h, w = spatial
            base = torch.ones((1, h, w), dtype=torch.float32, device=device)
        else:
            base = mask_tensor.detach().clone().to(device=device, dtype=torch.float32)
        if base.ndim == 2:
            base = base.unsqueeze(0)
        return base.clamp(0.0, 1.0).unsqueeze(-1)

    def _image_to_mask(self, tensor: torch.Tensor, spatial: Tuple[int, int]) -> torch.Tensor:
        mask = tensor[..., 0]
        mask = mask.clamp(0.0, 1.0)
        if mask.shape[1] != spatial[0] or mask.shape[2] != spatial[1]:
            mask = F.interpolate(mask.unsqueeze(0), size=spatial, mode="bilinear", align_corners=False)[0]
        return mask

    def _resize_image(self, tensor: torch.Tensor, spatial: Tuple[int, int]) -> torch.Tensor:
        if tensor.shape[1] == spatial[0] and tensor.shape[2] == spatial[1]:
            return tensor

        chw = tensor.permute(0, 3, 1, 2)
        resized = F.interpolate(chw, size=spatial, mode="bilinear", align_corners=False)
        return resized.permute(0, 2, 3, 1)

    def _default_mask(self, tensor: torch.Tensor) -> torch.Tensor:
        _, h, w, _ = tensor.shape
        return torch.ones((1, h, w), dtype=torch.float32, device=tensor.device)

    def _landmarks_to_tensor(self, data) -> Optional[torch.Tensor]:
        if data is None:
            return None
        if isinstance(data, torch.Tensor):
            tensor = data.detach().clone().float()
        else:
            arr = np.asarray(data, dtype=np.float32)
            if arr.ndim != 2 or arr.shape[1] < 2:
                return None
            tensor = torch.from_numpy(arr[:, :2])
        return tensor

    def _unwrap_landmarks(self, pipe: FacePipe, unwrap_size: int) -> Optional[torch.Tensor]:
        model = ensure_face_model(pipe, unwrap_size, "unwrap_models")
        if not model:
            return None
        return torch.tensor(model["landmarks"], dtype=torch.float32)

