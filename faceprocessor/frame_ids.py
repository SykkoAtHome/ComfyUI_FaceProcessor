"""Helpers for resolving frame_id ordering within FacePipe metadata."""
from __future__ import annotations

from typing import List, Optional, Sequence

from .pipe import FacePipe


def resolve_frame_ids(
    pipe: FacePipe, count: int, fallback: Optional[Sequence[str]] = None
) -> List[str]:
    """Return the canonical ``frame_id`` ordering for the provided ``pipe``."""

    frame_order = pipe.meta.get("frame_order")
    if isinstance(frame_order, Sequence):
        if len(frame_order) < count:
            raise ValueError(
                "FacePipe meta frame_order does not contain enough entries for the current batch",
            )
        return [str(frame_order[idx]) for idx in range(count)]

    if fallback is not None:
        if len(fallback) != count:
            raise ValueError("Fallback frame_id list length must match the requested count")
        return list(fallback)

    raise ValueError(
        "FacePipe meta is missing frame_order information; run FaceFitAndRestore in Fit mode first",
    )
