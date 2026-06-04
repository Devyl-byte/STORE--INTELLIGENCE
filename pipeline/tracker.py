from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Track:
    track_id: str
    first_frame: int
    last_frame: int
    motion_score: float
    first_center: tuple[float, float] | None = None
    last_center: tuple[float, float] | None = None
    hits: int = 1


class CentroidTracker:
    """Tiny tracker used by the OpenCV fallback detector.

    It is intentionally conservative: the detector emits low-confidence track
    candidates rather than pretending to solve full re-identification.
    """

    def __init__(self) -> None:
        self._counter = 0

    def new_track(
        self,
        frame_index: int,
        score: float,
        center: tuple[float, float] | None = None,
    ) -> Track:
        self._counter += 1
        return Track(
            track_id=f"TRK_{self._counter:05d}",
            first_frame=frame_index,
            last_frame=frame_index,
            motion_score=score,
            first_center=center,
            last_center=center,
        )
