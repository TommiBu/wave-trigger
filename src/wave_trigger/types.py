from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import numpy as np


class KP(IntEnum):
    NOSE = 0
    L_EYE = 1
    R_EYE = 2
    L_EAR = 3
    R_EAR = 4
    L_SHOULDER = 5
    R_SHOULDER = 6
    L_ELBOW = 7
    R_ELBOW = 8
    L_WRIST = 9
    R_WRIST = 10
    L_HIP = 11
    R_HIP = 12
    L_KNEE = 13
    R_KNEE = 14
    L_ANKLE = 15
    R_ANKLE = 16


N_KEYPOINTS = 17

SKELETON: tuple[tuple[int, int], ...] = (
    (KP.L_SHOULDER, KP.R_SHOULDER),
    (KP.L_SHOULDER, KP.L_ELBOW),
    (KP.L_ELBOW, KP.L_WRIST),
    (KP.R_SHOULDER, KP.R_ELBOW),
    (KP.R_ELBOW, KP.R_WRIST),
    (KP.L_SHOULDER, KP.L_HIP),
    (KP.R_SHOULDER, KP.R_HIP),
    (KP.L_HIP, KP.R_HIP),
    (KP.L_HIP, KP.L_KNEE),
    (KP.L_KNEE, KP.L_ANKLE),
    (KP.R_HIP, KP.R_KNEE),
    (KP.R_KNEE, KP.R_ANKLE),
    (KP.NOSE, KP.L_EYE),
    (KP.NOSE, KP.R_EYE),
    (KP.L_EYE, KP.L_EAR),
    (KP.R_EYE, KP.R_EAR),
)


@dataclass(frozen=True)
class Frame:
    index: int
    t: float
    image: np.ndarray


@dataclass(frozen=True)
class Pose:
    xy: np.ndarray
    conf: np.ndarray
    box: np.ndarray


@dataclass(frozen=True)
class PoseResult:
    frame_index: int
    t: float
    pose: Pose | None

    @property
    def has_pose(self) -> bool:
        return self.pose is not None


@dataclass(frozen=True)
class WaveEvent:
    """A detected gesture. Passed to your callback or yielded by `events()`."""

    name: str
    t: float
    score: float
    frame_index: int = -1

    def __str__(self) -> str:
        return f"{self.name}@{self.t:.2f}s score={self.score:.2f}"


@dataclass(frozen=True)
class TraceStep:
    """One processed frame. Yielded by `WaveTrigger.trace()` for custom rendering."""

    frame: Frame
    pose: Pose | None
    score: float
    event: WaveEvent | None
    infer_ms: float
