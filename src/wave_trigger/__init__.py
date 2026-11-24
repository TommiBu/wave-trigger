"""Wave gesture detection on top of YOLO pose estimation."""

from __future__ import annotations

from .config import Config
from .detectors import GestureDetector, RuleBasedWave, TorchWave, build_detector
from .trigger import WaveTrigger
from .types import Frame, Pose, TraceStep, WaveEvent

__version__ = "0.1.0"

__all__ = [
    "WaveTrigger",
    "WaveEvent",
    "TraceStep",
    "Config",
    "GestureDetector",
    "RuleBasedWave",
    "TorchWave",
    "build_detector",
    "Frame",
    "Pose",
    "__version__",
]
