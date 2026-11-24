from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass
class Config:
    model: str = "yolo11n-pose.pt"
    device: str = "auto"
    imgsz: int = 640
    conf: float = 0.5

    camera_index: int = 0
    cam_width: int = 1280
    cam_height: int = 720
    mirror: bool = True

    window_seconds: float = 2.0
    n_steps: int = 32

    smoother: str = "ema"
    ema_alpha: float = 0.4

    detector: str = "rule"
    threshold: float = 0.5
    cooldown_s: float = 1.5
    min_reversals: int = 3
    min_amplitude: float = 0.15
    weights: str = "runs/wave.pt"

    def to_dict(self) -> dict:
        return asdict(self)
