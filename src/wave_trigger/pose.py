from __future__ import annotations

import time

import numpy as np

from .types import Frame, Pose, PoseResult


class PoseEstimator:
    """Wraps a YOLO pose model and returns the largest person's pose."""

    def __init__(
        self,
        model: str = "yolo11n-pose.pt",
        device: str = "cpu",
        imgsz: int = 640,
        conf: float = 0.5,
    ):
        from ultralytics import YOLO

        self.model = YOLO(model)
        self.device = device
        self.imgsz = imgsz
        self.conf = conf
        self.last_infer_ms: float = 0.0

    def __call__(self, frame: Frame) -> PoseResult:
        t0 = time.perf_counter()
        results = self.model.predict(
            frame.image,
            imgsz=self.imgsz,
            conf=self.conf,
            device=self.device,
            verbose=False,
        )
        self.last_infer_ms = (time.perf_counter() - t0) * 1000.0

        r = results[0]
        if r.keypoints is None or len(r.boxes) == 0:
            return PoseResult(frame.index, frame.t, None)

        boxes = r.boxes.xyxy.cpu().numpy()
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        i = int(np.argmax(areas))

        xy = r.keypoints.xy[i].cpu().numpy().astype(np.float32)
        kconf = r.keypoints.conf
        conf = (
            kconf[i].cpu().numpy().astype(np.float32)
            if kconf is not None
            else np.ones(len(xy), dtype=np.float32)
        )
        return PoseResult(
            frame.index,
            frame.t,
            Pose(xy=xy, conf=conf, box=boxes[i].astype(np.float32)),
        )


def resolve_device(prefer: str = "auto") -> str:
    """Pick a torch device. Measure before assuming mps beats cpu for small models."""
    if prefer != "auto":
        return prefer
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"
