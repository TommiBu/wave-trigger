from __future__ import annotations

import time
from collections import deque
from pathlib import Path
from typing import Callable, Iterator

import numpy as np

from . import features as F
from .buffer import RingBuffer
from .config import Config
from .detectors import GestureDetector, build_detector
from .smoothing import OneEuro, build_smoother
from .types import Frame, Pose, TraceStep, WaveEvent

Action = Callable[[WaveEvent], None]


class WaveTrigger:
    """Detects waving and runs an action.

    Callback style::

        WaveTrigger(on_wave=lambda e: print("hello world")).run()

    Generator style, where you keep the loop::

        for event in WaveTrigger().events():
            print("hello world")

    Args:
        on_wave: callable, or list of callables, invoked with a WaveEvent.
        source: a Source, a video path, or None for the default webcam.
        config: Config overriding model, thresholds, window length.
        detector: a GestureDetector overriding the one named in config.
        estimator: a pose estimator overriding the default YOLO one.
    """

    def __init__(
        self,
        on_wave: Action | list[Action] | None = None,
        *,
        source=None,
        config: Config | None = None,
        detector: GestureDetector | None = None,
        estimator=None,
    ):
        self.config = config or Config()
        self.actions: list[Action] = []
        if on_wave is not None:
            for a in (on_wave if isinstance(on_wave, (list, tuple)) else [on_wave]):
                self.actions.append(a)

        self._source_arg = source
        self._source = source if not isinstance(source, (str, Path, type(None))) else None
        self._estimator = estimator
        self._detector = detector
        self._buffer = RingBuffer(seconds=self.config.window_seconds)
        self._smoother = build_smoother(self.config.smoother, self.config.ema_alpha)
        self._last_fire = -1e9
        self._processed = 0
        self._fps_hist: deque[float] = deque(maxlen=30)
        self.event_count = 0

    def add_action(self, action: Action) -> "WaveTrigger":
        self.actions.append(action)
        return self

    @property
    def detector(self) -> GestureDetector:
        if self._detector is None:
            self._detector = build_detector(self.config)
        return self._detector

    @property
    def source(self):
        if self._source is None:
            from .sources import VideoFileSource, WebcamSource

            arg = self._source_arg
            if arg is None:
                self._source = WebcamSource(
                    self.config.camera_index, self.config.cam_width, self.config.cam_height
                )
            else:
                self._source = VideoFileSource(arg)
        return self._source

    @property
    def estimator(self):
        if self._estimator is None:
            from .pose import PoseEstimator, resolve_device

            self.config.device = resolve_device(self.config.device)
            self._estimator = PoseEstimator(
                self.config.model,
                device=self.config.device,
                imgsz=self.config.imgsz,
                conf=self.config.conf,
            )
        return self._estimator

    @property
    def fps(self) -> float:
        return float(np.mean(self._fps_hist)) if self._fps_hist else 0.0

    @property
    def dropped_frames(self) -> int:
        captured = getattr(self.source, "captured_count", self._processed)
        return max(captured - self._processed, 0)

    def trace(self) -> Iterator[TraceStep]:
        """Yield every processed frame with its pose, score and any event."""
        cfg = self.config
        t_prev = time.monotonic()

        for frame in self.source.frames():
            if cfg.mirror:
                import cv2

                frame = Frame(frame.index, frame.t, cv2.flip(frame.image, 1))

            result = self.estimator(frame)
            score = 0.0
            pose_out: Pose | None = None

            if result.has_pose:
                pose = result.pose
                if isinstance(self._smoother, OneEuro):
                    xy = self._smoother(pose.xy, frame.t)
                elif self._smoother is not None:
                    xy = self._smoother(pose.xy)
                else:
                    xy = pose.xy
                pose_out = Pose(xy=xy, conf=pose.conf, box=pose.box)
                self._buffer.push(frame.t, F.normalize_pose(pose_out))

                if len(self._buffer) > 3:
                    win = self._buffer.as_array()
                    if win is not None:
                        score = float(self.detector(win[0], win[1]))
            else:
                self._buffer.push(frame.t, None)

            event = self._maybe_fire(score, frame)

            self._processed += 1
            now = time.monotonic()
            self._fps_hist.append(1.0 / max(now - t_prev, 1e-6))
            t_prev = now

            yield TraceStep(
                frame=frame,
                pose=pose_out,
                score=score,
                event=event,
                infer_ms=self.estimator.last_infer_ms,
            )

    def events(self) -> Iterator[WaveEvent]:
        """Yield only the detected gestures."""
        for step in self.trace():
            if step.event is not None:
                yield step.event

    def run(self) -> None:
        """Consume the stream and invoke the registered actions."""
        try:
            for _ in self.events():
                pass
        finally:
            self.close()

    def _maybe_fire(self, score: float, frame: Frame) -> WaveEvent | None:
        cfg = self.config
        if score < cfg.threshold or (frame.t - self._last_fire) < cfg.cooldown_s:
            return None
        self._last_fire = frame.t
        self.event_count += 1
        event = WaveEvent(name="wave", t=frame.t, score=score, frame_index=frame.index)
        for action in self.actions:
            try:
                action(event)
            except Exception as exc:
                print(f"[wave_trigger] action {action!r} failed: {exc}")
        return event

    def reset(self) -> None:
        self._buffer.clear()
        self.detector.reset()
        if self._smoother is not None:
            self._smoother.reset()
        self._last_fire = -1e9

    def close(self) -> None:
        if self._source is not None:
            self._source.close()
            self._source = None

    def __enter__(self) -> "WaveTrigger":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
