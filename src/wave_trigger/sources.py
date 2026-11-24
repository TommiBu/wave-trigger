from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Iterator, Protocol

import cv2
import numpy as np

from .types import Frame


class Source(Protocol):
    def frames(self) -> Iterator[Frame]: ...

    def close(self) -> None: ...


class WebcamSource:
    """Webcam reader that keeps only the newest frame, dropping backlog."""

    def __init__(self, camera_index: int = 0, width: int = 1280, height: int = 720):
        self.cap = cv2.VideoCapture(camera_index)
        if not self.cap.isOpened():
            raise RuntimeError(
                f"cannot open camera {camera_index}; on macOS check "
                "System Settings > Privacy & Security > Camera for your terminal or IDE"
            )
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self._lock = threading.Lock()
        self._latest: tuple[int, float, np.ndarray] | None = None
        self._running = True
        self._captured = 0
        self._t0 = time.monotonic()
        self._thread = threading.Thread(target=self._reader, daemon=True)
        self._thread.start()

    def _reader(self) -> None:
        while self._running:
            ok, img = self.cap.read()
            if not ok:
                time.sleep(0.005)
                continue
            self._captured += 1
            with self._lock:
                self._latest = (self._captured, time.monotonic() - self._t0, img)

    def frames(self) -> Iterator[Frame]:
        last_index = -1
        while self._running:
            with self._lock:
                latest = self._latest
            if latest is None or latest[0] == last_index:
                time.sleep(0.002)
                continue
            last_index = latest[0]
            yield Frame(index=latest[0], t=latest[1], image=latest[2])

    @property
    def captured_count(self) -> int:
        return self._captured

    def close(self) -> None:
        self._running = False
        self._thread.join(timeout=1.0)
        self.cap.release()


class VideoFileSource:
    """Video file reader. Processes every frame; time comes from the video."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.cap = cv2.VideoCapture(str(self.path))
        if not self.cap.isOpened():
            raise RuntimeError(f"cannot open video: {self.path}")
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.n_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.captured_count = 0

    def frames(self) -> Iterator[Frame]:
        i = 0
        while True:
            ok, img = self.cap.read()
            if not ok:
                break
            self.captured_count = i + 1
            yield Frame(index=i, t=i / self.fps, image=img)
            i += 1

    def close(self) -> None:
        self.cap.release()
