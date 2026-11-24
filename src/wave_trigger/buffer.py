from __future__ import annotations

from collections import deque

import numpy as np


class RingBuffer:
    """Keeps the last `seconds` of feature vectors. None marks a missing pose."""

    def __init__(self, seconds: float = 2.0, max_len: int = 300):
        self.seconds = seconds
        self._t: deque[float] = deque(maxlen=max_len)
        self._x: deque[np.ndarray | None] = deque(maxlen=max_len)

    def push(self, t: float, x: np.ndarray | None) -> None:
        self._t.append(t)
        self._x.append(x)
        while self._t and t - self._t[0] > self.seconds:
            self._t.popleft()
            self._x.popleft()

    def __len__(self) -> int:
        return len(self._t)

    @property
    def span(self) -> float:
        return (self._t[-1] - self._t[0]) if len(self._t) > 1 else 0.0

    @property
    def gap_ratio(self) -> float:
        if not self._x:
            return 1.0
        return sum(1 for x in self._x if x is None) / len(self._x)

    def as_array(self, fill_missing: bool = True) -> tuple[np.ndarray, np.ndarray] | None:
        if not self._x:
            return None
        valid = [x for x in self._x if x is not None]
        if not valid:
            return None
        if not fill_missing and len(valid) != len(self._x):
            return None
        dim = valid[0].shape[0]

        rows: list[np.ndarray] = []
        last: np.ndarray | None = None
        for x in self._x:
            if x is not None:
                last = x
            rows.append(last if last is not None else np.zeros(dim, dtype=np.float32))
        return np.stack(rows).astype(np.float32), np.asarray(self._t, dtype=np.float64)

    def dt(self) -> np.ndarray:
        t = np.asarray(self._t, dtype=np.float64)
        if len(t) < 2:
            return np.zeros(len(t), dtype=np.float32)
        d = np.diff(t)
        return np.concatenate([[np.median(d)], d]).astype(np.float32)

    def clear(self) -> None:
        self._t.clear()
        self._x.clear()


def resample_window(window: np.ndarray, times: np.ndarray, n_out: int) -> np.ndarray:
    """Linearly resample a window to a fixed number of steps in time."""
    if len(window) == 0:
        raise ValueError("empty window")
    if len(window) == 1:
        return np.repeat(window, n_out, axis=0).astype(np.float32)
    t_norm = (times - times[0]) / max(times[-1] - times[0], 1e-6)
    grid = np.linspace(0.0, 1.0, n_out)
    out = np.empty((n_out, window.shape[1]), dtype=np.float32)
    for d in range(window.shape[1]):
        out[:, d] = np.interp(grid, t_norm, window[:, d])
    return out
