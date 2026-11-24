from __future__ import annotations

import math

import numpy as np


class EMA:
    def __init__(self, alpha: float = 0.4):
        self.alpha = alpha
        self._state: np.ndarray | None = None

    def __call__(self, x: np.ndarray) -> np.ndarray:
        if self._state is None:
            self._state = x.astype(np.float32).copy()
        else:
            self._state = self.alpha * x + (1.0 - self.alpha) * self._state
        return self._state

    def reset(self) -> None:
        self._state = None


class OneEuro:
    """Adaptive causal filter: strong smoothing when slow, weak when fast."""

    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.05, d_cutoff: float = 1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self._x_prev: np.ndarray | None = None
        self._dx_prev: np.ndarray | None = None
        self._t_prev: float | None = None

    @staticmethod
    def _alpha(cutoff: float, dt: float) -> float:
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, x: np.ndarray, t: float) -> np.ndarray:
        x = x.astype(np.float32)
        if self._t_prev is None:
            self._x_prev, self._dx_prev, self._t_prev = x.copy(), np.zeros_like(x), t
            return x

        dt = max(t - self._t_prev, 1e-3)
        dx = (x - self._x_prev) / dt
        a_d = self._alpha(self.d_cutoff, dt)
        dx_hat = a_d * dx + (1 - a_d) * self._dx_prev

        cutoff = self.min_cutoff + self.beta * np.abs(dx_hat)
        tau = 1.0 / (2.0 * np.pi * cutoff)
        a = 1.0 / (1.0 + tau / dt)
        x_hat = a * x + (1 - a) * self._x_prev

        self._x_prev, self._dx_prev, self._t_prev = x_hat, dx_hat, t
        return x_hat

    def reset(self) -> None:
        self._x_prev = self._dx_prev = self._t_prev = None


def smooth_offline(seq: np.ndarray, window: int = 5) -> np.ndarray:
    """Zero-phase centred moving average. Non-causal: offline use only."""
    if window <= 1:
        return seq
    pad = window // 2
    padded = np.pad(seq, ((pad, pad), (0, 0)), mode="edge")
    kernel = np.ones(window, dtype=np.float32) / window
    out = np.empty_like(seq, dtype=np.float32)
    for d in range(seq.shape[1]):
        out[:, d] = np.convolve(padded[:, d], kernel, mode="valid")
    return out


def build_smoother(kind: str, alpha: float = 0.4):
    if kind == "ema":
        return EMA(alpha)
    if kind == "oneeuro":
        return OneEuro()
    if kind in ("none", "", None):
        return None
    raise ValueError(f"unknown smoother: {kind!r}")
