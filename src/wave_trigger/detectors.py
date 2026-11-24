from __future__ import annotations

from typing import Protocol

import numpy as np

from . import features as F


class GestureDetector(Protocol):
    """Scores a time window of normalized features in [0, 1]. Stateless in time."""

    name: str

    def __call__(self, window: np.ndarray, times: np.ndarray) -> float: ...

    def reset(self) -> None: ...


def _moving_average(x: np.ndarray, k: int) -> np.ndarray:
    if k <= 1 or len(x) < k:
        return x
    kernel = np.ones(k, dtype=np.float32) / k
    pad = k // 2
    return np.convolve(np.pad(x, (pad, pad), mode="edge"), kernel, mode="valid")[: len(x)]


def _count_reversals(x: np.ndarray, min_amplitude: float) -> tuple[int, float]:
    """Direction changes with hysteresis, so detector noise is not counted."""
    if len(x) < 3:
        return 0, 0.0

    reversals = 0
    anchor = float(x[0])
    direction = 0
    max_swing = 0.0

    for value in x[1:]:
        delta = float(value) - anchor
        if abs(delta) < min_amplitude:
            continue
        step_direction = 1 if delta > 0 else -1
        if direction != 0 and step_direction != direction:
            reversals += 1
        direction = step_direction
        max_swing = max(max_swing, abs(delta))
        anchor = float(value)

    return reversals, max_swing


class RuleBasedWave:
    """Hand above the shoulder and wrist oscillating sideways. No training, no weights."""

    name = "rule"

    def __init__(
        self,
        min_reversals: int = 3,
        min_amplitude: float = 0.15,
        smooth_window: int = 3,
    ):
        self.min_reversals = max(1, min_reversals)
        self.min_amplitude = min_amplitude
        self.smooth_window = smooth_window
        self.debug: dict[str, float] = {}
        self._idx_cache: dict[str, int] | None = None

    def reset(self) -> None:
        self.debug = {}

    def _indices(self) -> dict[str, int]:
        if self._idx_cache is None:
            self._idx_cache = {
                "lw_x": F.feature_index("l_wrist_x"),
                "lw_y": F.feature_index("l_wrist_y"),
                "rw_x": F.feature_index("r_wrist_x"),
                "rw_y": F.feature_index("r_wrist_y"),
                "ls_y": F.feature_index("l_shoulder_y"),
                "rs_y": F.feature_index("r_shoulder_y"),
            }
        return self._idx_cache

    def _hand_score(self, window: np.ndarray, x_i: int, y_i: int, sh_i: int):
        above = float(np.mean(window[:, y_i] < window[:, sh_i]))
        x = _moving_average(window[:, x_i].astype(np.float32), self.smooth_window)
        reversals, swing = _count_reversals(x, self.min_amplitude)
        score = above * min(1.0, reversals / self.min_reversals)
        return score, {"above": above, "reversals": float(reversals), "swing": swing}

    def __call__(self, window: np.ndarray, times: np.ndarray) -> float:
        if window.ndim != 2 or len(window) < 3:
            return 0.0

        idx = self._indices()
        left, dbg_l = self._hand_score(window, idx["lw_x"], idx["lw_y"], idx["ls_y"])
        right, dbg_r = self._hand_score(window, idx["rw_x"], idx["rw_y"], idx["rs_y"])

        best, dbg = (right, dbg_r) if right >= left else (left, dbg_l)
        self.debug = {**dbg, "hand": 1.0 if right >= left else 0.0, "score": best}
        return float(np.clip(best, 0.0, 1.0))


class TorchWave:
    """Trained window classifier loaded from a checkpoint produced by scripts/train.py."""

    name = "torch"

    def __init__(self, weights: str, device: str = "cpu", strict_contract: bool = True):
        import torch

        from .models import build_model

        self.device = device
        ckpt = torch.load(weights, map_location=device, weights_only=False)

        self.n_steps = int(ckpt["n_steps"])
        self.use_velocity = bool(ckpt.get("use_velocity", False))
        self.arch = ckpt.get("arch", "cnn")
        trained_names: list[str] = list(ckpt.get("feature_names", []))

        if strict_contract and trained_names and trained_names != F.FEATURE_NAMES:
            raise RuntimeError(
                "input contract mismatch: the checkpoint was trained on different features\n"
                f"  trained on: {trained_names}\n"
                f"  current:    {F.FEATURE_NAMES}"
            )

        n_features = len(trained_names or F.FEATURE_NAMES)
        if self.use_velocity:
            n_features *= 2

        self.model = build_model(self.arch, self.n_steps, n_features)
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.to(device)
        self.model.eval()

        self.debug: dict[str, float] = {}

    def __call__(self, window: np.ndarray, times: np.ndarray) -> float:
        import torch

        if window.ndim != 2 or len(window) < 3:
            return 0.0

        x = F.prepare_window(window, times, self.n_steps, self.use_velocity)
        tensor = torch.from_numpy(x).unsqueeze(0).to(self.device)

        with torch.inference_mode():
            prob = torch.sigmoid(self.model(tensor)).item()

        self.debug = {"prob": prob}
        return float(prob)

    def reset(self) -> None:
        self.debug = {}


def build_detector(cfg) -> GestureDetector:
    if cfg.detector == "rule":
        return RuleBasedWave(
            min_reversals=cfg.min_reversals,
            min_amplitude=cfg.min_amplitude,
        )
    if cfg.detector == "torch":
        return TorchWave(weights=cfg.weights, device=cfg.device)
    raise ValueError(f"unknown detector: {cfg.detector!r}")
