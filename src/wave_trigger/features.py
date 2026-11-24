from __future__ import annotations

import numpy as np

from .types import KP, Pose

CONF_THRESHOLD = 0.3
MIN_SCALE_PX = 1e-3

_USED_KEYPOINTS: tuple[KP, ...] = (
    KP.NOSE,
    KP.L_SHOULDER, KP.R_SHOULDER,
    KP.L_ELBOW, KP.R_ELBOW,
    KP.L_WRIST, KP.R_WRIST,
    KP.L_HIP, KP.R_HIP,
)

_SHORT = {
    KP.NOSE: "nose",
    KP.L_SHOULDER: "l_shoulder", KP.R_SHOULDER: "r_shoulder",
    KP.L_ELBOW: "l_elbow", KP.R_ELBOW: "r_elbow",
    KP.L_WRIST: "l_wrist", KP.R_WRIST: "r_wrist",
    KP.L_HIP: "l_hip", KP.R_HIP: "r_hip",
}

FEATURE_NAMES: list[str] = [f"{_SHORT[k]}_{ax}" for k in _USED_KEYPOINTS for ax in ("x", "y")]
FEATURE_DIM: int = len(FEATURE_NAMES)


def feature_index(name: str) -> int:
    try:
        return FEATURE_NAMES.index(name)
    except ValueError as exc:
        raise KeyError(f"no such feature: {name!r}; have {FEATURE_NAMES}") from exc


def pose_scale(xy: np.ndarray, conf: np.ndarray) -> float:
    shoulder_width = float(np.linalg.norm(xy[KP.L_SHOULDER] - xy[KP.R_SHOULDER]))
    torso = 0.0
    if conf[KP.L_HIP] >= CONF_THRESHOLD and conf[KP.R_HIP] >= CONF_THRESHOLD:
        shoulder_mid = (xy[KP.L_SHOULDER] + xy[KP.R_SHOULDER]) / 2.0
        hip_mid = (xy[KP.L_HIP] + xy[KP.R_HIP]) / 2.0
        torso = float(np.linalg.norm(hip_mid - shoulder_mid))
    return max(shoulder_width, torso)


def normalize_pose(pose: Pose) -> np.ndarray | None:
    """Pixel keypoints to a translation- and scale-invariant vector, or None."""
    xy = pose.xy.astype(np.float32)
    conf = pose.conf.astype(np.float32)

    if conf[KP.L_SHOULDER] < CONF_THRESHOLD or conf[KP.R_SHOULDER] < CONF_THRESHOLD:
        return None

    origin = (xy[KP.L_SHOULDER] + xy[KP.R_SHOULDER]) / 2.0
    scale = pose_scale(xy, conf)
    if scale < MIN_SCALE_PX:
        return None

    normalized = (xy - origin) / scale
    idx = [int(k) for k in _USED_KEYPOINTS]
    selected = normalized[idx]
    trusted = conf[idx] >= CONF_THRESHOLD
    selected = np.where(trusted[:, None], selected, 0.0)
    out = selected.reshape(-1).astype(np.float32)

    if not np.all(np.isfinite(out)):
        return None
    return out


def window_features(window: np.ndarray, dt: np.ndarray) -> np.ndarray:
    """Append per-feature velocity in units per second. (T, D) -> (T, 2D)."""
    if len(window) < 2:
        return np.concatenate([window, np.zeros_like(window)], axis=1)
    safe_dt = np.maximum(dt, 1e-3).astype(np.float32)[:, None]
    velocity = np.zeros_like(window)
    velocity[1:] = (window[1:] - window[:-1]) / safe_dt[1:]
    velocity[0] = velocity[1]
    return np.concatenate([window, velocity], axis=1).astype(np.float32)


def prepare_window(
    window: np.ndarray, times: np.ndarray, n_steps: int, use_velocity: bool = False
) -> np.ndarray:
    """The single input contract shared by training and inference."""
    from .buffer import resample_window

    if use_velocity:
        first_dt = (times[1] - times[0]) if len(times) > 1 else 0.033
        dt = np.diff(times, prepend=times[0] - first_dt)
        window = window_features(window, dt.astype(np.float32))
    return resample_window(window, times, n_steps)
