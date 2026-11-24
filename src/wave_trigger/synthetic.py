from __future__ import annotations

import numpy as np

from .types import KP, N_KEYPOINTS, Pose

_CANON = {
    KP.NOSE: (0.00, -0.55),
    KP.L_EYE: (-0.08, -0.60),
    KP.R_EYE: (0.08, -0.60),
    KP.L_EAR: (-0.16, -0.55),
    KP.R_EAR: (0.16, -0.55),
    KP.L_SHOULDER: (-0.50, 0.00),
    KP.R_SHOULDER: (0.50, 0.00),
    KP.L_ELBOW: (-0.62, 0.55),
    KP.R_ELBOW: (0.62, 0.55),
    KP.L_WRIST: (-0.68, 1.05),
    KP.R_WRIST: (0.68, 1.05),
    KP.L_HIP: (-0.32, 1.15),
    KP.R_HIP: (0.32, 1.15),
    KP.L_KNEE: (-0.34, 1.95),
    KP.R_KNEE: (0.34, 1.95),
    KP.L_ANKLE: (-0.35, 2.70),
    KP.R_ANKLE: (0.35, 2.70),
}


def make_pose(
    center: tuple[float, float] = (640.0, 360.0),
    scale: float = 180.0,
    right_wrist: tuple[float, float] | None = None,
    conf: float = 0.9,
) -> Pose:
    """Synthetic pose in pixels. `right_wrist` is in canonical units."""
    xy = np.zeros((N_KEYPOINTS, 2), dtype=np.float32)
    canon = dict(_CANON)
    if right_wrist is not None:
        canon[KP.R_WRIST] = right_wrist
        sx, sy = canon[KP.R_SHOULDER]
        wx, wy = right_wrist
        canon[KP.R_ELBOW] = ((sx + wx) / 2 + 0.15, (sy + wy) / 2 + 0.1)

    for k, (x, y) in canon.items():
        xy[int(k)] = (center[0] + x * scale, center[1] + y * scale)

    return Pose(
        xy=xy,
        conf=np.full(N_KEYPOINTS, conf, dtype=np.float32),
        box=np.array([xy[:, 0].min(), xy[:, 1].min(), xy[:, 0].max(), xy[:, 1].max()],
                     dtype=np.float32),
    )


def _noisy(poses: list[Pose], sigma_px: float, rng) -> list[Pose]:
    if sigma_px <= 0:
        return poses
    return [
        Pose(xy=p.xy + rng.normal(0, sigma_px, p.xy.shape).astype(np.float32),
             conf=p.conf, box=p.box)
        for p in poses
    ]


def waving_sequence(n=60, fps=30.0, hz=2.0, amplitude=0.55,
                    center=(640.0, 360.0), scale=180.0, noise_px=0.0, seed=0):
    rng = np.random.default_rng(seed)
    poses, times = [], []
    for i in range(n):
        t = i / fps
        x = 0.75 + amplitude * np.sin(2 * np.pi * hz * t)
        poses.append(make_pose(center, scale, right_wrist=(float(x), -0.80)))
        times.append(t)
    return _noisy(poses, noise_px, rng), np.asarray(times)


def idle_sequence(n=60, fps=30.0, jitter_px=2.0, center=(640.0, 360.0), scale=180.0, seed=0):
    rng = np.random.default_rng(seed)
    poses, times = [], []
    for i in range(n):
        poses.append(make_pose(center, scale))
        times.append(i / fps)
    return _noisy(poses, jitter_px, rng), np.asarray(times)


def scratching_sequence(n=60, fps=30.0, center=(640.0, 360.0), scale=180.0,
                        noise_px=0.0, seed=0):
    """Hard negative: hand up near the head, almost no sideways motion."""
    rng = np.random.default_rng(seed)
    poses, times = [], []
    for i in range(n):
        t = i / fps
        x = 0.30 + 0.04 * np.sin(2 * np.pi * 1.5 * t)
        poses.append(make_pose(center, scale, right_wrist=(float(x), -0.62)))
        times.append(t)
    return _noisy(poses, noise_px, rng), np.asarray(times)


def reaching_sequence(n=60, fps=30.0, center=(640.0, 360.0), scale=180.0,
                      noise_px=0.0, seed=0):
    """Hard negative: one large up-and-down motion instead of oscillation."""
    rng = np.random.default_rng(seed)
    poses, times = [], []
    for i in range(n):
        t = i / fps
        phase = np.sin(np.pi * i / max(n - 1, 1))
        y = 1.05 - 1.9 * phase
        x = 0.68 + 0.25 * phase
        poses.append(make_pose(center, scale, right_wrist=(float(x), float(y))))
        times.append(t)
    return _noisy(poses, noise_px, rng), np.asarray(times)


def synthetic_clips(n_per_class: int = 40, seed: int = 0):
    """Labelled synthetic clips for `scripts/train.py --synthetic`."""
    rng = np.random.default_rng(seed)
    clips = []

    def rand_view():
        return (
            (float(rng.uniform(300, 1000)), float(rng.uniform(200, 520))),
            float(rng.uniform(100, 300)),
        )

    for i in range(n_per_class):
        center, scale = rand_view()
        poses, times = waving_sequence(
            n=int(rng.integers(50, 80)),
            hz=float(rng.uniform(1.2, 3.0)),
            amplitude=float(rng.uniform(0.35, 0.7)),
            center=center, scale=scale,
            noise_px=float(rng.uniform(0.5, 3.0)), seed=int(rng.integers(1e6)),
        )
        clips.append((poses, times, "wave", f"syn_wave_{i:03d}"))

    makers = [idle_sequence, scratching_sequence, reaching_sequence]
    for i in range(n_per_class):
        center, scale = rand_view()
        maker = makers[i % len(makers)]
        kw = dict(n=int(rng.integers(50, 80)), center=center, scale=scale,
                  seed=int(rng.integers(1e6)))
        if maker is idle_sequence:
            kw["jitter_px"] = float(rng.uniform(1.0, 4.0))
        else:
            kw["noise_px"] = float(rng.uniform(0.5, 3.0))
        poses, times = maker(**kw)
        clips.append((poses, times, "other", f"syn_other_{i:03d}"))

    return clips
