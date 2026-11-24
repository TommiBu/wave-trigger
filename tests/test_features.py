from __future__ import annotations

import numpy as np
import pytest

from wave_trigger import features as F
from wave_trigger.synthetic import make_pose

TOL = 1e-3


def _normalize(pose):
    out = F.normalize_pose(pose)
    assert out is not None
    return np.asarray(out)


def test_feature_metadata_matches():
    assert F.FEATURE_DIM > 0
    assert len(F.FEATURE_NAMES) == F.FEATURE_DIM


def test_shape_and_dtype():
    out = _normalize(make_pose())
    assert out.shape == (F.FEATURE_DIM,)
    assert out.dtype == np.float32
    assert np.all(np.isfinite(out))


def test_translation_invariance():
    a = _normalize(make_pose(center=(300.0, 200.0)))
    b = _normalize(make_pose(center=(900.0, 500.0)))
    assert np.allclose(a, b, atol=TOL), f"max diff {np.abs(a - b).max():.4f}"


def test_scale_invariance():
    a = _normalize(make_pose(scale=120.0))
    b = _normalize(make_pose(scale=320.0))
    assert np.allclose(a, b, atol=TOL), f"max diff {np.abs(a - b).max():.4f}"


def test_translation_and_scale_invariance():
    a = _normalize(make_pose(center=(200.0, 150.0), scale=100.0))
    b = _normalize(make_pose(center=(1000.0, 600.0), scale=300.0))
    assert np.allclose(a, b, atol=TOL)


def test_gesture_changes_features():
    down = _normalize(make_pose())
    up = _normalize(make_pose(right_wrist=(0.9, -0.8)))
    assert not np.allclose(down, up, atol=1e-2)


def test_degenerate_pose_does_not_crash():
    pose = make_pose()
    degenerate = type(pose)(xy=np.full_like(pose.xy, 500.0), conf=pose.conf, box=pose.box)
    out = F.normalize_pose(degenerate)
    if out is not None:
        assert np.all(np.isfinite(out))


def test_low_confidence_never_yields_nan():
    out = F.normalize_pose(make_pose(conf=0.05))
    if out is not None:
        assert np.all(np.isfinite(np.asarray(out)))


def test_prepare_window_fixed_length():
    poses = [make_pose(right_wrist=(0.5 + 0.1 * i, -0.8)) for i in range(11)]
    feats = np.stack([F.normalize_pose(p) for p in poses]).astype(np.float32)
    times = np.arange(11) / 30.0
    assert F.prepare_window(feats, times, 32).shape == (32, F.FEATURE_DIM)
    assert F.prepare_window(feats, times, 32, use_velocity=True).shape == (32, 2 * F.FEATURE_DIM)
