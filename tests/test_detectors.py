from __future__ import annotations

import numpy as np
import pytest

from wave_trigger import features as F
from wave_trigger.detectors import RuleBasedWave
from wave_trigger.synthetic import (
    idle_sequence,
    reaching_sequence,
    scratching_sequence,
    waving_sequence,
)


def _score(gen, **kw) -> float:
    poses, times = gen(**kw)
    window = np.stack([F.normalize_pose(p) for p in poses]).astype(np.float32)
    det = RuleBasedWave()
    det.reset()
    return float(det(window, times))


def test_wave_fires():
    assert _score(waving_sequence, n=60, hz=2.0, amplitude=0.55) >= 0.5


def test_idle_does_not_fire():
    assert _score(idle_sequence, n=60, jitter_px=2.0) < 0.5


def test_scratching_does_not_fire():
    assert _score(scratching_sequence, n=60) < 0.5


def test_reaching_does_not_fire():
    assert _score(reaching_sequence, n=60) < 0.5


def test_faster_wave_scores_at_least_as_high():
    assert _score(waving_sequence, n=60, hz=2.5) >= _score(waving_sequence, n=60, hz=0.4)


@pytest.mark.parametrize("center,scale", [((250.0, 180.0), 110.0), ((980.0, 520.0), 300.0)])
def test_independent_of_position_and_distance(center, scale):
    assert _score(waving_sequence, n=60, hz=2.0, center=center, scale=scale) >= 0.5


def test_short_window_returns_zero():
    det = RuleBasedWave()
    assert det(np.zeros((2, F.FEATURE_DIM), np.float32), np.array([0.0, 0.1])) == 0.0
