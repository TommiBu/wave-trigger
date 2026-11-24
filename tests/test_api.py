from __future__ import annotations

import numpy as np

from wave_trigger import Config, WaveEvent, WaveTrigger
from wave_trigger.synthetic import idle_sequence, waving_sequence
from wave_trigger.types import Frame, PoseResult


class FakeSource:
    """Replays synthetic poses as frames, so tests need no camera."""

    def __init__(self, poses, times):
        self.poses = poses
        self.times = times
        self.captured_count = len(poses)
        self.closed = False

    def frames(self):
        blank = np.zeros((4, 4, 3), np.uint8)
        for i, t in enumerate(self.times):
            yield Frame(index=i, t=float(t), image=blank)

    def close(self):
        self.closed = True


class FakeEstimator:
    """Returns the pre-made pose for each frame instead of running YOLO."""

    def __init__(self, poses):
        self.poses = poses
        self.last_infer_ms = 0.0

    def __call__(self, frame: Frame) -> PoseResult:
        return PoseResult(frame.index, frame.t, self.poses[frame.index])


def _trigger(poses, times, **cfg_kw):
    cfg = Config(mirror=False, **cfg_kw)
    return WaveTrigger(
        config=cfg,
        source=FakeSource(poses, times),
        estimator=FakeEstimator(poses),
    )


def test_generator_yields_events_on_wave():
    poses, times = waving_sequence(n=120, hz=2.0)
    events = list(_trigger(poses, times).events())
    assert events, "waving produced no events"
    assert all(isinstance(e, WaveEvent) for e in events)
    assert all(e.name == "wave" for e in events)


def test_no_events_when_idle():
    poses, times = idle_sequence(n=120, jitter_px=2.0)
    assert list(_trigger(poses, times).events()) == []


def test_callback_is_invoked():
    poses, times = waving_sequence(n=120, hz=2.0)
    seen: list[WaveEvent] = []
    trigger = _trigger(poses, times)
    trigger.add_action(seen.append)
    trigger.run()
    assert seen
    assert len(seen) == trigger.event_count


def test_cooldown_limits_event_rate():
    poses, times = waving_sequence(n=300, hz=2.0)
    fast = list(_trigger(poses, times, cooldown_s=0.5).events())
    slow = list(_trigger(poses, times, cooldown_s=5.0).events())
    assert len(fast) > len(slow)


def test_context_manager_closes_source():
    poses, times = waving_sequence(n=30, hz=2.0)
    source = FakeSource(poses, times)
    with WaveTrigger(
        config=Config(mirror=False), source=source, estimator=FakeEstimator(poses)
    ):
        pass
    assert source.closed


def test_trace_exposes_every_frame():
    poses, times = waving_sequence(n=60, hz=2.0)
    steps = list(_trigger(poses, times).trace())
    assert len(steps) == len(poses)
    assert all(0.0 <= s.score <= 1.0 for s in steps)
