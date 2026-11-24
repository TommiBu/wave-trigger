#!/usr/bin/env python3
"""Evaluate a detector on recorded clips: recall and false positives per minute."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from wave_trigger import features as F
from wave_trigger.buffer import RingBuffer
from wave_trigger.config import Config
from wave_trigger.detectors import build_detector
from wave_trigger.smoothing import smooth_offline
from wave_trigger.types import Pose


@dataclass
class ClipResult:
    name: str
    label: str
    duration_s: float
    events: list[float]
    first_event_s: float | None


def clip_to_features(npz_path: Path):
    d = np.load(npz_path, allow_pickle=False)
    label = str(d["label"])
    feats, times = [], []
    for i in range(len(d["t"])):
        if not d["valid"][i]:
            continue
        pose = Pose(xy=d["xy"][i], conf=d["conf"][i], box=np.zeros(4, np.float32))
        f = F.normalize_pose(pose)
        if f is not None:
            feats.append(f)
            times.append(float(d["t"][i]))
    if not feats:
        return np.empty((0, 0), np.float32), np.empty(0), label
    return np.stack(feats).astype(np.float32), np.asarray(times), label


def run_detector_on_clip(detector, seq, times, cfg, causal: bool = True) -> list[float]:
    if len(seq) < 5:
        return []
    if not causal:
        seq = smooth_offline(seq, window=5)

    detector.reset()
    buf = RingBuffer(seconds=cfg.window_seconds)
    events: list[float] = []
    last_fire = -1e9
    for i, t in enumerate(times):
        buf.push(float(t), seq[i])
        if len(buf) < 5:
            continue
        win = buf.as_array()
        if win is None:
            continue
        if float(detector(win[0], win[1])) >= cfg.threshold and (t - last_fire) >= cfg.cooldown_s:
            last_fire = float(t)
            events.append(float(t))
    return events


def summarize(results: list[ClipResult]) -> dict:
    waves = [r for r in results if r.label == "wave"]
    others = [r for r in results if r.label != "wave"]
    detected = sum(1 for r in waves if r.events)
    fp_events = sum(len(r.events) for r in others)
    fp_minutes = sum(r.duration_s for r in others) / 60.0
    lats = [r.first_event_s for r in waves if r.first_event_s is not None]
    return {
        "n_wave": len(waves),
        "n_other": len(others),
        "recall": detected / len(waves) if waves else float("nan"),
        "fp_per_min": fp_events / fp_minutes if fp_minutes > 0 else float("nan"),
        "fp_events": fp_events,
        "other_minutes": fp_minutes,
        "median_latency_s": float(np.median(lats)) if lats else float("nan"),
    }


def print_summary(tag: str, s: dict) -> None:
    print(f"\n=== {tag} ===")
    print(f"  clips:        {s['n_wave']} wave / {s['n_other']} other")
    print(f"  recall:       {s['recall']:.1%}")
    print(f"  FP / minute:  {s['fp_per_min']:.2f}  "
          f"({s['fp_events']} in {s['other_minutes']:.1f} min)")
    print(f"  latency:      {s['median_latency_s']:.2f} s median")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/npz")
    p.add_argument("--detector", default="rule", choices=["rule", "torch"])
    p.add_argument("--weights", default="runs/wave.pt")
    p.add_argument("--threshold", type=float, default=None)
    p.add_argument("--sweep", action="store_true")
    p.add_argument("--postprocess", action="store_true")
    args = p.parse_args()

    cfg = Config(detector=args.detector, weights=args.weights)
    if args.threshold is not None:
        cfg.threshold = args.threshold

    paths = sorted(Path(args.data).rglob("*.npz"))
    if not paths:
        raise SystemExit(f"no data in {args.data}")

    clips = []
    for path in paths:
        seq, times, label = clip_to_features(path)
        if len(seq) >= 5:
            clips.append((path.name, label, seq, times))
    print(f"{len(clips)} clips loaded")

    thresholds = np.round(np.arange(0.1, 0.95, 0.1), 2) if args.sweep else [cfg.threshold]
    for th in thresholds:
        cfg.threshold = float(th)
        detector = build_detector(cfg)
        for causal in ([True, False] if args.postprocess else [True]):
            results = []
            for name, label, seq, times in clips:
                ev = run_detector_on_clip(detector, seq, times, cfg, causal=causal)
                results.append(
                    ClipResult(name, label, float(times[-1] - times[0]), ev,
                               (ev[0] - float(times[0])) if ev else None)
                )
            mode = "causal (realtime)" if causal else "non-causal (offline)"
            print_summary(f"threshold {th:.2f} | {mode}", summarize(results))


if __name__ == "__main__":
    main()
