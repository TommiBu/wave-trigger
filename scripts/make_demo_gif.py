#!/usr/bin/env python3
"""Render docs/demo.gif from synthetic poses. No camera or YOLO weights needed.

Replace the result with a real webcam recording when you have one:

    python examples/overlay_demo.py        # record your screen, convert to gif
"""

from __future__ import annotations

import argparse

import cv2
import numpy as np
from PIL import Image

from wave_trigger import Config, WaveTrigger
from wave_trigger.overlay import draw_panel, draw_pose, flash_event
from wave_trigger.synthetic import idle_sequence, waving_sequence
from wave_trigger.types import Frame, PoseResult

W, H = 800, 460
CENTER = (400.0, 150.0)
SCALE = 95.0
FPS = 20.0


class ListSource:
    def __init__(self, n: int):
        self.n = n
        self.captured_count = n

    def frames(self):
        canvas = np.zeros((H, W, 3), np.uint8)
        for i in range(self.n):
            yield Frame(index=i, t=i / FPS, image=canvas.copy())

    def close(self):
        pass


class ListEstimator:
    def __init__(self, poses):
        self.poses = poses
        self.last_infer_ms = 5.6

    def __call__(self, frame: Frame) -> PoseResult:
        return PoseResult(frame.index, frame.t, self.poses[frame.index])


def build_poses() -> list:
    lead, _ = idle_sequence(n=14, fps=FPS, jitter_px=1.5, center=CENTER, scale=SCALE)
    wave, _ = waving_sequence(n=46, fps=FPS, hz=1.6, amplitude=0.6,
                              center=CENTER, scale=SCALE, noise_px=1.5)
    tail, _ = idle_sequence(n=16, fps=FPS, jitter_px=1.5, center=CENTER, scale=SCALE, seed=3)
    return [*lead, *wave, *tail]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="docs/demo.gif")
    args = p.parse_args()

    poses = build_poses()
    cfg = Config(mirror=False, threshold=0.5, cooldown_s=1.5)
    trigger = WaveTrigger(
        config=cfg, source=ListSource(len(poses)), estimator=ListEstimator(poses)
    )

    frames: list[Image.Image] = []
    last_event_t = -1e9

    for step in trigger.trace():
        img = step.frame.image
        img[:] = (24, 22, 20)
        cv2.putText(img, "wave_trigger - synthetic poses, simulated timings",
                    (16, H - 18), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (110, 110, 110), 1, cv2.LINE_AA)

        if step.pose is not None:
            draw_pose(img, step.pose)
        if step.event is not None:
            last_event_t = step.frame.t

        draw_panel(
            img,
            score=step.score,
            threshold=cfg.threshold,
            fps=FPS,
            infer_ms=step.infer_ms,
            dropped=0,
            detector_debug=getattr(trigger.detector, "debug", None),
        )
        if step.frame.t - last_event_t < 0.9:
            flash_event(img)

        frames.append(Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))

    from pathlib import Path

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        out,
        save_all=True,
        append_images=frames[1:],
        duration=int(1000 / FPS),
        loop=0,
        optimize=True,
    )
    print(f"{out}  {len(frames)} frames  {out.stat().st_size / 1024:.0f} KB  "
          f"{trigger.event_count} events")


if __name__ == "__main__":
    main()
