#!/usr/bin/env python3
"""Run pose estimation over recorded clips and cache keypoints as .npz."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from wave_trigger.extract import extract_video
from wave_trigger.pose import PoseEstimator, resolve_device


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="inp", default="data/raw")
    p.add_argument("--out", default="data/npz")
    p.add_argument("--model", default="yolo11n-pose.pt")
    p.add_argument("--device", default="auto")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    estimator = PoseEstimator(args.model, device=resolve_device(args.device), imgsz=args.imgsz)
    in_root, out_root = Path(args.inp), Path(args.out)
    videos = sorted(in_root.rglob("*.mp4"))
    if not videos:
        raise SystemExit(f"no videos in {in_root}")

    print(f"{len(videos)} videos -> {out_root}")
    for i, v in enumerate(videos, 1):
        rel = v.relative_to(in_root).with_suffix(".npz")
        dst = out_root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists() and not args.force:
            print(f"  [{i}/{len(videos)}] skip {rel}")
            continue
        data = extract_video(v, estimator)
        np.savez_compressed(dst, label=rel.parts[0], source=str(v), **data)
        print(f"  [{i}/{len(videos)}] {rel}  T={len(data['t'])}  "
              f"pose in {data['valid'].mean():.0%} of frames")


if __name__ == "__main__":
    main()
