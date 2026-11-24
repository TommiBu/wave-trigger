#!/usr/bin/env python3
"""Record labelled clips for training. Vary distance, side, speed and lighting."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2


def fourcc(code: str = "mp4v") -> int:
    for fn in (getattr(cv2.VideoWriter, "fourcc", None), getattr(cv2, "VideoWriter_fourcc", None)):
        if fn is not None:
            return int(fn(*code))
    raise RuntimeError("no fourcc helper in this OpenCV build")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--label", required=True, help="wave | other")
    p.add_argument("--n", type=int, default=20)
    p.add_argument("--seconds", type=float, default=3.0)
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--out", default="data/raw")
    p.add_argument("--fps", type=float, default=30.0)
    args = p.parse_args()

    out_dir = Path(args.out) / args.label
    out_dir.mkdir(parents=True, exist_ok=True)
    existing = len(list(out_dir.glob("clip_*.mp4")))

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit("cannot open camera")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    codec = fourcc("mp4v")

    print(f"recording {args.n} clips of {args.seconds}s into {out_dir}")
    print("SPACE starts a clip, q quits. Change position between clips.")

    made = 0
    try:
        while made < args.n:
            while True:
                ok, img = cap.read()
                if not ok:
                    continue
                img = cv2.flip(img, 1)
                cv2.putText(img, f"[{args.label}] clip {made+1}/{args.n} - SPACE",
                            (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (80, 220, 120), 2)
                cv2.imshow("record", img)
                k = cv2.waitKey(1) & 0xFF
                if k == ord(" "):
                    break
                if k == ord("q"):
                    return

            t0 = time.monotonic()
            while time.monotonic() - t0 < 1.5:
                ok, img = cap.read()
                if not ok:
                    continue
                img = cv2.flip(img, 1)
                left = 1.5 - (time.monotonic() - t0)
                cv2.putText(img, f"{left:.1f}", (img.shape[1] // 2 - 40, img.shape[0] // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 3.0, (60, 220, 235), 5)
                cv2.imshow("record", img)
                cv2.waitKey(1)

            path = out_dir / f"clip_{existing + made:03d}.mp4"
            writer = cv2.VideoWriter(str(path), codec, args.fps, (w, h))
            t0 = time.monotonic()
            while time.monotonic() - t0 < args.seconds:
                ok, img = cap.read()
                if not ok:
                    continue
                img = cv2.flip(img, 1)
                writer.write(img)
                shown = img.copy()
                cv2.circle(shown, (40, 40), 14, (60, 60, 235), -1)
                cv2.imshow("record", shown)
                cv2.waitKey(1)
            writer.release()
            made += 1
            print(f"  saved {path}")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print(f"done, {made} clips in {out_dir}")


if __name__ == "__main__":
    main()
