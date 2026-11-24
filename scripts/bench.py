#!/usr/bin/env python3
"""Measure pose inference latency. Run this before anything else."""

from __future__ import annotations

import argparse
import time

import numpy as np


def bench_model(model_name: str, device: str, imgsz: int, n: int = 30) -> dict:
    from ultralytics import YOLO

    model = YOLO(model_name)
    img = (np.random.rand(720, 1280, 3) * 255).astype(np.uint8)

    for _ in range(5):
        model.predict(img, imgsz=imgsz, device=device, verbose=False)

    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        model.predict(img, imgsz=imgsz, device=device, verbose=False)
        ts.append((time.perf_counter() - t0) * 1000)
    a = np.asarray(ts)
    return {
        "model": model_name,
        "device": device,
        "imgsz": imgsz,
        "median_ms": float(np.median(a)),
        "p95_ms": float(np.percentile(a, 95)),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--models", default="yolo11n-pose.pt")
    p.add_argument("--devices", default="")
    p.add_argument("--imgsz", default="640")
    p.add_argument("--n", type=int, default=30)
    args = p.parse_args()

    import torch

    if args.devices:
        devices = args.devices.split(",")
    else:
        devices = ["cpu"]
        if torch.backends.mps.is_available():
            devices.append("mps")
        if torch.cuda.is_available():
            devices.append("cuda")

    rows = []
    for m in args.models.split(","):
        for d in devices:
            for s in (int(x) for x in str(args.imgsz).split(",")):
                try:
                    rows.append(bench_model(m.strip(), d, s, args.n))
                except Exception as exc:
                    print(f"  {m} / {d} / {s}: failed ({exc})")

    print(f"\n{'model':<22}{'device':<8}{'imgsz':<8}{'median ms':>10}{'p95 ms':>9}{'fps':>8}")
    print("-" * 65)
    for r in sorted(rows, key=lambda r: r["median_ms"]):
        print(f"{r['model']:<22}{r['device']:<8}{r['imgsz']:<8}"
              f"{r['median_ms']:>10.1f}{r['p95_ms']:>9.1f}{1000 / r['median_ms']:>8.1f}")


if __name__ == "__main__":
    main()
