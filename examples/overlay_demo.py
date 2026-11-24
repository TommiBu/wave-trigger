"""Debug overlay: skeleton, score bar, live feature values. Keys: q f r"""

from __future__ import annotations

import argparse

import cv2

from wave_trigger import Config, WaveTrigger
from wave_trigger import features as F
from wave_trigger.overlay import draw_panel, draw_pose, flash_event


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--detector", default="rule", choices=["rule", "torch"])
    p.add_argument("--weights", default="runs/wave.pt")
    p.add_argument("--model", default="yolo11n-pose.pt")
    p.add_argument("--device", default="auto")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--video", default=None)
    args = p.parse_args()

    cfg = Config(
        detector=args.detector,
        weights=args.weights,
        model=args.model,
        device=args.device,
        imgsz=args.imgsz,
        threshold=args.threshold,
    )
    trigger = WaveTrigger(on_wave=lambda e: print("hello world"), config=cfg, source=args.video)

    show_features = False
    last_event_t = -1e9

    try:
        for step in trigger.trace():
            img = step.frame.image.copy()
            if step.pose is not None:
                draw_pose(img, step.pose)

            feat_view = None
            if show_features and step.pose is not None:
                f = F.normalize_pose(step.pose)
                if f is not None:
                    feat_view = dict(zip(F.FEATURE_NAMES, f.tolist()))

            if step.event is not None:
                last_event_t = step.frame.t

            draw_panel(
                img,
                score=step.score,
                threshold=cfg.threshold,
                fps=trigger.fps,
                infer_ms=step.infer_ms,
                dropped=trigger.dropped_frames,
                detector_debug=getattr(trigger.detector, "debug", None),
                features=feat_view,
                cooldown_left=0.0,
            )
            if step.frame.t - last_event_t < 0.6:
                flash_event(img)

            cv2.imshow("wave_trigger", img)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("f"):
                show_features = not show_features
            if key == ord("r"):
                trigger.reset()
    finally:
        trigger.close()
        cv2.destroyAllWindows()
        print(f"{trigger.event_count} events, {trigger.dropped_frames} frames dropped")


if __name__ == "__main__":
    main()
