from __future__ import annotations

import cv2
import numpy as np

from .types import SKELETON, Pose

GREEN = (80, 220, 120)
RED = (60, 60, 235)
WHITE = (245, 245, 245)
GREY = (140, 140, 140)
YELLOW = (60, 220, 235)


def _pt(p) -> tuple[int, int]:
    return int(round(float(p[0]))), int(round(float(p[1])))


def draw_pose(img: np.ndarray, pose: Pose, conf_threshold: float = 0.3) -> None:
    for a, b in SKELETON:
        if pose.conf[a] < conf_threshold or pose.conf[b] < conf_threshold:
            continue
        cv2.line(img, _pt(pose.xy[a]), _pt(pose.xy[b]), GREEN, 2, cv2.LINE_AA)

    for i, p in enumerate(pose.xy):
        good = pose.conf[i] >= conf_threshold
        cv2.circle(img, _pt(p), 4, WHITE if good else GREY, -1, cv2.LINE_AA)

    x1, y1, x2, y2 = (int(round(float(v))) for v in pose.box)
    cv2.rectangle(img, (x1, y1), (x2, y2), (70, 70, 70), 1)


def draw_panel(
    img: np.ndarray,
    *,
    score: float,
    threshold: float,
    fps: float,
    infer_ms: float,
    dropped: int,
    detector_debug: dict[str, float] | None = None,
    features: dict[str, float] | None = None,
    cooldown_left: float = 0.0,
) -> None:
    lines: list[tuple[str, tuple[int, int, int]]] = [
        (f"FPS {fps:5.1f}   infer {infer_ms:5.1f} ms", WHITE),
        (f"dropped frames: {dropped}", GREY),
    ]
    if cooldown_left > 0:
        lines.append((f"cooldown {cooldown_left:.1f}s", YELLOW))
    if detector_debug:
        lines.append(("-- detector --", GREY))
        lines += [(f"{k}: {v:.3f}", WHITE) for k, v in detector_debug.items()]
    if features:
        lines.append(("-- features --", GREY))
        lines += [(f"{k}: {v:+.3f}", WHITE) for k, v in features.items()]

    pad, lh = 10, 18
    h = pad * 2 + lh * len(lines)
    box = img[pad : pad + h, pad : pad + 300]
    if box.size:
        cv2.addWeighted(box, 0.35, np.zeros_like(box), 0.65, 0, box)

    y = pad + 16
    for text, color in lines:
        cv2.putText(img, text, (pad + 8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
        y += lh

    _draw_score_bar(img, score, threshold)


def _draw_score_bar(img: np.ndarray, score: float, threshold: float) -> None:
    h, w = img.shape[:2]
    bx, by, bw, bh = w - 60, 40, 28, h - 100
    cv2.rectangle(img, (bx, by), (bx + bw, by + bh), (60, 60, 60), -1)
    filled = int(bh * float(np.clip(score, 0.0, 1.0)))
    color = GREEN if score >= threshold else (90, 140, 220)
    cv2.rectangle(img, (bx, by + bh - filled), (bx + bw, by + bh), color, -1)
    ty = by + bh - int(bh * float(np.clip(threshold, 0.0, 1.0)))
    cv2.line(img, (bx - 6, ty), (bx + bw + 6, ty), RED, 2)
    cv2.putText(img, f"{score:.2f}", (bx - 8, by - 12), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, WHITE, 1, cv2.LINE_AA)


def flash_event(img: np.ndarray, text: str = "HELLO WORLD") -> None:
    h, w = img.shape[:2]
    cv2.rectangle(img, (0, 0), (w - 1, h - 1), GREEN, 8)
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.4, 3)
    cv2.putText(img, text, ((w - tw) // 2, (h + th) // 2), cv2.FONT_HERSHEY_SIMPLEX,
                1.4, GREEN, 3, cv2.LINE_AA)
