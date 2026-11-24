from __future__ import annotations

from pathlib import Path

import numpy as np

from .pose import PoseEstimator
from .sources import VideoFileSource


def extract_video(video: Path, estimator: PoseEstimator) -> dict[str, np.ndarray]:
    """Run pose estimation over a video and return raw keypoints and times."""
    src = VideoFileSource(video)
    xys, confs, ts, ok = [], [], [], []
    try:
        for frame in src.frames():
            res = estimator(frame)
            ts.append(frame.t)
            if res.has_pose:
                xys.append(res.pose.xy)
                confs.append(res.pose.conf)
                ok.append(True)
            else:
                xys.append(np.full((17, 2), np.nan, np.float32))
                confs.append(np.zeros(17, np.float32))
                ok.append(False)
    finally:
        src.close()
    return {
        "xy": np.stack(xys).astype(np.float32),
        "conf": np.stack(confs).astype(np.float32),
        "t": np.asarray(ts, np.float64),
        "valid": np.asarray(ok, bool),
    }
