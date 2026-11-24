from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

from . import features as F
from .types import Pose

LABELS = {"other": 0, "wave": 1}


@dataclass
class Clip:
    feats: np.ndarray
    times: np.ndarray
    label: int
    clip_id: str


def poses_to_clip(poses, times, label: str, clip_id: str) -> Clip | None:
    feats, ts = [], []
    for pose, t in zip(poses, times):
        f = F.normalize_pose(pose)
        if f is not None:
            feats.append(f)
            ts.append(float(t))
    if len(feats) < 5:
        return None
    return Clip(np.stack(feats).astype(np.float32), np.asarray(ts), LABELS[label], clip_id)


def npz_to_clip(path: Path) -> Clip | None:
    d = np.load(path, allow_pickle=False)
    label = str(d["label"])
    if label not in LABELS:
        return None
    poses = [
        Pose(xy=d["xy"][i], conf=d["conf"][i], box=np.zeros(4, np.float32))
        for i in range(len(d["t"]))
        if d["valid"][i]
    ]
    times = d["t"][d["valid"]]
    return poses_to_clip(poses, times, label, path.stem)


def clip_to_windows(
    clip: Clip, window_seconds: float, n_steps: int, stride_seconds: float, use_velocity: bool
) -> list[np.ndarray]:
    windows = []
    t0, t_end = clip.times[0], clip.times[-1]
    start = t0
    while start + window_seconds <= t_end + 1e-6:
        i0 = int(np.searchsorted(clip.times, start, side="left"))
        i1 = int(np.searchsorted(clip.times, start + window_seconds, side="right"))
        if i1 - i0 >= 5:
            windows.append(
                F.prepare_window(clip.feats[i0:i1], clip.times[i0:i1], n_steps, use_velocity)
            )
        start += stride_seconds
    return windows


def build_arrays(clips, window_seconds, n_steps, stride, use_velocity):
    xs, ys = [], []
    for clip in clips:
        for w in clip_to_windows(clip, window_seconds, n_steps, stride, use_velocity):
            xs.append(w)
            ys.append(clip.label)
    if not xs:
        return np.zeros((0, n_steps, 1), np.float32), np.zeros(0, np.float32)
    return np.stack(xs).astype(np.float32), np.asarray(ys, np.float32)


def split_by_clip(clips: list[Clip], val_frac: float, seed: int):
    """Split by clip, never by window: overlapping windows would leak."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(clips))
    n_val = max(1, int(round(len(clips) * val_frac)))
    val_idx = set(order[:n_val].tolist())
    train = [c for i, c in enumerate(clips) if i not in val_idx]
    val = [c for i, c in enumerate(clips) if i in val_idx]
    return train, val


class WindowDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.float32))

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, i: int):
        return self.X[i], self.y[i]


def train_one_epoch(model, loader, optimizer, criterion, device) -> float:
    model.train()
    total, n = 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
        total += loss.item() * len(y)
        n += len(y)
    return total / max(n, 1)


@torch.no_grad()
def evaluate_epoch(model, loader, criterion, device, threshold: float = 0.5) -> dict:
    model.eval()
    total, n = 0.0, 0
    tp = fp = fn = tn = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        total += criterion(logits, y).item() * len(y)
        n += len(y)
        pred = (torch.sigmoid(logits) >= threshold).float()
        tp += int(((pred == 1) & (y == 1)).sum())
        fp += int(((pred == 1) & (y == 0)).sum())
        fn += int(((pred == 0) & (y == 1)).sum())
        tn += int(((pred == 0) & (y == 0)).sum())

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    return {
        "loss": total / max(n, 1),
        "acc": (tp + tn) / max(n, 1),
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / max(precision + recall, 1e-9),
    }


def overfit_check(model, X, y, device, steps: int = 300, verbose: bool = True) -> float:
    """Memorise 8 samples. Loss must approach zero or the code is wrong."""
    idx = np.concatenate([np.where(y == 1)[0][:4], np.where(y == 0)[0][:4]])
    xb = torch.from_numpy(X[idx]).to(device)
    yb = torch.from_numpy(y[idx]).to(device)

    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    crit = nn.BCEWithLogitsLoss()
    loss = torch.tensor(float("nan"))
    for step in range(steps):
        opt.zero_grad()
        loss = crit(model(xb), yb)
        loss.backward()
        opt.step()
        if verbose and step % 50 == 0:
            print(f"    step {step:3d}   loss {loss.item():.5f}")
    return float(loss.item())
