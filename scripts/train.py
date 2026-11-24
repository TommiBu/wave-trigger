#!/usr/bin/env python3
"""Train the window classifier.

    python scripts/train.py --synthetic --overfit
    python scripts/train.py --synthetic --arch cnn --epochs 25
    python scripts/train.py --data data/npz --arch cnn --epochs 40 --device mps
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from wave_trigger import features as F
from wave_trigger.config import Config
from wave_trigger.models import build_model, count_parameters
from wave_trigger.training import (
    WindowDataset,
    build_arrays,
    evaluate_epoch,
    npz_to_clip,
    overfit_check,
    poses_to_clip,
    split_by_clip,
    train_one_epoch,
)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/npz")
    p.add_argument("--synthetic", action="store_true")
    p.add_argument("--n-per-class", type=int, default=40)
    p.add_argument("--arch", default="cnn", choices=["mlp", "cnn", "gru"])
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--n-steps", type=int, default=32)
    p.add_argument("--stride", type=float, default=0.2)
    p.add_argument("--val-frac", type=float, default=0.25)
    p.add_argument("--velocity", action="store_true")
    p.add_argument("--device", default="cpu")
    p.add_argument("--out", default="runs/wave.pt")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--overfit", action="store_true")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    cfg = Config()
    device = args.device

    if args.synthetic:
        from wave_trigger.synthetic import synthetic_clips

        raw = synthetic_clips(args.n_per_class, seed=args.seed)
        clips = [c for c in (poses_to_clip(*r) for r in raw) if c is not None]
    else:
        paths = sorted(Path(args.data).rglob("*.npz"))
        if not paths:
            raise SystemExit(
                f"no data in {args.data}; record clips first, or try --synthetic"
            )
        clips = [c for c in (npz_to_clip(p) for p in paths) if c is not None]

    n_wave = sum(c.label == 1 for c in clips)
    print(f"clips: {len(clips)}  (wave {n_wave} / other {len(clips) - n_wave})")
    if n_wave == 0 or n_wave == len(clips):
        raise SystemExit("both classes are required")

    train_clips, val_clips = split_by_clip(clips, args.val_frac, args.seed)
    Xtr, ytr = build_arrays(train_clips, cfg.window_seconds, args.n_steps,
                            args.stride, args.velocity)
    Xva, yva = build_arrays(val_clips, cfg.window_seconds, args.n_steps,
                            args.stride, args.velocity)
    if len(Xtr) == 0 or len(Xva) == 0:
        raise SystemExit("clips shorter than the window; lower window_seconds")

    n_features = Xtr.shape[2]
    print(f"windows: train {Xtr.shape}  val {Xva.shape}  wave share {ytr.mean():.1%}")

    model = build_model(args.arch, args.n_steps, n_features).to(device)
    print(f"{args.arch}: {count_parameters(model):,} params on {len(Xtr)} samples")

    if args.overfit:
        print("\noverfit check on 8 samples")
        final = overfit_check(model, Xtr, ytr, device)
        verdict = "ok" if final < 0.01 else "FAILED - fix the code before training"
        print(f"final loss {final:.6f}  ->  {verdict}")
        return

    pos = float(ytr.sum())
    pos_weight = torch.tensor([(len(ytr) - pos) / max(pos, 1.0)], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    train_loader = DataLoader(WindowDataset(Xtr, ytr), batch_size=args.batch, shuffle=True)
    val_loader = DataLoader(WindowDataset(Xva, yva), batch_size=args.batch)

    best_key, best_f1, best_state, best_epoch = (-1.0, -1e9), -1.0, None, -1
    history = []
    t0 = time.time()
    print(f"\n{'ep':>3} {'train':>8} {'val':>8} {'acc':>7} {'prec':>7} {'rec':>7} {'f1':>7}")
    print("-" * 52)
    for epoch in range(1, args.epochs + 1):
        tr_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        m = evaluate_epoch(model, val_loader, criterion, device)
        history.append({"epoch": epoch, "train_loss": tr_loss, **m})
        print(f"{epoch:>3} {tr_loss:>8.4f} {m['loss']:>8.4f} {m['acc']:>7.1%} "
              f"{m['precision']:>7.1%} {m['recall']:>7.1%} {m['f1']:>7.1%}")

        key = (m["f1"], -m["loss"])
        if key > best_key:
            best_key, best_f1, best_epoch = key, m["f1"], epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    print(f"\nbest f1 {best_f1:.1%} at epoch {best_epoch} in {time.time() - t0:.1f}s")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": best_state,
            "arch": args.arch,
            "n_steps": args.n_steps,
            "use_velocity": args.velocity,
            "feature_names": F.FEATURE_NAMES,
            "window_seconds": cfg.window_seconds,
            "val_f1": best_f1,
            "history": history,
            "synthetic": args.synthetic,
        },
        out,
    )
    out.with_suffix(".json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(f"saved {out}")
    if args.synthetic:
        print("trained on synthetic data; expect worse results on real video")


if __name__ == "__main__":
    main()
