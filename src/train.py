"""Train one model and save its validation / test probabilities.

Model selection uses balanced accuracy on the validation set (the official
ISIC 2018 Task 3 metric). The test set is only predicted once, at the end.
"""
import argparse
import copy
import json
import math
import os
import random
import time

import numpy as np
import timm
import torch
from sklearn.metrics import balanced_accuracy_score
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm.auto import tqdm

from data import LesionDataset, build_transforms
from splits import CLASSES, load_or_build_splits


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_scaler(enabled):
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


@torch.no_grad()
def predict(model, df, args, device, tta):
    """Class probabilities; with tta=True the four flip views are averaged."""
    loader = DataLoader(LesionDataset(df, build_transforms(args.img_size, False)),
                        batch_size=args.batch_size * 2, shuffle=False, num_workers=args.workers)
    model.eval()
    out = []
    for x, _ in loader:
        x = x.to(device)
        views = [x] if not tta else [x, x.flip(3), x.flip(2), x.flip(2).flip(3)]
        probs = 0
        for v in views:
            with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                logits = model(v)
            probs = probs + torch.softmax(logits.float(), dim=1)
        out.append((probs / len(views)).cpu().numpy())
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True, help="run name, used for output files")
    ap.add_argument("--arch", default="convnext_tiny", help="any timm model name")
    ap.add_argument("--img_size", type=int, default=384)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=0.05)
    ap.add_argument("--loss", choices=["ce", "weighted_ce"], default="ce")
    ap.add_argument("--label_smoothing", type=float, default=0.0)
    ap.add_argument("--sampler", choices=["none", "balanced"], default="none")
    ap.add_argument("--aug", choices=["basic", "strong"], default="strong")
    ap.add_argument("--no_tta", action="store_true")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--out_dir", default="results")
    ap.add_argument("--ckpt_dir", default="checkpoints")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.ckpt_dir, exist_ok=True)

    df = load_or_build_splits(args.data_dir, os.path.join(args.out_dir, "splits.csv"), args.seed)
    train, val, test = (df[df["split"] == s].reset_index(drop=True) for s in ("train", "val", "test"))
    print(df.attrs["protocol"], f"| train={len(train)} val={len(val)} test={len(test)}")

    counts = np.bincount(train["label"], minlength=len(CLASSES))
    sampler = None
    if args.sampler == "balanced":                       # every class equally likely per batch
        weights = (1.0 / counts)[train["label"].values]
        sampler = WeightedRandomSampler(torch.as_tensor(weights, dtype=torch.double),
                                        num_samples=len(train), replacement=True)
    train_loader = DataLoader(LesionDataset(train, build_transforms(args.img_size, True, args.aug)),
                              batch_size=args.batch_size, sampler=sampler, shuffle=sampler is None,
                              num_workers=args.workers, drop_last=True, pin_memory=True)

    model = timm.create_model(args.arch, pretrained=True, num_classes=len(CLASSES)).to(device)
    class_weight = None
    if args.loss == "weighted_ce":                       # inverse class frequency
        class_weight = torch.tensor(len(train) / (len(CLASSES) * counts), dtype=torch.float,
                                    device=device)
    loss_fn = nn.CrossEntropyLoss(weight=class_weight, label_smoothing=args.label_smoothing)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total, warmup = len(train_loader) * args.epochs, len(train_loader)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: (s + 1) / warmup if s < warmup
        else 0.5 * (1 + math.cos(math.pi * (s - warmup) / max(1, total - warmup))))
    scaler = make_scaler(use_amp)

    history, best, best_state, start = [], -1.0, None, time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        running, seen = 0.0, 0
        for x, y in tqdm(train_loader, desc=f"{args.name} epoch {epoch}/{args.epochs}", leave=False):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device.type, dtype=torch.float16, enabled=use_amp):
                logits = model(x)
            loss = loss_fn(logits.float(), y)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            running += loss.item() * y.size(0)
            seen += y.size(0)

        val_pred = predict(model, val, args, device, tta=False).argmax(1)
        bal_acc = balanced_accuracy_score(val["label"], val_pred)
        history.append({"epoch": epoch, "train_loss": running / seen,
                        "val_balanced_acc": float(bal_acc),
                        "val_acc": float((val_pred == val["label"].values).mean())})
        print(history[-1])
        if bal_acc > best:
            best = bal_acc
            best_state = copy.deepcopy({k: v.cpu() for k, v in model.state_dict().items()})

    model.load_state_dict(best_state)
    torch.save({"state_dict": best_state, "arch": args.arch, "img_size": args.img_size,
                "classes": CLASSES}, os.path.join(args.ckpt_dir, f"{args.name}.pth"))

    tta = not args.no_tta
    np.savez(os.path.join(args.out_dir, f"{args.name}_probs.npz"),
             val_probs=predict(model, val, args, device, tta), val_labels=val["label"].values,
             test_probs=predict(model, test, args, device, tta), test_labels=test["label"].values,
             test_images=test["image"].values.astype(str))
    with open(os.path.join(args.out_dir, f"{args.name}_train.json"), "w") as fh:
        json.dump({"name": args.name, "config": vars(args), "protocol": df.attrs["protocol"],
                   "n_train": len(train), "n_val": len(val), "n_test": len(test),
                   "train_class_counts": dict(zip(CLASSES, map(int, counts))),
                   "best_val_balanced_acc": float(best), "history": history, "tta": tta,
                   "train_minutes": round((time.time() - start) / 60, 1)}, fh, indent=2)
    print(f"done: best val balanced accuracy {best:.4f}")


if __name__ == "__main__":
    main()
