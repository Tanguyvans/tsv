"""Stage 2 — classifieur binaire has_panel / no_panel sur crops du stage 1.

Réutilise build_classifier (timm) avec un encoder léger (efficientnet_b0).
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from src.data.preprocessing import eval_transforms, train_transforms
from src.models.classifier import build_classifier


class CropDataset(Dataset):
    def __init__(self, items: list[tuple[str, int]], transform):
        self.items = items
        self.transform = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, label = self.items[i]
        img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
        return self.transform(image=img)["image"], label


def collect(pos_dir: Path, neg_dir: Path) -> list[tuple[str, int]]:
    items = []
    for p in pos_dir.rglob("*"):
        if p.suffix.lower() in {".jpg", ".png"}:
            items.append((str(p), 1))
    for p in neg_dir.rglob("*"):
        if p.suffix.lower() in {".jpg", ".png"}:
            items.append((str(p), 0))
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--has-panel", default="data/gerald_augmented/has_panel")
    ap.add_argument("--bare-poles", default="data/gerald_augmented/bare_poles/images")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--out", default="outputs/signals/stage2_classifier")
    args = ap.parse_args()

    items = collect(Path(args.has_panel), Path(args.bare_poles))
    if not items:
        print("No data — run extract_masts and generate_bare_poles first")
        return
    rng = np.random.default_rng(0)
    idx = np.arange(len(items)); rng.shuffle(idx)
    split = int(0.85 * len(idx))
    tr = [items[i] for i in idx[:split]]
    va = [items[i] for i in idx[split:]]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_loader = DataLoader(CropDataset(tr, train_transforms(160)), batch_size=args.batch, shuffle=True, num_workers=4)
    val_loader = DataLoader(CropDataset(va, eval_transforms(160)), batch_size=args.batch, num_workers=4)

    model = build_classifier("efficientnet_b0", num_classes=2).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    crit = nn.CrossEntropyLoss()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    best = 0.0
    for ep in range(args.epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(); crit(model(x), y).backward(); opt.step()
        model.eval(); correct = total = 0
        with torch.no_grad():
            for x, y in val_loader:
                pred = model(x.to(device)).argmax(1).cpu()
                correct += (pred == y).sum().item(); total += y.numel()
        acc = correct / max(1, total)
        print(f"epoch {ep:02d} val_acc={acc:.4f}")
        if acc > best:
            best = acc
            torch.save(model.state_dict(), out / "best.pt")
    print(f"best val acc = {best:.4f}")


if __name__ == "__main__":
    main()
