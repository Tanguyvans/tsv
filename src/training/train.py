"""Boucle d'entraînement classification déséquilibrée.

Focal Loss + WeightedRandomSampler + AdamW + CosineAnnealingWarmRestarts
+ early stopping macro-F1.

Usage:
  python src/training/train.py --config configs/train_efficientnet.yaml
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter

from src.data.dataset import RailDefectDataset
from src.data.preprocessing import eval_transforms, train_transforms
from src.models.classifier import build_classifier
from src.utils.metrics import compute_metrics


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, weight: torch.Tensor | None = None):
        super().__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, logits, target):
        ce = F.cross_entropy(logits, target, weight=self.weight, reduction="none")
        pt = torch.exp(-ce)
        return ((1 - pt) ** self.gamma * ce).mean()


def make_sampler(dataset: RailDefectDataset) -> WeightedRandomSampler:
    counts = dataset.class_counts()
    counts = np.where(counts == 0, 1, counts)
    class_weights = 1.0 / counts
    sample_weights = np.array([class_weights[lbl] for _, lbl, _ in dataset.samples])
    return WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)


def evaluate(model, loader, device, class_names):
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            logits = model(x)
            ps.append(logits.argmax(1).cpu().numpy())
            ys.append(y.numpy())
    return compute_metrics(np.concatenate(ys), np.concatenate(ps), class_names)


def train(cfg: dict):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(out_dir)

    train_ds = RailDefectDataset(cfg["train_csv"], train_transforms(cfg["img_size"]))
    val_ds = RailDefectDataset(cfg["val_csv"], eval_transforms(cfg["img_size"]))
    class_names = RailDefectDataset.class_names()

    sampler = make_sampler(train_ds)
    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], sampler=sampler, num_workers=cfg.get("num_workers", 4))
    val_loader = DataLoader(val_ds, batch_size=cfg["batch_size"], shuffle=False, num_workers=cfg.get("num_workers", 4))

    model = build_classifier(cfg["model"], num_classes=len(class_names)).to(device)

    counts = train_ds.class_counts().astype(float)
    counts = np.where(counts == 0, 1, counts)
    cls_w = torch.tensor(counts.sum() / (len(counts) * counts), dtype=torch.float32, device=device)
    criterion = FocalLoss(gamma=cfg.get("focal_gamma", 2.0), weight=cls_w)

    optim = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg.get("weight_decay", 1e-4))
    sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optim, T_0=cfg.get("t0", 10))

    best_f1 = -1.0
    patience = cfg.get("patience", 10)
    bad = 0
    for epoch in range(cfg["epochs"]):
        model.train()
        running = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optim.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optim.step()
            running += loss.item() * x.size(0)
        sched.step()
        train_loss = running / len(train_ds)
        metrics = evaluate(model, val_loader, device, class_names)
        writer.add_scalar("loss/train", train_loss, epoch)
        writer.add_scalar("f1/macro_val", metrics["macro_f1"], epoch)
        print(f"epoch {epoch:03d} loss={train_loss:.4f} macro_f1={metrics['macro_f1']:.4f}")

        if metrics["macro_f1"] > best_f1:
            best_f1 = metrics["macro_f1"]
            bad = 0
            torch.save(model.state_dict(), out_dir / "best.pt")
            (out_dir / "best_metrics.json").write_text(json.dumps(metrics, indent=2))
        else:
            bad += 1
            if bad >= patience:
                print(f"early stop @ epoch {epoch}")
                break

    print(f"best val macro_f1 = {best_f1:.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    train(cfg)


if __name__ == "__main__":
    main()
