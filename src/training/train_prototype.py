"""Episodic training N-way K-shot pour PrototypeNet sur les classes rares."""
from __future__ import annotations

import argparse
import random
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.tensorboard import SummaryWriter

from src.data.dataset import RailDefectDataset
from src.data.preprocessing import eval_transforms, train_transforms
from src.models.prototype_net import PrototypeNet


def load_image(path: str, transform) -> torch.Tensor:
    img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
    return transform(image=img)["image"]


def sample_episode(by_class, classes, n_way, k_shot, q_query, transform, device):
    chosen = random.sample(classes, n_way)
    sup, sup_lbl, qry, qry_lbl = [], [], [], []
    for ci, cls in enumerate(chosen):
        items = random.sample(by_class[cls], k_shot + q_query) if len(by_class[cls]) >= k_shot + q_query \
            else random.choices(by_class[cls], k=k_shot + q_query)
        for p in items[:k_shot]:
            sup.append(load_image(p, transform)); sup_lbl.append(ci)
        for p in items[k_shot:]:
            qry.append(load_image(p, transform)); qry_lbl.append(ci)
    return (
        torch.stack(sup).to(device),
        torch.tensor(sup_lbl, device=device),
        torch.stack(qry).to(device),
        torch.tensor(qry_lbl, device=device),
    )


def index_by_class(ds: RailDefectDataset) -> dict[str, list[str]]:
    out: dict[str, list[str]] = defaultdict(list)
    names = ds.class_names()
    for path, lbl, _ in ds.samples:
        out[names[lbl]].append(path)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(cfg["out_dir"]); out_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(out_dir)

    train_ds = RailDefectDataset(cfg["train_csv"], None)
    by_cls = index_by_class(train_ds)
    classes = cfg.get("classes") or list(by_cls.keys())
    classes = [c for c in classes if len(by_cls[c]) >= cfg["k_shot"] + 1]
    print(f"Classes used: {classes}")

    tf_train = train_transforms(cfg["img_size"])
    tf_eval = eval_transforms(cfg["img_size"])

    model = PrototypeNet(cfg.get("backbone", "resnet50")).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=1e-4)

    n_way, k_shot, q_query = cfg["n_way"], cfg["k_shot"], cfg["q_query"]
    n_way = min(n_way, len(classes))

    best_acc = 0.0
    for it in range(cfg["episodes"]):
        model.train()
        sup, sl, qry, ql = sample_episode(by_cls, classes, n_way, k_shot, q_query, tf_train, device)
        logits = model(sup, sl, qry, n_way) * cfg.get("scale", 10.0)
        loss = F.cross_entropy(logits, ql)
        optim.zero_grad(); loss.backward(); optim.step()

        if it % cfg.get("log_every", 50) == 0:
            acc = (logits.argmax(1) == ql).float().mean().item()
            writer.add_scalar("loss/episodic", loss.item(), it)
            writer.add_scalar("acc/episodic", acc, it)
            print(f"ep {it:05d} loss={loss.item():.3f} acc={acc:.3f}")
            if acc > best_acc:
                best_acc = acc
                torch.save(model.state_dict(), out_dir / "best.pt")

    print(f"best episodic acc = {best_acc:.4f}")


if __name__ == "__main__":
    main()
