"""Évalue un checkpoint sur le test split et écrit un rapport markdown."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.dataset import RailDefectDataset
from src.data.preprocessing import eval_transforms
from src.models.classifier import build_classifier
from src.utils.metrics import compute_metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--test-csv", default="data/splits/test.csv")
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--out", default="outputs/predictions/report.md")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ds = RailDefectDataset(args.test_csv, eval_transforms(args.img_size))
    loader = DataLoader(ds, batch_size=32, num_workers=4)
    class_names = RailDefectDataset.class_names()

    model = build_classifier(args.model, num_classes=len(class_names)).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    model.eval()

    ys, ps = [], []
    with torch.no_grad():
        for x, y in loader:
            ps.append(model(x.to(device)).argmax(1).cpu().numpy())
            ys.append(y.numpy())
    metrics = compute_metrics(np.concatenate(ys), np.concatenate(ps), class_names)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    md = [f"# Eval: {args.model}", "", f"- Macro F1: **{metrics['macro_f1']:.4f}**",
          f"- Weighted F1: **{metrics['weighted_f1']:.4f}**", "", "## Per class", ""]
    md.append("| Class | P | R | F1 | Support |")
    md.append("|---|---|---|---|---|")
    for c, m in metrics["per_class"].items():
        md.append(f"| {c} | {m['precision']:.3f} | {m['recall']:.3f} | {m['f1']:.3f} | {m['support']} |")
    md += ["", "## Confusion matrix", "```", str(metrics["confusion_matrix"]), "```"]
    out.write_text("\n".join(md))
    print(metrics["report"])
    print(f"\nReport → {out}")
    (out.with_suffix(".json")).write_text(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
