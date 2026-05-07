"""Filtre les frames pour ne garder que les vraies vues cab (depuis le train).

Utilise CLIP pour comparer chaque frame à des images de référence stockées dans
`data/cabview/_refs/cabview/` (positifs) et `data/cabview/_refs/not_cabview/`
(négatifs : drones, intros, plans extérieurs, gares...).

Une frame est cab-view si max sim aux refs cabview > max sim aux refs négatifs
+ marge.

Usage:
    PYTHONPATH=. python src/cabview/check_cabview.py \\
        --frames data/cabview/fr/frames/bordeaux_nantes_2023 \\
        --copy
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import clip
import torch
from PIL import Image
from tqdm import tqdm

REFS_ROOT = Path("data/cabview/_refs")


def load_clip(device: str):
    model, preprocess = clip.load("ViT-B/32", device=device)
    return model, preprocess


@torch.no_grad()
def encode_images(model, preprocess, paths: list[Path], device: str,
                  batch: int = 32) -> torch.Tensor:
    feats_all = []
    for i in range(0, len(paths), batch):
        chunk = paths[i:i + batch]
        imgs = torch.stack(
            [preprocess(Image.open(p).convert("RGB")) for p in chunk]
        ).to(device)
        f = model.encode_image(imgs)
        f = f / f.norm(dim=-1, keepdim=True)
        feats_all.append(f)
    return torch.cat(feats_all, dim=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None,
                    help="Dossier sortie (default: <parent>/frames_cabview/<name>)")
    ap.add_argument("--margin", type=float, default=0.05,
                    help="sim_pos doit dépasser sim_neg de cette marge")
    ap.add_argument("--copy", action="store_true")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else (
            "mps" if torch.backends.mps.is_available() else "cpu")
    else:
        device = args.device
    print(f"device: {device}")

    pos_refs = sorted((REFS_ROOT / "cabview").glob("*.jpg")) + \
               sorted((REFS_ROOT / "cabview").glob("*.png"))
    neg_refs = sorted((REFS_ROOT / "not_cabview").glob("*.jpg")) + \
               sorted((REFS_ROOT / "not_cabview").glob("*.png"))
    if not pos_refs or not neg_refs:
        raise SystemExit(
            f"Manque des refs dans {REFS_ROOT}. "
            f"Trouvé pos={len(pos_refs)} neg={len(neg_refs)}"
        )
    print(f"refs cabview: {len(pos_refs)} | not_cabview: {len(neg_refs)}")

    frames = sorted(args.frames.glob("*.jpg"))
    if not frames:
        raise SystemExit(f"Aucune frame dans {args.frames}")
    print(f"{len(frames)} frames à scorer")

    model, preprocess = load_clip(device)

    print("encodage refs...")
    pos_feats = encode_images(model, preprocess, pos_refs, device)
    neg_feats = encode_images(model, preprocess, neg_refs, device)

    out_dir = args.out or args.frames.parent.parent / "frames_cabview" / args.frames.name
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = []
    n_pos = 0
    batch = 32
    for i in tqdm(range(0, len(frames), batch), desc="CLIP scoring"):
        chunk = frames[i:i + batch]
        feats = encode_images(model, preprocess, chunk, device, batch=len(chunk))
        sim_pos = (feats @ pos_feats.T).max(dim=-1).values
        sim_neg = (feats @ neg_feats.T).max(dim=-1).values
        for p, sp, sn in zip(chunk, sim_pos.cpu().tolist(), sim_neg.cpu().tolist()):
            is_cab = sp > sn + args.margin
            manifest.append({
                "frame": p.name,
                "sim_cab": round(sp, 4),
                "sim_not_cab": round(sn, 4),
                "is_cabview": is_cab,
            })
            if is_cab:
                n_pos += 1
                if args.copy:
                    shutil.copy2(p, out_dir / p.name)

    (out_dir / "_scores.json").write_text(json.dumps(manifest, indent=2))
    print(f"\n{n_pos} / {len(frames)} cab-view ({100*n_pos/len(frames):.1f}%)")
    print(f"Manifest: {out_dir}/_scores.json")
    if args.copy:
        print(f"Frames cab-view copiées dans {out_dir}")


if __name__ == "__main__":
    main()
