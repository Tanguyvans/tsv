"""Test Depth Anything 3 sur des frames de cabview ferroviaire.

Installation préalable (une fois) :
    pip install xformers
    pip install -e git+https://github.com/ByteDance-Seed/Depth-Anything-3.git#egg=depth_anything_3

Utilisation :
    PYTHONPATH=. python src/depth/test_da3.py
    PYTHONPATH=. python src/depth/test_da3.py --src data/cabview/fr/frames_cabview/bordeaux_nantes_2023 --n 5
    PYTHONPATH=. python src/depth/test_da3.py --model DA3-BASE --n 3

Modèles disponibles (HuggingFace `depth-anything/...`) :
    - DA3MONO-LARGE-1.1     : profondeur relative mono, recommandé pour 1ère exploration
    - DA3METRIC-LARGE-1.1   : profondeur métrique (mètres réels)
    - DA3-BASE / DA3-LARGE  : any-view foundation, supporte multi-frame
    - DA3NESTED-GIANT-LARGE-1.1 : qualité max, lourd
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image


def colorize_depth(depth: np.ndarray, near: float | None = None, far: float | None = None) -> np.ndarray:
    """Carte de profondeur → image RGB (turbo colormap, near=rouge, far=bleu)."""
    import matplotlib.cm as cm

    valid = np.isfinite(depth)
    near = near if near is not None else float(np.nanpercentile(depth[valid], 2))
    far = far if far is not None else float(np.nanpercentile(depth[valid], 98))
    norm = np.clip((depth - near) / max(far - near, 1e-6), 0, 1)
    colored = cm.turbo(norm)[..., :3]
    colored[~valid] = 0
    return (colored * 255).astype(np.uint8)


def save_ply(points: np.ndarray, colors: np.ndarray | None, path: Path) -> None:
    """Sauve un nuage de points (N,3) en .ply ASCII pour MeshLab/CloudCompare."""
    pts = points.reshape(-1, 3)
    mask = np.isfinite(pts).all(axis=1)
    pts = pts[mask]
    cols = None
    if colors is not None:
        cols = colors.reshape(-1, 3)[mask]
    n = len(pts)

    with path.open("w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {n}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        if cols is not None:
            f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        if cols is not None:
            for (x, y, z), (r, g, b) in zip(pts, cols):
                f.write(f"{x} {y} {z} {int(r)} {int(g)} {int(b)}\n")
        else:
            for x, y, z in pts:
                f.write(f"{x} {y} {z}\n")


def depth_to_points(depth: np.ndarray, K: np.ndarray) -> np.ndarray:
    """Reprojection profondeur → (H,W,3) en repère caméra (Z avant, Y bas, X droite)."""
    H, W = depth.shape
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    Z = depth.astype(np.float32)
    X = (u - cx) * Z / fx
    Y = (v - cy) * Z / fy
    return np.stack([X, Y, Z], axis=-1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/cabview/fr/frames_cabview/bordeaux_nantes_2023",
                    help="dossier d'images ou chemin d'une image")
    ap.add_argument("--model", default="DA3METRIC-LARGE-1.1",
                    help="nom du modèle (suffixe HF depth-anything/...)")
    ap.add_argument("--n", type=int, default=3, help="nombre d'images à traiter")
    ap.add_argument("--out", default="outputs/depth_da3", help="dossier de sortie")
    ap.add_argument("--device", default=None, help="cuda / mps / cpu (auto sinon)")
    args = ap.parse_args()

    # device
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"[info] device = {device}")

    # collecte des images
    src = Path(args.src)
    if src.is_file():
        images = [src]
    else:
        images = sorted([p for p in src.iterdir()
                         if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])[:args.n]
    if not images:
        raise SystemExit(f"Aucune image trouvée dans {src}")
    print(f"[info] {len(images)} image(s) à traiter depuis {src}")

    # chargement modèle
    print(f"[info] chargement de depth-anything/{args.model} ...")
    from depth_anything_3.api import DepthAnything3
    model = DepthAnything3.from_pretrained(f"depth-anything/{args.model}")
    model = model.to(device).eval()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    for img_path in images:
        print(f"\n[run] {img_path.name}")
        t0 = time.time()
        with torch.inference_mode():
            pred = model.inference([str(img_path)])
        elapsed = time.time() - t0

        # extraction tensor → numpy
        depth = pred.depth
        if hasattr(depth, "detach"):
            depth = depth.detach().cpu().numpy()
        depth = np.squeeze(depth)
        print(f"  depth shape={depth.shape} min={depth.min():.2f} max={depth.max():.2f} ({elapsed*1000:.0f} ms)")

        # sauvegarde profondeur colorisée
        rgb_img = np.array(Image.open(img_path).convert("RGB"))
        depth_vis = colorize_depth(depth)
        side = np.concatenate([rgb_img, depth_vis], axis=1) \
            if rgb_img.shape[:2] == depth_vis.shape[:2] \
            else depth_vis
        Image.fromarray(side).save(out_dir / f"{img_path.stem}_depth.jpg", quality=92)

        # nuage de points si intrinsics dispo
        K = getattr(pred, "intrinsics", None)
        if K is not None:
            K = np.squeeze(K.detach().cpu().numpy() if hasattr(K, "detach") else K)
            print(f"  intrinsics: fx={K[0,0]:.1f} fy={K[1,1]:.1f} cx={K[0,2]:.1f} cy={K[1,2]:.1f}")
            # downsample pour limiter taille ply
            stride = max(1, depth.shape[0] // 256)
            d_ds = depth[::stride, ::stride]
            K_ds = K.copy()
            K_ds[0] /= stride
            K_ds[1] /= stride
            pts = depth_to_points(d_ds, K_ds)
            rgb_ds = np.array(Image.open(img_path).convert("RGB").resize(
                (d_ds.shape[1], d_ds.shape[0]), Image.BILINEAR))
            save_ply(pts, rgb_ds, out_dir / f"{img_path.stem}.ply")
            print(f"  -> {img_path.stem}_depth.jpg + {img_path.stem}.ply")
        else:
            print(f"  -> {img_path.stem}_depth.jpg (pas d'intrinsics, pas de PLY)")

    print(f"\n[done] sorties dans {out_dir}")


if __name__ == "__main__":
    main()
