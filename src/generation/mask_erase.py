"""Helpers réutilisables : SAM 3 mask generation + Bria Eraser.

Ce module factorise les briques communes aux pipelines de génération qui
doivent retirer un objet d'une image :
  - Génération des images Normal (retire les déchets des Flakings)
  - Génération des bare poles (retire le panneau d'un signal)

Pattern:
  url = upload_file(img_path)
  mask = sam3_mask(url, size=(W, H), prompts=["trash", ...])
  result_url = bria_erase(url, mask, out_dir / "debug_mask.png")
"""
from __future__ import annotations

import io
from pathlib import Path

import cv2
import numpy as np
import requests
from PIL import Image

from src.generation.fal_wrapper import submit_and_wait, upload_file

MODEL_SAM = "fal-ai/sam-3/image"
MODEL_ERASER = "fal-ai/bria/eraser"


def download(url: str) -> bytes:
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    return r.content


def _png_bytes_to_mask(png_bytes: bytes, size: tuple[int, int]) -> np.ndarray:
    """Décode un PNG de mask (possiblement avec alpha) en uint8 binaire."""
    img = Image.open(io.BytesIO(png_bytes))
    if img.size != size:
        img = img.resize(size, Image.NEAREST)
    if img.mode == "RGBA":
        arr = np.array(img.split()[-1])
    else:
        arr = np.array(img.convert("L"))
    return (arr > 127).astype(np.uint8) * 255


def sam3_mask(
    image_url: str,
    size: tuple[int, int],
    prompts: list[str],
    dilate_px: int = 8,
    max_masks: int = 5,
    verbose: bool = True,
) -> np.ndarray | None:
    """Union des masks SAM 3 pour chaque prompt texte.

    Args:
        image_url: URL fal de l'image (retournée par upload_file)
        size: (W, H) de l'image source
        prompts: liste de concepts à segmenter (ex: ["trash", "plastic bag"])
        dilate_px: dilatation du mask final (0 pour désactiver)
        max_masks: nombre max de masks par prompt
        verbose: log les détections

    Returns:
        Mask uint8 (255=objet, 0=fond) ou None si rien détecté.
    """
    W, H = size
    union = np.zeros((H, W), dtype=np.uint8)
    found_any = False

    for prompt in prompts:
        result = submit_and_wait(
            MODEL_SAM,
            {
                "image_url": image_url,
                "prompt": prompt,
                "apply_mask": False,
                "return_multiple_masks": True,
                "max_masks": max_masks,
                "output_format": "png",
            },
        )
        masks = result.get("masks") or []
        if masks and verbose:
            print(f"    SAM3[{prompt}]: {len(masks)} mask(s)")
        for m in masks:
            murl = m.get("url") if isinstance(m, dict) else None
            if not murl:
                continue
            try:
                union = np.maximum(union, _png_bytes_to_mask(download(murl), (W, H)))
                found_any = True
            except Exception as e:
                print(f"    mask download failed: {e}")

    if not found_any:
        return None

    if dilate_px > 0:
        k = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (dilate_px * 2 + 1, dilate_px * 2 + 1)
        )
        union = cv2.dilate(union, k)

    if verbose:
        coverage = union.sum() / 255 / (W * H) * 100
        print(f"    mask coverage: {coverage:.1f}%")
    return union


def _result_url(result: dict) -> str | None:
    """Extrait l'URL de l'image résultat (compatible Bria/LaMa/autres)."""
    images = result.get("images") or []
    if images:
        return images[0].get("url") if isinstance(images[0], dict) else None
    img = result.get("image")
    if isinstance(img, dict):
        return img.get("url")
    return None


def bria_erase(image_url: str, mask: np.ndarray, mask_path: Path,
               src_path: Path | None = None) -> bytes | None:
    """Applique Bria Eraser avec un mask binaire.

    Sauvegarde le mask sur disque (pour upload + debug), appelle Bria, et
    retourne les bytes de l'image résultat.
    """
    mask_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(mask_path), mask)
    mask_url = upload_file(str(mask_path))

    image_paths = [str(mask_path)]
    if src_path is not None:
        image_paths.insert(0, str(src_path))

    result = submit_and_wait(
        MODEL_ERASER,
        {
            "image_url": image_url,
            "mask_url": mask_url,
            "mask_type": "manual",
        },
        image_paths=image_paths,
    )
    url = _result_url(result)
    if not url:
        return None
    return download(url)


def collect_images(src: Path, n: int | None = None) -> list[Path]:
    """Liste les images dans un dossier (ou retourne [src] si c'est un fichier)."""
    if src.is_file():
        return [src]
    exts = {".jpg", ".jpeg", ".png"}
    files = sorted(p for p in src.rglob("*") if p.suffix.lower() in exts)
    return files[:n] if n else files
