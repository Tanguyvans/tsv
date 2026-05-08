"""Helpers de visualisation : grilles d'images avec labels."""
from __future__ import annotations

import cv2
import numpy as np

LABEL_H = 40
PAD = 10
STRIP_BG = (240, 240, 240)
TEXT_COLOR = (20, 20, 20)
SEP_COLOR = (255, 255, 255)
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.8
FONT_THICK = 2


def resize_to_height(img: np.ndarray, target_h: int) -> np.ndarray:
    r = target_h / img.shape[0]
    return cv2.resize(img, (int(img.shape[1] * r), target_h), interpolation=cv2.INTER_AREA)


def label_strip(width: int, text: str) -> np.ndarray:
    strip = np.full((LABEL_H, width, 3), STRIP_BG, dtype=np.uint8)
    cv2.putText(strip, text, (10, LABEL_H - 12),
                FONT, FONT_SCALE, TEXT_COLOR, FONT_THICK, cv2.LINE_AA)
    return strip


def hstack_with_labels(panels: list[tuple[str, np.ndarray]],
                       target_h: int = 540) -> np.ndarray:
    """Compose N labelled images horizontally. Each panel = (label, BGR image)."""
    cols = []
    for label, img in panels:
        resized = resize_to_height(img, target_h)
        cols.append(np.vstack([label_strip(resized.shape[1], label), resized]))
    sep = np.full((target_h + LABEL_H, PAD, 3), SEP_COLOR, dtype=np.uint8)
    out = cols[0]
    for c in cols[1:]:
        out = np.hstack([out, sep, c])
    return out
