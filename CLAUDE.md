# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Railway surface defect dataset and analysis tools. The project deals with classifying rail surface defects into 7 classes (Flakings, Squats, Spallings, Shellings, Cracks, Joints, Grooves). The dataset is heavily imbalanced — Flakings+Squats account for ~91% of images.

## Setup

- Python 3.11 with virtualenv at `./venv`
- Activate: `source venv/bin/activate`
- Dependencies in `requirements.txt` : `pip install -r requirements.txt`
- API key fal.ai requise : copier `.env.example` → `.env` et ajouter `FAL_KEY`

## Running

```bash
# Play color video (default)
python read_video.py

# Play specific video
python read_video.py Images/depth_0.mkv
```

## Generation d'images Normal

Les images Flakings contiennent des déchets (plastique, papier, emballages) sur le ballast.
Le pipeline retire ces déchets via SAM 3 (détection) + object-removal (LaMa, remplissage texture).
Le rail, les boulons et le ballast restent pixel-identiques à l'original.

```bash
# Générer N images Normal à partir des Flakings
PYTHONPATH=. python src/generation/generate_normal.py --src data/surface/Flakings --n 50

# Tester sur une seule image
PYTHONPATH=. python src/generation/generate_normal.py --src data/surface/Flakings/image.JPEG

# Sortie custom
PYTHONPATH=. python src/generation/generate_normal.py --src data/surface/Flakings --n 100 --out data/surface/Normal
```

Résultats dans `data/normal_synthetic/Normal/` par défaut, masks de debug dans `_debug/`.

## Modules src/generation/

| Script | Description |
|--------|-------------|
| `generate_normal.py` | Pipeline Normal : SAM 3 trash detection + object-removal |
| `generate_rare.py` | Génération d'images synthétiques pour les classes rares (Cracks/Joints/Grooves) |
| `augment_from_rsdds.py` | Transplantation de défauts RSDDs sur des rails sains |
| `fal_wrapper.py` | Wrapper fal-client avec cache disque, retry et logging |
| `quality_filter.py` | Filtre qualité : CLIP-score + LPIPS diversity |
| `make_comparison.py` | Génère des grilles de comparaison avant/après |

## Data Layout

- `data/surface/` — 5,153 images across 7 defect class subdirectories
- `data/normal_synthetic/` — images Normal générées
- `data/_raw/cabview_refs/` — frames de référence caméra frontale train
- `Images/` — video files (color_0.mkv, depth_0.mkv) and timing data (time_0.time)

## Language

Project documentation and code comments are in French.
