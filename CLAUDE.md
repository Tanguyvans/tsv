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
Le pipeline retire ces déchets via SAM 3 (détection) + Bria Eraser (ControlNet inpaint).
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

## Generation de bare poles (signalisation)

À partir du dataset GERALD (5000 images de lignes ferroviaires allemandes avec
annotations PASCAL VOC), le pipeline retire les panneaux de signaux lumineux
pour simuler un scénario "panneau tombé" (poteau seul, pas de signal).

```bash
# Télécharger GERALD (~4.2 GB)
PYTHONPATH=. python src/signals/download_gerald.py

# Générer N bare poles depuis GERALD
PYTHONPATH=. python src/signals/generate_bare_poles.py --n 50
```

Pipeline :
1. Parse les bboxes VOC des classes de signaux (Ks, Hp, Vr, Zs, Ne, Lf, Signal_*)
2. SAM 3 avec box prompts raffine les masks des panneaux
3. Bria Eraser retire les panneaux (poteaux préservés)
4. Génère les labels YOLO du bare_pole (classe 0) via heuristique
   (pole centré sur signal, 30% de largeur, étendu vers le bas)

Résultats dans `data/gerald_augmented/bare_poles/{images,labels}/`.

## Modules src/generation/

| Script | Description |
|--------|-------------|
| `generate_normal.py` | Pipeline Normal : SAM 3 trash detection + Bria Eraser |
| `generate_rare.py` | Génération d'images synthétiques pour les classes rares (Cracks/Joints/Grooves) |
| `augment_from_rsdds.py` | Transplantation de défauts RSDDs sur des rails sains |
| `mask_erase.py` | Helpers réutilisables (SAM 3 + Bria Eraser) |
| `compare_erasers.py` | Compare LaMa, Bria et Nano Banana Pro sur une image |
| `fal_wrapper.py` | Wrapper fal-client avec cache disque, retry et logging |
| `quality_filter.py` | Filtre qualité : CLIP-score + LPIPS diversity |
| `make_comparison.py` | Génère des grilles de comparaison avant/après |

## Modules src/signals/

| Script | Description |
|--------|-------------|
| `download_gerald.py` | Télécharge le dataset GERALD depuis RWTH (4.2 GB) |
| `generate_bare_poles.py` | Génère des bare poles (SAM 3 + Bria) depuis GERALD |
| `voc_to_yolo.py` | Convertit les annotations VOC GERALD au format YOLO |
| `extract_masts.py` | Extrait les crops de mâts pour entraîner un classifieur stage 2 |
| `stage1_yolo_pole.py` | Entraîne YOLOv8 pour la détection de mâts |
| `stage2_classifier.py` | Classifieur binaire has_panel / no_panel |
| `pipeline_a.py` / `pipeline_b.py` | Pipelines de détection comparatifs |
| `eval_pipelines.py` | Évaluation A vs B |

## Data Layout

- `data/surface/` — 5,153 images across 7 defect class subdirectories
- `data/normal_synthetic/` — images Normal générées
- `data/_raw/cabview_refs/` — frames de référence caméra frontale train
- `Images/` — video files (color_0.mkv, depth_0.mkv) and timing data (time_0.time)

## Language

Project documentation and code comments are in French.
