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
| `mask_erase.py` | Helpers réutilisables (SAM 3 + Bria Eraser) |
| `compare_erasers.py` | Compare LaMa, Bria et Nano Banana Pro sur une image |
| `fal_wrapper.py` | Wrapper fal-client avec cache disque, retry et logging |

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
| `make_before_after.py` | Grilles de comparaison avant/après bare-pole generation |

## Pipeline cabview (dataset de signalisation par région)

Construit un dataset de panneaux de signalisation à partir de cab-view
YouTube par région (FR, CH...). Le pipeline filtre les frames pour ne
garder que les vraies vues depuis le train (rejette drones, intros, etc.).

```bash
# 1. Télécharger une vidéo listée dans sources.yaml
PYTHONPATH=. python src/cabview/download.py --region fr

# 2. Extraire les frames (1 fps par défaut)
PYTHONPATH=. python src/cabview/extract_frames.py --region fr --id bordeaux_nantes_2023

# 3. Filtrer pour ne garder que les cab-view (CLIP image-image vs refs)
PYTHONPATH=. python src/cabview/check_cabview.py \
    --frames data/cabview/fr/frames/bordeaux_nantes_2023 --copy

# 4. Catalogue HTML des icônes de signaux Wikimedia
PYTHONPATH=. python src/cabview/fetch_signal_icons.py
PYTHONPATH=. python src/cabview/build_catalog.py
```

Refs cab-view dans `data/cabview/_refs/{cabview,not_cabview}/` (ajouter
des images supplémentaires pour améliorer le classifieur).

| Script | Description |
|--------|-------------|
| `download.py` | Télécharge les vidéos YouTube listées dans `sources.yaml` |
| `extract_frames.py` | Extrait des frames via ffmpeg (fps configurable) |
| `check_cabview.py` | Filtre CLIP : compare aux refs cabview vs not_cabview |
| `fetch_signal_icons.py` | Scrape Wikimedia Commons pour les diagrammes de signaux |
| `build_catalog.py` | Génère un catalogue HTML des icônes de signaux |

**Note** : la détection automatique de panneaux (zero-shot via VLMs ou
détecteurs ouverts) a été testée et abandonnée — qualité insuffisante.
Pour une vraie détection, entraîner un YOLO sur le dataset FRSign
(100k images SNCF annotées, https://frsign.irt-systemx.fr/).

## Data Layout

- `data/surface/` — 5,153 images across 7 defect class subdirectories
- `data/normal_synthetic/` — images Normal générées
- `data/cabview/_refs/` — frames de référence pour le filtre cab-view (positives + négatives)
- `Images/` — video files (color_0.mkv, depth_0.mkv) and timing data (time_0.time)

## Language

Project documentation and code comments are in French.
