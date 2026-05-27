# Pipeline de détection et classification des signaux ferroviaires

## Vue d'ensemble

```
Vidéo cab-view YouTube
        │
        ▼
  extract_frames.py          (ffmpeg, 1 fps)
        │
        ▼
  check_cabview.py           (filtre CLIP — rejette drones / intros / écrans noirs)
        │
        ▼
 Frames cab-view filtrées
        │
        ▼
  detect_yolo26.py           (YOLOv26-s pré-entraîné — détecte "il y a un panneau ici")
        │        │
        │        └──► viewer HTML  (frames détectées vs manquées)
        ▼
  Crops de panneaux (bbox + marge)
        │
        ▼
  match_signals.py  ◄─────── Corpus Wikimedia SVG (SNCF + CFF)   [EN COURS]
        │                     fetch_signal_icons.py + build_catalog.py
        │                     201 SVG France · 53 SVG Suisse
        ▼
  Top-K matches CLIP
  viewer HTML : crop → top-5 signaux candidats
```

---

## Étape 1 — Acquisition et filtrage des frames

```bash
# Télécharger une vidéo cab-view YouTube
PYTHONPATH=. python src/cabview/download.py --region fr --id bordeaux_nantes_2023

# Extraire les frames (1 fps)
PYTHONPATH=. python src/cabview/extract_frames.py --region fr --id bordeaux_nantes_2023

# Filtrer : ne garder que les vraies vues cab-view
PYTHONPATH=. python src/cabview/check_cabview.py \
    --frames data/cabview/fr/frames/bordeaux_nantes_2023 --copy

# Ou tout en une commande
PYTHONPATH=. python src/cabview/process_video.py --region fr --id bordeaux_nantes_2023 --copy
```

Le filtre compare chaque frame aux refs `data/cabview/_refs/` par similarité CLIP
(cabview vs drones / cartes / slides / écrans noirs).

| Vidéo | Frames | Cab-view conservées | Précision |
|---|---|---|---|
| Bordeaux → Nantes (FR) | 14 560 | 14 538 | **99.8 %** |
| Fribourg → Ins (CH) | 2 245 | 2 203 | **98.1 %** |

---

## Étape 2 — Détection de panneaux (YOLOv26)

Modèle : [`Otmane42/yolo26s-railway-signs-detector`](https://huggingface.co/Otmane42/yolo26s-railway-signs-detector)
Single-class agnostique — retourne uniquement des bboxes "panneau" sans classification fine.

```bash
# Télécharger les poids (~20 MB)
git clone https://huggingface.co/Otmane42/yolo26s-railway-signs-detector \
    models/yolo26s-railway-signs-detector

# Détecter sur les frames cab-view
PYTHONPATH=. python src/cabview/detect_yolo26.py \
    --frames data/cabview/fr/frames_cabview/bordeaux_nantes_2023 \
    --conf 0.25 --imgsz 960

# Visualiser (HTML : frames détectées vs sans détection)
PYTHONPATH=. python src/cabview/build_viewer.py \
    --frames data/cabview/fr/frames_cabview/bordeaux_nantes_2023 \
    --detections data/cabview/fr/detections/bordeaux_nantes_2023
# → ouvrir data/cabview/fr/detections/bordeaux_nantes_2023/viewer.html
```

| Vidéo | Frames cab-view | Avec ≥1 signal | Vitesse (CPU M3 Pro) |
|---|---|---|---|
| Bordeaux → Nantes (FR) | 14 529 | **779 (5.4 %)** | ~10 min |
| Fribourg → Ins (CH) | 2 220 | **420 (18.9 %)** | ~1.5 min |

Le modèle entraîné sur données SNCF transfère bien aux signaux CFF suisses.

### Exemples de détections (Bordeaux → Nantes)

<table>
<tr>
  <td><img src="data/cabview/fr/detections/bordeaux_nantes_2023/_thumbs/det_bordeaux_nantes_2023_000049.jpg" width="220"/><br><sub>Campagne, signal distant gauche</sub></td>
  <td><img src="data/cabview/fr/detections/bordeaux_nantes_2023/_thumbs/det_bordeaux_nantes_2023_000974.jpg" width="220"/><br><sub>Panneau vitesse 40 en forêt</sub></td>
  <td><img src="data/cabview/fr/detections/bordeaux_nantes_2023/_thumbs/det_bordeaux_nantes_2023_001023.jpg" width="220"/><br><sub>Viaduc, signal Z</sub></td>
</tr>
<tr>
  <td><img src="data/cabview/fr/detections/bordeaux_nantes_2023/_thumbs/det_bordeaux_nantes_2023_004441.jpg" width="220"/><br><sub>Marqueur 515 km, bord de voie</sub></td>
  <td><img src="data/cabview/fr/detections/bordeaux_nantes_2023/_thumbs/det_bordeaux_nantes_2023_005784.jpg" width="220"/><br><sub>Approche de gare, signal carré</sub></td>
  <td><img src="data/cabview/fr/detections/bordeaux_nantes_2023/_thumbs/det_bordeaux_nantes_2023_007499.jpg" width="220"/><br><sub>Zone urbaine, feu à droite</sub></td>
</tr>
<tr>
  <td><img src="data/cabview/fr/detections/bordeaux_nantes_2023/_thumbs/det_bordeaux_nantes_2023_010620.jpg" width="220"/><br><sub>Passage à niveau, signal droit</sub></td>
  <td colspan="2"></td>
</tr>
</table>

---

## Étape 3 — Corpus de référence Wikimedia `[EN COURS]`

### Construction du corpus

```bash
# Télécharger les SVG depuis Wikimedia Commons
PYTHONPATH=. python src/cabview/fetch_signal_icons.py --region fr   # SNCF
PYTHONPATH=. python src/cabview/fetch_signal_icons.py --region ch   # CFF

# Catalogue HTML de navigation visuelle
PYTHONPATH=. python src/cabview/build_catalog.py
# → ouvrir data/signals_ref/catalog.html
```

| Région | Source Wikimedia | SVG téléchargés |
|---|---|---|
| France (SNCF) | *Diagrams of railway signals in France* | **201** |
| Suisse (CFF) | *Diagrams of railway signals in Switzerland* | **53** |

### Matching CLIP (crop YOLO → catalogue)

```bash
PYTHONPATH=. python src/cabview/match_signals.py \
    --frames data/cabview/fr/frames_cabview/bordeaux_nantes_2023 \
    --detections data/cabview/fr/detections/bordeaux_nantes_2023 \
    --refs-dir data/signals_ref/fr_diagrams \
    --conf-min 0.3 --top-k 5
# → ouvrir data/cabview/fr/matches/bordeaux_nantes_2023/viewer.html
```

**Principe :**
1. Les SVG du catalogue sont rendu en PNG (cache disque) et encodés par CLIP
2. Chaque crop YOLO est encodé par CLIP
3. Similarité cosinus crop ↔ catalogue → top-K signaux candidats
4. Viewer HTML : crop détecté + 5 signaux candidats avec score de similarité

---

## Statut

| Étape | Statut |
|---|---|
| Téléchargement vidéos YouTube | ✅ opérationnel |
| Extraction + filtrage cab-view | ✅ validé FR + CH (≥98 %) |
| Détection panneaux YOLOv26 | ✅ bbox correctes FR + CH |
| Corpus SVG Wikimedia | ✅ 201 SNCF + 53 CFF téléchargés |
| Matching CLIP crop → catalogue | 🔄 en cours d'évaluation |
| Classification fine des panneaux | ⬜ à faire |

---

## Structure des données

```
data/
├── cabview/
│   ├── fr/
│   │   ├── sources.yaml
│   │   ├── raw/                    vidéos .mp4
│   │   ├── frames/                 toutes les frames extraites
│   │   ├── frames_cabview/         frames filtrées cab-view
│   │   ├── detections/             JSONs YOLOv26 + viewer.html
│   │   └── matches/                résultats CLIP + viewer.html
│   ├── ch/                         idem pour la Suisse
│   └── _refs/
│       ├── cabview/                images de référence positives
│       └── not_cabview/            images de référence négatives
└── signals_ref/
    ├── fr_diagrams/                201 SVG SNCF
    ├── ch_diagrams/                53 SVG CFF
    └── catalog.html                catalogue visuel
```

---

## Modules

| Script | Description |
|---|---|
| `src/cabview/process_video.py` | Wrapper end-to-end (download → extract → filter) |
| `src/cabview/download.py` | Téléchargement vidéos YouTube (yt-dlp) |
| `src/cabview/extract_frames.py` | Extraction frames (ffmpeg) |
| `src/cabview/check_cabview.py` | Filtre CLIP image-image |
| `src/cabview/detect_yolo26.py` | Détection panneaux YOLOv26 |
| `src/cabview/build_viewer.py` | Viewer HTML détections |
| `src/cabview/fetch_signal_icons.py` | Scrape Wikimedia Commons |
| `src/cabview/build_catalog.py` | Catalogue HTML des SVG |
| `src/cabview/match_signals.py` | Matching CLIP crop → catalogue |

## Prérequis

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
# yt-dlp et ffmpeg sont inclus dans requirements.txt
```

---

## Autres pipelines du repo

- **Bare poles** (`src/signals/`) — retire les panneaux des images GERALD pour simuler des poteaux tombés
- **Normal generation** (`src/generation/`) — génère des images "Normal" en effaçant les déchets des Flakings via SAM 3 + Bria Eraser
- **Surface defects** (`src/data/`, `src/models/`) — classification de 5 153 images de défauts rail en 7 classes
