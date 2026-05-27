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

---

## Pipeline 2 — Estimation de distance par profondeur monoculaire

Compare la distance estimée par 3 modèles de depth estimation contre la référence stéréo **MultiSense M1** (fx = 1288.33 px).

### Méthodes comparées

| Méthode | Modèle | Type | Référence |
|---|---|---|---|
| **msense** | MultiSense M1 (stéréo) | Métrique (référence) | Caméra embarquée |
| **dav2** | Depth Anything V2 Metric Outdoor Small | Métrique (mètres) | HuggingFace `depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf` |
| **da3** | Depth Anything 3 Metric Large | Métrique (mètres) | HuggingFace `depth-anything/DA3METRIC-LARGE-1.1` |
| **vda** | Video Depth Anything ViT-S | Relatif (sans échelle) | GitHub `DepthAnything/Video-Depth-Anything` |

### Résultats (560 frames, 28 séquences, fx = 1288.33 px)

| Méthode | Distance médiane | Erreur vs msense |
|---|---|---|
| msense | ~15 m | — (référence) |
| **dav2** | 3.24 m | **23 %** ✅ meilleur |
| da3 | 26.4 m | 153 % |
| vda | relatif | non comparable |

### Lancer le benchmark (GPU, cluster Lucia)

```bash
# Soumettre le job SLURM
sbatch benchmark_gpu_focal.sh

# Ou en interactif
PYTHONPATH=. python src/depth/benchmark_msense_vs_depth_anything.py \
  --data-root $DATA \
  --ckpt yolo_signs_best.pt \
  --seqs 0,1,2,3 --start 0 --end 200 --stride 5 \
  --conf 0.05 \
  --methods msense,dav2,da3 \
  --focal-px 1288.33 \
  --out depth_benchmark_focal_full
```

### Générer la vidéo de comparaison

```bash
PYTHONPATH=. python src/depth/make_method_comparison_video.py \
  --benchmark depth_benchmark_focal_full \
  --fps 10 --cell-w 960 --cell-h 540
# → depth_benchmark_focal_full/comparison_depth_methods/*.mp4
```

Layout : grille 2×2 (1920×1080) — msense avec bbox, depth colormaps pour dav2/da3/vda.

### Modules depth

| Script | Description |
|---|---|
| `src/depth/benchmark_msense_vs_depth_anything.py` | Orchestrateur principal — lance toutes les méthodes |
| `src/depth/estimate_dav2_distance.py` | Inférence Depth Anything V2 |
| `src/depth/estimate_da3_distance.py` | Inférence Depth Anything 3 |
| `src/depth/estimate_vda_distance.py` | Inférence Video Depth Anything |
| `src/depth/benchmark_depth_methods.py` | Agrégation et métriques |
| `src/depth/make_method_comparison_video.py` | Vidéo de comparaison 2×2 |
| `src/depth/track_distance_timeseries.py` | Courbe de distance dans le temps |

---

## Pipeline 3 — Détection de signaux GERALD

Pipeline deux étapes sur le dataset GERALD (images de signaux ferroviaires annotés VOC) :

```
Images GERALD (VOC)
      │
      ▼
stage1_yolo_pole.py    Détecte les mâts (YOLO)
      │
      ▼
stage2_classifier.py   Classifie has_panel / no_panel (EfficientNet-B0)
```

### Modules signals

| Script | Description |
|---|---|
| `src/signals/download_gerald.py` | Téléchargement dataset GERALD |
| `src/signals/stage1_yolo_pole.py` | Détection mâts (YOLO) |
| `src/signals/stage2_classifier.py` | Classification panneau (EfficientNet) |
| `src/signals/pipeline_a.py` | Pipeline YOLO → EfficientNet end-to-end |
| `src/signals/pipeline_b.py` | Variante pipeline |
| `src/signals/eval_pipelines.py` | Évaluation comparative A vs B |
| `src/signals/extract_masts.py` | Extraction crops de mâts |
| `src/signals/voc_to_yolo.py` | Conversion annotations VOC → YOLO |
| `src/signals/estimate_msense_signal_distance.py` | Distance signal via MSense |
| `src/signals/make_before_after.py` | Visualisation avant/après |
