# TSV — Analyse Ferroviaire (Défauts de Surface + Signalisation)

Outils d'analyse et de génération de datasets ferroviaires :

1. **Surface defects** : classification des défauts de rail en 7 classes
2. **Bare poles** : génération de poteaux sans signal (à partir de GERALD)
3. **Cabview** : extraction de frames depuis cab-views YouTube par région (FR, CH...)

## Prérequis

- Python 3.11 + `pip install -r requirements.txt`
- Pour génération SAM 3 / Bria : clé fal.ai dans `.env` (`cp .env.example .env`)
- Pour cabview : `yt-dlp` et `ffmpeg` (auto via pip / brew)

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## 1. Pipeline Cabview (signalisation par région)

Construit un dataset de panneaux de signalisation à partir de cab-views YouTube
par région. Le pipeline filtre automatiquement les frames pour ne garder que
les vraies vues depuis le poste de conduite (rejette intros, drone shots,
slides, écrans noirs).

### Workflow

```
URL YouTube
    ↓ download.py
data/cabview/{region}/raw/{id}.mp4
    ↓ extract_frames.py (ffmpeg, fps=1.0)
data/cabview/{region}/frames/{id}/*.jpg
    ↓ check_cabview.py (CLIP image-image vs refs)
data/cabview/{region}/frames_cabview/{id}/*.jpg + _scores.json
```

### Utilisation rapide

```bash
# Pipeline complet en une commande (download → extract → filter)
PYTHONPATH=. python src/cabview/process_video.py --region fr --id bordeaux_nantes_2023 --copy

# Ou étape par étape
PYTHONPATH=. python src/cabview/download.py        --region fr --id bordeaux_nantes_2023
PYTHONPATH=. python src/cabview/extract_frames.py  --region fr --id bordeaux_nantes_2023 --fps 1.0
PYTHONPATH=. python src/cabview/check_cabview.py   --frames data/cabview/fr/frames/bordeaux_nantes_2023 --copy
```

### Ajouter une nouvelle région ou vidéo

1. Éditer `data/cabview/{region}/sources.yaml` :
   ```yaml
   videos:
     - id: my_video_id
       url: https://www.youtube.com/watch?v=XXXXXXXXX
       title: "..."
       duration_sec: 1800
   ```
2. Lancer `process_video.py --region X --id my_video_id --copy`

### Refs cab-view (configurables)

Le filtre CLIP compare chaque frame à un dossier de refs positives (vraies
cab-views) vs négatives (drones, intros, slides, etc.) :

```
data/cabview/_refs/
├── cabview/                       # Refs positives
│   ├── bordeaux_nantes_*.jpg      # Cab-view classique
│   ├── cabview_countryside.jpg    # Variante campagne
│   ├── cabview_grille.jpg         # Avec grille pare-soleil
│   ├── cabview_station_*.jpg      # En gare
│   └── cabview_station_with_trains.jpg  # Avec trains adjacents
└── not_cabview/                   # Refs négatives
    ├── bordeaux_nantes_000093.jpg # Drone shot type 1
    ├── drone_train_2.jpg          # Drone shot type 2
    ├── intro_map_ch.jpg           # Carte intro
    ├── slide_credits_text.jpg     # Slide outro
    └── black_screen.jpg           # Écran noir technique
```

Pour améliorer le filtre : ajouter d'autres refs dans ces dossiers.

### Performance mesurée

| Vidéo | Durée | Frames | Cab-view | Précision |
|---|---|---|---|---|
| Bordeaux-Nantes (FR) | 4h05 | 14 560 | 14 538 | 99.8% |
| Fribourg-Ins (CH) | 37 min | 2 245 | 2 203 | 98.1% |

### Catalogue de signaux (référence visuelle)

```bash
PYTHONPATH=. python src/cabview/fetch_signal_icons.py
PYTHONPATH=. python src/cabview/build_catalog.py
# Ouvrir : data/signals_ref/catalog.html
```

Récupère 201 SVG SNCF + 53 SVG CFF depuis Wikimedia Commons.

### Modules

| Script | Description |
|---|---|
| `process_video.py` | **Wrapper end-to-end** (download → extract → filter) |
| `download.py` | Télécharge les vidéos YouTube de `sources.yaml` (yt-dlp) |
| `extract_frames.py` | Extrait des frames via ffmpeg (fps configurable) |
| `check_cabview.py` | Filtre CLIP image-image (cabview vs not_cabview) |
| `fetch_signal_icons.py` | Scrape Wikimedia Commons pour les diagrammes |
| `build_catalog.py` | Génère un catalogue HTML des icônes |

> **Note** : la détection automatique de panneaux (zero-shot via VLMs ou
> détecteurs ouverts type GroundingDINO/Florence-2/Moondream) a été testée
> et abandonnée — qualité insuffisante. La voie sérieuse passe par
> l'entraînement d'un YOLO sur GERALD (déjà dans le projet) ou
> [FRSign](https://frsign.irt-systemx.fr/) (289 GB, demande accès).

---

## 2. Pipeline Bare Poles (signaux retirés)

Génère des images de "poteaux nus" (panneau retiré) à partir du dataset
GERALD allemand. Sert de classe synthétique pour entraîner un modèle
qui détecte les panneaux tombés / manquants.

```bash
# Télécharger GERALD (~4.2 GB)
PYTHONPATH=. python src/signals/download_gerald.py

# Générer N bare poles (SAM 3 + Bria Eraser)
PYTHONPATH=. python src/signals/generate_bare_poles.py --n 50
```

**Pipeline :**
1. Parse les bboxes VOC des classes de signaux (Ks, Hp, Vr, Zs, Ne, Lf, Signal_*)
2. SAM 3 avec box prompts raffine les masks des panneaux
3. Bria Eraser retire les panneaux (poteaux préservés pixel-identiques)
4. Génère les labels YOLO du bare_pole (classe 0) via heuristique

Sortie : `data/gerald_augmented/bare_poles/{images,labels}/`.

**Modules `src/signals/`** : `download_gerald`, `generate_bare_poles`,
`voc_to_yolo`, `extract_masts`, `stage1_yolo_pole`, `stage2_classifier`,
`pipeline_a/b`, `eval_pipelines`, `make_before_after`.

---

## 3. Génération Normal (défauts retirés)

Le dataset surface ne contient pas de classe "Normal" (rail sans défaut).
Le pipeline retire les déchets (plastique, papier) des images Flakings
pour créer des images Normal :

1. **SAM 3** (`fal-ai/sam-3/image`) détecte les déchets via prompts texte
2. **Bria Eraser** (`fal-ai/bria/eraser`) remplace par du ballast cohérent

Le rail, les boulons et le ballast restent **pixel-identiques** à l'original.

```bash
# Générer 50 images Normal à partir des Flakings
PYTHONPATH=. python src/generation/generate_normal.py --src data/surface/Flakings --n 50

# Tester sur une seule image
PYTHONPATH=. python src/generation/generate_normal.py --src data/surface/Flakings/image.JPEG
```

Sortie dans `data/normal_synthetic/Normal/` par défaut, masks de debug
dans `_debug/`.

**Modules `src/generation/`** : `generate_normal`, `mask_erase` (helpers
SAM 3 + Bria), `compare_erasers` (LaMa vs Bria vs Nano Banana Pro),
`fal_wrapper` (cache disque + retry + logging).

---

## 4. Lecteur vidéo (utility)

```bash
python read_video.py                  # vidéo couleur par défaut
python read_video.py Images/depth_0.mkv
```

| Touche | Action |
|---|---|
| ESPACE | Play / Pause |
| d / a | Frame suivante / précédente (en pause) |
| q | Quitter |

---

## Structure du projet

```
tsv/
├── read_video.py
├── src/
│   ├── cabview/              # Pipeline cabview YouTube par région
│   ├── signals/              # GERALD bare poles + YOLO multi-stage
│   ├── generation/           # SAM 3 + Bria Eraser (Normal class)
│   ├── data/                 # Dataset surface (RSDDs splits)
│   ├── models/               # Classifieurs (EfficientNet, ViT, PrototypeNet)
│   ├── training/             # Train + evaluate surface defects
│   ├── depth/                # Tests Depth Anything 3 sur cabview
│   └── utils/                # viz (grilles d'images), metrics
├── configs/                  # YAML configs (train, GERALD)
├── data/
│   ├── cabview/              # Vidéos + frames + refs par région (FR, CH)
│   ├── surface/              # 5 153 images défauts (7 classes)
│   ├── normal_synthetic/     # Images Normal générées
│   ├── gerald_augmented/     # Bare poles générés
│   ├── signals_ref/          # Catalogue SVG signaux SNCF + CFF
│   ├── splits/               # CSV train/val/test surface
│   └── _raw/                 # Datasets bruts (GERALD, RSDDs)
├── Images/                   # Vidéos couleur + profondeur
├── outputs/                  # Logs, comparaisons, fal cache
└── requirements.txt
```

---

## Dataset Surface

### Statistiques

| Classe | Nom français | Images | Proportion |
|---|---|---|---|
| Flakings | Écaillages | 2 829 | 54.9% |
| Squats | Squats | 1 844 | 35.8% |
| Spallings | Déchets | 291 | 5.6% |
| Shellings | Décollements | 130 | 2.5% |
| Cracks | Fissures | 40 | 0.8% |
| Joints | Joints | 11 | 0.2% |
| Grooves | Rainures | 8 | 0.2% |
| **Total** | | **5 153** | **100%** |

Dataset **fortement déséquilibré** : Flakings + Squats = 90.7%. Trois
classes (Grooves, Joints, Cracks) sont très sous-représentées (<1%
chacune), ce qui justifie les pipelines de génération synthétique
(`src/generation/`) et few-shot learning (`src/models/prototype_net.py`).

---

## Statut & next steps

- ✅ **Cabview pipeline** : opérationnel, validé FR + CH (98%+ précision filtre)
- ✅ **Bare poles** : pipeline GERALD complet, ~5000 exemples générés
- ✅ **Normal generation** : SAM 3 + Bria fonctionnel
- 🔄 **Détection panneaux** : à faire — entraîner YOLO sur GERALD comme
  baseline, puis évaluer transfert FR/CH
- 🔄 **Surface classification** : splits prêts, training à lancer
- 🔬 **Depth Anything 3** : exploration en cours (`src/depth/`)
