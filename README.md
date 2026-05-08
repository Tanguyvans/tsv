# TSV - Analyse de Défauts de Surface Ferroviaire

Outils d'analyse et de visualisation pour un dataset d'images de défauts de surface de rails, classés en 7 catégories.

## Prérequis

- Python 3.11
- Dépendances listées dans `requirements.txt`
- Clé API [fal.ai](https://fal.ai) (pour la génération d'images)

## Installation

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Configurer la clé fal.ai
cp .env.example .env
# Éditer .env et ajouter FAL_KEY=...
```

## Utilisation

### Lecteur vidéo

```bash
# Lire la vidéo couleur (par défaut)
python read_video.py

# Lire une vidéo spécifique
python read_video.py Images/depth_0.mkv
```

**Contrôles :**

| Touche  | Action                         |
|---------|--------------------------------|
| ESPACE  | Play / Pause                   |
| d       | Frame suivante (en pause)      |
| a       | Frame précédente (en pause)    |
| q       | Quitter                        |

## Génération d'images Normal

Le dataset ne contient pas de classe "Normal" (rail sans défaut). Les images Flakings contiennent des déchets (plastique, papier) sur le ballast autour du rail. Le pipeline les retire automatiquement pour créer des images Normal :

1. **SAM 3** (`fal-ai/sam-3/image`) détecte les déchets via prompts texte
2. **Bria Eraser** (`fal-ai/bria/eraser`) remplace les zones détectées par du ballast cohérent via Bria 2.3 + ControlNet Inpaint (training sur data licensée, remplissage de meilleure qualité que LaMa)

Le rail, les boulons et le ballast restent **pixel-identiques** à l'original.

```bash
# Générer 50 images Normal à partir des Flakings
PYTHONPATH=. python src/generation/generate_normal.py --src data/surface/Flakings --n 50

# Tester sur une seule image
PYTHONPATH=. python src/generation/generate_normal.py --src data/surface/Flakings/image.JPEG

# Sortie dans un dossier custom
PYTHONPATH=. python src/generation/generate_normal.py --src data/surface/Flakings --n 100 --out data/surface/Normal
```

## Structure du projet

```
tsv/
├── read_video.py              # Lecteur vidéo avec contrôles play/pause
├── src/
│   ├── generation/
│   │   ├── generate_normal.py # Pipeline Normal : SAM 3 + Bria Eraser
│   │   ├── compare_erasers.py # Compare LaMa, Bria, Nano Banana Pro
│   │   ├── mask_erase.py      # Helpers SAM 3 + Bria Eraser réutilisables
│   │   └── fal_wrapper.py     # Wrapper fal-client (cache, retry, logging)
│   ├── models/                # Classifieurs (EfficientNet, ViT, PrototypeNet)
│   ├── training/              # Scripts d'entraînement
│   ├── signals/               # Pipeline détection signaux (GERALD)
│   ├── cabview/                # Pipeline cabview YouTube par région (FR, CH)
│   ├── depth/                  # Tests Depth Anything 3 sur cabview
│   └── utils/                  # Helpers viz, métriques
├── configs/                   # Fichiers de configuration YAML
├── data/
│   ├── surface/               # 5 153 images de défauts (7 classes)
│   ├── normal_synthetic/      # Images Normal générées
│   ├── cabview/               # Vidéos cab-view + frames + refs (FR, CH)
│   └── gerald_augmented/      # Bare poles générés depuis GERALD
├── Images/                    # Vidéos (color_0.mkv, depth_0.mkv)
└── venv/                      # Environnement virtuel Python
```

## Dataset

### Statistiques

| Classe | Nom français | Images | Proportion |
|---|---|---|---|
| Flakings | Écaillages | 2 829 | 54.9% |
| Squats | Squats | 1 844 | 35.8% |
| Spallings | Déchets | 291 | 5.6% |
| Shellings | Décollements | 130 | 2.5% |
| Cracks | Fissures | 40 | 0.8% |
| Joints | Joints | 11 | 0.2% |
| Grooves | Rainures/Sillons | 8 | 0.2% |
| **Total** | | **5 153** | **100%** |

### Répartition

Le dataset est **fortement déséquilibré** :

- **Flakings** et **Squats** représentent **90.7%** des données.
- **Grooves**, **Joints** et **Cracks** sont très sous-représentées (< 1% chacune).
