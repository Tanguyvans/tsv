# TSV - Analyse de Défauts de Surface Ferroviaire

Outils d'analyse et de visualisation pour un dataset d'images de défauts de surface de rails, classés en 7 catégories.

## Prérequis

- Python 3.11
- OpenCV (`opencv-python`)
- NumPy

## Installation

```bash
python -m venv venv
source venv/bin/activate
pip install opencv-python numpy
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

## Structure du projet

```
tsv/
├── read_video.py          # Lecteur vidéo avec contrôles play/pause
├── data/
│   └── surface/           # 5 153 images de défauts (7 classes)
├── Images/
│   ├── color_0.mkv        # Vidéo couleur
│   ├── depth_0.mkv        # Vidéo de profondeur
│   └── time_0.time        # Données de synchronisation
└── venv/                  # Environnement virtuel Python
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
