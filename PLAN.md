# Plan de Projet : Classification et Detection de Defauts de Rails

## 1. Etat des Lieux

### Dataset actuel
- **5 153 images** (1280x720, JPEG) reparties en 7 classes
- **Desequilibre severe** : Flakings+Squats = 90.7%, les 5 autres classes = 9.3%
- Donnees video disponibles (couleur + profondeur) avec synchronisation temporelle
- Aucun code de ML/DL existant, uniquement un lecteur video

### Problemes identifies
1. Classes minoritaires extremement sous-representees (Grooves: 8, Joints: 11, Cracks: 40)
2. Pas de split train/val/test
3. Pas d'annotations de localisation (bounding boxes ou masques de segmentation)
4. Pas de pipeline de preprocessing

---

## 2. Architecture du Projet

```
tsv/
├── configs/                    # Fichiers de configuration (hyperparametres, chemins)
│   ├── train_config.yaml
│   └── augmentation_config.yaml
├── src/
│   ├── data/
│   │   ├── dataset.py          # Dataset PyTorch + DataLoader
│   │   ├── preprocessing.py    # Redimensionnement, normalisation
│   │   ├── augmentation.py     # Augmentations classiques + avancees
│   │   ├── split.py            # Stratified split train/val/test
│   │   └── balance.py          # Strategies de reequilibrage
│   ├── models/
│   │   ├── classifier.py       # Modeles de classification (ResNet, EfficientNet)
│   │   ├── detector.py         # Modeles de detection (YOLOv8, Faster R-CNN)
│   │   └── segmentor.py        # Modeles de segmentation (U-Net, DeepLabV3)
│   ├── training/
│   │   ├── train.py            # Boucle d'entrainement
│   │   ├── evaluate.py         # Metriques et evaluation
│   │   └── callbacks.py        # Early stopping, checkpointing
│   ├── generation/
│   │   ├── diffusion.py        # Generation par diffusion (Stable Diffusion)
│   │   ├── cutmix_mosaic.py    # CutMix, MixUp, Mosaic
│   │   └── style_transfer.py   # Transfert de defauts entre images
│   └── utils/
│       ├── visualization.py    # Affichage des resultats
│       └── metrics.py          # Metriques adaptees au desequilibre
├── notebooks/
│   ├── 01_eda.ipynb            # Analyse exploratoire
│   ├── 02_augmentation.ipynb   # Visualisation des augmentations
│   ├── 03_training.ipynb       # Entrainement interactif
│   └── 04_results.ipynb        # Analyse des resultats
├── data/
│   └── surface/                # Dataset existant
├── outputs/
│   ├── models/                 # Poids sauvegardes
│   ├── logs/                   # Logs TensorBoard/W&B
│   └── predictions/            # Predictions et visualisations
```

---

## 3. Phases du Projet

### Phase 1 : Analyse Exploratoire et Preparation (Semaine 1)

**Objectif** : Comprendre les donnees et preparer l'infrastructure

**Taches** :
1. **EDA (Exploratory Data Analysis)**
   - Distribution des classes (histogramme)
   - Analyse des dimensions, couleurs moyennes, variance par classe
   - Visualisation d'echantillons representatifs par classe
   - Analyse des similarites inter-classes (t-SNE sur features pretrained)

2. **Split stratifie**
   - Train 70% / Val 15% / Test 15%
   - Stratification par classe pour preserver les proportions
   - Pour les classes tres petites (Grooves, Joints) : utiliser k-fold cross-validation

3. **Pipeline de preprocessing**
   - Redimensionnement : 224x224 (classification) ou 640x640 (detection)
   - Normalisation ImageNet (mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
   - Dataset PyTorch avec DataLoader multi-worker

**Dependances** : `torch`, `torchvision`, `albumentations`, `scikit-learn`, `matplotlib`, `seaborn`

---

### Phase 2 : Enrichissement du Dataset (Semaines 2-3)

**Objectif** : Augmenter les classes minoritaires pour atteindre un minimum de ~500 images par classe

#### 2A. Augmentations classiques (rapide, fiable)

Utiliser **Albumentations** pour :
- Rotations (0, 90, 180, 270 degres) — les defauts vus du dessus sont independants de l'orientation
- Flips horizontaux et verticaux
- Ajustements de luminosite/contraste (simule differentes conditions d'eclairage)
- Ajout de bruit gaussien (simule le bruit du capteur)
- Flou gaussien leger
- Elastic transform (deformations realistes des defauts)
- CLAHE (amelioration locale du contraste)

**Multiplicateur attendu** : x4 a x8 pour les classes minoritaires

#### 2B. Augmentations avancees (CutMix, MixUp, Mosaic)

- **CutMix** : Decouper une region d'une image de defaut et la coller sur une autre
- **MixUp** : Melange lineaire de deux images avec leurs labels
- **Mosaic** : Combiner 4 images en une seule (technique YOLOv4)
- **Copy-Paste** : Isoler un defaut et le copier sur un fond de rail sain

#### 2C. Generation par IA (pour les classes < 50 images)

**Approche recommandee : Fine-tuning de Stable Diffusion**

1. **Stable Diffusion + LoRA** (Low-Rank Adaptation)
   - Fine-tuner un LoRA par classe minoritaire (Cracks, Joints, Grooves)
   - Prompt : "top view photograph of railway rail surface showing [defect_type] defect"
   - ~20-30 images suffisent pour un LoRA de qualite
   - Generer 200-500 images par classe
   - **Outil** : `diffusers` (HuggingFace) ou `kohya_ss` pour le fine-tuning

2. **Filtrage qualite des images generees**
   - Verification manuelle d'un echantillon
   - Filtrage automatique via un classifieur pre-entraine sur les vraies images
   - FID (Frechet Inception Distance) pour mesurer la qualite globale

3. **Alternative : DreamBooth**
   - Plus lourd mais meilleure fidelite au style des images originales
   - A utiliser si LoRA donne des resultats insuffisants

#### 2D. Transposition de defauts (segmentation + collage)

**Principe** : Segmenter le defaut sur une image, l'extraire, et le transposer sur une autre zone de rail

1. **Segmentation du defaut** (semi-automatique)
   - Utiliser SAM (Segment Anything Model) pour isoler les defauts
   - Ou annotation manuelle avec LabelMe/CVAT pour les classes rares
   - Obtenir des masques binaires defaut/fond

2. **Transposition**
   - Extraire le defaut segmente
   - Le placer sur une image de rail saine ou avec un autre defaut
   - Harmonisation des bords via Poisson blending (cv2.seamlessClone)
   - Varier position, echelle et rotation

**Avantage** : Genere des images tres realistes car le defaut est reel

#### Objectif de repartition apres enrichissement

| Classe | Avant | Apres (cible) | Methode principale |
|--------|-------|---------------|-------------------|
| Flakings | 2829 | 2829 (inchange) | — |
| Squats | 1844 | 1844 (inchange) | — |
| Spallings | 291 | ~800 | Augmentation classique |
| Shellings | 130 | ~600 | Augmentation + CutMix |
| Cracks | 40 | ~500 | Generation IA + transposition |
| Joints | 11 | ~400 | Generation IA + transposition |
| Grooves | 8 | ~400 | Generation IA + transposition |

---

### Phase 3 : Classification (Semaines 3-5)

**Objectif** : Classifier les images dans les 7 classes de defauts

#### 3A. Approche baseline : Transfer Learning

1. **Modele** : EfficientNet-B3 ou ResNet-50, pre-entraine sur ImageNet
2. **Fine-tuning progressif** :
   - Etape 1 : Geler le backbone, entrainer uniquement la tete de classification (5-10 epochs, lr=1e-3)
   - Etape 2 : Degeler les dernieres couches, fine-tuner (20-30 epochs, lr=1e-4)
   - Etape 3 : Degeler tout, fine-tuner avec lr tres faible (10 epochs, lr=1e-5)

3. **Gestion du desequilibre** (cumuler plusieurs techniques) :
   - **Weighted CrossEntropy** : poids inversement proportionnels a la frequence
   - **Focal Loss** : penalise plus les exemples mal classifies (gamma=2)
   - **Oversampling** : WeightedRandomSampler de PyTorch
   - **Class-balanced sampling** : chaque batch contient des echantillons de chaque classe

4. **Hyperparametres** :
   - Optimizer : AdamW (lr=1e-4, weight_decay=1e-4)
   - Scheduler : CosineAnnealingWarmRestarts
   - Batch size : 32
   - Image size : 224x224 (EfficientNet) ou 256x256
   - Epochs : 50 avec early stopping (patience=10)

#### 3B. Approche avancee : Vision Transformers

- **ViT-B/16** ou **DeiT** si le dataset enrichi est suffisant (>5000 images)
- **Swin Transformer** : bon compromis performance/cout
- Necessitent plus de donnees que les CNN

#### 3C. Ensemble

- Combiner EfficientNet + ResNet + ViT par vote majoritaire ou stacking
- Chaque modele peut capturer des patterns differents

#### Metriques d'evaluation

- **Macro F1-Score** (metrique principale — traite chaque classe egalement)
- **Weighted F1-Score**
- **Matrice de confusion**
- **Precision/Recall par classe**
- **AUC-ROC multi-classe**
- Ne PAS utiliser l'accuracy seule (trompeuse avec desequilibre)

---

### Phase 4 : Detection d'Objets (Semaines 5-7)

**Objectif** : Localiser les defauts dans l'image avec des bounding boxes

**Prerequis** : Annotation des bounding boxes (travail le plus long)

#### 4A. Annotation

1. **Outil recommande** : CVAT (open-source, supporte YOLO/COCO formats)
   - Alternative : LabelImg, Roboflow, Label Studio
2. **Strategie** : Annoter au minimum 200-300 images par classe
3. **Format** : YOLO (txt) ou COCO (json)
4. **Astuce** : Utiliser SAM pour pre-annoter automatiquement, puis corriger manuellement

#### 4B. Modeles

1. **YOLOv8** (recommande en premier)
   - Rapide a entrainer, bons resultats out-of-the-box
   - `ultralytics` pip package
   - Pre-entraine sur COCO, fine-tuner sur notre dataset
   - Variantes : YOLOv8n (nano, rapide) a YOLOv8x (extra-large, precis)

2. **Faster R-CNN** (plus precis, plus lent)
   - Backbone : ResNet-50-FPN
   - Meilleur pour les petits defauts
   - Via `torchvision.models.detection`

3. **DETR / RT-DETR** (transformer-based)
   - Approche end-to-end sans NMS
   - Bon pour les defauts de formes variees

#### Metriques
- mAP@0.5 (mean Average Precision)
- mAP@0.5:0.95
- Precision/Recall par classe

---

### Phase 5 : Segmentation (Semaines 7-9)

**Objectif** : Segmenter pixel par pixel les zones de defaut

#### 5A. Annotation de segmentation

1. **Outil** : CVAT (polygones) ou SAM (semi-automatique)
2. **Format** : Masques PNG (1 canal, valeur = id de classe) ou COCO segmentation
3. **Strategie** :
   - Utiliser SAM pour generer des masques initiaux
   - Correction manuelle des masques
   - Au minimum 100-200 masques annotes par classe

#### 5B. Modeles

1. **U-Net** (baseline solide)
   - Encoder : ResNet-34 ou EfficientNet pre-entraine
   - Decoder : upsampling classique
   - Bibliotheque : `segmentation_models_pytorch`

2. **DeepLabV3+**
   - Meilleur pour les contours precis (Atrous Spatial Pyramid Pooling)
   - Backbone : ResNet-101 ou Xception

3. **Mask R-CNN** (segmentation d'instance)
   - Combine detection + segmentation
   - Distingue les instances individuelles de defauts
   - Via `detectron2` (Meta)

#### 5C. Transposition basee sur la segmentation

Utiliser les masques de segmentation pour :
1. Extraire le defaut segmente d'une image source
2. Le placer sur une image cible (rail sain ou autre)
3. Harmoniser avec Poisson blending
4. Cela cree des donnees d'entrainement supplementaires ET des masques de segmentation associes automatiquement

#### Metriques
- **mIoU** (mean Intersection over Union) — metrique principale
- **Dice Score** (equivalent au F1 pixel-wise)
- IoU par classe

---

### Phase 6 : Exploitation des Donnees Video/Profondeur (Semaine 9-10)

**Objectif** : Exploiter les videos couleur + profondeur pour enrichir les donnees

1. **Extraction de frames** depuis color_0.mkv et depth_0.mkv
   - Synchroniser via time_0.time
   - Extraire les frames contenant des defauts visibles

2. **Fusion couleur + profondeur**
   - Les images de profondeur revelent la geometrie 3D des defauts
   - Creer des images 4 canaux (RGB + Depth)
   - Ou utiliser la profondeur comme canal supplementaire dans le modele

3. **Profil de profondeur** pour la classification
   - Les defauts ont des signatures de profondeur distinctes
   - Cracks : creux lineaires
   - Squats : depressions localisees
   - Peut servir de feature supplementaire

---

### Phase 7 : Optimisation et Deploiement (Semaine 10-11)

1. **Optimisation**
   - Test Time Augmentation (TTA) : augmenter l'image au moment de l'inference
   - Ensemble de modeles (vote ou stacking)
   - Optimisation des hyperparametres avec Optuna

2. **Export**
   - ONNX pour portabilite
   - TensorRT pour inference rapide sur GPU
   - TorchScript pour deploiement

3. **Pipeline d'inference**
   - Charger une image ou un flux video
   - Preprocessing → Modele → Post-processing → Visualisation
   - Mesurer le temps d'inference (objectif : <100ms/image)

---

## 4. Dependances a Installer

```bash
# Framework principal
pip install torch torchvision torchaudio

# Augmentation
pip install albumentations

# Modeles pre-entraines
pip install timm                     # EfficientNet, ViT, Swin, etc.
pip install segmentation-models-pytorch  # U-Net, DeepLabV3, etc.
pip install ultralytics              # YOLOv8

# Generation d'images
pip install diffusers transformers accelerate  # Stable Diffusion
pip install segment-anything         # SAM

# Evaluation et visualisation
pip install scikit-learn
pip install matplotlib seaborn
pip install tensorboard              # ou wandb

# Annotation
pip install labelme                  # Annotation locale

# Optimisation
pip install optuna                   # Hyperparameter tuning
pip install onnx onnxruntime         # Export ONNX
```

---

## 5. Planning Resume

| Phase | Semaine | Priorite | Effort |
|-------|---------|----------|--------|
| 1. EDA + Preparation | S1 | Critique | Faible |
| 2. Enrichissement dataset | S2-S3 | Critique | Eleve |
| 3. Classification | S3-S5 | Haute | Moyen |
| 4. Detection (bounding boxes) | S5-S7 | Moyenne | Eleve (annotation) |
| 5. Segmentation | S7-S9 | Moyenne | Eleve (annotation) |
| 6. Video/Profondeur | S9-S10 | Basse | Moyen |
| 7. Optimisation/Deploiement | S10-S11 | Basse | Moyen |

**Chemin critique** : Phase 1 → Phase 2 → Phase 3 (classification minimale viable en ~5 semaines)

---

## 6. Risques et Mitigations

| Risque | Impact | Mitigation |
|--------|--------|------------|
| Classes trop petites meme apres enrichissement | Modele biaise | Generation IA aggressive + regrouper classes similaires (ex: Joints+Grooves) |
| Images generees de mauvaise qualite | Degrade le modele | Filtrage strict, validation manuelle, FID score |
| Annotation de detection/segmentation trop lente | Retarde phases 4-5 | Utiliser SAM pour pre-annotation, prioriser les classes frequentes |
| Overfitting sur dataset petit | Mauvaise generalisation | Regularisation (dropout, weight decay), augmentation forte, early stopping |
| Confusion inter-classes (Flakings vs Spallings) | Precision faible | Analyse des erreurs, features de profondeur, hierarchie de classes |

---

## 7. Quick Start — Premiere Tache a Implementer

La toute premiere chose a faire est le script EDA + split :

```python
# src/data/split.py — a implementer en premier
# 1. Charger toutes les images et labels depuis data/surface/
# 2. Afficher les statistiques
# 3. Creer un split stratifie train/val/test
# 4. Sauvegarder les chemins dans des CSV (train.csv, val.csv, test.csv)
```

Ensuite : le Dataset PyTorch avec augmentations, puis le premier modele de classification.
