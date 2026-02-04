# AA-SigLIP: Anomaly-Aware SigLIP pour la détection d'anomalies

## Vue d'ensemble du projet

Ce projet explore l'adaptation de l'approche AA-CLIP (CVPR 2025) pour la détection d'anomalies avec Moondream/SigLIP.

### Références clés
- [AA-CLIP Paper (CVPR 2025)](https://openaccess.thecvf.com/content/CVPR2025/papers/Ma_AA-CLIP_Enhancing_Zero-Shot_Anomaly_Detection_via_Anomaly-Aware_CLIP_CVPR_2025_paper.pdf)
- [AA-CLIP GitHub](https://github.com/Mwxinnn/AA-CLIP)
- [SigLIP SO400M HuggingFace](https://huggingface.co/google/siglip-so400m-patch14-384)
- [Moondream GitHub](https://github.com/vikhyat/moondream)

---

## ⚠️ ANALYSE CRITIQUE : Limitation Fondamentale

### Le Problème Central

**AA-CLIP ne peut PAS être directement appliqué à Moondream.**

AA-CLIP repose sur un **text encoder contrastif** aligné avec le vision encoder :
```
T_N = TextEncoder("a normal bottle")   → aligné avec l'espace vision
T_A = TextEncoder("a damaged bottle")  → aligné avec l'espace vision
```

**Moondream n'a PAS ceci :**
- Vision encoder : SigLIP (présent)
- Text encoder contrastif : **ABSENT**
- LLM (Phi-1.5/MoE) : **PAS aligné contrastivement** avec le vision encoder

Le LLM de Moondream vit dans un espace d'embedding différent de SigLIP. Ils n'ont jamais été entraînés ensemble de manière contrastive.

### Observations Empiriques (Tests Préliminaires)

| Problème observé | Description |
|------------------|-------------|
| **Confusion de domaine** | Images MVTec (angles industriels) mal interprétées (ex: bouteille vue comme iris) |
| **Anomaly unawareness** | Moondream ne comprend pas ce qui définit une anomalie industrielle |
| **Hallucinations** | Décrit des défauts inexistants ou manque des défauts réels |

### Stratégies Alternatives

| Stratégie | Description | Avantages | Inconvénients |
|-----------|-------------|-----------|---------------|
| **AA-CLIP original** | Utiliser AA-CLIP tel quel (OpenCLIP) | Prouvé, SOTA | Pas d'explication, modèle séparé |
| **AA-SigLIP standalone** | Adapter AA-CLIP à SigLIP complet (text+vision) | Même backbone que Moondream | Non prouvé, pourquoi SigLIP > CLIP ? |
| **Fine-tuning Moondream** | Entraîner Moondream sur tâche anomaly QA | Simple, explicable | Perd zero-shot, besoin données |
| **Hybride** | AA-CLIP pour détection + Moondream pour explication | Best of both | Deux modèles |

### Recommandation

**Avant tout développement**, évaluer l'anomaly awareness de Moondream avec les tests diagnostiques (Section 11).

---

## 0. Analyse architecturale détaillée

Cette section documente l'architecture des trois composants clés avant toute implémentation.

### 0.1 AA-CLIP (CVPR 2025)

#### Backbone: OpenCLIP ViT-L/14-336px

| Composant | Spécification |
|-----------|---------------|
| **Vision Encoder** | |
| Hidden size | 1024 |
| Layers | 24 |
| Attention heads | 16 |
| FFN intermediate | 4096 |
| Patch size | 14 |
| Resolution | 336×336 |
| Num patches | (336/14)² = 576 |
| **Text Encoder** | |
| Hidden size | 1024 |
| Layers | 12 |
| Attention heads | 16 |
| **Projection** | |
| Embedding dim | 512 |

#### Residual Adapters

```python
# SimpleAdapter: injection dans les couches transformer
class SimpleAdapter(nn.Module):
    def __init__(self, c_in=1024, c_out=1024):
        self.fc = nn.Sequential(
            nn.Linear(c_in, c_out, bias=False),
            nn.LeakyReLU(negative_slope=0.01)
        )
```

**Placement des adapters:**
- Text encoder: couches **1, 2, 3** (K_T = 3, soit 25% des 12 couches)
- Vision encoder: couches **1, 2, 3, 4, 5, 6** (K_I = 6, soit 25% des 24 couches)

**Intégration résiduelle:**
```
x_enhanced = λ × adapter(x) + (1 - λ) × x_original
λ = 0.1 (par défaut)
```

#### Extraction multi-niveau (Vision)

Features extraits aux couches: **6, 12, 18, 24** (quarts du réseau)

```python
# Pour chaque niveau, projection vers l'espace partagé
proj_i = SimpleProj(1024 → 512)
patch_features = mean(proj_6(f6), proj_12(f12), proj_18(f18), proj_24(f24))
patch_features = L2_normalize(patch_features)
```

#### Text Anchors

5 templates par classe pour normal et anormal:
```
Normal: ["a photo of a {cls}", "a {cls}", "a good {cls}", "a normal {cls}", "a perfect {cls}"]
Anomaly: ["a damaged {cls}", "a broken {cls}", "a defective {cls}", "a {cls} with defect", ...]

T_N = mean(encode(normal_templates))  # (512,)
T_A = mean(encode(anomaly_templates)) # (512,)
```

#### Calcul de l'anomaly map

```python
# Classification globale
score = cos_sim(det_features, T_A) - cos_sim(det_features, T_N)

# Segmentation patch-level
anomaly_map = cos_sim(patch_features, T_A) - cos_sim(patch_features, T_N)
# Shape: (batch, 576) → reshape (batch, 24, 24) → upsample (batch, 336, 336)
```

---

### 0.2 Moondream v2

#### Architecture globale

| Composant | Spécification |
|-----------|---------------|
| **Total params** | **1.86B** |
| Vision encoder | ~86M (SigLIP) |
| Projection | ~460M |
| LLM decoder | 1.3B (Phi-1.5) |

#### Vision Encoder: SigLIP ViT-B/16

⚠️ **Note importante**: Moondream v2 utilise **SigLIP-B** (Base), pas SigLIP-SO400M !

| Paramètre | Valeur |
|-----------|--------|
| Variant | ViT-B-16-SigLIP-384 |
| Hidden size | **768** |
| Layers | **12** |
| Attention heads | 12 |
| FFN intermediate | 3072 |
| Patch size | 16 |
| Resolution | 384×384 |
| Num patches | (384/16)² = 576 |
| Params | ~86M |

#### LLM Decoder: Phi-1.5

| Paramètre | Valeur |
|-----------|--------|
| Params | 1.3B |
| Layers | 24 |
| Hidden size | 2048 |
| Attention heads | 32 |
| FFN intermediate | 8192 |
| Max sequence | 2048 tokens |
| Vocab size | 51200 |
| Positional | Rotary (RoPE) |

#### Vision-Text Connection

```
Image → SigLIP (768-dim patches) → Pixel Shuffle → MLP Projection → LLM tokens
```

- Projection MLP mappe 768 → 2048 (dimension Phi-1.5)
- Pixel shuffle réorganise les patches spatialement avant projection
- ~460M params dans la projection

#### Méthodes spéciales

- `caption(image, length)`: génération de descriptions
- `query(image, question)`: VQA
- `detect(image, object)`: détection avec bounding boxes

---

### 0.3 Moondream v3

#### Architecture globale

| Composant | Spécification |
|-----------|---------------|
| **Total params** | **9B** (MoE) |
| **Active params** | **2B** par forward |
| Vision encoder | SigLIP (hérité de v2) |
| LLM decoder | MoE custom |

#### Changements majeurs vs v2

| Aspect | v2 | v3 |
|--------|-----|-----|
| Architecture LLM | Dense (1.3B) | **MoE (9B total, 2B actif)** |
| Context length | 2K | **32K** |
| Experts | - | 64 experts, 8 actifs/token |
| Vision encoder | SigLIP-B | SigLIP (même) |
| Initialisation | From scratch | **Drop upcycling** depuis v2 |

#### LLM Decoder: MoE Custom

| Paramètre | Valeur |
|-----------|--------|
| Total layers | 24 |
| Dense layers | 4 (premiers) |
| MoE layers | 20 (restants) |
| Experts par couche | 64 |
| Experts actifs/token | 8 |
| Hidden size | 2048 |
| FFN type | GeGLU |
| Inner/gate dim | 1024 |
| Tokenizer | SuperBPE (Starmie-v1) |

#### Innovations techniques

- **Attention scaling apprise**: température dépendante de la position et des données
- **FlexAttention**: optimisation pour inférence rapide
- **Multi-crop channel concatenation**: traitement efficace haute résolution

---

### 0.4 Tableau comparatif complet

| Paramètre | AA-CLIP | Moondream v2 | Moondream v3 |
|-----------|---------|--------------|--------------|
| **Vision Encoder** | | | |
| Type | OpenCLIP ViT-L/14 | SigLIP ViT-B/16 | SigLIP (hérité v2) |
| Hidden dim | 1024 | **768** | 768 |
| Layers | 24 | **12** | 12 |
| Patch size | 14 | **16** | 16 |
| Resolution | 336×336 | 384×384 | 384×384 |
| Num patches | 576 | 576 | 576 |
| **Text/LLM** | | | |
| Type | CLIP Text Enc | Phi-1.5 | MoE Custom |
| Hidden dim | 1024 | 2048 | 2048 |
| Layers | 12 | 24 | 24 (4 dense + 20 MoE) |
| **Pré-entraînement** | Softmax (InfoNCE) | Sigmoid (SigLIP) | Sigmoid (SigLIP) |
| **Total params** | ~428M | 1.86B | 9B (2B actif) |

---

### 0.5 Défi principal : Génération des Text Anchors

AA-CLIP repose sur un **text encoder contrastif** pour générer les ancres T_N et T_A :
```
T_N = TextEncoder("a normal bottle")   → (512,)
T_A = TextEncoder("a damaged bottle")  → (512,)
```

**Problème avec Moondream** : Pas de text encoder séparé, seulement un LLM (Phi/MoE).

#### Solutions possibles

**Solution 1: Embeddings du LLM**
Extraire les embeddings de tokens du LLM Phi-1.5 :
```python
# Utiliser les embeddings de la couche d'entrée du LLM
text_embeddings = phi.embed_tokens(tokenize("a damaged bottle"))
T_A = project(text_embeddings.mean(dim=1))  # Projection vers l'espace vision
```
⚠️ Problème : Ces embeddings ne sont pas alignés avec l'espace vision de SigLIP.

**Solution 2: Apprendre les anchors directement**
Ne pas utiliser de prompts texte, mais apprendre T_N et T_A comme paramètres :
```python
T_N = nn.Parameter(torch.randn(768))  # Anchor "normal" appris
T_A = nn.Parameter(torch.randn(768))  # Anchor "anomaly" appris
```
✅ Avantage : Pas besoin de text encoder
⚠️ Inconvénient : Perd la capacité zero-shot (anchors spécifiques par dataset)

**Solution 3: Utiliser les features vision comme proxy**
Calculer T_N à partir d'images normales d'entraînement :
```python
T_N = mean([encode_image(img) for img in normal_images])  # Centroïde "normal"
T_A = apprendre via contrastive loss
```
✅ Avantage : Aligné naturellement avec l'espace vision

**Solution 4: Cross-attention du LLM**
Utiliser la capacité de Moondream à raisonner sur les images :
```python
# Prompt Moondream pour générer une représentation
response = moondream.query(image, "Is this item normal or defective?")
# Extraire les hidden states comme features
```
⚠️ Inconvénient : Lent, pas de localisation patch-level

---

### 0.6 Adaptation des Residual Adapters pour Moondream

#### Paramètres adaptés (AA-CLIP → Moondream)

| Paramètre | AA-CLIP | Moondream |
|-----------|---------|-----------|
| Vision hidden dim | 1024 | **768** |
| Vision layers | 24 | **12** |
| K_I (layers adaptées) | 6 (25%) | **3** (25%) |
| Feature extraction | [6,12,18,24] | **[3, 6, 9, 12]** |
| Adapter dim | 1024→1024 | **768→768** |
| Projection dim | 512 | **512 ou 768** |
| λ (residual weight) | 0.1 | 0.1 |

#### Architecture proposée

```
┌─────────────────────────────────────────────────────────┐
│  Moondream Vision Encoder (SigLIP-B, 12 layers)        │
│                                                         │
│  [Patch Embed] → [L1+Adapter] → [L2+Adapter] → [L3+Adapter]
│                       ↓              ↓              ↓
│                   features       features       features
│                       ↓              ↓              ↓
│  → [L4] → [L5] → [L6] → ... → [L12]                   │
│              ↓                    ↓                    │
│          features             features                 │
│                                                         │
│  Multi-level aggregation: [3, 6, 9, 12]               │
│              ↓                                         │
│  patch_features (576, 768) → normalize                │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│  Anomaly Detection Head                                 │
│                                                         │
│  T_N, T_A = learned anchors (768,) each                │
│                                                         │
│  anomaly_map = sim(patches, T_A) - sim(patches, T_N)  │
│  → reshape (576,) → (24, 24) → upsample (384, 384)    │
└─────────────────────────────────────────────────────────┘
```

---

### 0.7 Recommandation : Approche en 2 phases

**Phase 1 : Validation sur vision encoder seul**
- Extraire le SigLIP-B de Moondream v2
- Implémenter les Residual Adapters (K_I=3)
- Utiliser des **anchors appris** (Solution 2) ou **centroïdes** (Solution 3)
- Entraîner sur VisA, évaluer sur MVTec AD

**Phase 2 : Intégration avec le LLM**
- Si les résultats Phase 1 sont bons :
  - Utiliser le LLM pour générer des explications des anomalies détectées
  - Explorer l'extraction d'embeddings texte depuis Phi-1.5

---

## 1. Comparaison architecturale CLIP vs SigLIP

### Paramètres architecturaux

| Paramètre | OpenCLIP ViT-L/14 (AA-CLIP) | SigLIP ViT-SO400M/14 |
|-----------|----------------------------|----------------------|
| Hidden size | 1024 | **1152** |
| Num layers (vision) | 24 | **27** |
| Num layers (text) | 12 | **27** |
| Attention heads | 16 | 16 |
| Intermediate size | 4096 | **4304** |
| Patch size | 14 | 14 |
| Image resolution | 336×336 | **384×384** |
| Loss function | Softmax (InfoNCE) | **Sigmoid** |

### Différence fondamentale : Softmax vs Sigmoid Loss

```
CLIP (Softmax/InfoNCE):
  L = -log(exp(sim(i,t)/τ) / Σ exp(sim(i,t')/τ))
  → Normalisation globale sur le batch
  → Chaque paire comparée à TOUTES les autres

SigLIP (Sigmoid):
  L = -Σ log(σ(y_ij * sim(i,j)/τ))  où y_ij ∈ {-1, +1}
  → Traitement indépendant par paire
  → Pas de normalisation inter-batch
```

**Hypothèse à tester**: La sigmoid loss pourrait produire un espace latent avec une meilleure séparation intrinsèque normal/anormal car elle n'impose pas de comparaison relative au batch.

---

## 2. Architecture AA-SigLIP proposée

### 2.1 Vue d'ensemble

```
┌─────────────────────────────────────────────────────────────────┐
│                        AA-SigLIP                                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  STAGE 1: Text Anchor Disentanglement                           │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ SigLIP Text Encoder (27 layers, d=1152)                    │ │
│  │                                                            │ │
│  │ [Input] → [Embed] → [Layer 1] → [Layer 2] → [Layer 3] →...│ │
│  │                        ↓            ↓           ↓          │ │
│  │                   +Adapter    +Adapter    +Adapter         │ │
│  │                   (K_T=3)                                  │ │
│  │                                                            │ │
│  │ ... → [Layer 27] → [LN] → [Proj Adapter] → T_N, T_A       │ │
│  │                                            (anchors)       │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  STAGE 2: Visual Feature Alignment                              │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ SigLIP Vision Encoder (27 layers, d=1152)                  │ │
│  │                                                            │ │
│  │ [Patch Embed] → [Layer 1] → ... → [Layer 7] → ... →       │ │
│  │                     ↓                 ↓                    │ │
│  │                +Adapter          +Adapter                  │ │
│  │                (K_I=7)                                     │ │
│  │                                                            │ │
│  │ Feature extraction at layers: 7, 14, 21, 27               │ │
│  │                     ↓      ↓      ↓      ↓                 │ │
│  │                 [Proj] [Proj] [Proj] [Proj]                │ │
│  │                     ↘     ↓      ↓     ↙                   │ │
│  │                    V_patch (aggregated patches)            │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  INFERENCE: CosSim(V_patch, [T_N, T_A]) → anomaly_map + score   │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 Residual Adapter (de AA-CLIP)

Le Residual Adapter est simple et agnostique à l'architecture :

```python
import torch.nn as nn

class SimpleAdapter(nn.Module):
    """Adapter léger pour injection dans les couches transformer."""
    def __init__(self, c_in, c_out=768):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(c_in, c_out, bias=False),
            nn.LeakyReLU()
        )

    def forward(self, x):
        return self.fc(x)


class SimpleProj(nn.Module):
    """Projection pour aligner les dimensions."""
    def __init__(self, c_in, c_out=768, relu=True):
        super().__init__()
        if relu:
            self.fc = nn.Sequential(
                nn.Linear(c_in, c_out, bias=False),
                nn.LeakyReLU()
            )
        else:
            self.fc = nn.Linear(c_in, c_out, bias=False)

    def forward(self, x):
        return self.fc(x)
```

**Intégration résiduelle** (Eq. du paper):
```python
# x = sortie de la couche transformer
# adapter = SimpleAdapter instance
x_adapted = adapter(x)
x_enhanced = λ * x_adapted + (1 - λ) * x  # λ = 0.1 par défaut
```

### 2.3 Adaptation des hyperparamètres pour SigLIP

| Paramètre | AA-CLIP (CLIP) | AA-SigLIP (proposé) | Justification |
|-----------|---------------|---------------------|---------------|
| K_T (couches texte adaptées) | 3 | 3 | Même ratio ~10% |
| K_I (couches vision adaptées) | 6 | **7** | 6/24 ≈ 7/27 (~25%) |
| Adapter dimension | 1024 | **1152** | Dimension SigLIP |
| Projection output | 768 | **768** (ou 1152) | À ablater |
| Feature layers | [6, 12, 18, 24] | **[7, 14, 21, 27]** | Quarts du réseau |
| λ (residual weight) | 0.1 | 0.1 | Point de départ |
| γ (disentangle weight) | 0.1 | 0.1 | À ajuster |
| Image resolution | 336×336 | **384×384** | Natif SigLIP |

---

## 3. Implémentation AA-SigLIP

### 3.1 Classe principale AdaptedSigLIP

```python
import torch
import torch.nn as nn
from transformers import SiglipModel, SiglipProcessor

class AdaptedSigLIP(nn.Module):
    """
    SigLIP adapté pour la détection d'anomalies (style AA-CLIP).
    """
    def __init__(
        self,
        model_name: str = "google/siglip-so400m-patch14-384",
        text_adapt_weight: float = 0.1,
        image_adapt_weight: float = 0.1,
        text_adapt_until: int = 3,      # K_T: nombre de couches texte adaptées
        image_adapt_until: int = 7,     # K_I: nombre de couches vision adaptées
        feature_levels: list = [7, 14, 21, 27],  # Couches d'extraction
        proj_dim: int = 768,            # Dimension de projection
    ):
        super().__init__()

        # Charger le modèle SigLIP pré-entraîné
        self.siglip = SiglipModel.from_pretrained(model_name)
        self.processor = SiglipProcessor.from_pretrained(model_name)

        # Geler les poids originaux
        for param in self.siglip.parameters():
            param.requires_grad = False

        # Configuration
        self.hidden_size = 1152  # SigLIP SO400M
        self.text_adapt_until = text_adapt_until
        self.image_adapt_until = image_adapt_until
        self.t_w = text_adapt_weight
        self.i_w = image_adapt_weight
        self.feature_levels = feature_levels

        # Adapters pour l'encodeur d'image
        self.image_adapter = nn.ModuleDict({
            "layer_adapters": nn.ModuleList([
                SimpleAdapter(self.hidden_size, self.hidden_size)
                for _ in range(image_adapt_until)
            ]),
            "seg_proj": nn.ModuleList([
                SimpleProj(self.hidden_size, proj_dim, relu=True)
                for _ in range(len(feature_levels))
            ]),
            "det_proj": SimpleProj(self.hidden_size, proj_dim, relu=True),
        })

        # Adapters pour l'encodeur de texte
        self.text_adapter = nn.ModuleList([
            SimpleAdapter(self.hidden_size, self.hidden_size)
            for _ in range(text_adapt_until)
        ] + [
            SimpleProj(self.hidden_size, proj_dim, relu=True)  # Final projection
        ])

        self._init_weights_()

    def _init_weights_(self):
        """Initialisation Xavier pour les adapters."""
        for module in [self.image_adapter, self.text_adapter]:
            for p in module.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)

    def encode_text(self, text_inputs, adapt_text: bool = True):
        """
        Encode le texte avec adaptation optionnelle.

        Args:
            text_inputs: Tokens de texte (input_ids, attention_mask)
            adapt_text: Appliquer les adapters texte

        Returns:
            text_features: (batch, proj_dim)
        """
        # Accéder à l'encodeur texte de SigLIP
        text_encoder = self.siglip.text_model

        # Embedding initial
        hidden_states = text_encoder.embeddings(text_inputs["input_ids"])

        # Passer par les couches transformer avec adaptation
        for idx, layer in enumerate(text_encoder.encoder.layers):
            hidden_states = layer(hidden_states)[0]

            if adapt_text and idx < self.text_adapt_until:
                # Appliquer l'adapter résiduel
                adapted = self.text_adapter[idx](hidden_states)
                hidden_states = self.t_w * adapted + (1 - self.t_w) * hidden_states

        # Normalisation finale
        hidden_states = text_encoder.final_layer_norm(hidden_states)

        # Extraire le token EOS (fin de séquence) pour SigLIP
        # SigLIP utilise le dernier token non-padding
        sequence_lengths = text_inputs["attention_mask"].sum(dim=1) - 1
        batch_size = hidden_states.shape[0]
        pooled = hidden_states[
            torch.arange(batch_size, device=hidden_states.device),
            sequence_lengths
        ]

        # Projection finale
        if adapt_text:
            pooled = self.text_adapter[-1](pooled)  # SimpleProj final

        # Normalisation L2
        pooled = pooled / pooled.norm(dim=-1, keepdim=True)

        return pooled

    def encode_image(self, pixel_values, return_patches: bool = True):
        """
        Encode l'image avec adaptation et extraction multi-niveau.

        Args:
            pixel_values: Images prétraitées (batch, 3, 384, 384)
            return_patches: Retourner les features patch-level

        Returns:
            det_features: (batch, proj_dim) pour classification
            patch_features: (batch, num_patches, proj_dim) pour segmentation
            multi_level_features: Liste de features à différents niveaux
        """
        vision_encoder = self.siglip.vision_model

        # Patch embedding
        hidden_states = vision_encoder.embeddings(pixel_values)

        multi_level_features = []

        # Passer par les couches transformer avec adaptation
        for idx, layer in enumerate(vision_encoder.encoder.layers):
            hidden_states = layer(hidden_states)[0]

            # Appliquer adapter aux premières couches
            if idx < self.image_adapt_until:
                adapted = self.image_adapter["layer_adapters"][idx](hidden_states)
                hidden_states = self.i_w * adapted + (1 - self.i_w) * hidden_states

            # Extraire features aux niveaux spécifiés (1-indexed)
            if (idx + 1) in self.feature_levels:
                level_idx = self.feature_levels.index(idx + 1)
                proj_features = self.image_adapter["seg_proj"][level_idx](hidden_states)
                multi_level_features.append(proj_features)

        # Post-normalization
        hidden_states = vision_encoder.post_layernorm(hidden_states)

        # Features pour la détection (global)
        # SigLIP utilise attention pooling, on peut utiliser la moyenne des patches
        det_features = hidden_states.mean(dim=1)
        det_features = self.image_adapter["det_proj"](det_features)
        det_features = det_features / det_features.norm(dim=-1, keepdim=True)

        # Agréger les features multi-niveau pour la segmentation
        if return_patches and multi_level_features:
            # Somme pondérée des features multi-niveau
            patch_features = sum(multi_level_features) / len(multi_level_features)
            patch_features = patch_features / patch_features.norm(dim=-1, keepdim=True)
        else:
            patch_features = None

        return det_features, patch_features, multi_level_features

    def forward(self, images, texts=None):
        """
        Forward pass complet.

        Args:
            images: Tensor d'images (batch, 3, 384, 384)
            texts: Dict avec input_ids et attention_mask (optionnel)

        Returns:
            dict avec det_features, patch_features, text_features (si texts fourni)
        """
        det_features, patch_features, multi_level = self.encode_image(images)

        result = {
            "det_features": det_features,
            "patch_features": patch_features,
            "multi_level_features": multi_level,
        }

        if texts is not None:
            text_features = self.encode_text(texts)
            result["text_features"] = text_features

        return result
```

### 3.2 Génération des Text Anchors

```python
def get_text_anchors(model, class_name, device, templates=None):
    """
    Génère les ancres textuelles normal/anormal pour une classe.

    Args:
        model: AdaptedSigLIP model
        class_name: Nom de la catégorie (ex: "bottle", "cable")
        device: torch device
        templates: Templates de prompts personnalisés

    Returns:
        text_anchors: (2, proj_dim) - [normal_anchor, anomaly_anchor]
    """
    if templates is None:
        # Templates par défaut
        normal_templates = [
            f"a photo of a {class_name}",
            f"a {class_name}",
            f"a good {class_name}",
            f"a normal {class_name}",
            f"a perfect {class_name}",
        ]
        anomaly_templates = [
            f"a photo of a damaged {class_name}",
            f"a damaged {class_name}",
            f"a broken {class_name}",
            f"a defective {class_name}",
            f"a {class_name} with defect",
        ]
    else:
        normal_templates = templates["normal"]
        anomaly_templates = templates["anomaly"]

    processor = model.processor

    # Encoder les templates normaux
    normal_inputs = processor(text=normal_templates, return_tensors="pt", padding=True)
    normal_inputs = {k: v.to(device) for k, v in normal_inputs.items()}
    with torch.no_grad():
        normal_features = model.encode_text(normal_inputs)
    normal_anchor = normal_features.mean(dim=0, keepdim=True)
    normal_anchor = normal_anchor / normal_anchor.norm(dim=-1, keepdim=True)

    # Encoder les templates anomaux
    anomaly_inputs = processor(text=anomaly_templates, return_tensors="pt", padding=True)
    anomaly_inputs = {k: v.to(device) for k, v in anomaly_inputs.items()}
    with torch.no_grad():
        anomaly_features = model.encode_text(anomaly_inputs)
    anomaly_anchor = anomaly_features.mean(dim=0, keepdim=True)
    anomaly_anchor = anomaly_anchor / anomaly_anchor.norm(dim=-1, keepdim=True)

    # Stack: (2, proj_dim)
    text_anchors = torch.cat([normal_anchor, anomaly_anchor], dim=0)

    return text_anchors
```

### 3.3 Loss Functions

```python
import torch.nn.functional as F

class DisentangleLoss(nn.Module):
    """
    Loss pour désenchevêtrer les ancres normal/anormal.
    L_dis = |⟨T_N, T_A⟩|²
    """
    def forward(self, text_anchors):
        """
        Args:
            text_anchors: (num_classes, 2, proj_dim)
                          [:, 0, :] = normal anchors
                          [:, 1, :] = anomaly anchors
        """
        normal = text_anchors[:, 0, :]   # (num_classes, proj_dim)
        anomaly = text_anchors[:, 1, :]  # (num_classes, proj_dim)

        # Produit scalaire entre normal et anormal
        inner_product = (normal * anomaly).sum(dim=-1)  # (num_classes,)

        # Loss = moyenne des carrés des produits scalaires
        loss = (inner_product ** 2).mean()

        return loss


class FocalLoss(nn.Module):
    """Focal Loss pour les échantillons difficiles."""
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss

        if self.reduction == 'mean':
            return F_loss.mean()
        elif self.reduction == 'sum':
            return F_loss.sum()
        return F_loss


class BinaryDiceLoss(nn.Module):
    """Dice Loss pour la segmentation binaire."""
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        inputs = torch.sigmoid(inputs)

        # Flatten
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice = (2. * intersection + self.smooth) / (
            inputs.sum() + targets.sum() + self.smooth
        )

        return 1 - dice


class SegmentationLoss(nn.Module):
    """Loss combinée pour la segmentation (Dice + Focal)."""
    def __init__(self, dice_weight=1.0, focal_weight=1.0):
        super().__init__()
        self.dice = BinaryDiceLoss()
        self.focal = FocalLoss()
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight

    def forward(self, pred_mask, target_mask):
        dice_loss = self.dice(pred_mask, target_mask)
        focal_loss = self.focal(pred_mask, target_mask)
        return self.dice_weight * dice_loss + self.focal_weight * focal_loss
```

### 3.4 Training Pipeline

```python
import torch.optim as optim
from torch.utils.data import DataLoader

def train_text_adapter(
    model: AdaptedSigLIP,
    train_loader: DataLoader,
    device: torch.device,
    epochs: int = 5,
    lr: float = 1e-5,
    disentangle_weight: float = 0.1,
):
    """
    Stage 1: Entraîner les adapters texte pour désenchevêtrer normal/anormal.
    """
    # Ne rendre entraînables que les adapters texte
    for param in model.parameters():
        param.requires_grad = False
    for param in model.text_adapter.parameters():
        param.requires_grad = True

    optimizer = optim.Adam(
        model.text_adapter.parameters(),
        lr=lr,
        betas=(0.5, 0.999)
    )

    seg_loss_fn = SegmentationLoss()
    dis_loss_fn = DisentangleLoss()

    model.train()

    for epoch in range(epochs):
        total_loss = 0
        for batch in train_loader:
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)
            class_names = batch["class_name"]

            optimizer.zero_grad()

            # Obtenir les features d'image (gelées pour Stage 1)
            with torch.no_grad():
                _, patch_features, _ = model.encode_image(images)

            # Générer les ancres texte adaptées
            batch_anchors = []
            for cls_name in class_names:
                anchors = get_text_anchors(model, cls_name, device)
                batch_anchors.append(anchors)
            text_anchors = torch.stack(batch_anchors)  # (batch, 2, proj_dim)

            # Calculer les prédictions de segmentation
            # Similarité entre patches et ancres
            normal_anchor = text_anchors[:, 0:1, :]  # (batch, 1, proj_dim)
            anomaly_anchor = text_anchors[:, 1:2, :]

            sim_normal = torch.bmm(patch_features, normal_anchor.transpose(1, 2))
            sim_anomaly = torch.bmm(patch_features, anomaly_anchor.transpose(1, 2))

            # Anomaly score = différence de similarité
            anomaly_map = sim_anomaly - sim_normal  # (batch, num_patches, 1)

            # Reshape en grille spatiale (27 patches = racine non-entière, utiliser 729 patches pour 384/14)
            # 384/14 = 27.4, SigLIP padding → 729 patches (27x27) + 1 cls = 730 ou similaire
            # À adapter selon la config exacte

            # Loss de segmentation
            loss_seg = seg_loss_fn(anomaly_map.squeeze(-1), masks)

            # Loss de désenchevêtrement
            loss_dis = dis_loss_fn(text_anchors)

            # Loss totale
            loss = loss_seg + disentangle_weight * loss_dis

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Stage 1 - Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

    return model


def train_image_adapter(
    model: AdaptedSigLIP,
    text_anchors_dict: dict,  # {class_name: (2, proj_dim)}
    train_loader: DataLoader,
    device: torch.device,
    epochs: int = 20,
    lr: float = 5e-4,
):
    """
    Stage 2: Entraîner les adapters vision pour aligner avec les ancres texte.
    """
    # Geler les adapters texte, débloquer les adapters image
    for param in model.parameters():
        param.requires_grad = False
    for param in model.image_adapter.parameters():
        param.requires_grad = True

    optimizer = optim.Adam(
        model.image_adapter.parameters(),
        lr=lr,
        betas=(0.5, 0.999)
    )

    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[16000, 32000],
        gamma=0.5
    )

    seg_loss_fn = SegmentationLoss()
    cls_loss_fn = nn.CrossEntropyLoss()

    model.train()
    global_step = 0

    for epoch in range(epochs):
        total_loss = 0
        for batch in train_loader:
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)
            labels = batch["label"].to(device)  # 0=normal, 1=anomaly
            class_names = batch["class_name"]

            optimizer.zero_grad()

            # Encoder les images avec adaptation
            det_features, patch_features, _ = model.encode_image(images)

            # Récupérer les ancres texte (gelées)
            batch_anchors = torch.stack([
                text_anchors_dict[cls].to(device)
                for cls in class_names
            ])  # (batch, 2, proj_dim)

            # Classification: similarité avec les ancres
            normal_anchor = batch_anchors[:, 0, :]  # (batch, proj_dim)
            anomaly_anchor = batch_anchors[:, 1, :]

            sim_normal = (det_features * normal_anchor).sum(dim=-1, keepdim=True)
            sim_anomaly = (det_features * anomaly_anchor).sum(dim=-1, keepdim=True)
            cls_logits = torch.cat([sim_normal, sim_anomaly], dim=-1)  # (batch, 2)

            # Segmentation
            sim_normal_patch = torch.bmm(
                patch_features,
                normal_anchor.unsqueeze(-1)
            ).squeeze(-1)
            sim_anomaly_patch = torch.bmm(
                patch_features,
                anomaly_anchor.unsqueeze(-1)
            ).squeeze(-1)
            anomaly_map = sim_anomaly_patch - sim_normal_patch

            # Losses
            loss_cls = cls_loss_fn(cls_logits, labels.long())
            loss_seg = seg_loss_fn(anomaly_map, masks)

            loss = loss_cls + loss_seg

            loss.backward()
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            global_step += 1

        avg_loss = total_loss / len(train_loader)
        print(f"Stage 2 - Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

    return model
```

---

## 4. Phase 1 — Diagnostic de l'Anomaly Unawareness

### 4.1 Script de diagnostic

```python
"""
diagnostic_anomaly_unawareness.py

Reproduire les analyses diagnostiques d'AA-CLIP sur SigLIP vs OpenCLIP.
"""

import torch
import numpy as np
from transformers import SiglipModel, SiglipProcessor
import open_clip
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

def load_models(device):
    """Charger SigLIP et OpenCLIP pour comparaison."""
    # SigLIP
    siglip_model = SiglipModel.from_pretrained("google/siglip-so400m-patch14-384")
    siglip_processor = SiglipProcessor.from_pretrained("google/siglip-so400m-patch14-384")
    siglip_model = siglip_model.to(device).eval()

    # OpenCLIP (même que AA-CLIP)
    clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
        'ViT-L-14-336',
        pretrained='openai'
    )
    clip_tokenizer = open_clip.get_tokenizer('ViT-L-14-336')
    clip_model = clip_model.to(device).eval()

    return {
        "siglip": (siglip_model, siglip_processor),
        "clip": (clip_model, clip_preprocess, clip_tokenizer)
    }


def test_anomaly_awareness(models, images, class_names, device):
    """
    Test 1: Similarité cosinus image-texte sur images anomales.

    Métrique: Anomaly Awareness Score (AAS)
    = pourcentage d'images anomales où sim_anomaly > sim_normal
    """
    results = {"siglip": [], "clip": []}

    siglip_model, siglip_processor = models["siglip"]
    clip_model, clip_preprocess, clip_tokenizer = models["clip"]

    for img, cls_name in zip(images, class_names):
        normal_prompt = f"a photo of a normal {cls_name}"
        anomaly_prompt = f"a photo of a damaged {cls_name}"

        # --- SigLIP ---
        inputs = siglip_processor(
            text=[normal_prompt, anomaly_prompt],
            images=img,
            return_tensors="pt",
            padding=True
        ).to(device)

        with torch.no_grad():
            outputs = siglip_model(**inputs)
            # Similarité logits (pas softmax, sigmoid native)
            logits = outputs.logits_per_image[0]  # (2,)
            sim_normal_siglip = logits[0].item()
            sim_anomaly_siglip = logits[1].item()

        results["siglip"].append({
            "sim_normal": sim_normal_siglip,
            "sim_anomaly": sim_anomaly_siglip,
            "correct": sim_anomaly_siglip > sim_normal_siglip
        })

        # --- OpenCLIP ---
        img_tensor = clip_preprocess(img).unsqueeze(0).to(device)
        text_tokens = clip_tokenizer([normal_prompt, anomaly_prompt]).to(device)

        with torch.no_grad():
            img_features = clip_model.encode_image(img_tensor)
            text_features = clip_model.encode_text(text_tokens)
            img_features = img_features / img_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            similarities = (img_features @ text_features.T)[0]
            sim_normal_clip = similarities[0].item()
            sim_anomaly_clip = similarities[1].item()

        results["clip"].append({
            "sim_normal": sim_normal_clip,
            "sim_anomaly": sim_anomaly_clip,
            "correct": sim_anomaly_clip > sim_normal_clip
        })

    # Calculer les métriques
    metrics = {}
    for model_name in ["siglip", "clip"]:
        res = results[model_name]
        aas = sum(1 for r in res if r["correct"]) / len(res) * 100
        adg = np.mean([r["sim_normal"] - r["sim_anomaly"] for r in res])
        metrics[model_name] = {"AAS": aas, "ADG": adg}

    return metrics, results


def test_text_similarity_matrix(models, class_names, device):
    """
    Test 2: Matrice de similarité inter-prompts.

    Mesure la corrélation entre prompts normaux et anormaux.
    """
    siglip_model, siglip_processor = models["siglip"]
    clip_model, _, clip_tokenizer = models["clip"]

    all_prompts = []
    labels = []  # 0=normal, 1=anomaly
    categories = []

    for cls_name in class_names:
        normal_prompts = [
            f"a photo of a {cls_name}",
            f"a {cls_name}",
            f"a good {cls_name}",
            f"a normal {cls_name}",
        ]
        anomaly_prompts = [
            f"a damaged {cls_name}",
            f"a broken {cls_name}",
            f"a defective {cls_name}",
            f"a {cls_name} with defect",
        ]
        all_prompts.extend(normal_prompts + anomaly_prompts)
        labels.extend([0] * len(normal_prompts) + [1] * len(anomaly_prompts))
        categories.extend([cls_name] * (len(normal_prompts) + len(anomaly_prompts)))

    # Encoder avec SigLIP
    siglip_inputs = siglip_processor(
        text=all_prompts,
        return_tensors="pt",
        padding=True
    ).to(device)

    with torch.no_grad():
        siglip_features = siglip_model.get_text_features(**siglip_inputs)
        siglip_features = siglip_features / siglip_features.norm(dim=-1, keepdim=True)
        siglip_sim_matrix = (siglip_features @ siglip_features.T).cpu().numpy()

    # Encoder avec OpenCLIP
    clip_tokens = clip_tokenizer(all_prompts).to(device)
    with torch.no_grad():
        clip_features = clip_model.encode_text(clip_tokens)
        clip_features = clip_features / clip_features.norm(dim=-1, keepdim=True)
        clip_sim_matrix = (clip_features @ clip_features.T).cpu().numpy()

    return {
        "siglip": siglip_sim_matrix,
        "clip": clip_sim_matrix,
        "labels": labels,
        "categories": categories,
        "prompts": all_prompts
    }


def test_tsne_visualization(models, class_names, device, output_path="tsne_comparison.png"):
    """
    Test 3: Visualisation t-SNE de l'espace textuel.
    """
    sim_data = test_text_similarity_matrix(models, class_names, device)

    siglip_model, siglip_processor = models["siglip"]
    clip_model, _, clip_tokenizer = models["clip"]

    prompts = sim_data["prompts"]
    labels = np.array(sim_data["labels"])

    # Obtenir les embeddings
    siglip_inputs = siglip_processor(text=prompts, return_tensors="pt", padding=True).to(device)
    clip_tokens = clip_tokenizer(prompts).to(device)

    with torch.no_grad():
        siglip_emb = siglip_model.get_text_features(**siglip_inputs).cpu().numpy()
        clip_emb = clip_model.encode_text(clip_tokens).cpu().numpy()

    # t-SNE
    tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42)

    siglip_tsne = tsne.fit_transform(siglip_emb)
    clip_tsne = tsne.fit_transform(clip_emb)

    # Silhouette scores
    siglip_silhouette = silhouette_score(siglip_emb, labels)
    clip_silhouette = silhouette_score(clip_emb, labels)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, data, title, sil in [
        (axes[0], clip_tsne, "OpenCLIP", clip_silhouette),
        (axes[1], siglip_tsne, "SigLIP", siglip_silhouette)
    ]:
        colors = ['blue' if l == 0 else 'red' for l in labels]
        ax.scatter(data[:, 0], data[:, 1], c=colors, alpha=0.6)
        ax.set_title(f"{title} (Silhouette: {sil:.3f})")
        ax.legend(['Normal', 'Anomaly'])

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    return {
        "siglip_silhouette": siglip_silhouette,
        "clip_silhouette": clip_silhouette
    }
```

### 4.2 Résultats attendus

| Métrique | OpenCLIP (attendu) | SigLIP (hypothèse) |
|----------|-------------------|-------------------|
| AAS (% correct) | ~40-50% | À mesurer (>60% = meilleur) |
| ADG (écart moyen) | Négatif | À mesurer (positif = bon) |
| Similarité N-A intra-classe | ~0.9 | À mesurer (<0.8 = meilleur) |
| Silhouette Score | Faible | À mesurer (plus haut = meilleur) |

---

## 5. Intégration avec Moondream

### 5.1 Option A: Remplacement de l'encodeur

```python
from transformers import AutoModelForCausalLM

class MoondreamWithAASigLIP:
    """
    Moondream avec encodeur SigLIP adapté pour l'anomaly detection.
    """
    def __init__(self, aa_siglip_checkpoint: str):
        # Charger Moondream
        self.moondream = AutoModelForCausalLM.from_pretrained(
            "vikhyatk/moondream2",
            trust_remote_code=True
        )

        # Charger AA-SigLIP adapté
        self.aa_siglip = AdaptedSigLIP.from_pretrained(aa_siglip_checkpoint)

        # Remplacer l'encodeur vision (nécessite investigation de l'architecture)
        # self.moondream.vision_encoder = self.aa_siglip.siglip.vision_model

    def detect_anomaly(self, image, class_name):
        """Détecte les anomalies et génère une explication."""
        # 1. Obtenir l'anomaly map avec AA-SigLIP
        text_anchors = get_text_anchors(self.aa_siglip, class_name, self.device)
        det_features, patch_features, _ = self.aa_siglip.encode_image(image)

        # Calculer l'anomaly score
        anomaly_score = (det_features * text_anchors[1]).sum() - \
                       (det_features * text_anchors[0]).sum()

        # 2. Générer l'explication avec Moondream
        if anomaly_score > 0:
            prompt = f"Describe any defects or anomalies visible in this {class_name}."
        else:
            prompt = f"Confirm this {class_name} appears normal and defect-free."

        explanation = self.moondream.query(image, prompt)

        return {
            "anomaly_score": anomaly_score.item(),
            "is_anomaly": anomaly_score > 0,
            "explanation": explanation
        }
```

### 5.2 Option B: Pipeline hybride

```python
class HybridAnomalyDetector:
    """
    Pipeline hybride: AA-SigLIP pour détection + Moondream pour explication.
    """
    def __init__(self):
        self.aa_siglip = AdaptedSigLIP()
        self.moondream = AutoModelForCausalLM.from_pretrained(
            "vikhyatk/moondream2",
            trust_remote_code=True
        )
        self.text_anchors = {}  # Cache des ancres par classe

    def process(self, image, class_name):
        # Charger ou calculer les ancres
        if class_name not in self.text_anchors:
            self.text_anchors[class_name] = get_text_anchors(
                self.aa_siglip, class_name, self.device
            )

        # Détection avec AA-SigLIP
        det_features, patch_features, _ = self.aa_siglip.encode_image(image)
        anchors = self.text_anchors[class_name]

        # Score global
        score_normal = (det_features * anchors[0]).sum()
        score_anomaly = (det_features * anchors[1]).sum()
        anomaly_prob = torch.sigmoid(score_anomaly - score_normal)

        # Anomaly map pour localisation
        sim_normal = (patch_features @ anchors[0:1].T).squeeze(-1)
        sim_anomaly = (patch_features @ anchors[1:2].T).squeeze(-1)
        anomaly_map = torch.sigmoid(sim_anomaly - sim_normal)

        # Reshape en grille spatiale (ex: 27x27 pour 384/14)
        grid_size = int(np.sqrt(anomaly_map.shape[1]))
        anomaly_map_2d = anomaly_map.reshape(-1, grid_size, grid_size)

        # Si anomalie détectée, demander à Moondream de décrire
        if anomaly_prob > 0.5:
            description = self.moondream.query(
                image,
                "What defects or damage do you see in this image? Be specific."
            )
        else:
            description = "No anomalies detected."

        return {
            "anomaly_probability": anomaly_prob.item(),
            "anomaly_map": anomaly_map_2d,
            "description": description
        }
```

---

## 6. Benchmarking et évaluation

### 6.1 Datasets

| Dataset | Type | Catégories | Usage |
|---------|------|------------|-------|
| [MVTec AD](https://www.mvtec.com/company/research/datasets/mvtec-ad) | Industriel | 15 | Benchmark principal |
| [VisA](https://github.com/amazon-science/spot-diff) | Industriel | 12 | Entraînement |
| [BTAD](https://github.com/pankajmishra000/BTAD) | Industriel | 3 | Zero-shot test |
| [MPDD](https://github.com/stepanje/MPDD) | Industriel | 6 | Zero-shot test |

### 6.2 Métriques

```python
from sklearn.metrics import roc_auc_score

def evaluate_model(model, test_loader, device):
    """Évaluer les performances de détection d'anomalies."""
    all_labels = []
    all_scores = []
    all_masks = []
    all_maps = []

    model.eval()
    with torch.no_grad():
        for batch in test_loader:
            images = batch["image"].to(device)
            labels = batch["label"]
            masks = batch["mask"]
            class_names = batch["class_name"]

            det_features, patch_features, _ = model.encode_image(images)

            for i, cls_name in enumerate(class_names):
                anchors = get_text_anchors(model, cls_name, device)

                # Score de détection
                score = (det_features[i] * anchors[1]).sum() - \
                       (det_features[i] * anchors[0]).sum()
                all_scores.append(score.item())
                all_labels.append(labels[i].item())

                # Anomaly map
                sim_a = (patch_features[i] @ anchors[1]).squeeze()
                sim_n = (patch_features[i] @ anchors[0]).squeeze()
                amap = (sim_a - sim_n).cpu().numpy()
                all_maps.append(amap)
                all_masks.append(masks[i].numpy())

    # Image-level AUROC
    img_auroc = roc_auc_score(all_labels, all_scores)

    # Pixel-level AUROC
    flat_masks = np.concatenate([m.flatten() for m in all_masks])
    flat_maps = np.concatenate([m.flatten() for m in all_maps])
    pixel_auroc = roc_auc_score(flat_masks > 0, flat_maps)

    return {
        "image_auroc": img_auroc,
        "pixel_auroc": pixel_auroc
    }
```

### 6.3 Baselines de comparaison

| Méthode | Type | Image AUROC | Pixel AUROC |
|---------|------|-------------|-------------|
| AA-CLIP (original) | CLIP-adapted | ~95% | ~93% |
| WinCLIP | CLIP zero-shot | ~91% | ~85% |
| AnomalyCLIP | CLIP prompt-learning | ~93% | ~89% |
| SigLIP raw | Zero-shot | À mesurer | À mesurer |
| **AA-SigLIP (ours)** | SigLIP-adapted | **À mesurer** | **À mesurer** |

---

## 7. Optimisation et déploiement

### 7.1 Optimisation pour l'inférence

```python
import torch
from torch.quantization import quantize_dynamic

def optimize_for_inference(model, save_path):
    """
    Optimiser le modèle pour l'inférence (edge ou serveur).
    """
    model.eval()

    # 1. Quantification dynamique (INT8)
    quantized_model = quantize_dynamic(
        model,
        {torch.nn.Linear},
        dtype=torch.qint8
    )

    # 2. Export ONNX pour TensorRT
    dummy_image = torch.randn(1, 3, 384, 384)
    torch.onnx.export(
        model,
        dummy_image,
        f"{save_path}/aa_siglip.onnx",
        input_names=["image"],
        output_names=["det_features", "patch_features"],
        dynamic_axes={
            "image": {0: "batch_size"},
            "det_features": {0: "batch_size"},
            "patch_features": {0: "batch_size"}
        },
        opset_version=14
    )

    # 3. Sauvegarder les ancres texte pré-calculées
    # Pré-calculer les ancres pour éviter l'inférence texte à runtime
    text_anchors = {}
    # Pré-calculer les ancres pour les classes MVTec AD
    mvtec_classes = [
        "bottle", "cable", "capsule", "carpet", "grid", "hazelnut",
        "leather", "metal_nut", "pill", "screw", "tile", "toothbrush",
        "transistor", "wood", "zipper"
    ]
    for cls_name in mvtec_classes:
        text_anchors[cls_name] = get_text_anchors(model, cls_name, "cpu")
    torch.save(text_anchors, f"{save_path}/text_anchors.pt")

    return quantized_model
```

### 7.2 Configuration d'inférence

```yaml
# config_inference.yaml
model:
  checkpoint: "aa_siglip_quantized.pt"
  text_anchors: "text_anchors.pt"

inference:
  batch_size: 1
  image_size: 384
  use_tensorrt: true
  fp16: true

thresholds:
  anomaly_detection: 0.5
  anomaly_localization: 0.3
```

---

## 8. Structure du projet

```
aa-siglip/
├── CLAUDE.md                    # Ce fichier
├── configs/
│   ├── train_config.yaml
│   └── eval_config.yaml
├── data/
│   └── README.md               # Instructions pour télécharger MVTec, VisA
├── src/
│   ├── __init__.py
│   ├── model/
│   │   ├── __init__.py
│   │   ├── adapted_siglip.py   # Classe AdaptedSigLIP
│   │   ├── adapters.py         # SimpleAdapter, SimpleProj
│   │   └── losses.py           # DisentangleLoss, SegmentationLoss
│   ├── data/
│   │   ├── __init__.py
│   │   ├── mvtec.py            # MVTec AD dataset
│   │   └── visa.py             # VisA dataset
│   ├── training/
│   │   ├── __init__.py
│   │   ├── trainer.py          # Pipeline d'entraînement 2-stages
│   │   └── utils.py            # Helpers
│   └── evaluation/
│       ├── __init__.py
│       ├── metrics.py          # AUROC, etc.
│       └── visualization.py    # Anomaly maps, t-SNE
├── scripts/
│   ├── diagnose_anomaly_unawareness.py
│   ├── train.py
│   ├── evaluate.py
│   └── export_onnx.py
├── notebooks/
│   ├── 01_diagnostic.ipynb
│   ├── 02_training.ipynb
│   └── 03_evaluation.ipynb
├── requirements.txt
└── setup.py
```

---

## 9. Prochaines étapes (RÉVISÉES)

### Phase 0: Diagnostic Moondream (PRIORITAIRE)
- [ ] Implémenter les tests d'anomaly awareness pour Moondream (Section 11)
- [ ] Évaluer zero-shot sur MVTec AD (accuracy, hallucination rate, miss rate)
- [ ] Analyser la séparation des features vision (silhouette score)
- [ ] **DÉCISION** : Continuer avec Moondream ou utiliser AA-CLIP ?

### Phase 1: Si Moondream viable → Fine-tuning direct
- [ ] Préparer dataset QA anomaly (MVTec + VisA)
- [ ] Fine-tuner Moondream (LoRA) sur tâche classification + explication
- [ ] Évaluer les performances post-fine-tuning

### Phase 1-alt: Si Moondream non viable → AA-CLIP
- [ ] Utiliser AA-CLIP original (OpenCLIP)
- [ ] Optionnel: Pipeline hybride AA-CLIP + Moondream pour explication

### Phase 2: Évaluation finale
- [ ] Benchmark sur MVTec AD, BTAD, MPDD
- [ ] Comparaison avec baselines (WinCLIP, AnomalyCLIP, AA-CLIP)
- [ ] Documentation des résultats

---

## 10. Diagnostic de l'Anomaly Awareness pour Moondream

Cette section définit les tests pour évaluer si Moondream peut être utilisé pour la détection d'anomalies.

### 10.1 Contexte

Contrairement à CLIP qui peut être évalué via similarité cosinus text-image, Moondream est un modèle génératif. Nous devons adapter les métriques.

### 10.2 Test 1: Classification Binaire Zero-shot

```python
def test_binary_classification(moondream, test_images):
    """
    Mesure la capacité de Moondream à distinguer normal/anomaly.

    Métrique: Accuracy (random = 50%, bon > 80%)
    """
    results = []

    for img, label, class_name in test_images:
        response = moondream.query(
            img,
            f"Is this {class_name} normal or defective? Answer only 'normal' or 'defective'."
        )

        predicted = "defective" if "defective" in response.lower() else "normal"
        correct = (predicted == "defective") == (label == 1)
        results.append({
            "correct": correct,
            "predicted": predicted,
            "actual": "defective" if label == 1 else "normal",
            "response": response
        })

    accuracy = sum(r["correct"] for r in results) / len(results)
    return accuracy, results
```

### 10.3 Test 2: Taux de Hallucination (False Positives)

```python
def test_hallucination_rate(moondream, normal_images):
    """
    Sur des images NORMALES, combien de fois Moondream invente des défauts ?

    Métrique: Taux de faux positifs (bon < 10%)
    """
    false_positives = 0
    details = []

    for img, class_name in normal_images:
        response = moondream.query(
            img,
            f"Describe any defects you see on this {class_name}. If none, say 'no defects'."
        )

        no_defect_phrases = ["no defect", "normal", "none", "no visible", "appears normal", "no issue"]
        is_hallucination = not any(phrase in response.lower() for phrase in no_defect_phrases)

        if is_hallucination:
            false_positives += 1

        details.append({
            "class": class_name,
            "hallucination": is_hallucination,
            "response": response
        })

    rate = false_positives / len(normal_images)
    return rate, details
```

### 10.4 Test 3: Taux de Manqués (False Negatives)

```python
def test_miss_rate(moondream, anomaly_images):
    """
    Sur des images ANOMALES, combien de fois Moondream manque le défaut ?

    Métrique: Taux de faux négatifs (bon < 15%)
    """
    misses = 0
    details = []

    for img, class_name, defect_type in anomaly_images:
        response = moondream.query(
            img,
            f"Examine this {class_name} carefully. Is it normal or does it have any defects?"
        )

        # Miss = dit que c'est normal alors que c'est anomal
        is_miss = "normal" in response.lower() and "not normal" not in response.lower()

        if is_miss:
            misses += 1

        details.append({
            "class": class_name,
            "defect_type": defect_type,
            "missed": is_miss,
            "response": response
        })

    rate = misses / len(anomaly_images)
    return rate, details
```

### 10.5 Test 4: Séparation des Features Vision

```python
import numpy as np
from sklearn.metrics import silhouette_score

def test_vision_feature_separation(moondream, images, labels):
    """
    Test si le vision encoder de Moondream (SigLIP) sépare normal/anomaly.

    Métrique: Silhouette score (random ≈ 0, bon > 0.3)
    """
    features = []

    for img in images:
        with torch.no_grad():
            # Accéder directement au vision encoder de Moondream
            # Note: L'API exacte dépend de la version de Moondream
            vision_features = moondream.encode_image(img)

            # Pooling global si nécessaire
            if len(vision_features.shape) > 2:
                pooled = vision_features.mean(dim=1)
            else:
                pooled = vision_features

            features.append(pooled.cpu().numpy())

    features = np.vstack(features)
    labels = np.array(labels)

    # Silhouette: mesure de séparation des clusters
    score = silhouette_score(features, labels)

    return score
```

### 10.6 Test 5: Cohérence des Descriptions

```python
def test_description_consistency(moondream, anomaly_images_with_gt):
    """
    Vérifie si les descriptions de Moondream correspondent aux défauts réels.

    Métrique: Taux de correspondance (bon > 70%)
    """
    matches = 0
    details = []

    defect_keywords = {
        "scratch": ["scratch", "scratched", "line", "mark"],
        "crack": ["crack", "cracked", "broken", "fracture"],
        "contamination": ["contamination", "dirt", "stain", "spot"],
        "missing": ["missing", "absent", "hole", "gap"],
        "bent": ["bent", "deformed", "warped", "twisted"],
    }

    for img, class_name, defect_type in anomaly_images_with_gt:
        response = moondream.query(
            img,
            f"Describe the defect on this {class_name} in detail."
        )

        # Vérifier si le type de défaut est mentionné
        keywords = defect_keywords.get(defect_type.lower(), [defect_type.lower()])
        found = any(kw in response.lower() for kw in keywords)

        if found:
            matches += 1

        details.append({
            "class": class_name,
            "defect_type": defect_type,
            "matched": found,
            "response": response
        })

    rate = matches / len(anomaly_images_with_gt)
    return rate, details
```

### 10.7 Script de Diagnostic Complet

```python
def run_moondream_diagnostic(moondream, mvtec_path, output_path="diagnostic_results.json"):
    """
    Exécute tous les tests diagnostiques et génère un rapport.
    """
    from datasets import load_dataset
    import json

    # Charger MVTec AD (ou sous-ensemble)
    # Note: Adapter selon le format de données disponible

    results = {
        "model": "moondream-v2",  # ou v3
        "dataset": "mvtec-ad",
        "tests": {}
    }

    # Test 1: Classification binaire
    accuracy, details = test_binary_classification(moondream, test_images)
    results["tests"]["binary_classification"] = {
        "accuracy": accuracy,
        "threshold": 0.80,
        "passed": accuracy > 0.80
    }

    # Test 2: Hallucination
    hall_rate, details = test_hallucination_rate(moondream, normal_images)
    results["tests"]["hallucination"] = {
        "rate": hall_rate,
        "threshold": 0.10,
        "passed": hall_rate < 0.10
    }

    # Test 3: Miss rate
    miss_rate, details = test_miss_rate(moondream, anomaly_images)
    results["tests"]["miss_rate"] = {
        "rate": miss_rate,
        "threshold": 0.15,
        "passed": miss_rate < 0.15
    }

    # Test 4: Feature separation
    silhouette = test_vision_feature_separation(moondream, all_images, labels)
    results["tests"]["feature_separation"] = {
        "silhouette_score": silhouette,
        "threshold": 0.30,
        "passed": silhouette > 0.30
    }

    # Test 5: Description consistency
    consistency, details = test_description_consistency(moondream, anomaly_images_with_gt)
    results["tests"]["description_consistency"] = {
        "rate": consistency,
        "threshold": 0.70,
        "passed": consistency > 0.70
    }

    # Verdict global
    passed_tests = sum(1 for t in results["tests"].values() if t["passed"])
    results["summary"] = {
        "passed": passed_tests,
        "total": len(results["tests"]),
        "recommendation": "proceed_with_finetuning" if passed_tests >= 3 else "use_aa_clip_instead"
    }

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    return results
```

### 10.8 Résultats Attendus et Seuils

| Test | Métrique | Seuil "Bon" | Si échec |
|------|----------|-------------|----------|
| Classification binaire | Accuracy | > 80% | Moondream ne distingue pas normal/anomaly |
| Hallucination | FP Rate | < 10% | Trop de fausses alertes |
| Miss rate | FN Rate | < 15% | Manque trop de défauts |
| Feature separation | Silhouette | > 0.30 | Vision encoder non discriminant |
| Description consistency | Match rate | > 70% | Descriptions non fiables |

**Interprétation:**
- **≥ 4/5 tests passés** : Moondream viable → procéder au fine-tuning
- **2-3/5 tests passés** : Fine-tuning intensif nécessaire
- **< 2/5 tests passés** : Utiliser AA-CLIP à la place

---

## 11. Ressources et références

### Papers
1. [AA-CLIP (Ma et al., CVPR 2025)](https://arxiv.org/abs/2503.06661) - Architecture de base
2. [SigLIP (Zhai et al., 2023)](https://arxiv.org/abs/2303.15343) - Sigmoid loss pour vision-language
3. [WinCLIP (Jeong et al., CVPR 2023)](https://arxiv.org/abs/2303.14814) - Zero-shot avec CLIP
4. [AnomalyCLIP (Zhou et al., ICLR 2024)](https://arxiv.org/abs/2310.18961) - Prompt learning

### Code
- [AA-CLIP GitHub](https://github.com/Mwxinnn/AA-CLIP)
- [OpenCLIP](https://github.com/mlfoundations/open_clip)
- [Transformers SigLIP](https://huggingface.co/docs/transformers/en/model_doc/siglip)
- [Moondream](https://github.com/vikhyat/moondream)

### Datasets
- [MVTec AD](https://www.mvtec.com/company/research/datasets/mvtec-ad)
- [VisA](https://github.com/amazon-science/spot-diff)
- [BTAD](https://github.com/pankajmishra000/BTAD)
