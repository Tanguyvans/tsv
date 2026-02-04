# Comparaison des Architectures : Moondream v2, v3 et AA-CLIP

## Objectif

Analyser la faisabilité d'appliquer la méthode AA-CLIP (adapters pour anomaly detection) à Moondream.

---

## ⚠️ CONCLUSION CRITIQUE

**L'approche AA-CLIP ne peut PAS être directement appliquée à Moondream.**

| Composant requis par AA-CLIP | Moondream | Statut |
|------------------------------|-----------|--------|
| Vision encoder (ViT) | SigLIP ViT | ✅ Présent |
| Text encoder contrastif | LLM (Phi-1.5) | ❌ **NON ALIGNÉ** |
| Espace embeddings partagé | Séparés | ❌ **INCOMPATIBLE** |

Le LLM de Moondream n'est pas entraîné contrastivement avec le vision encoder. Les text anchors "normal"/"anomaly" ne peuvent pas être générés comme dans AA-CLIP.

**Alternatives viables :**
1. Utiliser AA-CLIP original (OpenCLIP) - prouvé, SOTA
2. Fine-tuner Moondream directement sur tâche QA anomaly - simple mais perd zero-shot
3. Pipeline hybride AA-CLIP + Moondream pour explication

---

## 1. Vue d'ensemble

| Composant | AA-CLIP | Moondream v2 | Moondream v3 |
|-----------|---------|--------------|--------------|
| **Vision Encoder** | OpenCLIP ViT-L/14 | SigLIP SoViT-400M | SigLIP (multi-crop) |
| **Text/LLM** | CLIP Text Encoder | Phi-1.5 (1.3B) | MoE (9B total, 2B actifs) |
| **Params totaux** | ~400M | ~1.86B | ~9B |
| **Résolution image** | 518×518 | 378×378 | Variable (multi-crop) |

---

## 2. Vision Encoder - Comparaison détaillée

### AA-CLIP (ViT-L/14)

| Spec | Valeur |
|------|--------|
| Architecture | Vision Transformer Large |
| Layers | **24** |
| Hidden dim | **1024** |
| Attention heads | 16 |
| MLP dim | 4096 |
| Patch size | 14×14 |
| Input resolution | 518×518 |
| Params | ~300M |

### Moondream v2 (SigLIP SoViT-400M)

| Spec | Valeur |
|------|--------|
| Architecture | Shape-Optimized ViT |
| Layers | **27-28** |
| Hidden dim | **1152** |
| Attention heads | 16 |
| MLP/FFN dim | 4304 |
| Patch size | 14×14 |
| Input resolution | 378×378 |
| Position embeddings | 729 (27×27 patches) |
| Params | ~400M |

### Moondream v3 (SigLIP + Multi-crop)

| Spec | Valeur |
|------|--------|
| Architecture | SigLIP avec channel concatenation |
| Layers | ~27-28 (similaire v2) |
| Hidden dim | **1152** (estimé) |
| Attention heads | 16 |
| Patch size | 14×14 |
| Input resolution | Variable (multi-crop) |
| Params | ~400M |

---

## 3. Architecture des Adapters AA-CLIP

### Structure de l'Adapter

**IMPORTANT** : AA-CLIP utilise des adapters **linéaires simples**, PAS des bottleneck adapters.

```
Adapter AA-CLIP (par layer):
┌─────────────────────────────────────────┐
│  Input x ∈ R^(N×d)                      │
│          ↓                              │
│  W^i ∈ R^(d×d)  [Transformation]        │
│          ↓                              │
│  Act(·)         [Activation]            │
│          ↓                              │
│  Norm(·)        [Normalization]         │
│          ↓                              │
│  x_residual                             │
│          ↓                              │
│  x_enhanced = λ·x_residual + (1-λ)·x    │
│  avec λ = 0.1                           │
└─────────────────────────────────────────┘
```

### Configuration AA-CLIP

| Composant | Layers avec Adapter | Params par adapter |
|-----------|--------------------|--------------------|
| Text Encoder | 3 premiers layers (K_T=3) | 1024×1024 = ~1M |
| Visual Encoder | 6 premiers layers (K_I=6) | 1024×1024 = ~1M |
| **Total trainable** | 9 layers | **~10-12M params** |

### Extraction Multi-granularité

Features extraites des layers : **6, 12, 18, 24**

```
Layer 6  ──→ Features bas niveau (textures)
Layer 12 ──→ Features moyen niveau (patterns)
Layer 18 ──→ Features haut niveau (objets)
Layer 24 ──→ Features sémantiques
      ↓
   Fusion → Anomaly Detection
```

---

## 4. Transposition à Moondream

### 4.1 Adaptation des Adapters

| AA-CLIP (ViT-L/14) | Moondream (SigLIP) | Adaptation |
|--------------------|---------------------|------------|
| d = 1024 | d = 1152 | **Changer dim adapter** |
| 24 layers | 27-28 layers | **Plus de layers disponibles** |
| K_I = 6 layers | K_I = 6-8 layers | Compatible |
| λ = 0.1 | λ = 0.1 | Identique |

### 4.2 Adapter pour Moondream

```python
class AAAdapter(nn.Module):
    """Adapter style AA-CLIP pour Moondream"""
    def __init__(self, dim=1152, lambda_=0.1):
        super().__init__()
        self.W = nn.Linear(dim, dim)  # d×d, pas de bottleneck
        self.act = nn.GELU()
        self.norm = nn.LayerNorm(dim)
        self.lambda_ = lambda_

    def forward(self, x):
        residual = self.norm(self.act(self.W(x)))
        return self.lambda_ * residual + (1 - self.lambda_) * x
```

### 4.3 Params Estimés pour Moondream

| Config | Layers | Params par adapter | Total |
|--------|--------|-------------------|-------|
| Minimal | 4 | 1152² = 1.33M | ~5.3M |
| Standard (comme AA-CLIP) | 6 | 1.33M | ~8M |
| Étendu | 8 | 1.33M | ~10.6M |

---

## 5. Différences Clés

### 5.1 Text Encoder vs LLM

| AA-CLIP | Moondream |
|---------|-----------|
| CLIP Text Encoder (frozen) | LLM Phi-1.5 ou MoE (frozen) |
| Adapters sur text encoder | Pas d'adapters sur LLM |
| Text prompts : "normal/anomaly" | LLM pour query/explain |

**Impact** : Moondream n'a pas besoin d'adapters sur le text encoder car on utilise le LLM différemment (pas de contrastive learning text-image).

### 5.2 Multi-granularité

```
AA-CLIP:      Layers [6, 12, 18, 24] sur 24 total
Moondream:    Layers [7, 14, 21, 27] sur 27 total (proposition)
```

### 5.3 Training Pipeline

**AA-CLIP (2 stages) :**
1. Stage 1 : Train text adapters (5 epochs, lr=1e-5)
2. Stage 2 : Train visual adapters (20 epochs, lr=5e-4)

**Moondream adapté (1 stage) :**
1. Train visual adapters seulement (pas de text encoder à adapter)
2. ~20 epochs, lr=5e-4

---

## 6. Ce qui est FAISABLE

| Fonctionnalité | Faisable | Notes |
|----------------|----------|-------|
| Adapters dans vision encoder | ✅ OUI | SigLIP = ViT, même principe |
| Structure adapter linéaire d×d | ✅ OUI | Juste changer d=1024 → d=1152 |
| Multi-granularité | ✅ OUI | Adapter les indices de layers |
| Fusion factor λ=0.1 | ✅ OUI | Identique |
| Gel du vision encoder | ✅ OUI | Comme AA-CLIP |
| Text encoder adapters | ❌ NON NÉCESSAIRE | Moondream utilise LLM différemment |

---

## 7. Ce qui CHANGE

| Aspect | AA-CLIP | Moondream + AA |
|--------|---------|----------------|
| Loss contrastive text-image | Oui | Non (pas de text encoder CLIP) |
| Anomaly score | Similarité text-image | Tête de détection dédiée |
| Segmentation | Patch-level similarity | detect()/segment() ou tête dédiée |
| Explicabilité | Aucune | query() pour explication textuelle |

---

## 8. Architecture Proposée : AA-Moondream

```
┌─────────────────────────────────────────────────────────────────┐
│                        AA-Moondream                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │         SigLIP Vision Encoder (FROZEN)                  │   │
│  │                                                         │   │
│  │  [Block 1] ──→ [Block 2] ──→ ... ──→ [Block 27]        │   │
│  │      ↓            ↓                      ↓              │   │
│  │   +Adapt_1     +Adapt_2    ...      (no adapt)         │   │
│  │      │            │           │          │              │   │
│  │      ↓            ↓           ↓          ↓              │   │
│  │   [feat_7]    [feat_14]  [feat_21]  [feat_27]          │   │
│  │      └────────────┴──────────┴──────────┘              │   │
│  │                        ↓                                │   │
│  │              Multi-scale Fusion                         │   │
│  └─────────────────────────────────────────────────────────┘   │
│                           ↓                                     │
│              ┌────────────┴────────────┐                       │
│              ↓                         ↓                       │
│  ┌───────────────────┐    ┌───────────────────────────────┐   │
│  │  Anomaly Head     │    │  Projection → LLM (FROZEN)    │   │
│  │  (TRAINABLE)      │    │                               │   │
│  │                   │    │  detect("defect")             │   │
│  │  - Score: 0-1     │    │  segment("anomaly")           │   │
│  │  - Anomaly Map    │    │  query("What is wrong?")      │   │
│  └───────────────────┘    └───────────────────────────────┘   │
│         ↓                              ↓                       │
│    FAST (~20ms)                  DETAILED (~200ms)             │
│    Detection                     Explanation                   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘

Composants TRAINABLES:
- Adapters dans layers [1-6] du vision encoder (~8M params)
- Anomaly Detection Head (~2M params)
- Total: ~10M params (comme AA-CLIP)

Composants FROZEN:
- SigLIP Vision Encoder (~400M params)
- Projection MLP
- LLM Phi-1.5/MoE
```

---

## 9. Comparaison des Params Trainables

| Modèle | Vision Adapters | Text/Other | Total Trainable | Total Model |
|--------|-----------------|------------|-----------------|-------------|
| AA-CLIP | ~6M (6 layers) | ~3M (3 layers) | ~10-12M | ~400M |
| AA-Moondream v2 | ~8M (6 layers) | ~2M (head) | ~10M | ~1.86B |
| AA-Moondream v3 | ~8M (6 layers) | ~2M (head) | ~10M | ~9B |

---

## 10. Conclusion : Faisabilité (RÉVISÉE)

### ⚠️ AA-CLIP sur Moondream : NON DIRECTEMENT FAISABLE

L'approche AA-CLIP originale **ne fonctionne pas** avec Moondream car :

1. **Pas de text encoder contrastif** : Le LLM (Phi-1.5) n'est pas aligné avec SigLIP
2. **Espaces d'embeddings séparés** : Les text anchors T_N/T_A ne peuvent pas être générés
3. **Architecture fondamentalement différente** : VLM génératif ≠ modèle contrastif

### Ce qui EST faisable

| Approche | Faisabilité | Commentaire |
|----------|-------------|-------------|
| Adapters sur vision encoder seul | ✅ Oui | Mais besoin d'anchors appris (pas de text) |
| Fine-tuning Moondream QA | ✅ Oui | Approche simple et directe |
| AA-SigLIP standalone | ✅ Oui | Utilise SigLIP complet, pas Moondream |
| AA-CLIP original | ✅ Oui | Solution prouvée, SOTA |

### Recommandation finale

**Étape 1 : Diagnostic**
Évaluer Moondream zero-shot sur MVTec avec les tests de la Section 10 de CLAUDE.md

**Étape 2 : Décision**
- Si résultats corrects → Fine-tuner Moondream directement
- Si résultats mauvais → Utiliser AA-CLIP + Moondream en hybride

### Avantages potentiels de Moondream (si fine-tuné)

1. **Explicabilité** : Descriptions textuelles des défauts
2. **Contexte** : Comprend le type d'objet et le contexte
3. **Flexibilité** : Questions ouvertes possibles
4. **Segmentation** : Localisation via detect()/point()

### Inconvénients

1. **Pas de zero-shot prouvé** pour anomalies industrielles
2. **Domain gap** : Entraîné sur images naturelles, pas industrielles
3. **Pas de heatmap dense** : Localisation moins précise qu'AA-CLIP
4. **Non benchmarké** : Aucun résultat publié sur MVTec/VisA

---

## 11. Prochaines Étapes (RÉVISÉES)

### Phase 0 : Diagnostic (PRIORITAIRE)
1. [ ] Exécuter les tests d'anomaly awareness sur Moondream (CLAUDE.md Section 10)
2. [ ] Documenter les résultats (accuracy, hallucination, miss rate, silhouette)
3. [ ] **DÉCISION** : Choisir la stratégie selon les résultats

### Si Moondream viable (≥3/5 tests passés)
4. [ ] Préparer dataset QA anomaly (MVTec format conversationnel)
5. [ ] Fine-tuner Moondream avec LoRA
6. [ ] Évaluer post-fine-tuning

### Si Moondream non viable (<3/5 tests passés)
4. [ ] Déployer AA-CLIP original
5. [ ] Optionnel : Pipeline hybride AA-CLIP → Moondream pour explication

---

## Références

- [AA-CLIP Paper](https://arxiv.org/abs/2503.06661)
- [AA-CLIP GitHub](https://github.com/Mwxinnn/AA-CLIP)
- [Moondream v2 HuggingFace](https://huggingface.co/vikhyatk/moondream2)
- [Moondream v3 Blog](https://moondream.ai/blog/moondream-3-preview)
- [SigLIP Paper](https://arxiv.org/abs/2303.15343)
