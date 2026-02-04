# Anomaly Detection Project - Task List

## Status Legend
- [ ] Todo
- [x] Done
- [~] In Progress
- [!] Blocked

---

## Phase 0: Diagnostic Moondream (PRIORITAIRE)

### 0.1 Préparation
- [x] Télécharger MMAD dataset (4,562 QA samples, 8 catégories MVTec)
- [x] Télécharger MMAD images (28GB - DS-MVTec, MVTec-AD, VisA)
- [x] Créer script de diagnostic (`scripts/diagnostic_moondream.py`)
- [ ] Exécuter sur H100 avec Moondream 3

### 0.2 Tests d'Anomaly Awareness (via MMAD benchmark)
- [ ] **Test 1**: Defect Detection (binary classification)
  - Input: MMAD "defect_detection" questions (934 samples)
  - Output: Accuracy (seuil: >80%)

- [ ] **Test 2**: Hallucination Rate (false positives)
  - Input: Normal images from MMAD
  - Output: False Positive Rate (seuil: <10%)

- [ ] **Test 3**: Miss Rate (false negatives)
  - Input: Anomaly images from MMAD
  - Output: False Negative Rate (seuil: <15%)

- [ ] **Test 4**: Defect Type Classification
  - Input: MMAD "defect_type" questions (665 samples)
  - Output: Accuracy (seuil: >70%)

- [ ] **Test 5**: Defect Location
  - Input: MMAD "location" questions (805 samples)
  - Output: Accuracy (informational)

### 0.3 Analyse et Décision
- [ ] Compiler les résultats dans `diagnostic_results.json`
- [ ] Documenter les observations qualitatives (ex: confusion bottle/iris)
- [ ] **DÉCISION**: Choisir stratégie selon résultats
  - ≥4/5 tests passés → Phase 1A (Fine-tuning Moondream)
  - 2-3/5 tests passés → Phase 1A avec fine-tuning intensif
  - <2/5 tests passés → Phase 1B (AA-CLIP)

---

## Phase 1A: Fine-tuning Moondream (si diagnostic positif)

### 1A.1 Préparation données
- [ ] Convertir MVTec AD en format conversationnel
  ```json
  {
    "image": "path/to/image.png",
    "conversations": [
      {"role": "user", "content": "Is this bottle normal or defective?"},
      {"role": "assistant", "content": "Defective. There is a crack..."}
    ]
  }
  ```
- [ ] Créer split train/val/test
- [ ] Augmenter les données si nécessaire

### 1A.2 Fine-tuning
- [ ] Configurer LoRA pour Moondream
- [ ] Définir hyperparamètres (lr, epochs, batch_size)
- [ ] Entraîner sur tâche classification + description
- [ ] Sauvegarder checkpoints

### 1A.3 Évaluation post-fine-tuning
- [ ] Re-exécuter les 5 tests diagnostiques
- [ ] Calculer amélioration vs baseline
- [ ] Évaluer sur catégories non vues (zero-shot transfer)

---

## Phase 1B: AA-CLIP (si diagnostic négatif)

### 1B.1 Setup AA-CLIP
- [ ] Cloner repo AA-CLIP
- [ ] Télécharger weights pré-entraînés
- [ ] Vérifier reproduction des résultats paper

### 1B.2 Intégration (optionnel: hybride)
- [ ] Créer pipeline AA-CLIP → score + heatmap
- [ ] Si anomaly détectée → Moondream pour explication
- [ ] Benchmark du pipeline complet

---

## Phase 2: Évaluation Finale

### 2.1 Benchmarks
- [ ] MVTec AD (15 catégories)
  - [ ] Image-level AUROC
  - [ ] Pixel-level AUROC (si applicable)
- [ ] VisA (12 catégories)
- [ ] BTAD (3 catégories) - zero-shot
- [ ] MPDD (6 catégories) - zero-shot

### 2.2 Comparaison Baselines
- [ ] WinCLIP (zero-shot CLIP)
- [ ] AnomalyCLIP (prompt learning)
- [ ] AA-CLIP (si non utilisé comme méthode principale)

### 2.3 Analyse
- [ ] Performance par catégorie
- [ ] Analyse des échecs (quels types de défauts manqués)
- [ ] Temps d'inférence

---

## Phase 3: Documentation et Déploiement

### 3.1 Documentation
- [ ] Mettre à jour CLAUDE.md avec résultats finaux
- [ ] Créer README utilisateur
- [ ] Documenter API/usage

### 3.2 Optimisation (si nécessaire)
- [ ] Quantification (INT8)
- [ ] Export ONNX
- [ ] Benchmark latence

---

## Questions Ouvertes

1. **Moondream v2 vs v3** : Quelle version utiliser pour les tests ?
2. **Subset MVTec** : Tester sur toutes les catégories ou subset représentatif ?
3. **Seuils diagnostics** : Les seuils proposés sont-ils appropriés ?
4. **Few-shot** : Tester aussi le few-shot (k=1, 2, 4 examples) ?

---

## Ressources Nécessaires

- [ ] GPU pour inférence Moondream (~8GB VRAM minimum)
- [ ] Storage pour MVTec AD (~5GB)
- [ ] Storage pour VisA (~20GB)

---

## Journal de Décisions

| Date | Décision | Justification |
|------|----------|---------------|
| TBD | Choix stratégie | Basé sur résultats diagnostic Phase 0 |

---

## Notes

### Observations préliminaires (tests manuels)
- Moondream confond parfois les angles industriels (bottle → iris)
- Moondream ne semble pas comprendre "anomalie industrielle" vs "objet endommagé naturellement"
- Besoin de prompts très spécifiques pour obtenir des réponses utiles

### Références rapides
- AA-CLIP outputs: `score` (scalar) + `anomaly_map` (H×W heatmap)
- Moondream outputs: `text` (description) + `boxes/points` (localisation)
- MVTec AD: 15 categories, ~5000 images total
