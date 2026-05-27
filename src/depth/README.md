# Estimation de distance

Deux cas sont gardés dans le repo.

## 1. MultiSense avec profondeur capteur

Pour les enregistrements `cegelecRecordings`, utiliser :

```bash
python src/signals/estimate_msense_signal_distance.py
```

Le pipeline lit :

- `COLOR/color_i.mkv` pour la détection YOLO ;
- `DEPTH/depth_i.mkv` pour la profondeur `gray16le` en millimètres ;
- `TIME/avi_i.time` pour timestamp, position INS/GNSS et vitesse.

La distance retournée est la médiane des pixels de profondeur valides dans la
bbox détectée, convertie en mètres.

## 2. RGB seul avec Depth Anything

Quand il n'y a pas de carte de profondeur capteur, utiliser les estimateurs
monoculaires :

- `estimate_dav2_distance.py` : Depth Anything V2 metric, image par image ;
- `estimate_da3_distance.py` : Depth Anything 3 metric, image par image ;
- `estimate_vda_distance.py` : Video Depth Anything metric, cohérence temporelle.

Ces scripts ne sauvegardent plus les depth maps `.npy` par défaut. Ajouter
`--save-depth` uniquement si les cartes de profondeur sont nécessaires pour du
debug ou une analyse hors ligne.

## Benchmark

`benchmark_depth_methods.py` compare uniquement les méthodes basées profondeur :

```bash
PYTHONPATH=. python src/depth/benchmark_depth_methods.py \
  --methods vda,dav2,da3
```

Les anciennes estimations géométriques approximatives par taille apparente ou
homographie rail ont été retirées du pipeline propre.

Pour les enregistrements MultiSense, utiliser le benchmark dédié afin de
comparer la profondeur capteur aux méthodes Depth Anything sur les mêmes
frames et les mêmes bboxes :

```bash
PYTHONPATH=. python src/depth/benchmark_msense_vs_depth_anything.py \
  --data-root /gpfs/projects/acad/brainai/cegelecRecordings \
  --ckpt /gpfs/projects/acad/brainai/tvans_distance/yolo_signs_best.pt \
  --seqs 0,1,2,3 \
  --start 0 --end 99 --stride 1 \
  --methods msense,dav2,da3,vda \
  --out /gpfs/projects/acad/brainai/tvans_distance/depth_benchmark
```

Sorties principales :

- `distance_comparison_wide.csv` : distance capteur, prédictions Deep Anything
  et erreurs absolues/relatives ;
- `method_summary.csv` : erreurs médianes, FPS et RAM max ;
- `metrics.json` : métriques détaillées par étape.
