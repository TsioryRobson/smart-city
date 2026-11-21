# Suivi des expérimentations (MLflow)

Expérience : `novaria-air-anomaly`. Connexion : `novaria.common.tracking.setup_mlflow` utilise `MLFLOW_TRACKING_URI`
si elle est définie (serveur MLflow, PostgreSQL et MinIO au sprint 1 avec Docker Compose), sinon un suivi local
(`mlflow.db` et `mlartifacts/`, non versionnés). Le code est le même dans les deux cas.

## Ce qui est enregistré

| Élément | Entraînement (`training.py`) | Évaluation (`evaluation.py`) |
|---|---|---|
| Structure | 1 run parent `training` + 1 run par réglage (`<algorithme>-<n>`, 20 au total) | 1 run parent `evaluation` + 1 run par algorithme (`evaluation-<algorithme>`) |
| Hyperparamètres | `algorithm`, `scaler`, paramètres de l'algorithme (`n_estimators`, `max_samples`, `max_features`, `n_neighbors`, `gamma`, `nu`), `contamination`, `max_train_samples` | mêmes hyperparamètres que le modèle évalué ; parent : `injection_seeds`, `retrain_runs`, `train_fraction`, règle de sélection |
| Métriques | validation contrôlée : `val_detection_rate_*` (par type), `val_false_alarm_rate`, `val_youden_pic`, `val_roc_auc_pic`, `val_average_precision_pic` ; coût : `fit_time_s`, `inference_time_ms_per_row`, `latency_ms_single_row`, `latency_ok` | test contrôlé : `test_*_mean` et `test_*_std` sur 5 tirages, `retrain_youden_pic_std`, `latency_ms_single_row`, `model_size_mb` |
| Jeux de données (onglet *Datasets*) | `train` (ou `train_echantillon`) et `validation_controlee`, identifiés par l'empreinte SHA-256 du fichier, avec la graine et le nombre d'anomalies injectées | `test_controle_seed0` à `seed4`, empreinte SHA-256 et nombre d'anomalies par type |
| Artefacts | parent : `training_config.json`, `data/metadata.json`, `comparison.json`, `training_comparison.md` ; run retenu de chaque algorithme : `model/` | parent : `evaluation_config.json`, `evaluation.json`, `selection.json`, `evaluation.md`, `evaluation_stability.png` |
| Commit Git | tag `git_sha` (suffixe `-dirty` si modifications non commitées) ; tag automatique MLflow `mlflow.source.git.commit` en ligne de commande | tag `git_sha` sur le parent et chaque run d'algorithme |
| Données | tag `dataset_sha256` (jeu préparé) | tag `dataset_sha256`, empreintes des jeux contrôlés |
| Modèle | sur le run retenu : tags `selected=true`, `model_sha256` | tags `training_run_id`, `model_uri`, `model_sha256`, `selected` |

## Modèle : signature et dépendances

Chaque modèle retenu (`runs:/<run_id>/model`) contient :
- **signature** : 17 variables d'entrée (`MODEL_FEATURES`), toutes optionnelles (une valeur manquante est imputée),
  sortie = score d'anomalie ;
- **dépendances** : `requirements.txt` et `python_env.yaml` (mlflow, scikit-learn, pandas, numpy, skops, versions
  figées), et le code `novaria` embarqué (`code/`) ;
- **format skops** : seuls les types de `SKOPS_TRUSTED_TYPES` sont acceptés au chargement ;
- un exemple d'entrée (`input_example.json`).

Les 17 réglages non retenus n'enregistrent que leurs métriques et paramètres : stocker leur modèle n'apporte rien à
la comparaison et coûterait jusqu'à 100 Mo par modèle LOF.

## Comparer les 3 algorithmes dans l'interface

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db    # puis http://localhost:5000
```

- **Tous les réglages :** expérience `novaria-air-anomaly`, filtre `tags.step = 'training' and params.algorithm != ''`,
  puis sélectionner les runs et **Compare** (graphique `val_detection_rate_pic` en fonction de `val_false_alarm_rate`,
  coloré par `algorithm`).
- **Les 3 modèles retenus :** filtre `tags.selected = 'true' and tags.step = 'training'`.
- **Le test contrôlé :** filtre `tags.step = 'evaluation' and tags.algorithm != ''` ; le run marqué `selected=true`
  est le modèle choisi.

Vérifié via l'API du serveur MLflow : les 20 runs d'entraînement ont les mêmes 13 métriques, sans valeur manquante,
et les 3 runs d'évaluation désignent One-Class SVM (`selected=true`).

## Limites actuelles

- Suivi local en SQLite tant que Docker Compose (PostgreSQL + MinIO + serveur MLflow) n'est pas en place.
- Les runs existants ont été lancés avant commit (`git_sha` en `-dirty`) ; les runs d'entraînement existants datent
  d'avant l'enregistrement des jeux de données. Une relance après commit rendra la traçabilité complète.
- Le Registry (`air-anomaly-detector`, alias `@champion` / `@previous`) sera alimenté par `promotion.py`.
