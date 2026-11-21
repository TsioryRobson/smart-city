# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Novaria Smart City AI (POC)

Contexte de projet pour Claude Code. À lire en entier avant toute tâche.

## 1. Le projet en bref

Projet fil rouge du Master 2 EADL (Bloc 05 · MLOps et MLSecOps), réalisé en binôme.
Nous jouons l'équipe d'ingénierie du département R&D de **Novaria**. La ville centralise des données IoT dans **Urban Hub**, mais n'a pas encore d'IA.

**Objectif :** un Proof of Concept qui **détecte des mesures de pollution de l'air atypiques** et **affiche une alerte** quand un capteur IoT envoie une mesure anormale. La plateforme doit pouvoir accueillir plus tard d'autres services IA (eau, trafic, énergie) sans refonte.

Ce qui est évalué : une **démonstration fonctionnelle** et la **justification technique**. Le résultat attendu n'est pas seulement un modèle ML, mais un POC d'ingénierie logicielle complet : entraîné, déployé, automatisé, sécurisé.

Soutenance : 20 min de présentation et 10 min de questions (contexte 3, algorithmes 4, architecture 4, démo 6, bilan 3).

## 2. Données

- Dataset : **Beijing Multi-Site Air Quality**, 12 stations, un CSV par station dans `data/raw/`.
- Données **horaires** de 2013-03 à 2017-02.
- Colonnes : `No, year, month, day, hour, PM2.5, PM10, SO2, NO2, CO, O3, TEMP, PRES, DEWP, RAIN, wd, WSPM, station`.
- **Entraînement : toutes les stations de `data/raw/`.** La **démo** (simulateur, exemples de contrat) utilise la station **Aotizhongxin**.
- Le simulateur IoT **rejoue 2016-03 → 2017-02**, en accéléré. Cette période est **réservée à la démo** : elle n'est jamais utilisée pour l'entraînement, la validation ni le test.
- `data/raw/` n'est pas versionné. Les CSV bruts ne sont jamais modifiés.

Points de qualité à traiter et à documenter :
- valeurs manquantes ;
- doublons d'horodatage (par station) : supprimés, on garde la première occurrence ;
- unités (CO est en µg/m³ dans ce dataset, à vérifier) ;
- incohérences ;
- **saisonnalité forte** (chauffage l'hiver, poussières au printemps).

Traitement des incohérences :
- **Mises à NaN puis imputées :** valeurs négatives, valeurs hors bornes physiques (bornes définies dans la config), et PM2.5 > PM10 (les deux valeurs).
- **Signalé seulement, sans modification :** capteur figé (même valeur répétée).
- **Jamais modifiés :** les pics extrêmes mais physiquement possibles. Ce sont les anomalies à détecter.
- **Journal de vérification :** chaque problème détecté (manquant, doublon, incohérence, capteur figé) est écrit dans un fichier `.log` texte (module `logging`) dans `logs/`. Chaque entrée contient la station, l'horodatage, la colonne, la valeur, le type d'erreur et l'action (signalé, supprimé ou imputé).

## 3. Règles ML non négociables

1. **Split chronologique** entraînement / validation / test, jamais aléatoire, pour éviter les fuites. Découpage de référence (version 1) :
   - train : 2014-09 → 2015-08 (fenêtre de 12 mois, voir règle 8) ;
   - validation : 2015-09 → 2015-11 ;
   - test : 2015-12 → 2016-02 ;
   - 2016-03 → 2017-02 : réservé au simulateur, jamais vu par le modèle.

   Les dates sont définies dans la config, pas en dur dans le code.
2. **Trois algorithmes non supervisés** comparés avec le même protocole :
   - `IsolationForest` ;
   - `LocalOutlierFactor(novelty=True)` ;
   - `OneClassSVM` (échantillonner si le volume est trop gros, et le justifier).
3. Les anomalies réelles ne sont pas étiquetées. L'évaluation se fait sur un **jeu de test contrôlé** : des données réelles de référence, plus des **anomalies synthétiques injectées et identifiées** (pics, capteur figé, valeurs impossibles).
4. **Ne jamais entraîner sur les anomalies injectées.**
5. Métriques à calculer : taux de détection des anomalies injectées, taux de fausses alertes sur les données de référence, temps d'entraînement, temps d'inférence.
6. Un score d'anomalie n'est **pas** un dépassement de seuil sanitaire. La **règle métier** de l'API distingue :
   - `pollution_atypique` ;
   - `defaut_capteur` ;
   - `seuil_sanitaire`.
7. Ne jamais prétendre valider une performance réelle. Les limites de l'évaluation sont documentées.
8. Fenêtre d'entraînement **glissante de 12 mois**, pour couvrir toutes les saisons, suivie de 3 mois de validation puis 3 mois de test.
9. **Un seul module de préparation** (`src/novaria/preparation/`), utilisé à l'entraînement **et** en streaming (job PyFlink). Ne jamais dupliquer la logique de préparation.

## 4. Stack

| Domaine | Outils |
|---|---|
| Conteneurs | Docker, Docker Compose. Kubernetes local seulement au sprint 3 |
| ML | Python 3.12, pandas, scikit-learn. **Jupyter en local** pour l'exploration (pas de Google Colab) |
| MLOps | MLflow (Tracking et Registry), PostgreSQL pour le backend store, MinIO pour les artefacts. Kubeflow Pipelines au sprint 3 |
| Flux | Kafka (mode KRaft, sans Zookeeper), Kafka UI, Apache NiFi, Apache Flink (**PyFlink**), MongoDB |
| Applicatif | FastAPI, Pydantic v2, React (Vite) |
| DevSecOps (sprint 3) | GitHub Actions, SonarCloud, Snyk |

## 5. Architecture

Deux flux séparés, reliés par Kafka et MLflow.

**① Entraînement**

5 scripts Python, qui deviendront des composants Kubeflow au sprint 3 :

```
CSV --(NiFi)--> MinIO
  -> 1. ingestion    -> 2. preparation -> 3. training
  -> 4. evaluation   -> 5. promotion   -> MLflow Registry (@champion)
```

**② Données et registre**

- MinIO : brut archivé, datasets versionnés, artefacts MLflow.
- MongoDB : mesures propres, résultats, alertes, retours des opérateurs.
- MLflow, avec PostgreSQL comme backend.

**③ Inférence temps réel**

```
Simulateur IoT -> Kafka mesures.brutes -> Flink (préparation) -> Kafka mesures.propres
  -> Service IA <-> FastAPI /predict
  -> MongoDB + Kafka analyses.resultats
  -> passerelle WebSocket/SSE -> React (UrbanHub)
```

- Les retours des opérateurs repartent par `retours.operateurs`.
- NiFi archive `mesures.brutes` dans MinIO, en gardant la provenance de chaque donnée.

**③ bis Réentraînement (sprint 3)**

```
Moniteur Flink (fenêtres) -> Kafka entrainement.demandes -> Service entraîneur
  -> 5 scripts, puis Kubeflow
  -> Kafka entrainement.resultats -> l'API recharge @champion à chaud
```

**④ Interfaces**

- À développer, dans une seule application React avec plusieurs vues :
  - vue opérateur ;
  - vue mairie ;
  - panneau de pilotage de la démo ;
  - page habitant (en option).
- Outils existants : Kafka UI, NiFi UI, Flink UI, console MinIO, Mongo Express, MLflow UI, Swagger `/docs`.

## 6. Contrats (à respecter partout)

### API FastAPI

`POST /predict`

```json
// requête
{ "sensor_id": "IoT-003", "station_id": "Aotizhongxin", "timestamp": "2016-05-01T14:00:00Z",
  "pm25": 85.0, "pm10": 120.0, "no2": 60.0, "co": 900.0,
  "so2": 8.0, "o3": 70.0, "temp": 22.1, "pres": 1008.0, "dewp": 10.2, "rain": 0.0, "wd": "NE", "wspm": 2.1 }
// réponse
{ "sensor_id": "IoT-003", "timestamp": "2016-05-01T14:00:00Z", "score": -0.21,
  "is_anomaly": true, "alert_type": "pollution_atypique", "model_version": "2" }
```

- `alert_type` vaut `pollution_atypique`, `defaut_capteur`, `seuil_sanitaire` ou `null`.
- Validation Pydantic des types, unités, bornes physiques et valeurs manquantes. Erreur `422` documentée.

`GET /health`

```json
{ "status": "ok", "model_loaded": true, "model_name": "air-anomaly-detector",
  "model_version": "2", "model_alias": "champion" }
```

- Renvoie `503` si le modèle n'est pas chargé.

### Topics Kafka (messages JSON)

| Topic | Producteur → consommateur | Champs principaux |
|---|---|---|
| `mesures.brutes` | simulateur → Flink, NiFi | champs de la requête `/predict` + `injection` (`null` ou type d'anomalie injectée, pour la démo uniquement, jamais donné au modèle) |
| `mesures.propres` | Flink → service IA | mêmes champs, nettoyés + features |
| `analyses.resultats` | service IA → passerelle front, moniteur | réponse `/predict` + `alert_id`, `station_id`, `processed_at` |
| `retours.operateurs` | front → service IA, moniteur | `alert_id`, `sensor_id`, `verdict` (`confirmee` ou `fausse_alerte`), `operator_id`, `at` |
| `simulateur.commandes` | panneau de pilotage → simulateur | `command` (`start`, `stop`, `inject_anomaly`, `drift`, `freeze_sensor`), `params` |
| `entrainement.demandes` | moniteur, panneau → entraîneur | `request_id`, `reason` (`planifie`, `fausses_alertes`, `taux_alerte`, `derive`, `manuel`), `metrics`, `requested_at` |
| `entrainement.resultats` | entraîneur → API, front | `request_id`, `run_id`, `model_version`, `decision` (`promu` ou `refuse`), `metrics` |

Les schémas Pydantic de tous ces messages vivent dans `src/novaria/common/schemas.py`. C'est la source de vérité.

## 7. Conventions MLflow

- Expérience : `novaria-air-anomaly`.
- Modèle enregistré : `air-anomaly-detector`.
- Alias : `@champion` (en production) et `@previous` (pour le rollback).
- Chaque run enregistre :
  - les paramètres et les métriques ;
  - le **commit Git** (tag `git_sha`) ;
  - le hash du dataset ;
  - la signature du modèle et les dépendances ;
  - le hash SHA-256 du modèle (tag `model_sha256`).
- L'API vérifie ce hash au chargement.
- **Au moins 2 versions** dans le Registry, avec une vraie différence (fenêtre de données ou hyperparamètres).
- Promotion : le challenger remplace `@champion` seulement si, sur le test contrôlé :
  - son taux de détection est au moins celui du champion, à 2 points près ;
  - ses fausses alertes sont inférieures ou égales à celles du champion ;
  - sa latence reste sous le seuil.

  Sinon, il est enregistré mais refusé.

## 8. Règles de réentraînement (sprint 3)

- **Planifié :** mensuel (en temps simulé).
- **Déclencheurs automatiques :**
  - fausses alertes au-delà de 30 % sur 7 jours ;
  - taux d'alerte supérieur à 2 fois ou inférieur à la moitié de la contamination attendue, pendant 48 h ;
  - PSI au-dessus de 0,2 sur au moins 2 variables ;
  - demande manuelle.
- **Garde-fous :**
  - délai minimum de 7 jours entre deux réentraînements ; (SAUF MANUELLE, ON ACCEPTE TJRS SI MANUELLE)
  - les retours opérateurs pèsent au maximum 5 % des données et sont tracés ;
  - les anomalies confirmées sont exclues ;
  - les fausses alertes sont gardées comme données normales.
- Seul l'algorithme `@champion` est réentraîné. La comparaison des 3 algorithmes est relancée chaque trimestre, ou à la main.

## 9. Structure du dépôt

```
novaria-smart-city/
├── CLAUDE.md
├── README.md
├── docker-compose.yml
├── .env.example            # aucun secret réel dans Git
├── data/raw/               # gitignored
├── notebooks/              # exploration Jupyter
├── src/novaria/
│   ├── common/             # config, schemas.py (Pydantic), client Kafka, client MLflow
│   ├── preparation/        # module de préparation PARTAGÉ (batch + Flink)
│   ├── pipeline/           # ingestion.py, preparation.py, training.py, evaluation.py, promotion.py
│   ├── api/                # FastAPI (/predict, /health, passerelle WebSocket/SSE)
│   ├── simulator/          # simulateur IoT
│   ├── services/
│   │   ├── ia_service/
│   │   ├── monitor/        # sprint 3
│   │   └── trainer/        # sprint 3
│   └── streaming/          # jobs PyFlink
├── nifi/                   # flows exportés
├── frontend/               # React (Vite)
├── kubeflow/               # sprint 3
├── tests/
└── docs/                   # choix du modèle, architecture, mesures, limites
```

Chaque script de `pipeline/` :
- se lance en CLI (`python -m novaria.pipeline.training --input ... --output ...`) ;
- lit et écrit des **fichiers** (Parquet, JSON), sans état global ;
- prend toute sa configuration en arguments ou en variables d'environnement.

Ainsi, il se transforme directement en composant Kubeflow.

### État actuel et migration

L'arborescence ci-dessus est la **cible**. Aujourd'hui existent : `pyproject.toml` (paquet `novaria`, installé avec `pip install -e .`), `src/novaria/common/` (`config.py`, `io.py`, `tracking.py`, `injection.py`), `src/novaria/preparation/`, `src/novaria/models/detectors.py`, `src/novaria/pipeline/` (`ingestion.py`, `preparation.py`, `training.py`), `configs/training.yaml`, `notebooks/01_exploration.ipynb` et `docs/` (`preparation.md`, `training.md`). Pas encore de `docker-compose.yml`, de `schemas.py`, ni d'`evaluation.py` / `promotion.py`.

Entraînement (`training.py`, `models/detectors.py`) :
- **Protocole commun :** `build_model` = pipeline préprocesseur + `CalibratedDetector`. Le détecteur calibre son seuil au quantile `contamination` (1 %) des scores du train, pour les 3 algorithmes ; `decision_function` < 0 = anomalie (convention du score `/predict`).
- **Anomalies injectées** (`common/injection.py`) : `pic`, `valeur_impossible`, `capteur_fige` (mêmes valeurs que le champ `injection` du simulateur). Injectées seulement dans la validation (et plus tard le test), puis `fix_inconsistencies` est réappliqué. Jamais dans le train.
- **Choix du réglage** par algorithme : `val_youden_pic` (détection des pics − fausses alertes). Les modèles ne détectent ni le capteur figé ni les valeurs impossibles : ce sont des règles.
- **Sauvegarde** : format MLflow **skops** (pas pickle), avec les types fiables de `SKOPS_TRUSTED_TYPES` (à mettre à jour si une nouvelle classe entre dans le pipeline) et le code `novaria` embarqué (`code_paths`). Signature : 17 variables optionnelles. Tag `model_sha256` = empreinte du fichier sérialisé.
- **MLflow** : `common/tracking.setup_mlflow` utilise `MLFLOW_TRACKING_URI`, sinon le suivi local SQLite (`mlflow.db`, `mlartifacts/`). Les tests passent une base SQLite temporaire.

Module de préparation (`src/novaria/preparation/`) :
- **Format commun :** DataFrame indexé par `timestamp` (heure locale de Pékin, sans fuseau), colonnes `station_id` + les 12 mesures du contrat. `from_raw` y met les CSV, `standardize` les messages du contrat.
- **Fonctions sans état** (`standardize`, `fix_inconsistencies`, `add_features`) : partagées entre le batch et le streaming. `records.prepare_records` les appelle pour Flink ; `tests/preparation/test_records.py::test_streaming_matches_batch` garantit un résultat identique.
- **Fonctions avec historique** (doublons, heures manquantes, capteur figé) : batch seulement (`cleaning.clean`) ; en streaming, il faudra un état par station.
- **Imputation et normalisation** (`build_preprocessor`) : non appliquées par la préparation, mais mises dans le pipeline du modèle par `training.py`. Les fichiers `data/prepared/*.parquet` ne sont donc pas normalisés.
- `MODEL_FEATURES` (config) : 17 variables d'entrée du modèle (11 mesures numériques, `wd` en sin/cos, heure et mois en sin/cos).
- Le journal (`journal.py`) n'écrit que s'il est configuré par `setup_data_checks_log`.

### Commandes

Depuis la racine du dépôt, avec le venv activé (`source .venv/bin/activate`) :

```bash
pip install -r requirements.lock.txt && pip install -e .   # installation
python -m novaria.pipeline.ingestion     # data/raw/ -> data/ingested/ (Parquet + manifest.json)
python -m novaria.pipeline.preparation   # -> data/prepared/, reports/data_quality.json, logs/data_checks.log
python -m novaria.pipeline.training      # -> models/, reports/training_comparison.md, runs MLflow (≈ 8 min)
mlflow ui --backend-store-uri sqlite:///mlflow.db   # consulter les runs locaux
jupyter lab notebooks/01_exploration.ipynb
pytest -q                                # tests unitaires (données synthétiques, aucun CSV nécessaire)
pytest tests/preparation/test_records.py::test_streaming_matches_batch   # un seul test
ruff check .                             # lint
```

Pour réexécuter le notebook en ligne de commande : `jupyter nbconvert --to notebook --execute --inplace notebooks/01_exploration.ipynb`.

## 10. Ports locaux (Docker Compose)

| Service | Port |
|---|---|
| FastAPI | 8000 |
| Simulateur (commandes HTTP, si besoin) | 8001 |
| MLflow | 5000 |
| PostgreSQL | 5432 |
| MinIO API / console | 9000 / 9001 |
| Kafka | 9092 |
| Kafka UI | 8080 |
| MongoDB | 27017 |
| Mongo Express | 8081 |
| Flink UI | 8082 |
| NiFi | 8443 |
| React (Vite) | 5173 |

## 11. Sprints

### Sprint 1 · MLOps
**Objectif atteint quand :** les 3 algorithmes sont comparés dans MLflow, et le modèle retenu existe en 2 versions dans le Registry, avec `@champion` et `@previous`.

Ce sprint comprend :
- Docker Compose avec PostgreSQL, MinIO et MLflow ;
- l'exploration Jupyter ;
- les 5 scripts ;
- le générateur d'anomalies injectées ;
- MLflow Tracking et Registry ;
- la note de choix du modèle.

### Sprint 2 · Chaîne temps réel
**Objectif atteint quand :** une anomalie injectée depuis le panneau traverse Kafka, Flink et l'API, et s'affiche comme alerte dans React. NiFi archive le brut dans MinIO avec la provenance.

Ce sprint comprend :
- Kafka, Kafka UI, MongoDB, NiFi et Flink ;
- FastAPI ;
- le simulateur ;
- l'ETL, d'abord en Python puis en PyFlink ;
- les flows NiFi ;
- le service IA ;
- la passerelle WebSocket/SSE ;
- React (vues opérateur et mairie, panneau de pilotage).

### Sprint 3 · Réentraînement, Kubeflow, DevSecOps et finalisation
**Objectif atteint quand :**
- une dérive simulée déclenche un réentraînement exécuté dans Kubeflow, qui est soit promu, soit refusé ;
- le « avant / après » DevSecOps est documenté.

Ce sprint comprend :
- le moniteur Flink ;
- les topics `entrainement.*` et le service entraîneur ;
- le rechargement à chaud ;
- Kubernetes local et Kubeflow Pipelines standalone ;
- le tag `avant-devsecops`, puis GitHub Actions (pytest, ruff, ESLint, Vitest, build des images), SonarCloud et Snyk, et le scan « après » ;
- le dossier final : RAM/CPU, README, schéma, limites, plan d'industrialisation.

## 12. Répartition dans le binôme

- **Dev A · ML et applicatif :**
  - S1 : entraînement, évaluation, promotion, MLflow, Registry ;
  - S2 : FastAPI, service IA, passerelle WebSocket/SSE, React ;
  - S3 : Kubeflow, rechargement à chaud, tests API et front.
- **Dev B · Données et infrastructure :**
  - S1 : Docker Compose, exploration, ingestion, module de préparation, anomalies injectées ;
  - S2 : Kafka, MongoDB, NiFi, Flink, simulateur ;
  - S3 : moniteur Flink, entraîneur, garde-fous, CI et DevSecOps.
- Chaque sprint commence par **figer les contrats** (section 6) ensemble.

## 13. Règles pour Claude Code

- **Respecter le sprint en cours.** Pas de code Kubeflow ni de configuration DevSecOps (GitHub Actions, SonarCloud, Snyk) avant le sprint 3. C'est voulu : le prof veut voir le « avant / après ».
- **Ne pas utiliser Google Colab.** Tout tourne en local.
- **Ne jamais dupliquer la logique de préparation.** Elle vit dans `src/novaria/preparation/`.
- **Ne jamais entraîner sur des anomalies injectées.** Ne jamais faire de split aléatoire.
- **Ne jamais coder un secret en dur.** Tout passe par des variables d'environnement, avec `.env.example` comme modèle.
- Utiliser les schémas Pydantic de `common/schemas.py` pour tout message Kafka et toute requête API. Une modification de contrat se fait là, puis partout.
- Le code (noms de variables, fonctions, fichiers) est en **anglais**. Les docstrings, commentaires et documents de `docs/` sont en **français**.
- Python : typage, fonctions courtes, chaque fonction de préparation testée avec pytest.
- Quand une décision technique est prise (hyperparamètres, seuil, échantillonnage), la consigner dans `docs/` avec sa justification : elle servira à la soutenance.
- Si une demande contredit ce fichier, le signaler avant de coder.
- Commentaires pour chaque fonction : Description - args - returns