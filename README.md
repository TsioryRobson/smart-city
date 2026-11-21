# Novaria Smart City AI : POC de détection d'anomalies de qualité de l'air

Projet fil rouge Master 2 EADL (Bloc 05 · MLOps et MLSecOps). Auteur : Dania.

Détection non supervisée de mesures de pollution atypiques (Isolation Forest, LOF, One-Class SVM)
sur les mesures horaires des 12 stations du dataset UCI *Beijing Multi-Site Air Quality*,
avec affichage d'une alerte quand un capteur IoT envoie une mesure anormale.

Le contexte complet (règles ML, contrats d'API et Kafka, architecture, sprints) est dans [CLAUDE.md](CLAUDE.md).

## Stack

Python 3.12, pandas, scikit-learn, Jupyter, MLflow, FastAPI, Kafka, Flink (PyFlink), NiFi, MongoDB,
MinIO, PostgreSQL, Docker Compose. Kubernetes et Kubeflow Pipelines au sprint 3.

## Structure

```
src/novaria/
├── common/          # config (chemins, colonnes, bornes, dates du split), utilitaires fichiers
├── preparation/     # module de préparation PARTAGÉ entre l'entraînement et Flink
└── pipeline/        # scripts CLI : ingestion.py, preparation.py (puis training, evaluation, promotion)
notebooks/           # exploration Jupyter
docs/                # décisions techniques justifiées
reports/             # rapport qualité JSON, figures
tests/               # tests unitaires (données synthétiques)
data/                # données, non versionnées : raw/, ingested/, prepared/
logs/                # journal de vérification des données, non versionné
```

## Installation (WSL Ubuntu)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.lock.txt   # versions figées (ou requirements.txt)
pip install -e .                       # rend le paquet `novaria` importable
```

## Données

Télécharger le dataset sur https://archive.ics.uci.edu/dataset/501/beijing+multi+site+air+quality
et placer les 12 CSV `PRSA_Data_<station>_20130301-20170228.csv` dans `data/raw/`. Ils ne sont jamais modifiés.

| Période | Usage |
|---|---|
| 2014-09 → 2015-08 | entraînement (fenêtre de 12 mois) |
| 2015-09 → 2015-11 | validation |
| 2015-12 → 2016-02 | test |
| 2016-03 → 2017-02 | **réservée au simulateur IoT (démo)**, jamais vue par le modèle |

Les dates sont dans `src/novaria/common/config.py` et peuvent être passées en arguments.

## Pipeline de données (sprint 1)

```bash
python -m novaria.pipeline.ingestion     # data/raw/ -> data/ingested/
python -m novaria.pipeline.preparation   # data/ingested/ -> data/prepared/
```

Chaque script lit et écrit des fichiers et prend sa configuration en arguments (`--help`) :
il deviendra un composant Kubeflow au sprint 3.

**1. Ingestion** : lit les 12 CSV, vérifie leurs colonnes, et écrit sans modifier aucune valeur :
- `measurements.parquet` : toutes les stations, avec la colonne `source_file` (provenance) ;
- `manifest.json` : empreinte SHA-256 de chaque CSV et du jeu complet (`dataset_sha256`, pour MLflow).

**2. Préparation** : applique le module partagé et écrit :
- `train.parquet`, `val.parquet`, `test.parquet` : mesures nettoyées et variables dérivées, **non normalisées** ;
- `metadata.json` : bornes du split, variables, nombre de lignes, empreintes ;
- `reports/data_quality.json` : rapport qualité ;
- `logs/data_checks.log` : une ligne par problème détecté (station, horodatage, colonne, valeur, erreur, action).

### Règles de nettoyage

| Problème | Traitement |
|---|---|
| Doublon (station, horodatage) | supprimé, la première occurrence est gardée (seule suppression autorisée) |
| Valeur ou heure manquante | signalée ; la valeur sera imputée par le modèle |
| Concentration, pluie ou vent négatif | mis à NaN puis imputé |
| Valeur hors bornes physiques (bornes dans la config) | mise à NaN puis imputée |
| PM2.5 > PM10 | les deux valeurs mises à NaN puis imputées |
| Direction de vent inconnue | mise à NaN puis imputée |
| Capteur figé (même valeur ≥ 24 h) | signalé seulement |
| Pic extrême mais physiquement possible | **jamais modifié** : c'est une anomalie à détecter |

### Variables

Les noms suivent le contrat de l'API `/predict` (`station_id`, `timestamp`, mesures en snake_case).
Le modèle reçoit 17 variables (`MODEL_FEATURES`) :
- 11 mesures numériques : `pm25`, `pm10`, `no2`, `co`, `so2`, `o3` (µg/m³), `temp`, `pres`, `dewp`, `rain`, `wspm` ;
- direction du vent `wd` encodée en `wd_sin`, `wd_cos` ;
- heure et mois en encodage cyclique (`hour_sin`, `hour_cos`, `month_sin`, `month_cos`), pour la saisonnalité.

Les horodatages sont en heure locale de Pékin. Un message reçu avec un fuseau (ex. `Z`) est converti.

### Imputation et normalisation

`novaria.preparation.build_preprocessor()` (imputation médiane + `StandardScaler`) n'est **pas** appliqué
par la préparation : `training.py` le place dans le même pipeline scikit-learn que le modèle, enregistré dans MLflow.
Il est donc ajusté sur le train uniquement, et appliqué à l'identique en batch et en streaming.

### Réutilisation par Flink (sprint 2)

`novaria.preparation.prepare_records(messages)` prépare des messages du topic `mesures.brutes` et renvoie
des messages `mesures.propres`, en appelant exactement les mêmes fonctions que le batch (un test le vérifie).
Les contrôles qui demandent l'historique d'une station (doublons, heures manquantes, capteur figé)
demanderont un état par station dans Flink.

## Exploration

```bash
jupyter lab notebooks/01_exploration.ipynb
```

Valeurs manquantes, incohérences, unités, distributions, saisonnalité et dérive entre les jeux.
Les conclusions sont en fin de notebook, les figures dans `reports/figures/`, et les décisions dans
[docs/preparation.md](docs/preparation.md).

## Qualité

```bash
pytest -q                                              # tous les tests
pytest tests/preparation/test_records.py::test_streaming_matches_batch   # un seul test
ruff check .                                           # lint
```

## Points d'attention pour l'entraînement

- **Saisonnalité :** le test (2015-12 → 2016-02) est uniquement hivernal ; le taux de fausses alertes mesuré
  ne vaut que pour l'hiver. À documenter dans les limites de l'évaluation.
- **Queues de distribution très longues** sur PM2.5, PM10, CO et SO2 : comparer `StandardScaler` avec `RobustScaler`
  ou une transformation log.
- **Imputation médiane :** une valeur imputée paraît « normale » au modèle, surtout pour PM2.5 et PM10
  (5 % de lignes PM2.5 > PM10 mises à NaN).
