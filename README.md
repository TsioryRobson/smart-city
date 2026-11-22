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
├── common/          # config, fichiers, MLflow (tracking.py), générateur d'anomalies (injection.py)
├── preparation/     # module de préparation PARTAGÉ entre l'entraînement et Flink
├── models/          # détecteurs : les 3 algorithmes avec un seuil calibré commun
└── pipeline/        # scripts CLI : ingestion, preparation, training, injection, evaluation, promotion
configs/             # configuration de l'entraînement et de l'évaluation
notebooks/           # exploration Jupyter
docs/                # décisions techniques justifiées
reports/             # rapport qualité, comparaison des algorithmes, figures
tests/               # tests unitaires (données synthétiques)
data/                # données, non versionnées : raw/, ingested/, prepared/, controlled/
models/              # modèles entraînés, non versionnés
logs/                # journal de vérification des données, non versionné
```

## Premier démarrage après un clone

Ce que Git ne contient pas, volontairement : `.env` (secrets), `data/raw/` (CSV du dataset), les images Docker
(construites en local, sans registre) et les paquets téléchargés (`infra/mlflow/wheels/`, `kubeflow/check/wheels/`,
facultatifs). Dans l'ordre :

1. **Prérequis** : WSL Ubuntu, Python 3.12, Docker Desktop avec Kubernetes activé en mode **kubeadm**
   (Settings → Kubernetes ; contexte `docker-desktop`).
2. **Secrets** : `cp .env.example .env`, puis remplacer chaque `changeme` (`openssl rand -hex 24`).
   Si le port 5432 est déjà pris sur la machine, changer `POSTGRES_HOST_PORT`.
3. **Données** : placer les 12 CSV dans `data/raw/` (voir [Données](#données)).
4. **Environnements Python** :
   ```bash
   python3 -m venv .venv && source .venv/bin/activate
   pip install -r requirements.lock.txt && pip install -e .
   python3 -m venv .venv-kfp && .venv-kfp/bin/pip install -r kubeflow/requirements.txt
   ```
5. **Stack MLflow** : `docker compose up -d --build`, puis `export MLFLOW_TRACKING_URI=http://localhost:5000`.
6. **Sprint 1** (crée les modèles et `@champion` / `@previous` dans le Registry) : enchaîner les scripts de
   [Pipeline](#pipeline-sprint-1) à [Model Registry et promotion](#model-registry-et-promotion), v1 puis v2.
7. **Kubeflow** : section [Kubernetes et Kubeflow Pipelines](#kubernetes-et-kubeflow-pipelines-sprint-2), puis
   le pipeline de validation de [docs/kubeflow.md](docs/kubeflow.md) (il vérifie que `@champion` existe).
8. **Vérifier** : `pytest -q` et `ruff check .`.

Sans dossier `wheels/` rempli, les images téléchargent leurs paquets sur PyPI : c'est seulement plus lent.

## Infrastructure (Docker Compose)

PostgreSQL (métadonnées), MinIO (artefacts) et le serveur MLflow (suivi + Model Registry) :

```bash
cp .env.example .env      # remplacer chaque « changeme » (openssl rand -hex 24)
docker compose up -d --build
export MLFLOW_TRACKING_URI=http://localhost:5000
```

Interfaces : MLflow http://localhost:5000 · console MinIO http://localhost:9001. Images figées par digest,
ports sur 127.0.0.1 seulement, conteneurs non-root. Détails et choix : [docs/infrastructure.md](docs/infrastructure.md).
Sans `MLFLOW_TRACKING_URI`, les scripts utilisent un suivi local (`mlflow.db`).

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

## Pipeline (sprint 1)

Schéma du flux et de l'infrastructure (PlantUML) : [docs/workflow.puml](docs/workflow.puml).
Affichage : extension « PlantUML » de VS Code (Alt+D).

```bash
python -m novaria.pipeline.ingestion     # data/raw/ -> data/ingested/
python -m novaria.pipeline.preparation   # data/ingested/ -> data/prepared/
python -m novaria.pipeline.training      # data/prepared/ -> models/ + MLflow (≈ 8 min)
python -m novaria.pipeline.injection     # data/prepared/test -> data/controlled/ (anomalies injectées)
python -m novaria.pipeline.evaluation    # -> models/selection.json, reports/evaluation.md (≈ 3 min)
python -m novaria.pipeline.promotion     # -> Model Registry : version, @champion / @previous
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

`novaria.preparation.build_preprocessor()` (imputation médiane + `StandardScaler` ou `RobustScaler`) n'est **pas** appliqué
par la préparation : `training.py` le place dans le même pipeline scikit-learn que le modèle, enregistré dans MLflow.
Il est donc ajusté sur le train uniquement, et appliqué à l'identique en batch et en streaming.

### Réutilisation par Flink (sprint 2)

`novaria.preparation.prepare_records(messages)` prépare des messages du topic `mesures.brutes` et renvoie
des messages `mesures.propres`, en appelant exactement les mêmes fonctions que le batch (un test le vérifie).
Les contrôles qui demandent l'historique d'une station (doublons, heures manquantes, capteur figé)
demanderont un état par station dans Flink.

## Entraînement

**3. Entraînement** (`training.py`, configuration [configs/training.yaml](configs/training.yaml)) : pour Isolation Forest,
LOF et One-Class SVM, chaque combinaison de la grille d'hyperparamètres est entraînée sur le train et enregistrée
comme un run MLflow (paramètres, métriques, `git_sha`, `dataset_sha256`).

- **Protocole commun :** même pipeline (imputation + mise à l'échelle + détecteur), mêmes 17 variables, et seuil
  calibré pour alerter sur 1 % du train. Score négatif = anomalie (convention du contrat `/predict`).
- **Validation contrôlée :** des anomalies identifiées sont injectées dans la validation (pics, valeurs impossibles,
  capteur figé), jamais dans le train. Mesures : détection par type, fausses alertes, AUC, temps d'entraînement,
  temps d'inférence et latence unitaire.
- **Choix du réglage :** détection des pics − fausses alertes, par algorithme.
- **Sorties :** meilleur modèle par algorithme dans `models/<algorithme>/` et dans MLflow (format skops, empreinte
  `model_sha256`), `models/comparison.json`, et le tableau [reports/training_comparison.md](reports/training_comparison.md).

MLflow : si `MLFLOW_TRACKING_URI` est définie (serveur), elle est utilisée ; sinon le suivi est local
(`mlflow.db` et `mlartifacts/`, non versionnés). Pour consulter les runs en local :

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Résultats, protocole et limites : [docs/training.md](docs/training.md).
Ce qui est tracé dans MLflow (paramètres, métriques, jeux de données, commit, signatures) et comment comparer
les 3 algorithmes dans l'interface : [docs/experiment_tracking.md](docs/experiment_tracking.md).

## Évaluation et sélection

**Injection** (`injection.py`, configuration [configs/evaluation.yaml](configs/evaluation.yaml)) : écrit 5 jeux de test
contrôlés (test réel + anomalies injectées, une graine par fichier) dans `data/controlled/`, avec leurs empreintes.
Les 3 modèles sont évalués sur exactement les mêmes fichiers. L'injection est refusée sur le train.

**Évaluation** (`evaluation.py`) : pour le modèle retenu de chaque algorithme (empreinte vérifiée au chargement) :
- **performance** moyenne sur les 5 tirages : détection par type d'anomalie, fausses alertes, AUC ;
- **fausses alertes = alertes sur les heures ordinaires** : les heures où un polluant dépasse le 99e centile du train
  sont de vrais épisodes de pollution extrême, mesurés à part ;
- **stabilité** : écart-type entre tirages, entre 5 réentraînements sur 80 % du train, et fausses alertes par mois et
  par station ;
- **coût** : latence unitaire, temps d'inférence, taille du modèle.

La règle de sélection (latence ≤ 50 ms, puis meilleur « pics détectés − fausses alertes », puis moins de fausses
alertes à 0,02 près) écrit le modèle retenu dans `models/selection.json`, pour la promotion.

**Modèle retenu : One-Class SVM**, avec 96 % des pics détectés, 2,9 % de fausses alertes, 0,4 Mo et 8 ms par mesure.
La justification, l'alternative Isolation Forest et les limites sont dans [docs/model_choice.md](docs/model_choice.md).
Les résultats détaillés sont dans [reports/evaluation.md](reports/evaluation.md).

## Kubernetes et Kubeflow Pipelines (sprint 2)

Kubeflow Pipelines 2.18.0 standalone sur le Kubernetes de Docker Desktop (kubeadm), branché sur MLflow et MinIO :

```bash
./kubeflow/pull-images.sh && ./kubeflow/install.sh
set -a; . ./.env; set +a; python infra/minio/setup_pipeline_access.py && ./kubeflow/configure-access.sh
kubectl -n kubeflow port-forward svc/ml-pipeline-ui 8080:80     # http://localhost:8080
```

Les pods n'accèdent qu'à MLflow et au bucket `novaria-data` de MinIO (compte dédié, écriture seulement sous
`datasets/`) ; PostgreSQL n'est plus publié. Le SDK `kfp` est dans un venv séparé (`.venv-kfp`,
`kubeflow/requirements.txt`). Installation, choix et points de sécurité : [docs/kubeflow.md](docs/kubeflow.md).

## Model Registry et promotion

**Promotion** (`promotion.py`) : enregistre le modèle retenu comme nouvelle version de `air-anomaly-detector`
(tags : `model_sha256`, `git_sha`, `dataset_sha256`, `train_period`, hyperparamètres). Il devient `@champion`
seulement si, sur les mêmes jeux de test contrôlés que le champion actuel, sa détection des pics est au moins
celle du champion à 2 points près, ses fausses alertes sont inférieures ou égales et sa latence ≤ 50 ms.
L'ancien champion devient `@previous` (rollback). Sinon la version est enregistrée avec la décision `refuse`.
`--force` promeut malgré un refus, et c'est tracé (tag `forced`).

Deux versions du même modèle (One-Class SVM, mêmes hyperparamètres), qui ne diffèrent que par la fenêtre de données :

```bash
# v1 : fenêtre précédente (2013-09 -> 2014-08)
python -m novaria.pipeline.preparation --output data/prepared_window_2013 \
    --report data/prepared_window_2013/quality.json --log logs/data_checks_window_2013.log \
    --train-start 2013-09-01 --val-start 2014-09-01 --test-start 2014-12-01 --test-end 2015-03-01
python -m novaria.pipeline.training --input data/prepared_window_2013 --output models/window_2013 \
    --config configs/training_previous_window.yaml --report models/window_2013/training_comparison.md
python -m novaria.pipeline.promotion --models models/window_2013 --algorithm one_class_svm
# v2 : fenêtre actuelle (2014-09 -> 2015-08), modèle choisi par l'évaluation
python -m novaria.pipeline.promotion --models models
```

Les deux versions sont comparées sur le même test contrôlé (2015-12 → 2016-02), postérieur aux deux fenêtres.

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

## Points d'attention pour la suite

- **Fausses alertes météo :** 70 % des fausses alertes de One-Class SVM tombent sur une météo jamais vue au train
  (vague de froid de janvier 2016). La règle métier de l'API devra ne lever `pollution_atypique` que si un polluant
  est en cause.
- **Capteur figé et valeurs impossibles** ne sont pas détectés par les modèles : ils relèvent des règles
  (`defaut_capteur`), à mettre en place dans la règle métier de l'API.
- **Saisonnalité :** le test est uniquement hivernal ; les fausses alertes mesurées dépendent de la saison.
- **Imputation médiane :** une valeur imputée paraît « normale » au modèle, surtout pour PM2.5 et PM10
  (5 % de lignes PM2.5 > PM10 mises à NaN).
- **Pics injectés nets** (au-dessus du 99e centile) : les taux de détection sont optimistes.
