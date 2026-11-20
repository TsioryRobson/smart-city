# Smart City AI : POC de détection d'anomalies de qualité de l'air

Projet fil rouge Master 2 EADL. Auteur : Dania.

Détection d'anomalies non supervisée (Isolation Forest, LOF, One-Class SVM)
sur les mesures horaires de la station Changping (dataset UCI Beijing Multi-Site Air Quality).

## Stack
Python, Scikit-learn, MLflow, FastAPI, Docker Desktop, Kubernetes local, Kubeflow Pipelines.

## Structure
- `src/`       : chargement, prétraitement, entraînement, évaluation
- `api/`       : API FastAPI
- `simulator/` : simulateur de capteurs IoT
- `pipeline/`  : pipeline Kubeflow
- `data/`      : dataset (non versionné)
- `reports/`   : figures et résultats
- `tests/`     : tests unitaires

## Installation (WSL Ubuntu)
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Données
Télécharger le dataset sur https://archive.ics.uci.edu/dataset/501/beijing+multi+site+air+quality
puis placer le CSV de la station dans `data/`.

Le fichier utilisé est défini par `DATA_PATH` dans `src/config.py`.
Il pointe actuellement vers `PRSA_Data_Aotizhongxin_20130301-20170228.csv`, le seul CSV présent pour l'instant.
Pour utiliser Changping, placer `PRSA_Data_Changping_20130301-20170228.csv` dans `data/` et modifier `DATA_PATH`.

## Étape 2 : préparation des données

### Modules
- `src/config.py` : constantes centralisées (chemins, correspondance des colonnes, `FEATURES`, ratios de découpage).
- `src/data.py` : chargement, index horaire `timestamp`, sélection des variables, rapport qualité, suppression des doublons d'horodatage, découpage chronologique.
- `src/preprocess.py` : préprocesseur `SimpleImputer(median)` suivi de `StandardScaler`, qui renvoie des DataFrames.

### Variables retenues
Les noms snake_case ci-dessous seront aussi ceux du futur contrat d'API.

| Colonne d'origine | Nom utilisé |
|---|---|
| PM2.5 | `pm25` |
| PM10 | `pm10` |
| NO2 | `no2` |
| CO | `co` |
| SO2 | `so2` |
| O3 | `o3` |
| TEMP | `temp` |

### Choix de préparation
- **Aucune valeur extrême ou négative n'est supprimée ni modifiée.** Le but est de détecter les anomalies, donc on se contente de les signaler dans le rapport qualité.
- **Seule suppression autorisée :** les doublons d'horodatage. On garde la première occurrence.
- **Découpage chronologique 70 / 15 / 15** par position, sans mélange. Des `assert` vérifient qu'il n'y a aucun chevauchement entre train, val et test.
- **Pas de fuite :** le préprocesseur est ajusté sur le train uniquement, puis appliqué à val et test. Il n'est pas sauvegardé : il sera combiné au modèle à l'étape 3.

### Commandes
```bash
python -m src.data         # rapport qualité + résumé des splits -> reports/data_quality.json
python -m src.preprocess   # contrôle du préprocesseur (NaN, moyenne/écart-type, médianes)
pytest -q                  # tests unitaires sur données synthétiques
ruff check .               # lint
```

### Points d'attention pour l'étape 3
- **Saisonnalité :** le train couvre environ 3 cycles annuels. Le val va de l'hiver à l'été et le test de l'été à l'hiver, ce qui produit des écarts de distribution liés à la saison, à ne pas confondre avec des anomalies.
- **Queues de distribution très longues** sur PM2.5, PM10 et CO : le `StandardScaler` y est sensible. Un `RobustScaler` ou une transformation log sont à envisager.
- **Imputation par la médiane :** les valeurs imputées peuvent paraître artificiellement « normales » pour le modèle, surtout sur la variable la plus incomplète.
