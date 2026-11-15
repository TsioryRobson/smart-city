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
puis placer `PRSA_Data_Changping_20130301-20170228.csv` dans `data/`.
