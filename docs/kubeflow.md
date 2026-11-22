# Kubernetes et Kubeflow Pipelines (sprint 2, branche `chore/s2-kubeflow-setup`)

Kubeflow Pipelines **2.18.0 standalone** sur le Kubernetes de Docker Desktop. Les pods utilisent les services du
sprint 1 (MLflow, MinIO) lancés par Docker Compose.

## Architecture

| Élément | Où | Rôle |
|---|---|---|
| Kubernetes 1.34 | Docker Desktop, mode **kubeadm**, 1 nœud | Exécute Kubeflow et les étapes des pipelines |
| Kubeflow Pipelines | namespace `kubeflow` | API, interface, orchestrateur Argo, MySQL et SeaweedFS **internes** (métadonnées et fichiers échangés entre étapes) |
| MLflow | Docker Compose, `host.docker.internal:5000` | Suivi des runs et Model Registry (comme au sprint 1) |
| MinIO | Docker Compose, `host.docker.internal:9000` | Bucket `novaria-data` : CSV bruts (`raw/`), jeux produits par le pipeline (`datasets/`) |
| PostgreSQL | Docker Compose, **aucun port publié** | Utilisé seulement par MLflow ; inaccessible aux pods |

## Installation

```bash
./kubeflow/pull-images.sh        # télécharge les 17 images à l'avance (≈ 1,4 Go)
./kubeflow/install.sh            # Kubeflow Pipelines 2.18.0 (≈ 1 min si les images sont là)
set -a; . ./.env; set +a
python infra/minio/setup_pipeline_access.py   # bucket novaria-data, compte du pipeline, CSV
./kubeflow/configure-access.sh   # ConfigMap + Secret dans le namespace kubeflow
kubectl -n kubeflow port-forward svc/ml-pipeline-ui 8080:80   # interface : http://localhost:8080
```

Validation (pipeline à 2 étapes en parallèle, image `novaria/kfp-check`) :

```bash
docker build -t novaria/kfp-check:0.1.0 kubeflow/check
.venv-kfp/bin/python kubeflow/pipelines/validation_pipeline.py
.venv-kfp/bin/python kubeflow/pipelines/run.py kubeflow/pipelines/validation_pipeline.yaml
```

Résultat (2026-11-22) : run `SUCCEEDED`. `check-minio` lit les 12 CSV, écrit sous `datasets/`, se voit refuser
l'écriture sous `raw/` ; `check-mlflow` lit `air-anomaly-detector@champion` et enregistre un run dans l'expérience
MLflow `kfp-validation`.

## Choix

| Décision | Justification |
|---|---|
| Kubernetes de Docker Desktop en mode **kubeadm** | Le cluster utilise directement les images de Docker : nos images de composants sont construites en local, **sans registre**. Le mode kind aurait imposé le *containerd image store* et le retéléchargement de toutes les images. |
| KFP **2.18.0** (pas 2.18.1) | Version publiée depuis plus de deux semaines (2026-10-28) ; la 2.18.1 n'avait que 4 jours. |
| Manifestes **rendus et figés** dans `kubeflow/platform/upstream-2.18.0/` | Les manifestes officiels vont chercher ceux d'Argo sur GitHub au rendu, et kustomize abandonne après 27 s. Figés : installation sans réseau, reproductible, et manifestes relisibles (revue de sécurité). Régénération : `./kubeflow/render-manifests.sh`. |
| Déploiement `kfp-visualization-server` retiré | Image de 881 Mo, utilisée seulement par les visualisations de Kubeflow v1. Son **Service** est gardé (sans pod) : le serveur d'API refuse de démarrer sans la variable `ML_PIPELINE_VISUALIZATIONSERVER_SERVICE_HOST`. |
| Images téléchargées à l'avance (`pull-images.sh`) | Connexion lente : téléchargement reprenable, image par image, avant l'installation. |
| Pods → services Compose par `host.docker.internal` | Nom fourni par Docker Desktop pour joindre la machine ; ajouté à `--allowed-hosts` de MLflow (sinon 403). |
| **PostgreSQL non publié** (`docker-compose.sql-access.yml` pour un accès ponctuel) | Un port publié est joignable depuis les pods, et le réseau de Docker Desktop n'applique pas les NetworkPolicy. Seul MLflow doit parler à la base. |
| Compte MinIO **dédié** au pipeline (`novaria-pipeline`) | Lecture du bucket `novaria-data`, écriture seulement sous `datasets/` ; ni `raw/`, ni bucket `mlflow`, ni administration (vérifié par un test et par le pipeline de validation). |
| Secret injecté **seulement** dans les étapes qui en ont besoin | `check-mlflow` ne reçoit pas les identifiants MinIO (`kfp-kubernetes`, `use_secret_as_env`). MLflow sert lui-même les artefacts : les étapes n'ont besoin que de son URL. |
| SDK `kfp` dans un venv séparé (`.venv-kfp`) | `kfp` 2.18 impose `protobuf<7`, alors que MLflow utilise protobuf 7 dans `.venv`. Les pipelines lancent nos scripts dans des conteneurs : le SDK ne sert qu'à compiler et soumettre. |
| Composants « conteneur » avec **commande explicite** | Le lanceur KFP ignore l'`ENTRYPOINT` de l'image et exécute directement `command` + `args`. |
| Pas de mise en cache pour la validation | Chaque run de validation doit vraiment tester les accès. |

## Ressources mesurées

| Ensemble | Mémoire |
|---|---|
| Kubeflow Pipelines (13 pods) | ≈ 0,85 Go |
| Stack MLflow (PostgreSQL, MinIO, MLflow) | ≈ 0,47 Go |

## Sécurité : points relevés dans les manifestes officiels (à traiter au sprint 3)

- Stockage interne SeaweedFS : identifiants par défaut `minio` / `minio123` (Secret `mlpipeline-minio-artifact`).
- MySQL interne : compte `root` **sans mot de passe** (Secret `mysql-secret`).
- Ces services ne sont joignables que depuis le cluster (pas de port publié) ; l'interface n'est exposée que par
  `port-forward` sur la machine. Ce sont de bons exemples pour l'« avant / après » DevSecOps.
- Hors projet : un PostgreSQL installé sur Windows écoute sur toutes les interfaces (port 5432) et reste joignable
  depuis les pods.

## Reprendre après un arrêt

```bash
docker compose up -d                                         # MLflow, MinIO, PostgreSQL
kubectl -n kubeflow get pods                                 # Kubeflow redémarre avec Docker Desktop
kubectl -n kubeflow port-forward svc/ml-pipeline-ui 8080:80  # interface
```

Désinstaller : `kubectl delete -k kubeflow/platform && kubectl delete -f kubeflow/platform/upstream-2.18.0/cluster-scoped.yaml`.
