# Infrastructure MLOps du sprint 1 (Docker Compose)

`docker-compose.yml` démarre le serveur MLflow et ses deux stockages.

| Service | Image | Rôle | Port (machine locale) |
|---|---|---|---|
| `postgres` | `postgres:16.10-alpine` (figée par digest) | Métadonnées MLflow : expériences, runs, paramètres, métriques, Model Registry | aucun (accès ponctuel : `docker-compose.sql-access.yml`) |
| `minio` | `cgr.dev/chainguard/minio` (figée par digest) | Stockage S3 des artefacts : modèles, rapports, configurations | 127.0.0.1:9000 (API), 127.0.0.1:9001 (console) |
| `mlflow` | `novaria/mlflow:3.18.0` (construite depuis `infra/mlflow/`) | Serveur de suivi et Registry ; sert les artefacts par HTTP | 127.0.0.1:5000 |

## Démarrage

```bash
cp .env.example .env          # puis remplacer chaque « changeme » (openssl rand -hex 24)
docker compose up -d --build
export MLFLOW_TRACKING_URI=http://localhost:5000
```

Les scripts du pipeline (`novaria.pipeline.*`) tournent sur la machine et envoient tout au serveur : aucun changement
de code entre le suivi local (SQLite) et le serveur, seule la variable `MLFLOW_TRACKING_URI` change.

## Choix

| Décision | Justification |
|---|---|
| Artefacts servis par MLflow (`--serve-artifacts`, destination `s3://mlflow`) | Les scripts et l'API n'ont besoin que de l'URL MLflow, jamais des identifiants MinIO : un seul service détient ces secrets. |
| Bucket créé au démarrage du conteneur MLflow (`create_bucket.py`, boto3) | L'image MinIO Chainguard ne contient pas le client `mc` ; MLflow a déjà boto3. Le script réessaie tant que MinIO démarre. |
| Image MinIO Chainguard | MinIO a retiré ses images officielles de Docker Hub et de Quay fin 2025. L'image Chainguard est maintenue et minimale (peu de vulnérabilités) ; seul le tag `latest` est gratuit, d'où le digest. L'alternative `bitnamilegacy/minio` est archivée, sans correctifs de sécurité. |
| Images figées par digest SHA-256 | Construction reproductible ; une image modifiée en amont ne change pas silencieusement l'infrastructure. |
| Ports liés à 127.0.0.1 | Services accessibles depuis la machine seulement, pas depuis le réseau. |
| Conteneurs non-root (MLflow uid 10001, MinIO uid 65532) | Limite l'impact d'une compromission. |
| `--allowed-hosts` sur le serveur MLflow | Protection contre le *DNS rebinding* (en-têtes Host acceptés : localhost, le nom du service et `host.docker.internal` pour les pods Kubeflow). |
| Secrets dans `.env` (ignoré par git), modèle `.env.example` | Aucun secret dans le dépôt (CLAUDE.md, section 13). |
| Limites mémoire (512 Mo, 1 Go pour MLflow) | Cohabiter avec Kafka, Flink et NiFi au sprint 2 sur la même machine. Mesuré : MLflow ≈ 360 Mo, PostgreSQL et MinIO ≈ 65 Mo chacun. |
| MLflow : 1 processus de travail, `MLFLOW_SERVER_ENABLE_JOB_EXECUTION=false` | Par défaut, MLflow 3.18 lance un gestionnaire de tâches GenAI (≈ 8 processus de 240 Mo, soit 2 Go) inutile ici. Avec la limite mémoire, ces processus mouraient et MLflow arrêtait tout le serveur, qui redémarrait en boucle. |
| PostgreSQL sans port publié ; `docker-compose.sql-access.yml` pour un outil SQL (port `POSTGRES_HOST_PORT`, 5433 sur ce poste) | Un port publié serait joignable depuis les pods Kubeflow (`host.docker.internal`) ; seul MLflow doit utiliser la base. 5432 est pris par un PostgreSQL installé sur Windows. |
| Bucket `novaria-data` et compte MinIO `novaria-pipeline` (`infra/minio/setup_pipeline_access.py`) | Données d'entrée du pipeline Kubeflow ; compte limité à ce bucket (voir `docs/kubeflow.md`). |
| `mlflow==3.18.0` côté serveur | Même version que les scripts (`requirements.lock.txt`). |

## Construction hors ligne (`infra/mlflow/wheels/`)

Avec une connexion lente, la construction de l'image (MLflow et ses dépendances, environ 250 Mo) peut prendre
plus de 15 minutes. Les paquets peuvent être téléchargés une fois sur la machine, puis installés depuis
`infra/mlflow/wheels/` (non versionné) :

```bash
pip download -r infra/mlflow/requirements.txt -d infra/mlflow/wheels --only-binary=:all: \
    --python-version 3.12 --implementation cp --abi cp312 \
    --platform manylinux_2_28_x86_64 --platform manylinux_2_17_x86_64 --platform manylinux2014_x86_64 --platform any
```

Si le dossier contient des paquets, le Dockerfile les installe avec `--no-index` (aucun téléchargement : 40 s au lieu de
plus de 30 min sur une connexion à 40 Ko/s) ; s'il est vide, il télécharge depuis PyPI.

## Limites (POC)

- MLflow accède à MinIO avec le compte administrateur MinIO ; en production, un compte dédié limité au bucket.
- Pas de TLS entre les services ni vers les clients (réseau local uniquement).
- Pas d'authentification sur l'interface MLflow (exposée sur 127.0.0.1 seulement).
- Sauvegarde des volumes `pgdata` et `minio-data` non prévue.
