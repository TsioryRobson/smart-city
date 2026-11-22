#!/usr/bin/env bash
# Donne aux pods Kubeflow l'accès aux services du sprint 1 (MLflow, MinIO).
#
# - ConfigMap novaria-pipeline-config : adresses (aucun secret) ;
# - Secret novaria-minio-pipeline : identifiants du compte MinIO du pipeline,
#   limité au bucket novaria-data (voir infra/minio/setup_pipeline_access.py).
#
# Les pods n'ont jamais accès à PostgreSQL (non publié) ni au compte
# administrateur MinIO. MLflow sert lui-même les artefacts : les pods n'ont
# besoin que de son URL pour le suivi et le Model Registry.
#
# Usage : ./kubeflow/configure-access.sh   (depuis la racine, après install.sh)
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(dirname "$HERE")"
NAMESPACE="kubeflow"
CONTEXT="docker-desktop"

current="$(kubectl config current-context)"
if [ "$current" != "$CONTEXT" ]; then
  echo "Contexte kubectl « $current » : attendu « $CONTEXT ». Arrêt." >&2
  exit 1
fi

set -a
# shellcheck disable=SC1091
. "$ROOT/.env"
set +a
: "${MINIO_PIPELINE_USER:?à définir dans .env}"
: "${MINIO_PIPELINE_PASSWORD:?à définir dans .env}"

# Kubernetes de Docker Desktop : host.docker.internal désigne la machine, où
# Docker Compose publie MLflow (5000) et MinIO (9000) sur 127.0.0.1.
kubectl -n "$NAMESPACE" create configmap novaria-pipeline-config \
  --from-literal=MLFLOW_TRACKING_URI=http://host.docker.internal:5000 \
  --from-literal=MINIO_ENDPOINT=host.docker.internal:9000 \
  --from-literal=NOVARIA_DATA_BUCKET="${NOVARIA_DATA_BUCKET:-novaria-data}" \
  --dry-run=client -o yaml | kubectl apply -f -

kubectl -n "$NAMESPACE" create secret generic novaria-minio-pipeline \
  --from-literal=MINIO_ACCESS_KEY="$MINIO_PIPELINE_USER" \
  --from-literal=MINIO_SECRET_KEY="$MINIO_PIPELINE_PASSWORD" \
  --dry-run=client -o yaml | kubectl apply -f -

echo "Accès configurés dans le namespace $NAMESPACE."
