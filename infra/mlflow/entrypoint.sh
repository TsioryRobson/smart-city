#!/bin/sh
# Démarrage du serveur MLflow : crée le bucket MinIO, puis lance le serveur.
# Les artefacts sont servis par MLflow (--serve-artifacts) : les clients n'ont
# besoin que de MLFLOW_TRACKING_URI, jamais des identifiants MinIO.
set -eu

python /app/create_bucket.py

exec mlflow server \
    --host 0.0.0.0 \
    --port 5000 \
    --backend-store-uri "${MLFLOW_BACKEND_STORE_URI}" \
    --artifacts-destination "s3://${MLFLOW_BUCKET}" \
    --serve-artifacts \
    --allowed-hosts "${MLFLOW_ALLOWED_HOSTS}" \
    --workers 1
