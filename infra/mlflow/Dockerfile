# Serveur MLflow du projet : MLflow + pilote PostgreSQL + client S3 (MinIO).
# Image de base figée par digest ; exécution sans root.
#
# wheels/ (non versionné) peut contenir les paquets déjà téléchargés, pour
# construire sans tout retélécharger (connexion lente) :
#   pip download -r requirements.txt -d wheels --only-binary=:all: \
#       --python-version 3.12 --platform manylinux2014_x86_64 --platform any ...
# S'il est vide, pip télécharge depuis PyPI.
FROM python:3.12-slim@sha256:5e2bdb1b0d542d5039cffb8181c65930093949194a65c365dfb1b0fe13a4fa5b

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    MLFLOW_DISABLE_AGENT_HINT=1

WORKDIR /app
COPY requirements.txt .
COPY wheels/ /tmp/wheels/
# --no-index si wheels/ contient les paquets : aucun téléchargement (sinon PyPI).
RUN if ls /tmp/wheels/*.whl >/dev/null 2>&1; then \
        pip install --no-cache-dir --no-index --find-links /tmp/wheels -r requirements.txt; \
    else \
        pip install --no-cache-dir -r requirements.txt; \
    fi \
    && rm -rf /tmp/wheels \
    && useradd --system --uid 10001 --no-create-home mlflow

COPY --chmod=755 entrypoint.sh create_bucket.py ./

USER mlflow
EXPOSE 5000
ENTRYPOINT ["/app/entrypoint.sh"]
