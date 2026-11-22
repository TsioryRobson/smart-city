# Image de validation de l'installation Kubeflow : client MinIO + MLflow (léger).
# Construite en local, sans registre : Kubernetes de Docker Desktop (kubeadm)
# utilise directement les images de Docker.
#   docker build -t novaria/kfp-check:0.1.0 kubeflow/check
# wheels/ (non versionné, sauf .gitkeep) peut contenir les paquets téléchargés à
# l'avance (connexion lente) ; vide, l'installation passe par PyPI :
#   pip download minio==7.2.20 mlflow-skinny==3.18.0 -d kubeflow/check/wheels \
#     --only-binary=:all: --python-version 3.12 --platform manylinux2014_x86_64
FROM python:3.12-slim@sha256:5e2bdb1b0d542d5039cffb8181c65930093949194a65c365dfb1b0fe13a4fa5b

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    MLFLOW_DISABLE_AGENT_HINT=1

COPY wheels/ /tmp/wheels/
# --no-index si wheels/ contient les paquets : aucun téléchargement (sinon PyPI).
RUN if ls /tmp/wheels/*.whl >/dev/null 2>&1; then \
        pip install --no-index --find-links /tmp/wheels "minio==7.2.20" "mlflow-skinny==3.18.0"; \
    else \
        pip install "minio==7.2.20" "mlflow-skinny==3.18.0"; \
    fi \
    && rm -rf /tmp/wheels \
    && useradd --system --uid 10001 --no-create-home checker

COPY check_access.py /app/check_access.py
USER 10001
ENTRYPOINT ["python", "/app/check_access.py"]
