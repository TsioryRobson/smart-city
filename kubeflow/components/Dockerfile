# Image des composants du pipeline d'entraînement Kubeflow (novaria/pipeline).
# Contient le paquet novaria (src/) et les configurations (configs/) : chaque
# étape lance un script de novaria.pipeline avec ses arguments.
#
# Construite en local depuis la racine du dépôt, sans registre (Kubernetes de
# Docker Desktop en mode kubeadm utilise directement les images de Docker) :
#   docker build -f kubeflow/components/Dockerfile \
#     --build-arg GIT_SHA=$(git rev-parse HEAD) -t novaria/pipeline:0.1.0 .
#
# wheels/ (non versionné, sauf .gitkeep) peut contenir les paquets téléchargés à
# l'avance (connexion lente) ; vide, l'installation passe par PyPI. Les wheels
# sont montés le temps de l'installation : ils ne restent pas dans l'image.
FROM python:3.12-slim@sha256:5e2bdb1b0d542d5039cffb8181c65930093949194a65c365dfb1b0fe13a4fa5b

# Commit du code embarqué : tag git_sha des runs MLflow (pas de dépôt Git dans l'image).
ARG GIT_SHA=unknown

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    MLFLOW_DISABLE_AGENT_HINT=1 \
    PYTHONPATH=/app/src \
    HOME=/tmp \
    MPLCONFIGDIR=/tmp/matplotlib \
    NOVARIA_GIT_SHA=${GIT_SHA}

WORKDIR /app
COPY kubeflow/components/requirements.txt /app/requirements.txt
# --no-index si wheels/ contient les paquets : aucun téléchargement (sinon PyPI).
RUN --mount=type=bind,source=kubeflow/components/wheels,target=/wheels \
    if ls /wheels/*.whl >/dev/null 2>&1; then \
        pip install --no-index --find-links /wheels -r requirements.txt; \
    else \
        pip install -r requirements.txt; \
    fi \
    && useradd --system --uid 10001 --no-create-home novaria

# Code et configuration en lecture seule ; les sorties vont dans les dossiers
# d'artefacts donnés par Kubeflow.
COPY configs/ /app/configs/
COPY src/novaria/ /app/src/novaria/
USER 10001
