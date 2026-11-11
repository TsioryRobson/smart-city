# =============================================================================
# Conteneur de développement (VS Code Dev Containers)
# Contient les dépendances ML figées (requirements/ml.txt) + outils de dev.
# La future image de training Kubeflow réutilisera le même requirements/ml.txt.
# =============================================================================
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# git : utilisé par MLflow pour enregistrer le commit de chaque run
RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/* \
    && git config --system --add safe.directory /workspace

COPY requirements/ml.txt requirements/dev.txt /tmp/requirements/
RUN pip install -r /tmp/requirements/ml.txt -r /tmp/requirements/dev.txt \
    && rm -rf /tmp/requirements

# Utilisateur non-root ; UID 1000 = utilisateur par défaut de WSL2,
# ce qui évite les problèmes de droits sur le dossier monté.
RUN useradd --create-home --uid 1000 --shell /bin/bash dev
USER dev

WORKDIR /workspace
