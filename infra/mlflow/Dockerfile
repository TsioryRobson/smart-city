# =============================================================================
# Serveur MLflow (tracking + registry + proxy d'artefacts)
# Image construite localement : l'image officielle n'inclut pas le pilote
# PostgreSQL, et cela permet de figer exactement les versions.
# =============================================================================
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt && rm /tmp/requirements.txt

# Utilisateur non-root (bonne pratique DevSecOps, vérifiée par Snyk)
RUN useradd --create-home --uid 10001 mlflow \
    && mkdir -p /mlartifacts \
    && chown mlflow:mlflow /mlartifacts
USER mlflow

EXPOSE 5000

# La configuration (base, artefacts, hôtes autorisés...) vient entièrement
# des variables d'environnement MLFLOW_* définies dans docker-compose.yml.
CMD ["mlflow", "server"]
