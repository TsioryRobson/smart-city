"""Connexion à MLflow et traçabilité (commit Git) partagées par les scripts du pipeline.

Si la variable d'environnement ``MLFLOW_TRACKING_URI`` est définie (serveur
MLflow de Docker Compose), elle est utilisée. Sinon, le suivi est local :
base SQLite ``mlflow.db`` et artefacts dans ``mlartifacts/`` (non versionnés).
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import mlflow

from novaria.common.config import (
    LOCAL_ARTIFACTS_DIR,
    LOCAL_TRACKING_URI,
    MLFLOW_EXPERIMENT,
    PROJECT_ROOT,
)


def resolve_tracking_uri(tracking_uri: str | None = None) -> str:
    """Choisit l'URI de suivi MLflow : argument, puis variable d'environnement, puis local.

    Args:
        tracking_uri: URI explicite (ex. passée en argument CLI), prioritaire.

    Returns:
        L'URI de suivi à utiliser.
    """
    return tracking_uri or os.environ.get("MLFLOW_TRACKING_URI") or LOCAL_TRACKING_URI


def local_artifacts_dir(sqlite_uri: str) -> Path:
    """Donne le dossier d'artefacts associé à une base de suivi SQLite locale.

    Pour la base du projet, c'est ``LOCAL_ARTIFACTS_DIR`` ; pour une autre
    base (ex. dans un dossier temporaire de test), un dossier ``mlartifacts``
    placé à côté du fichier de base.

    Args:
        sqlite_uri: URI de la forme ``sqlite:///<chemin>``.

    Returns:
        Le dossier des artefacts.
    """
    db_path = Path(sqlite_uri.removeprefix("sqlite:///"))
    if (
        db_path.resolve()
        == Path(LOCAL_TRACKING_URI.removeprefix("sqlite:///")).resolve()
    ):
        return LOCAL_ARTIFACTS_DIR
    return db_path.parent / "mlartifacts"


def setup_mlflow(
    tracking_uri: str | None = None, experiment: str = MLFLOW_EXPERIMENT
) -> str:
    """Configure MLflow et sélectionne l'expérience, créée si besoin.

    En suivi local (SQLite), l'expérience est créée avec ses artefacts dans
    ``LOCAL_ARTIFACTS_DIR`` ; avec un serveur, c'est le serveur qui choisit
    l'emplacement des artefacts (MinIO).

    Args:
        tracking_uri: URI explicite ; sinon voir ``resolve_tracking_uri``.
        experiment: nom de l'expérience MLflow.

    Returns:
        L'identifiant de l'expérience.
    """
    uri = resolve_tracking_uri(tracking_uri)
    mlflow.set_tracking_uri(uri)
    existing = mlflow.get_experiment_by_name(experiment)
    if existing is not None:
        experiment_id = existing.experiment_id
    elif uri.startswith("sqlite:///"):
        artifacts = local_artifacts_dir(uri)
        artifacts.mkdir(parents=True, exist_ok=True)
        experiment_id = mlflow.create_experiment(
            experiment, artifact_location=artifacts.as_uri()
        )
    else:
        experiment_id = mlflow.create_experiment(experiment)
    mlflow.set_experiment(experiment_id=experiment_id)
    return experiment_id


def git_sha() -> str:
    """Renvoie le commit Git courant, suffixé de ``-dirty`` s'il y a des modifications.

    Returns:
        Le SHA du commit, ``<sha>-dirty`` si le dépôt contient des
        modifications non commitées, ou ``unknown`` hors d'un dépôt Git.
    """
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return f"{sha}-dirty" if status else sha
