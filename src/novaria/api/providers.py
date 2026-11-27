"""Chargement des pipelines entraînés ou du détecteur de démonstration."""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from novaria.common.config import LOCAL_TRACKING_URI, MLFLOW_MODEL_NAME, MODEL_FEATURES
from novaria.models.detectors import CalibratedDetector


class Scorer(Protocol):
    """Contrat du score calibré partagé par les trois algorithmes."""

    def decision_function(self, features: pd.DataFrame) -> np.ndarray:
        """Renvoie un score par ligne, négatif pour une anomalie."""
        ...


@dataclass(frozen=True)
class LoadedModel:
    """Modèle prêt à scorer, avec la version résolue au démarrage."""

    scorer: Scorer
    version: str
    alias: str | None = None


class ModelProvider(Protocol):
    """Source configurable d'un modèle d'inférence."""

    def load(self) -> LoadedModel:
        """Charge et vérifie la compatibilité du modèle."""
        ...


def validate_pipeline(model: Pipeline) -> None:
    """Vérifie l'ordre d'entraînement et la convention du détecteur."""
    if (
        not isinstance(model, Pipeline)
        or not isinstance(model.named_steps.get("detector"), CalibratedDetector)
        or list(getattr(model, "feature_names_in_", [])) != MODEL_FEATURES
    ):
        raise ValueError("Pipeline incompatible avec le contrat d'inférence Novaria")


class MLflowModelProvider:
    """Résout le champion et charge cette version après vérification SHA-256."""

    def load(self) -> LoadedModel:
        """Charge le modèle pointé par l'alias au démarrage."""
        from mlflow import MlflowClient

        from novaria.common.tracking import resolve_tracking_uri
        from novaria.pipeline.promotion import CHAMPION, load_registered_model

        tracking_uri = resolve_tracking_uri()
        registry_uri = os.environ.get("MLFLOW_REGISTRY_URI") or tracking_uri
        if (
            registry_uri == LOCAL_TRACKING_URI
            and not Path(registry_uri.removeprefix("sqlite:///")).is_file()
        ):
            raise FileNotFoundError("Registre SQLite local absent")
        client = MlflowClient(tracking_uri=tracking_uri, registry_uri=registry_uri)
        version = client.get_model_version_by_alias(MLFLOW_MODEL_NAME, CHAMPION)
        model = load_registered_model(
            MLFLOW_MODEL_NAME,
            version,
            tracking_uri=tracking_uri,
            registry_uri=registry_uri,
        )
        validate_pipeline(model)
        return LoadedModel(model, str(version.version), CHAMPION)


@dataclass(frozen=True)
class LocalModelProvider:
    """Charge une sortie locale de l'entraînement telle qu'utilisée en évaluation."""

    models_dir: Path
    algorithm: str | None = None

    def load(self) -> LoadedModel:
        """Sélectionne l'artefact et vérifie le hash du manifeste d'entraînement."""
        from novaria.pipeline.evaluation import load_selected_model
        from novaria.pipeline.promotion import candidate_entry

        algorithm, entry = candidate_entry(self.models_dir, self.algorithm)
        sha256 = entry["model_sha256"]
        model = load_selected_model(self.models_dir, algorithm, sha256)
        validate_pipeline(model)
        return LoadedModel(model, f"local:{algorithm}:{sha256[:12]}")


class MockModelProvider:
    """Simule des scores fixes pour les tests et la démonstration."""

    def load(self) -> LoadedModel:
        """Expose la règle déterministe avec une version dédiée au mock."""
        return LoadedModel(self, "mock-v1")

    def decision_function(self, features: pd.DataFrame) -> np.ndarray:
        """Simule une détection pour les pics et les combinaisons de la démo."""
        spike = features["pm25"] >= 300
        combination = (
            (features["pm25"] < 20)
            & (features["no2"] >= 250)
            & (features["co"] >= 7000)
        )
        return np.where(spike | combination, -1.0, 1.0)
