"""Tests des modèles sérialisés et d'un registre MLflow SQLite de test."""

import pandas as pd
import pytest
from fastapi.testclient import TestClient
from mlflow import MlflowClient

from novaria.api.main import create_app
from novaria.common.config import MLFLOW_MODEL_NAME, MODEL_FEATURES
from novaria.common.io import write_json
from novaria.models.detectors import build_model
from novaria.pipeline.training import save_model, serialized_model_path
from novaria.preparation.features import add_features
from novaria.preparation.records import prepare_record
from tests.api.test_endpoints import EVENT
from tests.factories import measurements


def train_artifact(directory, algorithm):
    """Entraîne et sérialise un pipeline avec les outils du dépôt."""
    train = add_features(measurements())[MODEL_FEATURES]
    pipeline = build_model(algorithm, {}, contamination=0.05, random_state=0).fit(train)
    sha = save_model(pipeline, directory / algorithm, train)
    write_json({"algorithm": algorithm}, directory / "selection.json")
    write_json(
        {"selected": {algorithm: {"model_sha256": sha}}}, directory / "comparison.json"
    )
    return pipeline, sha


@pytest.mark.parametrize(
    "algorithm", ["isolation_forest", "local_outlier_factor", "one_class_svm"]
)
def test_local_provider_preserves_preprocessing_and_predictions(
    tmp_path, monkeypatch, algorithm
):
    """Compare le score et la classe de l'API à ceux du pipeline pour chaque algorithme."""
    pipeline, sha = train_artifact(tmp_path, algorithm)
    monkeypatch.setenv("NOVARIA_MODEL_PROVIDER", "local")
    monkeypatch.setenv("NOVARIA_MODELS_DIR", str(tmp_path))
    monkeypatch.delenv("NOVARIA_MODEL_ALGORITHM", raising=False)
    with TestClient(create_app()) as client:
        health = client.get("/health")
        assert health.status_code == 200
        assert health.json()["model_version"] == f"local:{algorithm}:{sha[:12]}"
        for measures in ({}, {"pm25": 850, "pm10": 950, "wd": "NE", "pres": 1010}):
            event = EVENT | measures
            prepared = pd.DataFrame(
                [prepare_record(event)], columns=MODEL_FEATURES, dtype="float64"
            )
            result = client.post("/predict", json=event)
            assert result.status_code == 200
            assert result.json()["score"] == pytest.approx(
                pipeline.decision_function(prepared)[0]
            )
            assert result.json()["is_anomaly"] == bool(
                pipeline.predict(prepared)[0] == -1
            )
            assert not result.json()["is_mock"]


def test_local_tampered_hash_is_rejected(tmp_path, monkeypatch):
    """Refuse un fichier sérialisé modifié après l'entraînement."""
    algorithm = "isolation_forest"
    train_artifact(tmp_path, algorithm)
    serialized_model_path(tmp_path / algorithm).write_bytes(b"changed")
    monkeypatch.setenv("NOVARIA_MODEL_PROVIDER", "local")
    monkeypatch.setenv("NOVARIA_MODELS_DIR", str(tmp_path))
    with TestClient(create_app()) as client:
        assert client.get("/health").status_code == 503


def test_mlflow_provider_resolves_champion_and_checks_hash(tmp_path, monkeypatch):
    """Charge la version portant l'alias et refuse une empreinte de registre invalide."""
    algorithm = "one_class_svm"
    _, sha = train_artifact(tmp_path, algorithm)
    uri = f"sqlite:///{tmp_path / 'registry.db'}"
    monkeypatch.setenv("NOVARIA_MODEL_PROVIDER", "mlflow")
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
    monkeypatch.delenv("MLFLOW_REGISTRY_URI", raising=False)
    registry = MlflowClient(tracking_uri=uri, registry_uri=uri)
    registry.create_registered_model(MLFLOW_MODEL_NAME)
    source = (tmp_path / algorithm).as_uri()
    registry.create_model_version(MLFLOW_MODEL_NAME, source)
    version = registry.create_model_version(
        MLFLOW_MODEL_NAME, source, tags={"model_sha256": sha}
    )

    with TestClient(create_app()) as client:
        assert client.get("/health").status_code == 503

    registry.set_registered_model_alias(MLFLOW_MODEL_NAME, "champion", version.version)
    with TestClient(create_app()) as client:
        health = client.get("/health")
        assert health.status_code == 200
        assert health.json()["model_alias"] == "champion"
        assert health.json()["model_version"] == "2"
        result = client.post("/predict", json=EVENT)
        assert result.status_code == 200
        assert result.json()["model_provider"] == "mlflow"
        assert result.json()["model_version"] == "2"

    registry.set_model_version_tag(
        MLFLOW_MODEL_NAME, version.version, "model_sha256", "changed"
    )
    with TestClient(create_app()) as client:
        assert client.get("/health").status_code == 503
        assert client.post("/predict", json=EVENT).status_code == 503
