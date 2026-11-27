"""Tests de validation et d'analyse d'une mesure HTTP."""

import json

import pytest
from fastapi.testclient import TestClient

from novaria.api.main import create_app
from novaria.api.providers import MockModelProvider
from novaria.common.schemas import Health, Prediction

EVENT = {
    "event_id": "evt-001",
    "sensor_id": "IoT-003",
    "station_id": "Changping",
    "timestamp": "2017-01-15T08:00:00",
    "pm25": 31,
    "pm10": 38,
    "no2": 95,
    "co": 420,
    "so2": 40,
    "o3": 8,
    "temp": -3.5,
}


@pytest.fixture
def api(monkeypatch):
    """Démarre le service avec le fournisseur mock identifié dans ses réponses."""
    monkeypatch.setenv("NOVARIA_MODEL_PROVIDER", "mock")
    monkeypatch.delenv("NOVARIA_HOURLY_LIMITS", raising=False)
    with TestClient(create_app()) as client:
        yield client


def test_reference_and_health(api):
    """Vérifie le contrat complet, la mesure de référence et Swagger."""
    response = api.post("/predict", json=EVENT)
    assert response.status_code == 200
    assert Prediction.model_validate(response.json()).model_dump(mode="json") == {
        **{
            key: EVENT[key]
            for key in ("event_id", "sensor_id", "station_id", "timestamp")
        },
        "score": 1.0,
        "is_anomaly": False,
        "anomaly_type": None,
        "alert_type": None,
        "data_quality_issues": [],
        "health_exceedances": None,
        "model_version": "mock-v1",
        "model_provider": "mock",
        "is_mock": True,
    }
    health = api.get("/health")
    assert health.status_code == 200
    assert Health.model_validate(health.json()).model_loaded
    assert health.json()["is_mock"]
    assert api.get("/docs").status_code == 200


@pytest.mark.parametrize(
    "update",
    [
        {"sensor_id": " "},
        {"station_id": None},
        {"timestamp": "2017-01-15"},
        {"timestamp": "hier"},
        {"timestamp": 1484467200},
        {"pm25": "31"},
        {"pm25": True},
        {"pm25": float("nan")},
        {"pm25": float("inf")},
        {"injection": "pic"},
    ],
)
def test_invalid_request_returns_422(api, update):
    """Vérifie les types, valeurs finies, horodatages et champs inconnus."""
    response = api.post(
        "/predict",
        content=json.dumps(EVENT | update),
        headers={"Content-Type": "application/json"},
    )
    assert response.status_code == 422
    assert response.json()["detail"][0]["loc"][-1] == next(iter(update))


def test_legacy_event_without_id(api):
    """Conserve la compatibilité avec les messages du contrat initial."""
    request = {key: value for key, value in EVENT.items() if key != "event_id"}
    response = api.post("/predict", json=request)
    assert response.status_code == 200
    assert response.json()["event_id"]


@pytest.mark.parametrize(
    ("changes", "expected"),
    [
        ({"pm25": None}, [("pm25", "missing")]),
        ({"pm25": -5}, [("pm25", "invalid")]),
        ({"temp": 300}, [("temp", "invalid")]),
        ({"wd": "invalid"}, [("wd", "invalid")]),
        ({"pm25": 75}, [("pm25", "invalid"), ("pm10", "invalid")]),
    ],
)
def test_data_quality_is_reported_separately(api, changes, expected):
    """Identifie les valeurs nettoyées ou imputées par le traitement partagé."""
    response = api.post("/predict", json=EVENT | changes)
    assert response.status_code == 200
    result = response.json()
    assert result["anomaly_type"] == "defaut_capteur"
    assert not result["is_anomaly"]
    assert [
        (issue["field"], issue["reason"]) for issue in result["data_quality_issues"]
    ] == expected


def test_anomaly_independent_of_hourly_health_threshold(monkeypatch):
    """Sépare une alerte statistique et un dépassement du seuil horaire configuré."""
    monkeypatch.setenv("NOVARIA_MODEL_PROVIDER", "mock")
    monkeypatch.setenv("NOVARIA_HOURLY_LIMITS", '{"no2": 100}')
    with TestClient(create_app()) as client:
        spike = client.post("/predict", json=EVENT | {"pm25": 310, "pm10": 380}).json()
        hourly_alert = client.post("/predict", json=EVENT | {"no2": 120}).json()
        both = client.post(
            "/predict",
            json=EVENT | {"pm25": 310, "pm10": 380, "no2": 120, "wd": "invalid"},
        ).json()

    assert spike["is_anomaly"] and spike["anomaly_type"] == "pollution_atypique"
    assert spike["health_exceedances"] == []
    assert (
        not hourly_alert["is_anomaly"]
        and hourly_alert["anomaly_type"] == "seuil_sanitaire"
    )
    assert hourly_alert["health_exceedances"] == ["no2"]
    assert both["anomaly_type"] == "defaut_capteur"
    assert both["is_anomaly"] and both["health_exceedances"] == ["no2"]


def test_load_only_once(monkeypatch):
    """Vérifie que le fournisseur n'est appelé qu'au démarrage de l'API."""
    monkeypatch.setenv("NOVARIA_MODEL_PROVIDER", "mock")
    load = MockModelProvider.load
    loaded = []

    def track_load(self):
        loaded.append(True)
        return load(self)

    monkeypatch.setattr(MockModelProvider, "load", track_load)
    with TestClient(create_app()) as client:
        for _ in range(3):
            assert client.post("/predict", json=EVENT).status_code == 200
    assert len(loaded) == 1


def test_unavailable_model_returns_503_without_mock(monkeypatch, tmp_path):
    """Signale l'absence d'artefact local sans basculer vers le mode mock."""
    monkeypatch.setenv("NOVARIA_MODEL_PROVIDER", "local")
    monkeypatch.setenv("NOVARIA_MODELS_DIR", str(tmp_path))
    with TestClient(create_app()) as client:
        health = client.get("/health")
        assert health.status_code == 503
        result = Health.model_validate(health.json())
        assert result.status == "unavailable"
        assert not result.model_loaded and not result.is_mock
        assert result.model_version is None
        assert client.post("/predict", json=EVENT).status_code == 503
