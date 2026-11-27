"""API Novaria de détection d'anomalies sur une mesure brute."""

import logging
import math
import os
from contextlib import asynccontextmanager
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import TypeAdapter

from novaria.api.providers import (
    LoadedModel,
    LocalModelProvider,
    MLflowModelProvider,
    MockModelProvider,
    ModelProvider,
)
from novaria.common.config import (
    MEASURES,
    MLFLOW_MODEL_NAME,
    MODEL_FEATURES,
    MODELS_DIR,
    POLLUTANTS,
)
from novaria.common.schemas import (
    AlertType,
    DataQualityIssue,
    Health,
    Measure,
    Measurement,
    Prediction,
    ProviderMode,
)
from novaria.preparation.records import prepare_record

LOGGER = logging.getLogger(__name__)


def configured_provider(mode: ProviderMode) -> ModelProvider:
    """Construit le fournisseur de modèles choisi par l'environnement."""
    if mode == "local":
        return LocalModelProvider(
            Path(os.environ.get("NOVARIA_MODELS_DIR", str(MODELS_DIR))),
            os.environ.get("NOVARIA_MODEL_ALGORITHM") or None,
        )
    if mode == "mock":
        return MockModelProvider()
    return MLflowModelProvider()


def hourly_limits() -> dict[str, float]:
    """Lit les seuils métier horaires en µg/m³ pour les polluants configurés."""
    limits = TypeAdapter(dict[str, Measure]).validate_json(
        os.environ.get("NOVARIA_HOURLY_LIMITS") or "{}"
    )
    if any(name not in POLLUTANTS or value <= 0 for name, value in limits.items()):
        raise ValueError(
            "NOVARIA_HOURLY_LIMITS attend des polluants et des seuils positifs"
        )
    return limits


def predict_measurement(
    measurement: Measurement,
    model: LoadedModel,
    mode: ProviderMode,
    limits: dict[str, float],
) -> Prediction:
    """Prépare une mesure, calcule le score et applique les règles métier."""
    record = measurement.model_dump(mode="json", exclude_unset=True)
    prepared = prepare_record(record)
    issues = [
        DataQualityIssue(
            field=name, reason="missing" if record[name] is None else "invalid"
        )
        for name in MEASURES
        if name in record and prepared[name] is None
    ]
    exceedances = [
        name
        for name, limit in limits.items()
        if prepared[name] is not None and prepared[name] > limit
    ]
    features = pd.DataFrame(
        [{name: prepared[name] for name in MODEL_FEATURES}],
        columns=MODEL_FEATURES,
        dtype="float64",
    )
    score = float(model.scorer.decision_function(features)[0])
    if not math.isfinite(score):
        raise HTTPException(status_code=503, detail="Score du modèle indisponible")
    is_anomaly = score < 0
    alert: AlertType | None = None
    if issues:
        alert = "defaut_capteur"
    elif exceedances:
        alert = "seuil_sanitaire"
    elif is_anomaly:
        alert = "pollution_atypique"

    return Prediction(
        event_id=measurement.event_id,
        sensor_id=measurement.sensor_id,
        station_id=measurement.station_id,
        timestamp=measurement.timestamp,
        score=score,
        is_anomaly=is_anomaly,
        anomaly_type=alert,
        alert_type=alert,
        data_quality_issues=issues,
        health_exceedances=exceedances if limits else None,
        model_version=model.version,
        model_provider=mode,
        is_mock=mode == "mock",
    )


def create_app() -> FastAPI:
    """Crée le service et charge le modèle une fois par processus au démarrage."""
    mode = TypeAdapter(ProviderMode).validate_python(
        os.environ.get("NOVARIA_MODEL_PROVIDER", "mlflow")
    )
    provider = configured_provider(mode)
    limits = hourly_limits()
    model: LoadedModel | None = None

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        """Charge le modèle et maintient les routes disponibles si le chargement échoue."""
        nonlocal model
        try:
            model = provider.load()
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning(
                "Chargement du modèle %s impossible (%s)", mode, type(exc).__name__
            )
        try:
            yield
        finally:
            model = None

    app = FastAPI(title="Novaria Smart City AI", lifespan=lifespan)

    @app.exception_handler(RequestValidationError)
    async def invalid_request(_request, exc: RequestValidationError) -> JSONResponse:
        """Retourne une erreur 422 sérialisable, y compris pour NaN et Infinity."""
        return JSONResponse(
            status_code=422,
            content={
                "detail": [
                    {key: error[key] for key in ("type", "loc", "msg")}
                    for error in exc.errors()
                ]
            },
        )

    @app.get(
        "/health",
        responses={503: {"model": Health, "description": "Modèle indisponible"}},
    )
    def health(response: Response) -> Health:
        """Indique si le service a chargé son modèle."""
        if model is None:
            response.status_code = 503
        return Health(
            status="ok" if model else "unavailable",
            model_loaded=model is not None,
            model_name=MLFLOW_MODEL_NAME,
            model_version=model.version if model else None,
            model_alias=model.alias if model else None,
            model_provider=mode,
            is_mock=mode == "mock",
        )

    @app.post("/predict", responses={503: {"description": "Modèle indisponible"}})
    def predict(measurement: Measurement) -> Prediction:
        """Analyse une mesure après validation du contrat d'entrée."""
        if model is None:
            raise HTTPException(status_code=503, detail="Modèle indisponible")
        return predict_measurement(measurement, model, mode, limits)

    return app


app = create_app()
