"""Contrats JSON partagés par le simulateur et l'API."""

from datetime import datetime
from typing import Annotated, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, field_validator

Identifier = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
Measure = Annotated[float, Field(strict=True, allow_inf_nan=False)]
Injection = Literal[
    "pic", "valeur_impossible", "combinaison_inhabituelle", "valeur_manquante"
]


class Measurement(BaseModel):
    """Mesure brute avec concentrations en µg/m³ et température en °C."""

    model_config = ConfigDict(extra="forbid")

    event_id: Identifier = Field(default_factory=lambda: str(uuid4()))
    sensor_id: Identifier
    station_id: Identifier
    timestamp: datetime
    pm25: Measure | None
    pm10: Measure | None
    no2: Measure | None
    co: Measure | None
    so2: Measure | None
    o3: Measure | None
    temp: Measure | None
    pres: Measure | None = None
    dewp: Measure | None = None
    rain: Measure | None = None
    wd: str | None = None
    wspm: Measure | None = None

    @field_validator("timestamp", mode="before")
    @classmethod
    def require_iso_timestamp(cls, value: object) -> object:
        """Accepte un datetime ou une chaîne ISO 8601 avec une heure."""
        if not isinstance(value, datetime) and (
            not isinstance(value, str)
            or len(value) < 16
            or value[10] not in {"T", "t", " "}
        ):
            raise ValueError("horodatage ISO 8601 avec heure attendu")
        return value


class RawMeasurement(Measurement):
    """Événement de rejeu avec étiquette de scénario destinée à la démo."""

    injection: Injection | None = None


ProviderMode = Literal["mlflow", "local", "mock"]
AlertType = Literal["pollution_atypique", "defaut_capteur", "seuil_sanitaire"]


class DataQualityIssue(BaseModel):
    """Mesure manquante ou invalidée par la préparation partagée."""

    field: str
    reason: Literal["missing", "invalid"]


class Prediction(BaseModel):
    """Résultat du détecteur et états indépendants des règles métier."""

    event_id: str
    sensor_id: str
    station_id: str
    timestamp: datetime
    score: Measure
    is_anomaly: bool
    anomaly_type: AlertType | None
    alert_type: AlertType | None
    data_quality_issues: list[DataQualityIssue]
    health_exceedances: list[str] | None
    model_version: str
    model_provider: ProviderMode
    is_mock: bool


class Health(BaseModel):
    """Disponibilité du service et identité du modèle chargé."""

    status: Literal["ok", "unavailable"]
    model_loaded: bool
    model_name: str
    model_version: str | None
    model_alias: str | None
    model_provider: ProviderMode
    is_mock: bool
