"""Constantes centralisées du projet Novaria Smart City AI.

Les scripts de ``novaria.pipeline`` prennent ces valeurs comme défauts de leurs
arguments CLI : elles restent surchargeables sans modifier le code.
"""

from pathlib import Path

import pandas as pd

# --- Chemins -----------------------------------------------------------------
PROJECT_ROOT: Path = Path(__file__).resolve().parents[3]
DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DATA_DIR: Path = DATA_DIR / "raw"
INGESTED_DATA_DIR: Path = DATA_DIR / "ingested"
PREPARED_DATA_DIR: Path = DATA_DIR / "prepared"
# Jeux contrôlés (test + anomalies injectées), figés pour l'évaluation.
CONTROLLED_DATA_DIR: Path = DATA_DIR / "controlled"
# Un CSV par station (12 stations), jamais modifiés.
RAW_FILE_PATTERN: str = "PRSA_Data_*_20130301-20170228.csv"
REPORTS_DIR: Path = PROJECT_ROOT / "reports"
LOGS_DIR: Path = PROJECT_ROOT / "logs"
DATA_CHECKS_LOG_PATH: Path = LOGS_DIR / "data_checks.log"
CONFIGS_DIR: Path = PROJECT_ROOT / "configs"
TRAINING_CONFIG_PATH: Path = CONFIGS_DIR / "training.yaml"
EVALUATION_CONFIG_PATH: Path = CONFIGS_DIR / "evaluation.yaml"
MODELS_DIR: Path = PROJECT_ROOT / "models"

# --- MLflow (CLAUDE.md, section 7) --------------------------------------------
MLFLOW_EXPERIMENT: str = "novaria-air-anomaly"
MLFLOW_MODEL_NAME: str = "air-anomaly-detector"
# Utilisés seulement si la variable d'environnement MLFLOW_TRACKING_URI est
# absente : suivi local en SQLite, artefacts dans un dossier local.
LOCAL_TRACKING_URI: str = f"sqlite:///{PROJECT_ROOT / 'mlflow.db'}"
LOCAL_ARTIFACTS_DIR: Path = PROJECT_ROOT / "mlartifacts"

# Station utilisée par le simulateur et les exemples de contrat (démo).
DEMO_STATION: str = "Aotizhongxin"

# Fuseau des horodatages du dataset (heure locale de Pékin, sans fuseau dans
# les CSV). Un horodatage reçu avec un fuseau (ex. ``Z``) est converti vers
# ce fuseau avant le calcul des variables temporelles.
DATA_TIMEZONE: str = "Asia/Shanghai"

# --- Colonnes ----------------------------------------------------------------
# Colonnes des CSV bruts servant à construire l'horodatage.
DATETIME_COLUMNS: list[str] = ["year", "month", "day", "hour"]
# Colonnes obligatoires d'un CSV brut (contrôlées à l'ingestion).
RAW_REQUIRED_COLUMNS: list[str] = [
    *DATETIME_COLUMNS,
    "PM2.5",
    "PM10",
    "SO2",
    "NO2",
    "CO",
    "O3",
    "TEMP",
    "PRES",
    "DEWP",
    "RAIN",
    "wd",
    "WSPM",
    "station",
]

# Noms du contrat (section 6 de CLAUDE.md) : requête /predict et topics Kafka.
STATION_COLUMN: str = "station_id"
TIMESTAMP_COLUMN: str = "timestamp"

# Correspondance noms d'origine -> noms du contrat.
COLUMN_MAPPING: dict[str, str] = {
    "station": STATION_COLUMN,
    "PM2.5": "pm25",
    "PM10": "pm10",
    "NO2": "no2",
    "CO": "co",
    "SO2": "so2",
    "O3": "o3",
    "TEMP": "temp",
    "PRES": "pres",
    "DEWP": "dewp",
    "RAIN": "rain",
    "wd": "wd",
    "WSPM": "wspm",
}

# Mesures numériques du contrat (unités du dataset).
POLLUTANTS: list[str] = ["pm25", "pm10", "no2", "co", "so2", "o3"]  # µg/m³
WEATHER: list[str] = ["temp", "pres", "dewp", "rain", "wspm"]  # °C, hPa, °C, mm, m/s
NUMERIC_MEASURES: list[str] = [*POLLUTANTS, *WEATHER]
# Mesure catégorielle : direction du vent (16 points cardinaux).
WIND_DIRECTION_COLUMN: str = "wd"
WIND_DIRECTIONS: list[str] = [
    "N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
    "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW",
]  # fmt: skip
# Les 12 mesures du contrat.
MEASURES: list[str] = [*NUMERIC_MEASURES, WIND_DIRECTION_COLUMN]

# Variables dérivées (encodage cyclique : 23 h est proche de 0 h, décembre de janvier).
WIND_FEATURES: list[str] = ["wd_sin", "wd_cos"]
TIME_FEATURES: list[str] = ["hour_sin", "hour_cos", "month_sin", "month_cos"]
# Entrée du modèle : 11 mesures numériques + vent encodé + temps encodé.
MODEL_FEATURES: list[str] = [*NUMERIC_MEASURES, *WIND_FEATURES, *TIME_FEATURES]

# --- Contrôles de cohérence --------------------------------------------------
# Une valeur négative est impossible pour ces mesures.
# La température et le point de rosée peuvent être négatifs.
NON_NEGATIVE_MEASURES: list[str] = [*POLLUTANTS, "rain", "wspm"]

# Bornes physiques (min, max) inclusives. Au-delà, la valeur est impossible :
# elle est mise à NaN puis imputée. Les pics extrêmes mais possibles restent
# dans les bornes et ne sont jamais modifiés. Justification : docs/preparation.md.
PHYSICAL_BOUNDS: dict[str, tuple[float, float]] = {
    "pm25": (0.0, 1000.0),
    "pm10": (0.0, 1000.0),
    "no2": (0.0, 1000.0),
    "co": (0.0, 20000.0),
    "so2": (0.0, 1000.0),
    "o3": (0.0, 800.0),
    "temp": (-40.0, 50.0),
    "pres": (870.0, 1085.0),
    "dewp": (-60.0, 35.0),
    "rain": (0.0, 200.0),
    "wspm": (0.0, 75.0),
}

# Capteur figé : même valeur répétée au moins N heures de suite (signalé seulement).
# 24 h, car SO2 et CO sont mesurés par paliers et répètent souvent la même valeur.
FROZEN_SENSOR_MIN_HOURS: int = 24
# Mesures contrôlées pour le capteur figé (la pluie vaut 0 la plupart du temps).
FROZEN_SENSOR_MEASURES: list[str] = [m for m in NUMERIC_MEASURES if m != "rain"]

# --- Découpage chronologique -------------------------------------------------
# Bornes [début, fin[ : la fin d'un jeu est le début du suivant.
# Train = fenêtre glissante de 12 mois, puis 3 mois de validation et 3 mois de test.
TRAIN_START: pd.Timestamp = pd.Timestamp("2014-09-01")
VAL_START: pd.Timestamp = pd.Timestamp("2015-09-01")
TEST_START: pd.Timestamp = pd.Timestamp("2015-12-01")
# Début de la période rejouée par le simulateur : jamais vue par le modèle.
DEMO_START: pd.Timestamp = pd.Timestamp("2016-03-01")
