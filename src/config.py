"""Constantes centralisées du projet Smart City AI."""

from pathlib import Path

import pandas as pd

# --- Chemins -----------------------------------------------------------------
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DATA_DIR: Path = DATA_DIR / "raw"
# Un CSV par station (12 stations), jamais modifiés.
RAW_FILE_PATTERN: str = "PRSA_Data_*_20130301-20170228.csv"
REPORTS_DIR: Path = PROJECT_ROOT / "reports"
LOGS_DIR: Path = PROJECT_ROOT / "logs"
DATA_CHECKS_LOG_PATH: Path = LOGS_DIR / "data_checks.log"

# Station utilisée par le simulateur et les exemples de contrat (démo).
DEMO_STATION: str = "Aotizhongxin"

# --- Colonnes ----------------------------------------------------------------
# Colonnes servant à construire l'horodatage.
DATETIME_COLUMNS: list[str] = ["year", "month", "day", "hour"]
STATION_COLUMN: str = "station"

# Correspondance noms d'origine -> snake_case (noms du futur contrat d'API).
COLUMN_MAPPING: dict[str, str] = {
    "PM2.5": "pm25",
    "PM10": "pm10",
    "NO2": "no2",
    "CO": "co",
    "SO2": "so2",
    "O3": "o3",
    "TEMP": "temp",
}

FEATURES: list[str] = ["pm25", "pm10", "no2", "co", "so2", "o3", "temp"]

# --- Contrôles de cohérence --------------------------------------------------
# Concentrations (µg/m³) : une valeur négative est impossible.
# La température peut être négative, elle n'est donc pas dans cette liste.
NON_NEGATIVE_FEATURES: list[str] = ["pm25", "pm10", "no2", "co", "so2", "o3"]

# Bornes physiques (min, max) inclusives. Au-delà, la valeur est impossible :
# elle est mise à NaN puis imputée. Les pics extrêmes mais possibles restent
# dans les bornes et ne sont jamais modifiés.
# Maxima observés sur les 12 stations : PM2.5 et PM10 999 (saturation capteur),
# NO2 290, CO 10 000, SO2 500, O3 1 071, TEMP 41,6 °C.
PHYSICAL_BOUNDS: dict[str, tuple[float, float]] = {
    "pm25": (0.0, 1000.0),
    "pm10": (0.0, 1000.0),
    "no2": (0.0, 1000.0),
    "co": (0.0, 20000.0),
    "so2": (0.0, 1000.0),
    "o3": (0.0, 800.0),
    "temp": (-40.0, 50.0),
}

# Capteur figé : même valeur répétée au moins N heures de suite (signalé seulement).
# 24 h, car SO2 et CO sont mesurés par paliers et répètent souvent la même valeur.
FROZEN_SENSOR_MIN_HOURS: int = 24

# --- Découpage chronologique -------------------------------------------------
# Bornes [début, fin[ : la fin d'un jeu est le début du suivant.
# Train = fenêtre glissante de 12 mois, puis 3 mois de validation et 3 mois de test.
TRAIN_START: pd.Timestamp = pd.Timestamp("2014-09-01")
VAL_START: pd.Timestamp = pd.Timestamp("2015-09-01")
TEST_START: pd.Timestamp = pd.Timestamp("2015-12-01")
# Début de la période rejouée par le simulateur : jamais vue par le modèle.
DEMO_START: pd.Timestamp = pd.Timestamp("2016-03-01")
