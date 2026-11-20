"""Constantes centralisées du projet Smart City AI."""

from pathlib import Path

# --- Chemins -----------------------------------------------------------------
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
DATA_DIR: Path = PROJECT_ROOT / "data"
DATA_PATH: Path = DATA_DIR / "PRSA_Data_Aotizhongxin_20130301-20170228.csv"
REPORTS_DIR: Path = PROJECT_ROOT / "reports"

# --- Colonnes ----------------------------------------------------------------
# Colonnes servant à construire l'horodatage.
DATETIME_COLUMNS: list[str] = ["year", "month", "day", "hour"]

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

# --- Découpage chronologique -----------------------------------------------------
TRAIN_RATIO: float = 0.70
VAL_RATIO: float = 0.15
# Le jeu de test prend le reste (0.15).
