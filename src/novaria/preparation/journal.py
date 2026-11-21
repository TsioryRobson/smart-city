"""Journal de vérification des données (fichier ``.log`` texte).

Chaque problème détecté pendant la préparation (manquant, doublon, incohérence,
capteur figé) y est écrit avec la station, l'horodatage, la colonne, la valeur,
le type d'erreur et l'action appliquée. Le même journal sert en batch et en
streaming.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from novaria.common.config import DATA_CHECKS_LOG_PATH, STATION_COLUMN

LOGGER_NAME = "novaria.data_checks"
logger = logging.getLogger(LOGGER_NAME)


def setup_data_checks_log(path: str | Path = DATA_CHECKS_LOG_PATH) -> Path:
    """Configure le journal de vérification dans un fichier texte, écrasé à chaque appel.

    Les éventuels gestionnaires déjà attachés sont fermés et retirés, pour
    qu'un second appel n'écrive pas dans deux fichiers à la fois.

    Args:
        path: chemin du fichier de log ; le dossier parent est créé si besoin.

    Returns:
        Le chemin du fichier de log.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    close_data_checks_log()
    handler = logging.FileHandler(path, mode="w", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return path


def close_data_checks_log() -> None:
    """Ferme et retire tous les gestionnaires du journal de vérification.

    À appeler en fin de traitement pour libérer le fichier de log.
    """
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()


def log_issue(
    station: Any,
    timestamp: Any,
    column: str,
    value: Any,
    error: str,
    action: str,
) -> None:
    """Écrit un problème détecté dans le journal de vérification.

    Args:
        station: station concernée.
        timestamp: horodatage (ou période) concerné.
        column: variable concernée (``-`` pour une ligne entière).
        value: valeur d'origine.
        error: type d'erreur (``valeur_manquante``, ``doublon``, ``hors_bornes``...).
        action: action appliquée (``signale``, ``supprime`` ou ``impute``).
    """
    logger.info(
        "station=%s | timestamp=%s | column=%s | value=%s | error=%s | action=%s",
        station,
        timestamp,
        column,
        value,
        error,
        action,
    )


def log_mask(
    df: pd.DataFrame, mask: pd.Series | np.ndarray, column: str, error: str, action: str
) -> None:
    """Journalise chaque ligne de ``df`` sélectionnée par ``mask`` pour ``column``.

    Écrit une entrée par ligne sélectionnée, avec la station, l'horodatage et
    la valeur d'origine de ``column``. Ne fait rien si le journal est désactivé
    (aucun gestionnaire et niveau supérieur à INFO), pour rester rapide.

    Args:
        df: DataFrame indexé par ``timestamp`` avec une colonne ``station_id``.
        mask: masque booléen aligné sur ``df``, True pour les lignes à journaliser.
        column: variable concernée.
        error: type d'erreur.
        action: action appliquée.
    """
    if not logger.isEnabledFor(logging.INFO):
        return
    rows = df.loc[np.asarray(mask), [STATION_COLUMN, column]]
    for timestamp, station, value in zip(
        rows.index, rows[STATION_COLUMN], rows[column], strict=True
    ):
        log_issue(station, timestamp, column, value, error, action)
