"""Variables dérivées (sans état) : direction du vent et temps en encodage cyclique.

L'encodage sinus / cosinus garde la proximité circulaire : 23 h est proche de
0 h, décembre de janvier, NNW de N. Les variables temporelles donnent au modèle
le contexte saisonnier et journalier (chauffage l'hiver, pics de trafic), pour
qu'une pollution normale pour la saison ne soit pas prise pour une anomalie.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from novaria.common.config import WIND_DIRECTION_COLUMN, WIND_DIRECTIONS

# Angle de chaque direction en radians : N = 0, puis 22,5° par point cardinal.
WIND_ANGLES: dict[str, float] = {
    direction: 2 * np.pi * i / len(WIND_DIRECTIONS)
    for i, direction in enumerate(WIND_DIRECTIONS)
}


def cyclic_encoding(values: pd.Series, period: float) -> tuple[pd.Series, pd.Series]:
    """Encode une variable cyclique en sinus et cosinus.

    Args:
        values: valeurs numériques (NaN conservés).
        period: longueur du cycle (24 pour l'heure, 12 pour le mois...).

    Returns:
        Le couple ``(sinus, cosinus)``, chaque valeur dans [-1, 1].
    """
    angle = 2 * np.pi * values.astype("float64") / period
    return np.sin(angle), np.cos(angle)


def encode_wind_direction(df: pd.DataFrame) -> pd.DataFrame:
    """Ajoute ``wd_sin`` et ``wd_cos`` à partir de la direction du vent.

    Une direction manquante ou inconnue donne NaN, imputé ensuite par le
    préprocesseur du modèle.

    Args:
        df: trame de mesures avec la colonne ``wd``.

    Returns:
        Une copie de ``df`` avec les deux colonnes ajoutées.
    """
    out = df.copy()
    angle = out[WIND_DIRECTION_COLUMN].map(WIND_ANGLES).astype("float64")
    out["wd_sin"] = np.sin(angle)
    out["wd_cos"] = np.cos(angle)
    return out


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Ajoute l'heure et le mois en encodage cyclique, à partir de l'index.

    L'index est en heure locale de Pékin (voir ``cleaning.to_local_time``).

    Args:
        df: trame de mesures indexée par ``timestamp``.

    Returns:
        Une copie de ``df`` avec ``hour_sin``, ``hour_cos``, ``month_sin`` et
        ``month_cos``.
    """
    out = df.copy()
    index = pd.DatetimeIndex(out.index)
    out["hour_sin"], out["hour_cos"] = cyclic_encoding(
        pd.Series(index.hour, index=out.index), 24
    )
    out["month_sin"], out["month_cos"] = cyclic_encoding(
        pd.Series(index.month - 1, index=out.index), 12
    )
    return out


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Ajoute toutes les variables dérivées (vent puis temps).

    Après cette étape, la trame contient toutes les colonnes de
    ``MODEL_FEATURES``.

    Args:
        df: trame de mesures nettoyée.

    Returns:
        Une copie de ``df`` avec les variables dérivées.
    """
    return add_time_features(encode_wind_direction(df))
