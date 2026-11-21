"""Mise au format, vérification et correction des mesures.

Règles du projet (voir CLAUDE.md, section 2) :
- les pics extrêmes mais physiquement possibles ne sont jamais modifiés,
  puisque ce sont les anomalies à détecter ;
- les valeurs impossibles (négatives, hors bornes physiques, PM2.5 > PM10,
  direction de vent inconnue) sont mises à NaN, puis imputées par le
  préprocesseur du modèle ;
- un capteur figé est seulement signalé ;
- la seule suppression de lignes autorisée concerne les doublons
  (station, horodatage) ;
- chaque problème détecté est écrit dans le journal de vérification.

Format commun (« trame de mesures ») : un DataFrame indexé par ``timestamp``
(heure locale de Pékin, sans fuseau), avec ``station_id`` puis ``MEASURES``.
Les fonctions marquées « sans état » s'appliquent ligne à ligne et servent
aussi en streaming (voir ``novaria.preparation.records``).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from novaria.common.config import (
    COLUMN_MAPPING,
    DATA_TIMEZONE,
    DATETIME_COLUMNS,
    FROZEN_SENSOR_MEASURES,
    FROZEN_SENSOR_MIN_HOURS,
    MEASURES,
    NON_NEGATIVE_MEASURES,
    NUMERIC_MEASURES,
    PHYSICAL_BOUNDS,
    STATION_COLUMN,
    TIMESTAMP_COLUMN,
    WIND_DIRECTION_COLUMN,
    WIND_DIRECTIONS,
)
from novaria.preparation.journal import log_issue, log_mask

FROZEN_RUN_COLUMNS = [STATION_COLUMN, "column", "value", "start", "end", "hours"]


# --- Mise au format ----------------------------------------------------------
def to_local_time(timestamps: pd.Series) -> pd.Series:
    """Convertit des horodatages en heure locale du dataset, sans fuseau.

    Les horodatages sans fuseau sont considérés comme déjà en heure locale.
    Ceux avec un fuseau (ex. ``2016-05-01T06:00:00Z``) sont convertis vers
    ``DATA_TIMEZONE``, puis le fuseau est retiré. Des fuseaux différents
    dans la même série sont acceptés ; un mélange avec et sans fuseau ne
    l'est pas.

    Args:
        timestamps: horodatages (chaînes ISO 8601 ou dates).

    Returns:
        Les horodatages en heure locale, sans fuseau, à l'heure près.

    Raises:
        ValueError: si un horodatage est illisible, ou si la série mélange
            des horodatages avec et sans fuseau.
    """
    if pd.api.types.is_datetime64_any_dtype(timestamps):
        parsed = timestamps
    else:
        try:
            parsed = pd.to_datetime(timestamps, format="ISO8601")
        except ValueError:
            # Fuseaux différents : on passe par UTC.
            parsed = pd.to_datetime(timestamps, format="ISO8601", utc=True)
    if parsed.dt.tz is not None:
        parsed = parsed.dt.tz_convert(DATA_TIMEZONE).dt.tz_localize(None)
    return parsed.dt.floor("h")


def standardize(
    df: pd.DataFrame, keep_columns: list[str] | None = None
) -> pd.DataFrame:
    """Met une table de mesures au format commun (sans état).

    Utilisée pour les CSV bruts (après renommage) comme pour les messages
    du contrat. Construit l'index ``timestamp`` en heure locale, garde
    ``station_id`` puis ``MEASURES`` (une mesure absente devient NaN), convertit
    les mesures numériques (valeur non convertible -> NaN), met la direction
    du vent en majuscules et trie par date puis station (tri stable).

    Args:
        df: DataFrame avec les noms du contrat et une colonne ``timestamp``.
        keep_columns: colonnes supplémentaires à conserver telles quelles,
            placées en fin de trame.

    Returns:
        La trame de mesures, nouvelle copie.
    """
    columns = [TIMESTAMP_COLUMN, STATION_COLUMN, *MEASURES, *(keep_columns or [])]
    out = df.reindex(columns=columns).copy()
    out[TIMESTAMP_COLUMN] = to_local_time(out[TIMESTAMP_COLUMN])
    out[STATION_COLUMN] = out[STATION_COLUMN].astype("string")
    for col in NUMERIC_MEASURES:
        out[col] = pd.to_numeric(out[col], errors="coerce").astype("float64")
    out[WIND_DIRECTION_COLUMN] = (
        out[WIND_DIRECTION_COLUMN].astype("string").str.strip().str.upper()
    )
    out = out.sort_values([TIMESTAMP_COLUMN, STATION_COLUMN], kind="stable")
    return out.set_index(TIMESTAMP_COLUMN)


def from_raw(raw: pd.DataFrame) -> pd.DataFrame:
    """Met des lignes de CSV brut au format commun.

    Construit l'horodatage à partir de year/month/day/hour et renomme les
    colonnes avec les noms du contrat. Aucune valeur n'est corrigée ici.

    Args:
        raw: DataFrame avec les colonnes d'origine du dataset.

    Returns:
        La trame de mesures (voir ``standardize``).
    """
    renamed = raw.rename(columns=COLUMN_MAPPING)
    renamed[TIMESTAMP_COLUMN] = pd.to_datetime(raw[DATETIME_COLUMNS])
    return standardize(renamed)


# --- Détection ---------------------------------------------------------------
def duplicated_mask(df: pd.DataFrame) -> np.ndarray:
    """Repère les doublons (station, horodatage), en gardant la première occurrence.

    Args:
        df: trame de mesures.

    Returns:
        Un masque booléen, True pour les lignes en double.
    """
    keys = pd.MultiIndex.from_arrays([df[STATION_COLUMN], df.index])
    return keys.duplicated(keep="first")


def inconsistency_masks(df: pd.DataFrame) -> dict[str, dict[str, pd.Series]]:
    """Repère les valeurs impossibles, par type d'erreur et par variable (sans état).

    Les types sont exclusifs et évalués dans cet ordre : ``valeur_negative``,
    ``hors_bornes``, ``pm25_superieur_pm10`` (qui touche pm25 et pm10), puis
    ``direction_inconnue`` (direction du vent hors des 16 points cardinaux).

    Args:
        df: trame de mesures.

    Returns:
        ``{type_erreur: {variable: masque booléen}}``.
    """
    flagged = pd.DataFrame(False, index=df.index, columns=NUMERIC_MEASURES)

    negative = {col: (df[col] < 0).to_numpy() for col in NON_NEGATIVE_MEASURES}
    for col, mask in negative.items():
        flagged[col] |= mask

    out_of_bounds = {}
    for col, (low, high) in PHYSICAL_BOUNDS.items():
        mask = ((df[col] < low) | (df[col] > high)).to_numpy() & ~flagged[
            col
        ].to_numpy()
        out_of_bounds[col] = mask
        flagged[col] |= mask

    pm_mask = (
        (df["pm25"] > df["pm10"]).to_numpy()
        & ~flagged["pm25"].to_numpy()
        & ~flagged["pm10"].to_numpy()
    )
    wind = df[WIND_DIRECTION_COLUMN]
    unknown_wind = (wind.notna() & ~wind.isin(WIND_DIRECTIONS)).to_numpy(dtype=bool)
    return {
        "valeur_negative": negative,
        "hors_bornes": out_of_bounds,
        "pm25_superieur_pm10": {"pm25": pm_mask, "pm10": pm_mask},
        "direction_inconnue": {WIND_DIRECTION_COLUMN: unknown_wind},
    }


def frozen_sensor_runs(
    df: pd.DataFrame,
    min_hours: int = FROZEN_SENSOR_MIN_HOURS,
    measures: list[str] | None = None,
) -> pd.DataFrame:
    """Repère les capteurs figés : même valeur répétée au moins ``min_hours`` heures.

    Les NaN ne forment pas de série. Les heures doivent être consécutives :
    une heure absente coupe la série. Ce contrôle a besoin de l'historique
    de chaque station ; en streaming, il demandera un état par station.

    Args:
        df: trame de mesures, triée par date.
        min_hours: longueur minimale d'une série pour être signalée.
        measures: mesures à contrôler (défaut : ``FROZEN_SENSOR_MEASURES``).

    Returns:
        Une ligne par série : station_id, column, value, start, end, hours.
    """
    measures = FROZEN_SENSOR_MEASURES if measures is None else measures
    runs = []
    for station, group in df.groupby(STATION_COLUMN):
        times = group.index.to_series()
        new_hour = times.diff() != pd.Timedelta(hours=1)
        for col in measures:
            values = group[col]
            run_id = ((values != values.shift()) | new_hour | values.isna()).cumsum()
            sizes = values.groupby(run_id).size()
            firsts = values.groupby(run_id).first()
            keep = (sizes >= min_hours) & firsts.notna()
            if not keep.any():
                continue
            starts = times.groupby(run_id).min()
            ends = times.groupby(run_id).max()
            for rid in sizes.index[keep]:
                runs.append(
                    [station, col, firsts[rid], starts[rid], ends[rid], int(sizes[rid])]
                )
    return pd.DataFrame(runs, columns=FROZEN_RUN_COLUMNS)


# --- Correction --------------------------------------------------------------
def log_missing_values(df: pd.DataFrame) -> None:
    """Journalise chaque valeur manquante et chaque heure manquante (signalées).

    Args:
        df: trame de mesures.
    """
    for col in MEASURES:
        log_mask(df, df[col].isna(), col, "valeur_manquante", "signale")
    for station, group in df.groupby(STATION_COLUMN):
        full_range = pd.date_range(group.index.min(), group.index.max(), freq="h")
        for timestamp in full_range.difference(group.index.unique()):
            log_issue(station, timestamp, "-", "-", "heure_manquante", "signale")


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Supprime les doublons (station, horodatage) en gardant la première occurrence.

    C'est la SEULE suppression de lignes autorisée dans la préparation.
    Chaque ligne supprimée est journalisée.

    Args:
        df: trame de mesures.

    Returns:
        La trame sans doublon, nouvelle copie.
    """
    duplicated = duplicated_mask(df)
    for timestamp, station in zip(
        df.index[duplicated], df.loc[duplicated, STATION_COLUMN], strict=True
    ):
        log_issue(station, timestamp, "-", "-", "doublon", "supprime")
    return df.loc[~duplicated].copy()


def fix_inconsistencies(df: pd.DataFrame) -> pd.DataFrame:
    """Met à NaN les valeurs impossibles, qui seront imputées par le préprocesseur (sans état).

    Les pics extrêmes mais dans les bornes physiques ne sont pas modifiés.
    Chaque valeur corrigée est journalisée avec l'action ``impute``.

    Args:
        df: trame de mesures.

    Returns:
        La trame corrigée, nouvelle copie ; l'entrée n'est pas modifiée.
    """
    out = df.copy()
    for error, by_col in inconsistency_masks(df).items():
        for col, mask in by_col.items():
            log_mask(df, mask, col, error, "impute")
            out.loc[mask, col] = np.nan
    return out


def log_frozen_sensors(df: pd.DataFrame) -> pd.DataFrame:
    """Journalise les séries de capteur figé, sans modifier les données.

    Args:
        df: trame de mesures, triée par date.

    Returns:
        Les séries détectées (voir ``frozen_sensor_runs``).
    """
    runs = frozen_sensor_runs(df)
    for run in runs.itertuples(index=False):
        log_issue(
            run.station_id,
            f"{run.start} -> {run.end} ({run.hours} h)",
            run.column,
            run.value,
            "capteur_fige",
            "signale",
        )
    return runs


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Applique toutes les vérifications et corrections batch, avec journalisation.

    Ordre : valeurs et heures manquantes (signalées), doublons (supprimés),
    valeurs impossibles (mises à NaN), capteur figé (signalé).

    Args:
        df: trame de mesures (voir ``from_raw``).

    Returns:
        La trame nettoyée, avec des NaN que le préprocesseur imputera.
    """
    log_missing_values(df)
    df = remove_duplicates(df)
    df = fix_inconsistencies(df)
    log_frozen_sensors(df)
    return df
