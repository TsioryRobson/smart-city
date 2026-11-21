"""Chargement, contrôle qualité, correction et découpage chronologique des données.

Règles du projet (voir CLAUDE.md) :
- les pics extrêmes mais physiquement possibles ne sont jamais modifiés,
  puisque ce sont les anomalies à détecter ;
- les valeurs impossibles (concentration négative, hors bornes physiques,
  PM2.5 > PM10) sont mises à NaN, puis imputées par le préprocesseur ;
- un capteur figé est seulement signalé ;
- la seule suppression de lignes autorisée concerne les doublons
  (station, horodatage) ;
- chaque problème détecté est écrit dans le journal ``logs/data_checks.log``.

Exécution : ``python -m src.data``
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config import (
    COLUMN_MAPPING,
    DATA_CHECKS_LOG_PATH,
    DATETIME_COLUMNS,
    DEMO_START,
    FEATURES,
    FROZEN_SENSOR_MIN_HOURS,
    NON_NEGATIVE_FEATURES,
    PHYSICAL_BOUNDS,
    RAW_DATA_DIR,
    RAW_FILE_PATTERN,
    REPORTS_DIR,
    STATION_COLUMN,
    TEST_START,
    TRAIN_START,
    VAL_START,
)

logger = logging.getLogger("smart_city.data_checks")


# --- Journal de vérification -------------------------------------------------
def setup_data_checks_log(path: str | Path = DATA_CHECKS_LOG_PATH) -> Path:
    """Configure le journal de vérification (fichier ``.log`` texte, écrasé).

    Args:
        path: chemin du fichier de log.

    Returns:
        Le chemin du fichier de log.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()
    handler = logging.FileHandler(path, mode="w", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return path


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
        timestamp: horodatage concerné.
        column: variable concernée (``-`` pour une ligne entière).
        value: valeur d'origine.
        error: type d'erreur (``valeur_manquante``, ``doublon``...).
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


def _log_mask(
    df: pd.DataFrame, mask: pd.Series, column: str, error: str, action: str
) -> None:
    """Journalise chaque ligne de ``df`` sélectionnée par ``mask`` pour ``column``.

    Écrit une entrée par ligne sélectionnée dans le journal de vérification,
    avec la station, l'horodatage et la valeur d'origine de ``column``.

    Args:
        df: DataFrame indexé par ``timestamp`` avec une colonne ``station``.
        mask: masque booléen aligné sur ``df``, True pour les lignes à journaliser.
        column: variable concernée.
        error: type d'erreur (``valeur_manquante``, ``hors_bornes``...).
        action: action appliquée (``signale``, ``supprime`` ou ``impute``).
    """
    rows = df.loc[np.asarray(mask), [STATION_COLUMN, column]]
    for timestamp, station, value in zip(
        rows.index, rows[STATION_COLUMN], rows[column], strict=True
    ):
        log_issue(station, timestamp, column, value, error, action)


# --- Chargement --------------------------------------------------------------
def load_raw(path: str | Path) -> pd.DataFrame:
    """Lit un CSV brut de station.

    Les valeurs manquantes codées ``NA`` sont interprétées comme NaN
    (comportement par défaut de pandas).

    Args:
        path: chemin du fichier CSV.

    Returns:
        Le DataFrame brut, sans aucune transformation.
    """
    return pd.read_csv(path, na_values=["NA"])


def load_all_stations(raw_dir: str | Path = RAW_DATA_DIR) -> pd.DataFrame:
    """Lit et concatène les CSV de toutes les stations de ``raw_dir``.

    Args:
        raw_dir: dossier contenant un CSV par station.

    Returns:
        Le DataFrame brut de toutes les stations.

    Raises:
        FileNotFoundError: si aucun CSV ne correspond à ``RAW_FILE_PATTERN``.
    """
    paths = sorted(Path(raw_dir).glob(RAW_FILE_PATTERN))
    if not paths:
        raise FileNotFoundError(f"Aucun CSV '{RAW_FILE_PATTERN}' dans {raw_dir}")
    return pd.concat([load_raw(p) for p in paths], ignore_index=True)


def build_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    """Construit un ``DatetimeIndex`` nommé ``timestamp``, trié par date puis station.

    Le tri est stable, donc l'ordre d'origine des éventuels doublons est conservé.

    Args:
        df: DataFrame contenant les colonnes year, month, day, hour et station.

    Returns:
        Une copie du DataFrame indexée par ``timestamp`` et triée.
    """
    out = df.copy()
    out["timestamp"] = pd.to_datetime(out[DATETIME_COLUMNS])
    out = out.sort_values(["timestamp", STATION_COLUMN], kind="stable")
    return out.set_index("timestamp")


def select_features(df: pd.DataFrame) -> pd.DataFrame:
    """Renomme les colonnes en snake_case et ne garde que la station et ``FEATURES``.

    Chaque variable est convertie en numérique ; une valeur non convertible
    devient NaN (``errors="coerce"``). Aucune valeur n'est modifiée par ailleurs.

    Args:
        df: DataFrame avec les noms de colonnes d'origine.

    Returns:
        Un DataFrame dont les colonnes sont ``station`` puis ``FEATURES``.
    """
    out = df.rename(columns=COLUMN_MAPPING)[[STATION_COLUMN, *FEATURES]].copy()
    for col in FEATURES:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


# --- Contrôle qualité --------------------------------------------------------
def count_missing_hours(index: pd.DatetimeIndex) -> int:
    """Compte les heures absentes entre le premier et le dernier horodatage.

    Args:
        index: index temporel d'une seule station (horaire attendu).

    Returns:
        Le nombre d'heures manquantes dans la série.
    """
    if len(index) == 0:
        return 0
    full_range = pd.date_range(index.min(), index.max(), freq="h")
    return len(full_range.difference(index.unique()))


def missing_hours_by_station(df: pd.DataFrame) -> dict[str, int]:
    """Compte les heures manquantes de chaque station.

    Args:
        df: DataFrame indexé par ``timestamp`` avec une colonne ``station``.

    Returns:
        ``{station: nombre d'heures manquantes}``.
    """
    return {
        str(station): count_missing_hours(group.index)
        for station, group in df.groupby(STATION_COLUMN)
    }


def missing_summary(df: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Calcule le nombre et le pourcentage de valeurs manquantes par variable.

    Args:
        df: DataFrame à analyser.

    Returns:
        ``{variable: {"count": n, "pct": pourcentage}}``.
    """
    n_rows = len(df)
    counts = df.isna().sum()
    return {
        col: {
            "count": int(counts[col]),
            "pct": round(100 * counts[col] / n_rows, 2) if n_rows else 0.0,
        }
        for col in df.columns
    }


def duplicated_mask(df: pd.DataFrame) -> np.ndarray:
    """Repère les doublons (station, horodatage), en gardant la première occurrence.

    Args:
        df: DataFrame indexé par ``timestamp`` avec une colonne ``station``.

    Returns:
        Un masque booléen, True pour les lignes en double.
    """
    keys = pd.MultiIndex.from_arrays([df[STATION_COLUMN], df.index])
    return keys.duplicated(keep="first")


def inconsistency_masks(df: pd.DataFrame) -> dict[str, dict[str, pd.Series]]:
    """Repère les valeurs impossibles, par type d'erreur et par variable.

    Les types sont exclusifs et évalués dans cet ordre : ``valeur_negative``,
    puis ``hors_bornes``, puis ``pm25_superieur_pm10`` (qui touche pm25 et pm10).

    Args:
        df: DataFrame avec les colonnes ``FEATURES``.

    Returns:
        ``{type_erreur: {variable: masque booléen}}``.
    """
    flagged = pd.DataFrame(False, index=df.index, columns=FEATURES)

    negative = {col: df[col] < 0 for col in NON_NEGATIVE_FEATURES}
    for col, mask in negative.items():
        flagged[col] |= mask

    out_of_bounds = {}
    for col, (low, high) in PHYSICAL_BOUNDS.items():
        mask = ((df[col] < low) | (df[col] > high)) & ~flagged[col]
        out_of_bounds[col] = mask
        flagged[col] |= mask

    pm_mask = (df["pm25"] > df["pm10"]) & ~flagged["pm25"] & ~flagged["pm10"]
    return {
        "valeur_negative": negative,
        "hors_bornes": out_of_bounds,
        "pm25_superieur_pm10": {"pm25": pm_mask, "pm10": pm_mask},
    }


def frozen_sensor_runs(
    df: pd.DataFrame, min_hours: int = FROZEN_SENSOR_MIN_HOURS
) -> pd.DataFrame:
    """Repère les capteurs figés : même valeur répétée au moins ``min_hours`` heures.

    Les NaN ne forment pas de série. Les heures doivent être consécutives :
    une heure absente coupe la série.

    Args:
        df: DataFrame indexé par ``timestamp``, trié, avec une colonne ``station``.
        min_hours: longueur minimale d'une série pour être signalée.

    Returns:
        Une ligne par série : station, column, value, start, end, hours.
    """
    columns = ["station", "column", "value", "start", "end", "hours"]
    runs = []
    for station, group in df.groupby(STATION_COLUMN):
        times = group.index.to_series()
        new_hour = times.diff() != pd.Timedelta(hours=1)
        for col in FEATURES:
            values = group[col]
            new_run = (values != values.shift()) | new_hour | values.isna()
            run_id = new_run.cumsum()
            sizes = values.groupby(run_id).size()
            firsts = values.groupby(run_id).first()
            starts = times.groupby(run_id).min()
            ends = times.groupby(run_id).max()
            keep = (sizes >= min_hours) & firsts.notna()
            for rid in sizes.index[keep]:
                runs.append(
                    [station, col, firsts[rid], starts[rid], ends[rid], int(sizes[rid])]
                )
    return pd.DataFrame(runs, columns=columns)


def quality_report(df: pd.DataFrame, verbose: bool = True) -> dict[str, Any]:
    """Produit (et affiche) le rapport qualité des données brutes.

    Le rapport contient : types des colonnes, valeurs manquantes, doublons
    (station, horodatage), heures manquantes par station, statistiques
    descriptives (min, médiane, p99, max), valeurs impossibles par type et
    séries de capteur figé.

    Args:
        df: DataFrame indexé par ``timestamp`` avec ``station`` et ``FEATURES``.
        verbose: affiche le rapport si True.

    Returns:
        Le rapport sous forme de dictionnaire.
    """
    stats = df[FEATURES].quantile([0.0, 0.5, 0.99, 1.0]).T
    stats.columns = ["min", "median", "p99", "max"]
    masks = inconsistency_masks(df)
    frozen = frozen_sensor_runs(df)

    report: dict[str, Any] = {
        "n_rows": len(df),
        "n_columns": df.shape[1],
        "stations": sorted(df[STATION_COLUMN].unique().tolist()),
        "start": df.index.min(),
        "end": df.index.max(),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "missing": missing_summary(df),
        "duplicates": int(duplicated_mask(df).sum()),
        "missing_hours": missing_hours_by_station(df),
        "stats": {col: stats.loc[col].to_dict() for col in stats.index},
        "inconsistencies": {
            error: {col: int(m.sum()) for col, m in by_col.items()}
            for error, by_col in masks.items()
        },
        "frozen_sensor_runs": {
            "min_hours": FROZEN_SENSOR_MIN_HOURS,
            "by_column": frozen.groupby("column").size().to_dict(),
        },
    }

    if verbose:
        _print_report(report)
    return report


def _print_report(report: dict[str, Any]) -> None:
    """Affiche un rapport qualité de façon lisible dans la console.

    Affiche les dimensions, les stations, la période, les doublons, les heures
    manquantes, les valeurs manquantes, les statistiques descriptives, les
    valeurs impossibles et les séries de capteur figé.

    Args:
        report: rapport produit par ``quality_report``.
    """
    print("=" * 60)
    print("RAPPORT QUALITÉ")
    print("=" * 60)
    print(f"Lignes : {report['n_rows']}  |  Colonnes : {report['n_columns']}")
    print(f"Stations ({len(report['stations'])}) : {', '.join(report['stations'])}")
    print(f"Période : {report['start']} -> {report['end']}")
    print(f"Doublons (station, horodatage) : {report['duplicates']}")
    print(f"Heures manquantes : {sum(report['missing_hours'].values())}")

    print("\nValeurs manquantes :")
    for col, m in report["missing"].items():
        print(f"  {col:<8} {m['count']:>7}  ({m['pct']:.2f} %)")

    print("\nStatistiques descriptives :")
    print(f"  {'var':<6} {'min':>10} {'médiane':>10} {'p99':>10} {'max':>10}")
    for col, s in report["stats"].items():
        print(
            f"  {col:<6} {s['min']:>10.1f} {s['median']:>10.1f} "
            f"{s['p99']:>10.1f} {s['max']:>10.1f}"
        )

    print("\nValeurs impossibles (mises à NaN puis imputées) :")
    for error, by_col in report["inconsistencies"].items():
        counts = {col: n for col, n in by_col.items() if n}
        print(f"  {error:<20} {counts or 0}")

    frozen = report["frozen_sensor_runs"]
    print(f"\nCapteur figé >= {frozen['min_hours']} h (signalé seulement) :")
    print(f"  {frozen['by_column'] or 0}")
    print()


# --- Vérification et correction ----------------------------------------------
def log_missing_values(df: pd.DataFrame) -> None:
    """Journalise chaque valeur manquante et chaque heure manquante.

    Args:
        df: DataFrame indexé par ``timestamp`` avec ``station`` et ``FEATURES``.
    """
    for col in FEATURES:
        _log_mask(df, df[col].isna(), col, "valeur_manquante", "signale")
    for station, group in df.groupby(STATION_COLUMN):
        full_range = pd.date_range(group.index.min(), group.index.max(), freq="h")
        for timestamp in full_range.difference(group.index.unique()):
            log_issue(station, timestamp, "-", "-", "heure_manquante", "signale")


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Supprime les doublons (station, horodatage) en gardant la première occurrence.

    C'est la SEULE suppression de lignes autorisée dans la préparation.

    Args:
        df: DataFrame indexé par ``timestamp`` avec une colonne ``station``.

    Returns:
        Le DataFrame sans doublon.
    """
    duplicated = duplicated_mask(df)
    for timestamp, station in zip(
        df.index[duplicated], df.loc[duplicated, STATION_COLUMN], strict=True
    ):
        log_issue(station, timestamp, "-", "-", "doublon", "supprime")
    return df.loc[~duplicated].copy()


def fix_inconsistencies(df: pd.DataFrame) -> pd.DataFrame:
    """Met à NaN les valeurs impossibles, qui seront imputées par le préprocesseur.

    Les pics extrêmes mais dans les bornes physiques ne sont pas modifiés.

    Args:
        df: DataFrame avec ``station`` et ``FEATURES``.

    Returns:
        Une copie du DataFrame corrigée.
    """
    out = df.copy()
    for error, by_col in inconsistency_masks(df).items():
        for col, mask in by_col.items():
            _log_mask(df, mask, col, error, "impute")
            out.loc[np.asarray(mask), col] = np.nan
    return out


def log_frozen_sensors(df: pd.DataFrame) -> pd.DataFrame:
    """Journalise les séries de capteur figé, sans modifier les données.

    Args:
        df: DataFrame indexé par ``timestamp``, trié, avec ``station`` et ``FEATURES``.

    Returns:
        Les séries détectées (voir ``frozen_sensor_runs``).
    """
    runs = frozen_sensor_runs(df)
    for run in runs.itertuples(index=False):
        log_issue(
            run.station,
            f"{run.start} -> {run.end} ({run.hours} h)",
            run.column,
            run.value,
            "capteur_fige",
            "signale",
        )
    return runs


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Applique toutes les vérifications et corrections, avec journalisation.

    Ordre : valeurs et heures manquantes (signalées), doublons (supprimés),
    valeurs impossibles (mises à NaN), capteur figé (signalé).

    Args:
        df: DataFrame issu de ``select_features``.

    Returns:
        Le DataFrame nettoyé, avec des NaN que le préprocesseur imputera.
    """
    log_missing_values(df)
    df = remove_duplicates(df)
    df = fix_inconsistencies(df)
    log_frozen_sensors(df)
    return df


# --- Découpage ---------------------------------------------------------------
def split_chrono(
    df: pd.DataFrame,
    train_start: pd.Timestamp = TRAIN_START,
    val_start: pd.Timestamp = VAL_START,
    test_start: pd.Timestamp = TEST_START,
    test_end: pd.Timestamp = DEMO_START,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Découpe le jeu en train / val / test par dates, sans mélange.

    Chaque jeu couvre ``[début, fin[``. Les données avant ``train_start`` et à
    partir de ``test_end`` (période rejouée par le simulateur) sont écartées.

    Args:
        df: DataFrame indexé par ``timestamp``.
        train_start: début du train.
        val_start: début de la validation (fin du train).
        test_start: début du test (fin de la validation).
        test_end: fin du test, exclue.

    Returns:
        Le triplet ``(train, val, test)``.
    """
    assert train_start < val_start < test_start < test_end, "Bornes non ordonnées"
    index = df.index
    train = df[(index >= train_start) & (index < val_start)]
    val = df[(index >= val_start) & (index < test_start)]
    test = df[(index >= test_start) & (index < test_end)]

    assert train.index.max() < val.index.min(), "Chevauchement train / val"
    assert val.index.max() < test.index.min(), "Chevauchement val / test"
    assert test.index.max() < test_end, "Le test empiète sur la période de démo"
    return train, val, test


def load_and_split(
    raw_dir: str | Path = RAW_DATA_DIR,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Enchaîne chargement, index, sélection, nettoyage et découpage.

    Args:
        raw_dir: dossier des CSV bruts.

    Returns:
        Le triplet ``(train, val, test)``.
    """
    df = load_all_stations(raw_dir)
    df = build_datetime_index(df)
    df = select_features(df)
    df = clean(df)
    return split_chrono(df)


# --- Sorties -----------------------------------------------------------------
def to_serializable(obj: Any) -> Any:
    """Convertit récursivement types numpy / pandas en types JSON natifs.

    Args:
        obj: objet à convertir.

    Returns:
        Un objet sérialisable par ``json``.
    """
    if isinstance(obj, dict):
        return {str(k): to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_serializable(v) for v in obj]
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return None if np.isnan(obj) else float(obj)
    if isinstance(obj, float) and np.isnan(obj):
        return None
    return obj


def describe_split(name: str, df: pd.DataFrame) -> dict[str, Any]:
    """Affiche et retourne le résumé d'un jeu (dimensions, dates, manquants).

    Les manquants incluent les valeurs impossibles mises à NaN.

    Args:
        name: nom du jeu (train, val, test).
        df: DataFrame du jeu.

    Returns:
        Le résumé sous forme de dictionnaire.
    """
    summary = {
        "shape": list(df.shape),
        "n_stations": int(df[STATION_COLUMN].nunique()),
        "start": df.index.min(),
        "end": df.index.max(),
        "missing": missing_summary(df[FEATURES]),
    }
    print(f"--- {name.upper()} ---")
    print(f"Dimensions : {df.shape}  |  Stations : {summary['n_stations']}")
    print(f"Période    : {summary['start']} -> {summary['end']}")
    for col, m in summary["missing"].items():
        print(f"  {col:<6} {m['count']:>6}  ({m['pct']:.2f} %)")
    print()
    return summary


def main() -> None:
    """Rapport qualité, nettoyage journalisé, splits, et sauvegarde du rapport JSON."""
    log_path = setup_data_checks_log()

    df = select_features(build_datetime_index(load_all_stations()))
    report = quality_report(df)

    df = clean(df)
    train, val, test = split_chrono(df)
    report["splits"] = {
        name: describe_split(name, part)
        for name, part in (("train", train), ("val", val), ("test", test))
    }

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = REPORTS_DIR / "data_quality.json"
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(to_serializable(report), f, indent=2, ensure_ascii=False)
    print(f"Rapport qualité sauvegardé : {output_path}")
    print(f"Journal de vérification   : {log_path}")


if __name__ == "__main__":
    main()
