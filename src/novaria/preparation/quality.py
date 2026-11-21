"""Rapport qualité des mesures (batch) : manquants, doublons, incohérences, capteur figé."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from novaria.common.config import (
    FROZEN_SENSOR_MIN_HOURS,
    NUMERIC_MEASURES,
    STATION_COLUMN,
)
from novaria.preparation.cleaning import (
    duplicated_mask,
    frozen_sensor_runs,
    inconsistency_masks,
)


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
        df: trame de mesures.

    Returns:
        ``{station: nombre d'heures manquantes}``.
    """
    return {
        str(station): count_missing_hours(pd.DatetimeIndex(group.index))
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


def quality_report(df: pd.DataFrame, verbose: bool = True) -> dict[str, Any]:
    """Produit (et affiche) le rapport qualité des données avant nettoyage.

    Le rapport contient : stations, période, valeurs manquantes, doublons
    (station, horodatage), heures manquantes par station, statistiques
    descriptives (min, médiane, p99, max), valeurs impossibles par type et
    séries de capteur figé.

    Args:
        df: trame de mesures (voir ``cleaning.from_raw``).
        verbose: affiche le rapport si True.

    Returns:
        Le rapport sous forme de dictionnaire.
    """
    stats = df[NUMERIC_MEASURES].quantile([0.0, 0.5, 0.99, 1.0]).T
    stats.columns = ["min", "median", "p99", "max"]
    frozen = frozen_sensor_runs(df)

    report: dict[str, Any] = {
        "n_rows": len(df),
        "stations": sorted(df[STATION_COLUMN].dropna().unique().tolist()),
        "start": df.index.min(),
        "end": df.index.max(),
        "missing": missing_summary(df),
        "duplicates": int(duplicated_mask(df).sum()),
        "missing_hours": missing_hours_by_station(df),
        "stats": {col: stats.loc[col].to_dict() for col in stats.index},
        "inconsistencies": {
            error: {col: int(np.sum(mask)) for col, mask in by_col.items()}
            for error, by_col in inconsistency_masks(df).items()
        },
        "frozen_sensor_runs": {
            "min_hours": FROZEN_SENSOR_MIN_HOURS,
            "by_column": frozen.groupby("column").size().to_dict(),
        },
    }

    if verbose:
        print_report(report)
    return report


def print_report(report: dict[str, Any]) -> None:
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
    print(f"Lignes : {report['n_rows']}")
    print(f"Stations ({len(report['stations'])}) : {', '.join(report['stations'])}")
    print(f"Période : {report['start']} -> {report['end']}")
    print(f"Doublons (station, horodatage) : {report['duplicates']}")
    print(f"Heures manquantes : {sum(report['missing_hours'].values())}")

    print("\nValeurs manquantes :")
    for col, m in report["missing"].items():
        print(f"  {col:<10} {m['count']:>7}  ({m['pct']:.2f} %)")

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
