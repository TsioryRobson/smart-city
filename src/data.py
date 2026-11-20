"""Chargement, contrôle qualité et découpage chronologique des données.

Règle du projet : aucune valeur extrême ou négative n'est supprimée ni modifiée,
puisque l'objectif est justement de détecter les anomalies. On se contente de
les signaler. La seule suppression autorisée concerne les doublons d'horodatage.

Exécution : ``python -m src.data``
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config import (
    COLUMN_MAPPING,
    DATA_PATH,
    DATETIME_COLUMNS,
    FEATURES,
    REPORTS_DIR,
    TRAIN_RATIO,
    VAL_RATIO,
)


def load_raw(path: str | Path = DATA_PATH) -> pd.DataFrame:
    """Lit le CSV brut.

    Les valeurs manquantes codées ``NA`` sont interprétées comme NaN
    (comportement par défaut de pandas).

    Args:
        path: chemin du fichier CSV.

    Returns:
        Le DataFrame brut, sans aucune transformation.
    """
    return pd.read_csv(path, na_values=["NA"])


def build_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    """Construit un ``DatetimeIndex`` nommé ``timestamp`` et trie par date.

    L'horodatage est construit à partir des colonnes year/month/day/hour.
    Le tri est stable, donc l'ordre d'origine des éventuels doublons est conservé.

    Args:
        df: DataFrame contenant les colonnes year, month, day, hour.

    Returns:
        Une copie du DataFrame indexée par ``timestamp`` et triée.
    """
    out = df.copy()
    timestamp = pd.to_datetime(out[DATETIME_COLUMNS])
    out.index = pd.DatetimeIndex(timestamp, name="timestamp")
    return out.sort_index(kind="stable")


def select_features(df: pd.DataFrame) -> pd.DataFrame:
    """Renomme les colonnes en snake_case et ne garde que ``FEATURES``.

    Chaque variable est convertie en numérique ; une valeur non convertible
    devient NaN (``errors="coerce"``). Aucune valeur n'est modifiée par ailleurs.

    Args:
        df: DataFrame avec les noms de colonnes d'origine.

    Returns:
        Un DataFrame dont les colonnes sont exactement ``FEATURES``, dans cet ordre.
    """
    out = df.rename(columns=COLUMN_MAPPING)[FEATURES].copy()
    for col in FEATURES:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def count_missing_hours(index: pd.DatetimeIndex) -> int:
    """Compte les heures absentes entre le premier et le dernier horodatage.

    Args:
        index: index temporel (horaire attendu).

    Returns:
        Le nombre d'heures manquantes dans la série.
    """
    if len(index) == 0:
        return 0
    full_range = pd.date_range(index.min(), index.max(), freq="h")
    return len(full_range.difference(index.unique()))


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
    """Produit (et affiche) le rapport qualité d'un DataFrame indexé par date.

    Le rapport contient : types des colonnes, valeurs manquantes, doublons
    d'horodatage, heures manquantes, statistiques descriptives (min, médiane,
    p99, max) et nombre de valeurs négatives (signalement uniquement).

    Args:
        df: DataFrame indexé par ``timestamp``.
        verbose: affiche le rapport si True.

    Returns:
        Le rapport sous forme de dictionnaire.
    """
    numeric = df.select_dtypes(include="number")
    stats = numeric.quantile([0.0, 0.5, 0.99, 1.0]).T
    stats.columns = ["min", "median", "p99", "max"]

    report: dict[str, Any] = {
        "n_rows": len(df),
        "n_columns": df.shape[1],
        "start": df.index.min(),
        "end": df.index.max(),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "missing": missing_summary(df),
        "duplicate_timestamps": int(df.index.duplicated().sum()),
        "missing_hours": count_missing_hours(df.index),
        "stats": {col: stats.loc[col].to_dict() for col in stats.index},
        "negative_values": {col: int((numeric[col] < 0).sum()) for col in numeric},
    }

    if verbose:
        _print_report(report)
    return report


def _print_report(report: dict[str, Any]) -> None:
    """Affiche un rapport qualité de façon lisible."""
    print("=" * 60)
    print("RAPPORT QUALITÉ")
    print("=" * 60)
    print(f"Lignes : {report['n_rows']}  |  Colonnes : {report['n_columns']}")
    print(f"Période : {report['start']} -> {report['end']}")
    print(f"Doublons d'horodatage : {report['duplicate_timestamps']}")
    print(f"Heures manquantes dans la série : {report['missing_hours']}")

    print("\nTypes des colonnes :")
    for col, dtype in report["dtypes"].items():
        print(f"  {col:<6} {dtype}")

    print("\nValeurs manquantes :")
    for col, m in report["missing"].items():
        print(f"  {col:<6} {m['count']:>6}  ({m['pct']:.2f} %)")

    print("\nStatistiques descriptives :")
    print(f"  {'var':<6} {'min':>10} {'médiane':>10} {'p99':>10} {'max':>10}")
    for col, s in report["stats"].items():
        print(
            f"  {col:<6} {s['min']:>10.1f} {s['median']:>10.1f} "
            f"{s['p99']:>10.1f} {s['max']:>10.1f}"
        )

    print("\nValeurs négatives (signalement uniquement, rien n'est supprimé) :")
    for col, n in report["negative_values"].items():
        print(f"  {col:<6} {n}")
    print()


def remove_duplicate_timestamps(df: pd.DataFrame) -> pd.DataFrame:
    """Supprime les doublons d'horodatage en gardant la première occurrence.

    C'est la SEULE suppression de lignes autorisée dans la préparation.

    Args:
        df: DataFrame indexé par ``timestamp``.

    Returns:
        Le DataFrame sans doublon d'index.
    """
    duplicated = df.index.duplicated(keep="first")
    n_removed = int(duplicated.sum())
    print(f"Doublons d'horodatage retirés : {n_removed} ligne(s)")
    return df.loc[~duplicated].copy()


def split_chrono(
    df: pd.DataFrame,
    train_ratio: float = TRAIN_RATIO,
    val_ratio: float = VAL_RATIO,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Découpe le jeu en train / val / test par position, sans mélange.

    Le DataFrame doit être trié par date et sans doublon d'horodatage.
    Le jeu de test prend le reste après train et val.

    Args:
        df: DataFrame indexé par ``timestamp``, trié.
        train_ratio: proportion du jeu d'entraînement.
        val_ratio: proportion du jeu de validation.

    Returns:
        Le triplet ``(train, val, test)``.
    """
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)

    train = df.iloc[:train_end]
    val = df.iloc[train_end:val_end]
    test = df.iloc[val_end:]

    assert train.index.max() < val.index.min(), "Chevauchement train / val"
    assert val.index.max() < test.index.min(), "Chevauchement val / test"
    return train, val, test


def load_and_split(
    path: str | Path = DATA_PATH,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Enchaîne chargement, index, sélection, dédoublonnage et découpage.

    Args:
        path: chemin du fichier CSV.

    Returns:
        Le triplet ``(train, val, test)``.
    """
    df = load_raw(path)
    df = build_datetime_index(df)
    df = select_features(df)
    df = remove_duplicate_timestamps(df)
    return split_chrono(df)


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

    Args:
        name: nom du jeu (train, val, test).
        df: DataFrame du jeu.

    Returns:
        Le résumé sous forme de dictionnaire.
    """
    summary = {
        "shape": list(df.shape),
        "start": df.index.min(),
        "end": df.index.max(),
        "missing": missing_summary(df),
    }
    print(f"--- {name.upper()} ---")
    print(f"Dimensions : {df.shape}")
    print(f"Période    : {summary['start']} -> {summary['end']}")
    for col, m in summary["missing"].items():
        print(f"  {col:<6} {m['count']:>6}  ({m['pct']:.2f} %)")
    print()
    return summary


def main() -> None:
    """Affiche le rapport qualité, résume les splits et sauvegarde le rapport JSON."""
    df = build_datetime_index(load_raw(DATA_PATH))
    df = select_features(df)

    report = quality_report(df)

    df = remove_duplicate_timestamps(df)
    train, val, test = split_chrono(df)
    print()
    report["splits"] = {
        name: describe_split(name, part)
        for name, part in (("train", train), ("val", val), ("test", test))
    }

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = REPORTS_DIR / "data_quality.json"
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(to_serializable(report), f, indent=2, ensure_ascii=False)
    print(f"Rapport qualité sauvegardé : {output_path}")


if __name__ == "__main__":
    main()
