"""Étape 2 du pipeline : nettoyage, variables dérivées et découpage chronologique.

Lit le Parquet produit par l'ingestion, applique le module de préparation
partagé (``novaria.preparation``) et écrit :
- ``train.parquet``, ``val.parquet``, ``test.parquet`` : mesures nettoyées
  (valeurs impossibles à NaN) et variables dérivées, NON normalisées : le
  préprocesseur est ajusté avec le modèle dans ``training.py`` ;
- ``metadata.json`` : bornes du split, variables, nombre de lignes, empreintes ;
- le rapport qualité JSON et le journal de vérification.

Exécution :
``python -m novaria.pipeline.preparation --input data/ingested --output data/prepared``
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from novaria.common.config import (
    DATA_CHECKS_LOG_PATH,
    DEMO_START,
    INGESTED_DATA_DIR,
    MEASURES,
    MODEL_FEATURES,
    PREPARED_DATA_DIR,
    REPORTS_DIR,
    STATION_COLUMN,
    TEST_START,
    TIMESTAMP_COLUMN,
    TRAIN_START,
    VAL_START,
)
from novaria.common.io import combined_sha256, read_json, sha256_file, write_json
from novaria.pipeline.ingestion import MANIFEST_FILE, MEASUREMENTS_FILE
from novaria.preparation.cleaning import clean, from_raw
from novaria.preparation.features import add_features
from novaria.preparation.journal import close_data_checks_log, setup_data_checks_log
from novaria.preparation.quality import missing_summary, quality_report
from novaria.preparation.split import split_chrono

SPLIT_NAMES = ("train", "val", "test")
METADATA_FILE = "metadata.json"


def split_summary(df: pd.DataFrame) -> dict[str, Any]:
    """Résume un jeu découpé : dimensions, stations, période et manquants.

    Les manquants incluent les valeurs impossibles mises à NaN par le nettoyage.

    Args:
        df: jeu préparé indexé par ``timestamp``.

    Returns:
        Le résumé sous forme de dictionnaire.
    """
    return {
        "rows": len(df),
        "n_stations": int(df[STATION_COLUMN].nunique()),
        "start": df.index.min(),
        "end": df.index.max(),
        "missing": missing_summary(df[MODEL_FEATURES]),
    }


def write_split(df: pd.DataFrame, path: Path) -> str:
    """Écrit un jeu préparé en Parquet, l'horodatage redevenant une colonne.

    Args:
        df: jeu préparé indexé par ``timestamp``.
        path: chemin du fichier Parquet.

    Returns:
        L'empreinte SHA-256 du fichier écrit.
    """
    df.reset_index().to_parquet(path, index=False)
    return sha256_file(path)


def prepare(
    input_dir: str | Path,
    output_dir: str | Path,
    report_path: str | Path,
    log_path: str | Path,
    bounds: dict[str, pd.Timestamp],
    verbose: bool = True,
) -> dict[str, Any]:
    """Prépare les données ingérées et écrit les trois jeux et leurs métadonnées.

    Args:
        input_dir: dossier de sortie de l'ingestion (Parquet + manifeste).
        output_dir: dossier de sortie (créé si besoin).
        report_path: chemin du rapport qualité JSON.
        log_path: chemin du journal de vérification.
        bounds: bornes du split (``train_start``, ``val_start``, ``test_start``,
            ``test_end``), voir ``split_chrono``.
        verbose: affiche le rapport qualité et le résumé des jeux si True.

    Returns:
        Les métadonnées écrites dans ``metadata.json``.
    """
    input_dir, output_dir = Path(input_dir), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = input_dir / MANIFEST_FILE
    manifest = read_json(manifest_path) if manifest_path.exists() else {}

    setup_data_checks_log(log_path)
    try:
        measurements = from_raw(pd.read_parquet(input_dir / MEASUREMENTS_FILE))
        report = quality_report(measurements, verbose=verbose)
        prepared = add_features(clean(measurements))
    finally:
        close_data_checks_log()

    splits = dict(zip(SPLIT_NAMES, split_chrono(prepared, **bounds), strict=True))
    hashes = {
        name: write_split(df, output_dir / f"{name}.parquet")
        for name, df in splits.items()
    }
    summaries = {name: split_summary(df) for name, df in splits.items()}

    report["splits"] = summaries
    write_json(report, report_path)

    metadata = {
        "prepared_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "source_dataset_sha256": manifest.get("dataset_sha256"),
        "dataset_sha256": combined_sha256(hashes.values()),
        "split_bounds": bounds,
        "index": TIMESTAMP_COLUMN,
        "measures": MEASURES,
        "model_features": MODEL_FEATURES,
        "normalized": False,
        "files": {
            name: {"path": f"{name}.parquet", "sha256": hashes[name], **summaries[name]}
            for name in SPLIT_NAMES
        },
    }
    write_json(metadata, output_dir / METADATA_FILE)
    if verbose:
        print_splits(summaries)
    return metadata


def publish_prepared(
    output_dir: str | Path, metadata: dict[str, Any], bucket: str, prefix: str
) -> str:
    """Publie les jeux préparés dans MinIO, sous un dossier nommé par leur empreinte.

    Le même contenu va toujours au même endroit : un jeu publié est versionné
    par son empreinte (``dataset_sha256``) et le republier ne change rien.

    Args:
        output_dir: dossier de sortie de la préparation (Parquet + métadonnées).
        metadata: métadonnées renvoyées par ``prepare``.
        bucket: bucket des données (ex. ``novaria-data``).
        prefix: préfixe des jeux préparés (ex. ``datasets/prepared``).

    Returns:
        L'adresse du jeu publié (``minio://<bucket>/<préfixe>/<empreinte>/``).
    """
    from novaria.common.storage import client_from_env, upload_directory

    target = f"{prefix.rstrip('/')}/{metadata['dataset_sha256'][:16]}"
    upload_directory(client_from_env(), output_dir, bucket, target)
    return f"minio://{bucket}/{target}/"


def print_splits(summaries: dict[str, dict[str, Any]]) -> None:
    """Affiche le résumé de chaque jeu (lignes, stations, période, manquants).

    Args:
        summaries: résumés produits par ``split_summary``, par nom de jeu.
    """
    for name, s in summaries.items():
        print(f"--- {name.upper()} ---")
        print(f"Lignes : {s['rows']}  |  Stations : {s['n_stations']}")
        print(f"Période : {s['start']} -> {s['end']}")
        missing = {col: m["count"] for col, m in s["missing"].items() if m["count"]}
        print(f"NaN à imputer : {missing or 0}\n")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Lit les arguments de la ligne de commande.

    Args:
        argv: arguments à analyser (défaut : ceux du processus).

    Returns:
        Les arguments : dossiers d'entrée et de sortie, chemins du rapport et
        du journal, bornes du split et destination MinIO éventuelle.
    """
    parser = argparse.ArgumentParser(description="Préparation des données ingérées.")
    parser.add_argument(
        "--input", type=Path, default=INGESTED_DATA_DIR, help="sortie de l'ingestion"
    )
    parser.add_argument(
        "--output", type=Path, default=PREPARED_DATA_DIR, help="dossier de sortie"
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=REPORTS_DIR / "data_quality.json",
        help="rapport qualité JSON",
    )
    parser.add_argument(
        "--log", type=Path, default=DATA_CHECKS_LOG_PATH, help="journal de vérification"
    )
    parser.add_argument("--train-start", type=pd.Timestamp, default=TRAIN_START)
    parser.add_argument("--val-start", type=pd.Timestamp, default=VAL_START)
    parser.add_argument("--test-start", type=pd.Timestamp, default=TEST_START)
    parser.add_argument(
        "--test-end", type=pd.Timestamp, default=DEMO_START, help="exclue"
    )
    parser.add_argument(
        "--publish-bucket",
        default=None,
        help="publier aussi les jeux préparés dans ce bucket MinIO",
    )
    parser.add_argument(
        "--publish-prefix",
        default="datasets/prepared",
        help="préfixe des jeux publiés (suivi de leur empreinte)",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Point d'entrée CLI : prépare les données et affiche où sont les sorties.

    Args:
        argv: arguments de la ligne de commande (défaut : ceux du processus).
    """
    args = parse_args(argv)
    bounds = {
        "train_start": args.train_start,
        "val_start": args.val_start,
        "test_start": args.test_start,
        "test_end": args.test_end,
    }
    metadata = prepare(args.input, args.output, args.report, args.log, bounds)
    print(
        f"Jeux préparés : {args.output}  (empreinte {metadata['dataset_sha256'][:12]}…)"
    )
    print(f"Rapport qualité : {args.report}")
    print(f"Journal de vérification : {args.log}")
    if args.publish_bucket:
        location = publish_prepared(
            args.output, metadata, args.publish_bucket, args.publish_prefix
        )
        print(f"Jeux publiés : {location}")


if __name__ == "__main__":
    main()
