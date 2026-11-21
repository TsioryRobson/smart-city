"""Étape 1 du pipeline : ingestion des CSV bruts de toutes les stations.

Lit chaque CSV de station, vérifie ses colonnes, et écrit sans modifier aucune
valeur :
- ``measurements.parquet`` : toutes les stations concaténées, avec la colonne
  ``source_file`` (provenance de chaque ligne) ;
- ``manifest.json`` : empreinte SHA-256 de chaque CSV, nombre de lignes par
  station, et empreinte du jeu complet (``dataset_sha256``, pour MLflow).

Exécution :
``python -m novaria.pipeline.ingestion --input data/raw --output data/ingested``
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from novaria.common.config import (
    INGESTED_DATA_DIR,
    RAW_DATA_DIR,
    RAW_FILE_PATTERN,
    RAW_REQUIRED_COLUMNS,
)
from novaria.common.io import combined_sha256, sha256_file, write_json

MEASUREMENTS_FILE = "measurements.parquet"
MANIFEST_FILE = "manifest.json"
SOURCE_COLUMN = "source_file"


def list_station_files(
    input_dir: str | Path, pattern: str = RAW_FILE_PATTERN
) -> list[Path]:
    """Liste les CSV de station d'un dossier, triés par nom.

    Args:
        input_dir: dossier des CSV bruts.
        pattern: motif des noms de fichiers à retenir.

    Returns:
        Les chemins des CSV trouvés.

    Raises:
        FileNotFoundError: si aucun fichier ne correspond au motif.
    """
    paths = sorted(Path(input_dir).glob(pattern))
    if not paths:
        raise FileNotFoundError(f"Aucun CSV '{pattern}' dans {input_dir}")
    return paths


def read_station_csv(path: str | Path) -> pd.DataFrame:
    """Lit un CSV de station et vérifie qu'il contient les colonnes obligatoires.

    Les valeurs manquantes codées ``NA`` deviennent NaN. Aucune autre valeur
    n'est modifiée.

    Args:
        path: chemin du CSV.

    Returns:
        Le DataFrame brut, avec la colonne ``source_file`` ajoutée.

    Raises:
        ValueError: si des colonnes obligatoires manquent.
    """
    path = Path(path)
    df = pd.read_csv(path, na_values=["NA"])
    missing = [col for col in RAW_REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"{path.name} : colonnes manquantes {missing}")
    df[SOURCE_COLUMN] = path.name
    return df


def ingest(
    input_dir: str | Path, output_dir: str | Path, pattern: str = RAW_FILE_PATTERN
) -> dict[str, Any]:
    """Ingère tous les CSV de station et écrit le Parquet brut et son manifeste.

    Args:
        input_dir: dossier des CSV bruts.
        output_dir: dossier de sortie (créé si besoin).
        pattern: motif des CSV de station.

    Returns:
        Le manifeste écrit dans ``manifest.json``.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = list_station_files(input_dir, pattern)
    frames = [read_station_csv(p) for p in paths]
    measurements = pd.concat(frames, ignore_index=True)
    measurements.to_parquet(output_dir / MEASUREMENTS_FILE, index=False)

    files = [
        {
            "name": path.name,
            "sha256": sha256_file(path),
            "rows": len(frame),
            "stations": sorted(frame["station"].dropna().unique().tolist()),
        }
        for path, frame in zip(paths, frames, strict=True)
    ]
    manifest = {
        "ingested_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "source_dir": str(Path(input_dir).resolve()),
        "n_files": len(files),
        "n_rows": len(measurements),
        "dataset_sha256": combined_sha256(f["sha256"] for f in files),
        "output": MEASUREMENTS_FILE,
        "files": files,
    }
    write_json(manifest, output_dir / MANIFEST_FILE)
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Lit les arguments de la ligne de commande.

    Args:
        argv: arguments à analyser (défaut : ceux du processus).

    Returns:
        Les arguments : ``input``, ``output`` et ``pattern``.
    """
    parser = argparse.ArgumentParser(description="Ingestion des CSV bruts de station.")
    parser.add_argument(
        "--input", type=Path, default=RAW_DATA_DIR, help="dossier des CSV bruts"
    )
    parser.add_argument(
        "--output", type=Path, default=INGESTED_DATA_DIR, help="dossier de sortie"
    )
    parser.add_argument(
        "--pattern", default=RAW_FILE_PATTERN, help="motif des CSV de station"
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Point d'entrée CLI : ingère les CSV et affiche un résumé.

    Args:
        argv: arguments de la ligne de commande (défaut : ceux du processus).
    """
    args = parse_args(argv)
    manifest = ingest(args.input, args.output, args.pattern)
    print(f"{manifest['n_files']} fichiers, {manifest['n_rows']} lignes ingérées")
    print(f"Empreinte du jeu : {manifest['dataset_sha256']}")
    print(f"Sortie : {args.output / MEASUREMENTS_FILE}")


if __name__ == "__main__":
    main()
