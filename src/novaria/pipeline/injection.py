"""Injection d'anomalies artificielles : construit les jeux contrôlés figés.

Lit un jeu préparé (le test par défaut), y injecte des anomalies identifiées
(``pic``, ``valeur_impossible``, ``capteur_fige``) pour chaque graine de la
configuration, et écrit :
- ``<jeu>_seed<graine>.parquet`` : données contrôlées, avec la colonne
  ``injection`` (``None`` pour une ligne de référence, sinon le type) ;
- ``manifest.json`` : graines, nombre d'anomalies par type, empreintes.

Les fichiers sont figés : les 3 algorithmes sont évalués sur exactement les
mêmes données, et l'évaluation peut être rejouée à l'identique. Le train
n'est jamais modifié (CLAUDE.md, règle 4).

Exécution :
``python -m novaria.pipeline.injection --input data/prepared --output data/controlled``
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from novaria.common.config import (
    CONTROLLED_DATA_DIR,
    EVALUATION_CONFIG_PATH,
    PREPARED_DATA_DIR,
    TIMESTAMP_COLUMN,
)
from novaria.common.injection import (
    INJECTION_COLUMN,
    INJECTION_TYPES,
    build_controlled_set,
    injection_config_from_dict,
)
from novaria.common.io import read_json, sha256_file, write_json

MANIFEST_FILE = "manifest.json"
ALLOWED_SPLITS = ("val", "test")


def controlled_file_name(split: str, seed: int) -> str:
    """Donne le nom du fichier contrôlé d'un jeu et d'une graine.

    Args:
        split: jeu de départ (``val`` ou ``test``).
        seed: graine de l'injection.

    Returns:
        Le nom du fichier Parquet, par exemple ``test_seed0.parquet``.
    """
    return f"{split}_seed{seed}.parquet"


def read_controlled(path: str | Path) -> tuple[pd.DataFrame, pd.Series]:
    """Lit un jeu contrôlé et sépare les données des étiquettes d'injection.

    Args:
        path: chemin du fichier Parquet contrôlé.

    Returns:
        Le couple ``(données indexées par timestamp, étiquettes)``. Les
        étiquettes valent ``None`` pour une ligne de référence.
    """
    df = pd.read_parquet(path).set_index(TIMESTAMP_COLUMN)
    labels = df.pop(INJECTION_COLUMN).astype("object")
    labels = labels.where(labels.notna(), None)
    return df, labels


def write_controlled_sets(
    input_dir: str | Path,
    output_dir: str | Path,
    config_path: str | Path,
    split: str = "test",
) -> dict[str, Any]:
    """Écrit un jeu contrôlé par graine et le manifeste correspondant.

    Args:
        input_dir: dossier de sortie de la préparation.
        output_dir: dossier des jeux contrôlés (créé si besoin).
        config_path: configuration d'évaluation YAML (section ``injection``
            et liste ``injection_seeds``).
        split: jeu préparé de départ ; jamais le train.

    Returns:
        Le manifeste écrit dans ``manifest.json``.

    Raises:
        ValueError: si ``split`` n'est pas ``val`` ou ``test``.
    """
    if split not in ALLOWED_SPLITS:
        raise ValueError(
            f"Injection interdite sur {split!r} (autorisé : {ALLOWED_SPLITS})"
        )
    input_dir, output_dir = Path(input_dir), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with Path(config_path).open(encoding="utf-8") as f:
        config = yaml.safe_load(f)
    injection = injection_config_from_dict(config["injection"])
    source = pd.read_parquet(input_dir / f"{split}.parquet").set_index(TIMESTAMP_COLUMN)
    metadata = read_json(input_dir / "metadata.json")

    files = []
    for seed in config["injection_seeds"]:
        controlled, labels = build_controlled_set(source, injection, seed)
        path = output_dir / controlled_file_name(split, seed)
        controlled.assign(
            **{INJECTION_COLUMN: labels.to_numpy()}
        ).reset_index().to_parquet(path, index=False)
        files.append(
            {
                "path": path.name,
                "seed": seed,
                "rows": len(controlled),
                "injected": {k: int((labels == k).sum()) for k in INJECTION_TYPES},
                "sha256": sha256_file(path),
            }
        )

    manifest = {
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "split": split,
        "source_dataset_sha256": metadata["dataset_sha256"],
        "injection": config["injection"],
        "files": files,
    }
    write_json(manifest, output_dir / MANIFEST_FILE)
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Lit les arguments de la ligne de commande.

    Args:
        argv: arguments à analyser (défaut : ceux du processus).

    Returns:
        Les arguments : ``input``, ``output``, ``config`` et ``split``.
    """
    parser = argparse.ArgumentParser(description="Injection d'anomalies artificielles.")
    parser.add_argument(
        "--input", type=Path, default=PREPARED_DATA_DIR, help="sortie de la préparation"
    )
    parser.add_argument(
        "--output", type=Path, default=CONTROLLED_DATA_DIR, help="jeux contrôlés"
    )
    parser.add_argument(
        "--config", type=Path, default=EVALUATION_CONFIG_PATH, help="configuration YAML"
    )
    parser.add_argument(
        "--split", choices=ALLOWED_SPLITS, default="test", help="jeu de départ"
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Point d'entrée CLI : écrit les jeux contrôlés et affiche leur contenu.

    Args:
        argv: arguments de la ligne de commande (défaut : ceux du processus).
    """
    args = parse_args(argv)
    manifest = write_controlled_sets(args.input, args.output, args.config, args.split)
    for entry in manifest["files"]:
        print(
            f"{entry['path']} : {entry['rows']} lignes, injectées {entry['injected']}"
        )
    print(f"Manifeste : {args.output / MANIFEST_FILE}")


if __name__ == "__main__":
    main()
