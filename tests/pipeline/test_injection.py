"""Tests de novaria.pipeline.injection (jeux contrôlés figés)."""

import pandas as pd
import pytest
import yaml

from novaria.common.io import read_json, sha256_file
from novaria.pipeline.injection import (
    MANIFEST_FILE,
    main,
    read_controlled,
    write_controlled_sets,
)
from tests.pipeline.test_training import _prepared_dir

EVALUATION_CONFIG = {
    "injection": {
        "spike_rate": 0.1,
        "impossible_rate": 0.05,
        "frozen_runs": 1,
        "frozen_hours": [3, 4],
    },
    "injection_seeds": [0, 1],
    "stability": {"retrain_runs": 2, "train_fraction": 0.8},
    "selection": {
        "metric": "test_youden_pic",
        "tolerance": 0.02,
        "max_latency_ms": 1000,
    },
}


def write_evaluation_config(tmp_path, config: dict | None = None) -> str:
    """Écrit une configuration d'évaluation YAML dans un dossier temporaire.

    Args:
        tmp_path: dossier temporaire fourni par pytest.
        config: configuration à écrire (défaut : ``EVALUATION_CONFIG``).

    Returns:
        Le chemin du fichier YAML.
    """
    path = tmp_path / "evaluation.yaml"
    path.write_text(yaml.safe_dump(config or EVALUATION_CONFIG), encoding="utf-8")
    return str(path)


def test_writes_one_file_per_seed_with_labels(tmp_path) -> None:
    """Vérifie un fichier par graine, les étiquettes relues, et le manifeste.

    Les lignes de référence sont identiques au test préparé ; chaque fichier
    a la même taille que le test ; les empreintes du manifeste sont exactes.

    Args:
        tmp_path: dossier temporaire fourni par pytest.
    """
    prepared = _prepared_dir(tmp_path)
    out = tmp_path / "controlled"

    manifest = write_controlled_sets(prepared, out, write_evaluation_config(tmp_path))

    test = pd.read_parquet(prepared / "test.parquet").set_index("timestamp")
    assert [f["path"] for f in manifest["files"]] == [
        "test_seed0.parquet",
        "test_seed1.parquet",
    ]
    for entry in manifest["files"]:
        df, labels = read_controlled(out / entry["path"])
        assert len(df) == len(test)
        assert labels.notna().sum() == sum(entry["injected"].values())
        reference = labels.isna().to_numpy()
        pd.testing.assert_frame_equal(df[reference], test[reference], check_dtype=False)
        assert sha256_file(out / entry["path"]) == entry["sha256"]
    assert read_json(out / MANIFEST_FILE)["split"] == "test"


def test_same_seed_gives_same_file(tmp_path) -> None:
    """Vérifie que la même configuration redonne exactement les mêmes données.

    Args:
        tmp_path: dossier temporaire fourni par pytest.
    """
    prepared = _prepared_dir(tmp_path)
    config = write_evaluation_config(tmp_path)
    write_controlled_sets(prepared, tmp_path / "a", config)
    write_controlled_sets(prepared, tmp_path / "b", config)

    a, labels_a = read_controlled(tmp_path / "a" / "test_seed0.parquet")
    b, labels_b = read_controlled(tmp_path / "b" / "test_seed0.parquet")
    pd.testing.assert_frame_equal(a, b)
    pd.testing.assert_series_equal(labels_a, labels_b)


def test_refuses_train(tmp_path) -> None:
    """Vérifie que l'injection est refusée sur le train (règle 4).

    Args:
        tmp_path: dossier temporaire fourni par pytest.
    """
    with pytest.raises(ValueError, match="train"):
        write_controlled_sets(tmp_path, tmp_path / "out", "unused.yaml", split="train")


def test_cli(tmp_path, capsys) -> None:
    """Vérifie le lancement en ligne de commande.

    Args:
        tmp_path: dossier temporaire fourni par pytest.
        capsys: capture de la sortie console fournie par pytest.
    """
    prepared = _prepared_dir(tmp_path)
    main(
        [
            "--input", str(prepared),
            "--output", str(tmp_path / "controlled"),
            "--config", write_evaluation_config(tmp_path),
        ]
    )  # fmt: skip

    assert "test_seed1.parquet" in capsys.readouterr().out
