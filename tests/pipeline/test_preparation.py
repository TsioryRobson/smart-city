"""Tests de novaria.pipeline.preparation, enchaînée après l'ingestion."""

import numpy as np
import pandas as pd

from novaria.common.config import MODEL_FEATURES, STATION_COLUMN
from novaria.common.io import read_json, sha256_file
from novaria.pipeline.ingestion import ingest
from novaria.pipeline.preparation import METADATA_FILE, main, prepare
from tests.factories import write_raw_csvs

BOUNDS = {
    "train_start": pd.Timestamp("2020-01-02"),
    "val_start": pd.Timestamp("2020-01-05"),
    "test_start": pd.Timestamp("2020-01-06"),
    "test_end": pd.Timestamp("2020-01-07"),
}


def _ingested(tmp_path, corrupt: bool = False):
    """Écrit des CSV synthétiques (8 jours, 2 stations) et les ingère.

    Args:
        tmp_path: dossier temporaire fourni par pytest.
        corrupt: ajoute une concentration négative et un doublon si True.

    Returns:
        Le dossier de sortie de l'ingestion.
    """
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    paths = write_raw_csvs(raw_dir, n=24 * 8)
    if corrupt:
        frame = pd.read_csv(paths[0])
        frame.loc[30, "SO2"] = -2.0
        frame = pd.concat([frame, frame.iloc[[40]]])
        frame.to_csv(paths[0], index=False)
    ingest(raw_dir, tmp_path / "ingested")
    return tmp_path / "ingested"


def test_prepare_writes_splits_and_metadata(tmp_path) -> None:
    """Vérifie les trois jeux Parquet, leurs colonnes et les métadonnées.

    Les jeux contiennent ``timestamp``, ``station_id``, les mesures et
    ``MODEL_FEATURES`` ; ils ne sont pas normalisés ; les métadonnées
    reprennent l'empreinte de l'ingestion et celle de chaque fichier.

    Args:
        tmp_path: dossier temporaire fourni par pytest.
    """
    ingested, out = _ingested(tmp_path), tmp_path / "prepared"

    metadata = prepare(
        ingested,
        out,
        tmp_path / "report.json",
        tmp_path / "checks.log",
        BOUNDS,
        verbose=False,
    )

    train = pd.read_parquet(out / "train.parquet")
    assert {"timestamp", STATION_COLUMN, *MODEL_FEATURES} <= set(train.columns)
    assert len(train) == 3 * 24 * 2
    assert train["pres"].mean() > 900  # valeurs brutes, non normalisées
    assert metadata["normalized"] is False
    assert (
        metadata["source_dataset_sha256"]
        == read_json(ingested / "manifest.json")["dataset_sha256"]
    )
    assert metadata["files"]["test"]["sha256"] == sha256_file(out / "test.parquet")
    assert read_json(out / METADATA_FILE)["model_features"] == MODEL_FEATURES
    assert "splits" in read_json(tmp_path / "report.json")


def test_prepare_cleans_and_logs(tmp_path) -> None:
    """Vérifie que la préparation corrige, supprime le doublon et journalise.

    Args:
        tmp_path: dossier temporaire fourni par pytest.
    """
    ingested, out = _ingested(tmp_path, corrupt=True), tmp_path / "prepared"

    prepare(
        ingested,
        out,
        tmp_path / "report.json",
        tmp_path / "checks.log",
        BOUNDS,
        verbose=False,
    )

    data = pd.concat(
        pd.read_parquet(out / f"{n}.parquet") for n in ("train", "val", "test")
    )
    assert not data.duplicated(["timestamp", STATION_COLUMN]).any()
    assert not (data["so2"] < 0).any()
    log = (tmp_path / "checks.log").read_text(encoding="utf-8")
    assert "error=valeur_negative | action=impute" in log
    assert "error=doublon | action=supprime" in log
    report = read_json(tmp_path / "report.json")
    assert report["duplicates"] == 1


def test_cli_accepts_split_bounds(tmp_path) -> None:
    """Vérifie que les bornes du split se passent en arguments de ligne de commande.

    Args:
        tmp_path: dossier temporaire fourni par pytest.
    """
    ingested, out = _ingested(tmp_path), tmp_path / "prepared"

    main(
        [
            "--input", str(ingested),
            "--output", str(out),
            "--report", str(tmp_path / "report.json"),
            "--log", str(tmp_path / "checks.log"),
            "--train-start", "2020-01-02",
            "--val-start", "2020-01-05",
            "--test-start", "2020-01-06",
            "--test-end", "2020-01-07",
        ]
    )  # fmt: skip

    test = pd.read_parquet(out / "test.parquet")
    assert test["timestamp"].min() == pd.Timestamp("2020-01-06")
    assert np.isfinite(test["hour_sin"]).all()
