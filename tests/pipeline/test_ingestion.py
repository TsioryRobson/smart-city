"""Tests de novaria.pipeline.ingestion sur des CSV synthétiques."""

import pandas as pd
import pytest

from novaria.common.io import read_json, sha256_file
from novaria.pipeline.ingestion import (
    MANIFEST_FILE,
    MEASUREMENTS_FILE,
    ingest,
    ingest_from_minio,
    list_station_files,
    main,
    read_station_csv,
)
from tests.factories import FakeMinio, raw_csv_frame, write_raw_csvs


def test_ingest_writes_parquet_and_manifest(tmp_path) -> None:
    """Vérifie que l'ingestion concatène les stations sans modifier les valeurs.

    Contrôle le Parquet (lignes, provenance, valeurs identiques au CSV) et le
    manifeste (empreinte de chaque fichier, nombre de lignes, empreinte du jeu).

    Args:
        tmp_path: dossier temporaire fourni par pytest.
    """
    raw_dir, out_dir = tmp_path / "raw", tmp_path / "out"
    raw_dir.mkdir()
    paths = write_raw_csvs(raw_dir, n=24)
    (raw_dir / "notes.csv").write_text("ignored")

    manifest = ingest(raw_dir, out_dir)

    data = pd.read_parquet(out_dir / MEASUREMENTS_FILE)
    assert len(data) == 48
    assert set(data["source_file"]) == {p.name for p in paths}
    original = pd.read_csv(paths[0])
    pd.testing.assert_series_equal(
        data.loc[data["source_file"] == paths[0].name, "PM2.5"].reset_index(drop=True),
        original["PM2.5"],
    )
    assert read_json(out_dir / MANIFEST_FILE) == manifest
    assert manifest["n_rows"] == 48
    assert manifest["files"][0]["sha256"] == sha256_file(paths[0])
    assert len(manifest["dataset_sha256"]) == 64


def test_dataset_hash_changes_when_a_file_changes(tmp_path) -> None:
    """Vérifie que l'empreinte du jeu change dès qu'un CSV change.

    Args:
        tmp_path: dossier temporaire fourni par pytest.
    """
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    paths = write_raw_csvs(raw_dir)
    first = ingest(raw_dir, tmp_path / "a")["dataset_sha256"]

    frame = pd.read_csv(paths[0])
    frame.loc[0, "PM2.5"] += 1
    frame.to_csv(paths[0], index=False)

    assert ingest(raw_dir, tmp_path / "b")["dataset_sha256"] != first


def test_read_station_csv_rejects_missing_columns(tmp_path) -> None:
    """Vérifie qu'un CSV sans colonne obligatoire est refusé avec un message clair.

    Args:
        tmp_path: dossier temporaire fourni par pytest.
    """
    path = tmp_path / "PRSA_Data_X_20130301-20170228.csv"
    raw_csv_frame("X").drop(columns=["O3"]).to_csv(path, index=False)

    with pytest.raises(ValueError, match="O3"):
        read_station_csv(path)


def test_list_station_files_raises_without_csv(tmp_path) -> None:
    """Vérifie qu'un dossier sans CSV de station lève ``FileNotFoundError``.

    Args:
        tmp_path: dossier temporaire vide fourni par pytest.
    """
    with pytest.raises(FileNotFoundError):
        list_station_files(tmp_path)


def test_cli(tmp_path, capsys) -> None:
    """Vérifie le lancement en ligne de commande avec ``--input`` et ``--output``.

    Args:
        tmp_path: dossier temporaire fourni par pytest.
        capsys: capture de la sortie console fournie par pytest.
    """
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    write_raw_csvs(raw_dir)

    main(["--input", str(raw_dir), "--output", str(tmp_path / "out")])

    assert (tmp_path / "out" / MEASUREMENTS_FILE).exists()
    assert "2 fichiers" in capsys.readouterr().out


def test_ingest_from_minio(tmp_path, monkeypatch) -> None:
    """Vérifie l'ingestion depuis MinIO : mêmes empreintes qu'en local, provenance MinIO.

    Args:
        tmp_path: dossier temporaire fourni par pytest.
        monkeypatch: outil pytest pour remplacer le client MinIO.
    """
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    paths = write_raw_csvs(raw_dir)
    client = FakeMinio()
    for path in paths:
        client.fput_object("novaria-data", f"raw/{path.name}", str(path))
    monkeypatch.setattr("novaria.common.storage.client_from_env", lambda: client)

    manifest = ingest_from_minio("novaria-data", "raw/", tmp_path / "out")

    assert manifest["source_dir"] == "minio://novaria-data/raw/"
    assert (
        manifest["dataset_sha256"]
        == ingest(raw_dir, tmp_path / "local")["dataset_sha256"]
    )
    assert (tmp_path / "out" / MEASUREMENTS_FILE).exists()
