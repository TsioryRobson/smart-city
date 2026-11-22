"""Tests de novaria.common.storage (accès MinIO), avec un client en mémoire."""

import pytest
from minio import Minio

from novaria.common.storage import client_from_env, download_prefix, upload_directory
from tests.factories import FakeMinio


def test_download_prefix_filters_by_pattern(tmp_path) -> None:
    """Vérifie que seuls les objets du préfixe qui suivent le motif sont téléchargés.

    Les sous-dossiers et les autres préfixes sont ignorés.

    Args:
        tmp_path: dossier temporaire fourni par pytest.
    """
    client = FakeMinio()
    client.objects = {
        ("data", "raw/PRSA_Data_A.csv"): b"a",
        ("data", "raw/PRSA_Data_B.csv"): b"b",
        ("data", "raw/notes.txt"): b"x",
        ("data", "raw/old/PRSA_Data_C.csv"): b"c",
        ("data", "datasets/PRSA_Data_D.csv"): b"d",
    }

    paths = download_prefix(client, "data", "raw/", tmp_path, "PRSA_Data_*.csv")

    assert [p.name for p in paths] == ["PRSA_Data_A.csv", "PRSA_Data_B.csv"]
    assert (tmp_path / "PRSA_Data_B.csv").read_bytes() == b"b"


def test_download_prefix_raises_without_match(tmp_path) -> None:
    """Vérifie qu'un préfixe sans fichier correspondant lève une erreur explicite.

    Args:
        tmp_path: dossier temporaire fourni par pytest.
    """
    with pytest.raises(FileNotFoundError, match="raw/"):
        download_prefix(FakeMinio(), "data", "raw/", tmp_path, "*.csv")


def test_upload_directory_keeps_relative_paths(tmp_path) -> None:
    """Vérifie que l'arborescence du dossier est reproduite sous le préfixe.

    Args:
        tmp_path: dossier temporaire fourni par pytest.
    """
    (tmp_path / "sub").mkdir()
    (tmp_path / "train.parquet").write_bytes(b"t")
    (tmp_path / "sub" / "metadata.json").write_bytes(b"{}")
    client = FakeMinio()

    names = upload_directory(client, tmp_path, "data", "datasets/prepared/abc/")

    assert names == [
        "datasets/prepared/abc/sub/metadata.json",
        "datasets/prepared/abc/train.parquet",
    ]
    assert client.objects[("data", "datasets/prepared/abc/train.parquet")] == b"t"


def test_client_from_env_requires_credentials(monkeypatch) -> None:
    """Vérifie que le client lit ses identifiants dans l'environnement, sans valeur par défaut.

    Args:
        monkeypatch: outil pytest pour modifier l'environnement.
    """
    monkeypatch.setenv("MINIO_ENDPOINT", "localhost:9000")
    monkeypatch.delenv("MINIO_ACCESS_KEY", raising=False)
    with pytest.raises(KeyError):
        client_from_env()
    monkeypatch.setenv("MINIO_ACCESS_KEY", "user")
    monkeypatch.setenv("MINIO_SECRET_KEY", "secret")
    assert isinstance(client_from_env(), Minio)
