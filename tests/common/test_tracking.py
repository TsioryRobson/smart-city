"""Tests unitaires de novaria.common.tracking (connexion MLflow et traçabilité)."""

import mlflow
import pandas as pd

from novaria.common.config import LOCAL_ARTIFACTS_DIR, LOCAL_TRACKING_URI
from novaria.common.tracking import (
    git_sha,
    local_artifacts_dir,
    log_dataset,
    resolve_tracking_uri,
    setup_mlflow,
)


def test_resolve_tracking_uri_priority(monkeypatch) -> None:
    """Vérifie l'ordre de priorité : argument, puis variable d'environnement, puis local.

    Args:
        monkeypatch: outil pytest pour modifier l'environnement.
    """
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    assert resolve_tracking_uri() == LOCAL_TRACKING_URI
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    assert resolve_tracking_uri() == "http://mlflow:5000"
    assert resolve_tracking_uri("sqlite:///x.db") == "sqlite:///x.db"


def test_local_artifacts_dir(tmp_path) -> None:
    """Vérifie que les artefacts d'une base SQLite sont rangés à côté de la base.

    La base du projet garde le dossier ``mlartifacts/`` du projet.

    Args:
        tmp_path: dossier temporaire fourni par pytest.
    """
    assert local_artifacts_dir(LOCAL_TRACKING_URI) == LOCAL_ARTIFACTS_DIR
    assert (
        local_artifacts_dir(f"sqlite:///{tmp_path / 'm.db'}")
        == tmp_path / "mlartifacts"
    )


def test_git_sha_format() -> None:
    """Vérifie que le commit est un SHA de 40 caractères, suivi ou non de ``-dirty``."""
    sha = git_sha()

    assert sha == "unknown" or len(sha.removesuffix("-dirty")) == 40


def test_log_dataset_records_input_with_hash(tmp_path) -> None:
    """Vérifie qu'un jeu de données apparaît en entrée du run, avec son empreinte et ses tags.

    Args:
        tmp_path: dossier temporaire fourni par pytest.
    """
    setup_mlflow(f"sqlite:///{tmp_path / 'mlflow.db'}", experiment="test-datasets")
    df = pd.DataFrame(
        {"pm25": [1.0, 2.0]},
        index=pd.DatetimeIndex(["2020-01-01", "2020-01-02"], name="timestamp"),
    )
    sha = "a" * 64

    with mlflow.start_run() as run:
        log_dataset(
            df, "train", "entrainement", tmp_path / "train.parquet", sha, {"seed": 3}
        )

    inputs = mlflow.get_run(run.info.run_id).inputs.dataset_inputs
    assert len(inputs) == 1
    dataset = inputs[0].dataset
    tags = {t.key: t.value for t in inputs[0].tags}
    assert dataset.name == "train"
    assert dataset.digest == sha[:16]
    assert "train.parquet" in dataset.source
    assert dataset.source_type == "local"
    assert tags["sha256"] == sha
    assert tags["seed"] == "3"
    assert tags["mlflow.data.context"] == "entrainement"
