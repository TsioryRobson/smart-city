"""Tests de novaria.pipeline.training, sur des données synthétiques préparées."""

import mlflow
import numpy as np
import pandas as pd
import pytest
import yaml

from novaria.common.config import MODEL_FEATURES, STATION_COLUMN
from novaria.common.io import read_json, sha256_file
from novaria.models.metrics import detection_metrics
from novaria.pipeline.ingestion import ingest
from novaria.pipeline.preparation import prepare
from novaria.pipeline.training import (
    expand_grid,
    load_training_config,
    run_training,
    sample_training_rows,
    select_algorithms,
    select_best,
    serialized_model_path,
)
from tests.factories import measurements, write_raw_csvs

SMALL_CONFIG = {
    "contamination": 0.05,
    "random_state": 0,
    "validation_injection": {
        "spike_rate": 0.1,
        "impossible_rate": 0.05,
        "frozen_runs": 1,
        "frozen_hours": [3, 4],
    },
    "selection_metric": "val_youden_pic",
    "max_latency_ms": 1000,
    "algorithms": {
        "isolation_forest": {
            "max_train_samples": None,
            "grid": {"scaler": ["standard"], "n_estimators": [20, 40]},
        },
        "local_outlier_factor": {"max_train_samples": 60, "grid": {"n_neighbors": [5]}},
        "one_class_svm": {"max_train_samples": None, "grid": {"nu": [0.05]}},
    },
}


def _write_config(tmp_path, config: dict) -> str:
    """Écrit une configuration d'entraînement YAML dans un dossier temporaire.

    Args:
        tmp_path: dossier temporaire fourni par pytest.
        config: configuration à écrire.

    Returns:
        Le chemin du fichier YAML.
    """
    path = tmp_path / "training.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return str(path)


# --- Fonctions unitaires -----------------------------------------------------
def test_expand_grid() -> None:
    """Vérifie le produit cartésien de la grille, dans un ordre déterministe."""
    grid = {"b": [1, 2], "a": ["x"]}

    assert expand_grid(grid) == [{"a": "x", "b": 1}, {"a": "x", "b": 2}]
    assert expand_grid({}) == []


def test_load_training_config_rejects_errors(tmp_path) -> None:
    """Vérifie les erreurs de configuration : algorithme inconnu, contamination invalide.

    Args:
        tmp_path: dossier temporaire fourni par pytest.
    """
    assert (
        load_training_config(_write_config(tmp_path, SMALL_CONFIG))["contamination"]
        == 0.05
    )
    bad_algo = SMALL_CONFIG | {"algorithms": {"kmeans": {"grid": {"k": [2]}}}}
    with pytest.raises(ValueError, match="kmeans"):
        load_training_config(_write_config(tmp_path, bad_algo))
    with pytest.raises(ValueError, match="contamination"):
        load_training_config(
            _write_config(tmp_path, SMALL_CONFIG | {"contamination": 0.8})
        )


def test_select_algorithms() -> None:
    """Vérifie la restriction aux algorithmes demandés (une étape Kubeflow par algorithme).

    L'ordre de la configuration est conservé ; un algorithme absent est refusé.
    """
    assert select_algorithms(SMALL_CONFIG, None) is SMALL_CONFIG
    kept = select_algorithms(SMALL_CONFIG, ["one_class_svm", "isolation_forest"])
    assert list(kept["algorithms"]) == ["isolation_forest", "one_class_svm"]
    assert kept["contamination"] == SMALL_CONFIG["contamination"]
    assert len(SMALL_CONFIG["algorithms"]) == 3
    with pytest.raises(ValueError, match="absents"):
        select_algorithms(SMALL_CONFIG, ["kmeans"])


def test_sample_training_rows_keeps_station_balance() -> None:
    """Vérifie que l'échantillon garde la répartition des stations et ne fait rien si inutile."""
    df = measurements(n=200, stations=("A", "B"))

    sample = sample_training_rows(df, 100, seed=0)

    assert len(sample) == pytest.approx(100, abs=4)
    assert sample[STATION_COLUMN].value_counts().to_dict() == pytest.approx(
        {"A": 50, "B": 50}, abs=2
    )
    assert sample.index.is_monotonic_increasing
    assert sample_training_rows(df, None, seed=0) is df


def test_validation_metrics() -> None:
    """Vérifie les taux de détection et de fausses alertes sur des scores choisis à la main.

    4 lignes de référence dont 1 alerte (25 %), 2 pics dont 1 détecté (50 %).
    """
    scores = np.array([1.0, 0.5, 0.2, -0.1, -2.0, 0.3])
    labels = pd.Series([None, None, None, None, "pic", "pic"], dtype="object")

    metrics = detection_metrics(scores, labels)

    assert metrics["val_false_alarm_rate"] == 0.25
    assert metrics["val_detection_rate_pic"] == 0.5
    assert metrics["val_youden_pic"] == 0.25
    assert metrics["val_detection_rate_capteur_fige"] == 0.0


def test_select_best_breaks_ties_with_false_alarms() -> None:
    """Vérifie qu'à score égal, le réglage avec le moins de fausses alertes est choisi."""
    runs = [
        {"params": {"k": 1}, "metrics": {"m": 0.5, "val_false_alarm_rate": 0.03}},
        {"params": {"k": 2}, "metrics": {"m": 0.5, "val_false_alarm_rate": 0.01}},
        {"params": {"k": 3}, "metrics": {"m": 0.4, "val_false_alarm_rate": 0.0}},
    ]

    assert select_best(runs, "m")["params"] == {"k": 2}


# --- Bout en bout ------------------------------------------------------------
def _prepared_dir(tmp_path):
    """Produit des jeux préparés synthétiques (2 stations, 8 jours) avec le vrai pipeline.

    Args:
        tmp_path: dossier temporaire fourni par pytest.

    Returns:
        Le dossier de sortie de la préparation.
    """
    raw = tmp_path / "raw"
    raw.mkdir()
    write_raw_csvs(raw, n=24 * 8)
    ingest(raw, tmp_path / "ingested")
    bounds = {
        "train_start": pd.Timestamp("2020-01-01"),
        "val_start": pd.Timestamp("2020-01-05"),
        "test_start": pd.Timestamp("2020-01-07"),
        "test_end": pd.Timestamp("2020-01-08"),
    }
    prepare(
        tmp_path / "ingested",
        tmp_path / "prepared",
        tmp_path / "quality.json",
        tmp_path / "checks.log",
        bounds,
        verbose=False,
    )
    return tmp_path / "prepared"


def test_run_training_end_to_end(tmp_path) -> None:
    """Vérifie l'entraînement complet : runs MLflow, modèles retenus, comparaison.

    Contrôle : un run par réglage plus le run parent, un modèle retenu par
    algorithme (empreinte identique en local et dans MLflow, tags posés),
    rechargement du modèle avec une mesure manquante, et rapport Markdown.

    Args:
        tmp_path: dossier temporaire fourni par pytest.
    """
    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    summary = run_training(
        _prepared_dir(tmp_path),
        tmp_path / "models",
        _write_config(tmp_path, SMALL_CONFIG),
        tmp_path / "comparison.md",
        tracking_uri=uri,
        verbose=False,
    )

    assert set(summary["selected"]) == set(SMALL_CONFIG["algorithms"])
    assert len(summary["runs"]["isolation_forest"]) == 2
    assert summary["validation"]["injected"]["pic"] > 0
    assert (
        read_json(tmp_path / "models" / "comparison.json")["git_sha"]
        == summary["git_sha"]
    )
    assert "Modèles retenus" in (tmp_path / "comparison.md").read_text(encoding="utf-8")

    mlflow.set_tracking_uri(uri)
    assert len(mlflow.search_runs(search_all_experiments=True)) == 4 + 1
    for best in summary["selected"].values():
        model_dir = tmp_path / "models" / best["model_path"].split("/")[-1]
        assert sha256_file(serialized_model_path(model_dir)) == best["model_sha256"]
        tags = mlflow.get_run(best["run_id"]).data.tags
        assert tags["selected"] == "true"
        assert tags["model_sha256"] == best["model_sha256"]
        assert tags["dataset_sha256"] == summary["dataset_sha256"]

        model = mlflow.sklearn.load_model(best["model_uri"])
        row = pd.DataFrame([dict.fromkeys(MODEL_FEATURES, 1.0)])
        row.loc[0, "temp"] = np.nan
        assert np.isfinite(model.decision_function(row)).all()
    assert (tmp_path / "mlartifacts").exists()

    # Traçabilité : jeux de données en entrée, commit, signature et dépendances.
    for algorithm, best in summary["selected"].items():
        run = mlflow.get_run(best["run_id"])
        inputs = {i.dataset.name: i for i in run.inputs.dataset_inputs}
        if SMALL_CONFIG["algorithms"][algorithm]["max_train_samples"] is None:
            assert set(inputs) == {"train", "validation_controlee"}
            assert inputs["train"].dataset.digest == summary_train_sha(tmp_path)[:16]
        else:
            assert set(inputs) == {"train_echantillon", "validation_controlee"}
        assert run.data.tags["git_sha"] == summary["git_sha"]
        info = mlflow.models.get_model_info(best["model_uri"])
        assert [c.name for c in info.signature.inputs.inputs] == MODEL_FEATURES
        assert not any(c.required for c in info.signature.inputs.inputs)
        requirements = mlflow.artifacts.load_text(
            f"{best['model_uri']}/requirements.txt"
        )
        assert "scikit-learn==" in requirements and "skops==" in requirements


def summary_train_sha(tmp_path) -> str:
    """Lit l'empreinte du train préparé dans les métadonnées de la préparation.

    Args:
        tmp_path: dossier temporaire du test.

    Returns:
        L'empreinte SHA-256 du fichier ``train.parquet``.
    """
    return read_json(tmp_path / "prepared" / "metadata.json")["files"]["train"][
        "sha256"
    ]
