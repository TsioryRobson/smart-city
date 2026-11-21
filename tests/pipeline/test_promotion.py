"""Tests de novaria.pipeline.promotion (Model Registry et alias)."""

import mlflow
import pytest
from mlflow import MlflowClient

from novaria.common.config import MLFLOW_MODEL_NAME
from novaria.common.io import read_json
from novaria.pipeline.injection import write_controlled_sets
from novaria.pipeline.promotion import (
    PROMOTED,
    PROMOTION_FILE,
    candidate_entry,
    promotion_decision,
    run_promotion,
)
from novaria.pipeline.training import run_training
from tests.pipeline.test_injection import EVALUATION_CONFIG, write_evaluation_config
from tests.pipeline.test_training import SMALL_CONFIG, _prepared_dir, _write_config

CONFIG = EVALUATION_CONFIG | {"promotion": {"detection_tolerance": 0.02}}


def _metrics(
    detection: float, false_alarms: float, latency: float = 5.0
) -> dict[str, float]:
    """Construit des métriques de promotion minimales.

    Args:
        detection: taux de détection des pics.
        false_alarms: taux de fausses alertes.
        latency: latence unitaire en millisecondes.

    Returns:
        Les métriques au format de ``promotion_metrics``.
    """
    return {
        "detection_rate_pic": detection,
        "false_alarm_rate": false_alarms,
        "youden_pic": detection - false_alarms,
        "latency_ms_single_row": latency,
    }


# --- Règle de promotion ------------------------------------------------------
def test_first_version_is_promoted() -> None:
    """Vérifie qu'en l'absence de champion, la première version est promue."""
    accepted, reasons = promotion_decision(_metrics(0.9, 0.03), None, 0.02, 50)

    assert accepted
    assert any("Aucun champion" in r for r in reasons)


@pytest.mark.parametrize(
    ("candidate", "expected"),
    [
        (_metrics(0.89, 0.02), True),  # détection à moins de 2 points, moins d'alertes
        (_metrics(0.87, 0.02), False),  # détection trop basse
        (_metrics(0.95, 0.04), False),  # plus de fausses alertes
        (_metrics(0.95, 0.02, latency=80), False),  # trop lent
    ],
)
def test_rule_against_champion(candidate: dict, expected: bool) -> None:
    """Vérifie chaque critère de la règle face à un champion à 90 % / 3 %.

    Args:
        candidate: métriques du candidat.
        expected: décision attendue.
    """
    accepted, _ = promotion_decision(candidate, _metrics(0.90, 0.03), 0.02, 50)

    assert accepted is expected


def test_candidate_entry_requires_an_algorithm(tmp_path) -> None:
    """Vérifie qu'il faut un algorithme si aucune sélection n'a été faite.

    Args:
        tmp_path: dossier temporaire fourni par pytest.
    """
    with pytest.raises(ValueError, match="--algorithm"):
        candidate_entry(tmp_path, None)


# --- Bout en bout ------------------------------------------------------------
def test_two_versions_give_champion_and_previous(tmp_path) -> None:
    """Vérifie l'enregistrement de 2 versions et les alias ``@champion`` / ``@previous``.

    La v1 est promue (aucun champion). La v2 est forcée si la règle la refuse :
    dans tous les cas, v2 devient ``@champion`` et v1 ``@previous``, et la
    décision est tracée sur la version et dans un run MLflow.

    Args:
        tmp_path: dossier temporaire fourni par pytest.
    """
    prepared = _prepared_dir(tmp_path)
    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    config = write_evaluation_config(tmp_path, CONFIG)
    write_controlled_sets(prepared, tmp_path / "controlled", config)
    small = SMALL_CONFIG | {
        "algorithms": {"one_class_svm": SMALL_CONFIG["algorithms"]["one_class_svm"]}
    }
    for name in ("v1", "v2"):
        run_training(
            prepared,
            tmp_path / name,
            _write_config(tmp_path, small),
            tmp_path / f"{name}.md",
            tracking_uri=uri,
            verbose=False,
        )

    first = run_promotion(
        tmp_path / "v1",
        tmp_path / "controlled",
        prepared,
        config,
        "one_class_svm",
        tracking_uri=uri,
    )
    second = run_promotion(
        tmp_path / "v2",
        tmp_path / "controlled",
        prepared,
        config,
        "one_class_svm",
        force=True,
        tracking_uri=uri,
    )

    assert first["decision"] == PROMOTED and first["version"] == "1"
    assert second["decision"] == PROMOTED and second["previous_champion"] == "1"
    assert second["aliases"] == {"champion": "2", "previous": "1"}
    assert read_json(tmp_path / "v2" / PROMOTION_FILE)["version"] == "2"

    mlflow.set_tracking_uri(uri)
    client = MlflowClient()
    version = client.get_model_version_by_alias(MLFLOW_MODEL_NAME, "champion")
    assert version.tags["decision"] == PROMOTED
    assert version.tags["model_sha256"]
    assert version.tags["git_sha"]
    assert version.tags["train_period"]
    assert version.tags["forced"] in {"true", "false"}
    runs = mlflow.search_runs(
        search_all_experiments=True, filter_string="tags.step = 'promotion'"
    )
    assert len(runs) == 2
