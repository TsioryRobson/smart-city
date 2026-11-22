"""Tests de novaria.pipeline.evaluation, de l'entraînement jusqu'à la sélection."""

import mlflow
import pandas as pd
import pytest

from novaria.common.io import read_json, write_json
from novaria.pipeline.evaluation import (
    EVALUATION_FILE,
    SELECTION_FILE,
    load_controlled_sets,
    load_selected_model,
    merge_training_outputs,
    run_evaluation,
    select_model,
    summarize,
)
from novaria.pipeline.injection import write_controlled_sets
from novaria.pipeline.training import run_training
from tests.pipeline.test_injection import write_evaluation_config
from tests.pipeline.test_training import SMALL_CONFIG, _prepared_dir, _write_config

SELECTION = {"metric": "test_youden_pic", "tolerance": 0.02, "max_latency_ms": 50}


def _result(
    youden: float, false_alarms: float, latency: float = 5.0, std: float = 0.01
) -> dict:
    """Construit un résultat d'évaluation minimal pour tester la sélection.

    Args:
        youden: Youden des pics moyen.
        false_alarms: taux de fausses alertes moyen.
        latency: latence unitaire en millisecondes.
        std: écart-type du Youden entre tirages.

    Returns:
        Un dictionnaire au format des résultats de ``evaluate_algorithm``.
    """
    return {
        "latency_ms_single_row": latency,
        "injection": {
            "test_youden_pic": {"mean": youden, "std": std},
            "test_false_alarm_rate": {"mean": false_alarms, "std": 0.0},
        },
    }


def test_summarize() -> None:
    """Vérifie la moyenne, l'écart-type d'échantillon, le minimum et le maximum."""
    assert summarize([1.0, 2.0, 3.0]) == {
        "mean": 2.0,
        "std": 1.0,
        "min": 1.0,
        "max": 3.0,
    }
    assert summarize([4.0])["std"] == 0.0


def test_select_model_best_score() -> None:
    """Vérifie que le meilleur Youden l'emporte quand l'écart dépasse la tolérance."""
    results = {"a": _result(0.90, 0.03), "b": _result(0.70, 0.001)}

    chosen, reasons = select_model(results, SELECTION)

    assert chosen == "a"
    assert any("b" in r and "écarté" in r for r in reasons)


def test_select_model_tie_prefers_fewer_false_alarms() -> None:
    """Vérifie qu'à moins de la tolérance, le moins de fausses alertes l'emporte."""
    results = {"a": _result(0.93, 0.03), "b": _result(0.92, 0.01)}

    chosen, _ = select_model(results, SELECTION)

    assert chosen == "b"


def test_select_model_excludes_slow_models() -> None:
    """Vérifie qu'un modèle trop lent est écarté, et l'erreur si aucun n'est assez rapide."""
    results = {"a": _result(0.99, 0.0, latency=80), "b": _result(0.80, 0.02)}

    assert select_model(results, SELECTION)[0] == "b"
    with pytest.raises(ValueError):
        select_model({"a": _result(0.99, 0.0, latency=80)}, SELECTION)


# --- Bout en bout ------------------------------------------------------------
def _trained(tmp_path) -> tuple:
    """Entraîne les 3 algorithmes et injecte les jeux contrôlés, sur données synthétiques.

    Args:
        tmp_path: dossier temporaire fourni par pytest.

    Returns:
        Le triplet ``(dossier préparé, dossier des modèles, URI MLflow)``.
    """
    prepared = _prepared_dir(tmp_path)
    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    run_training(
        prepared,
        tmp_path / "models",
        _write_config(tmp_path, SMALL_CONFIG),
        tmp_path / "comparison.md",
        tracking_uri=uri,
        verbose=False,
    )
    write_controlled_sets(
        prepared, tmp_path / "controlled", write_evaluation_config(tmp_path)
    )
    return prepared, tmp_path / "models", uri


def test_run_evaluation_end_to_end(tmp_path) -> None:
    """Vérifie l'évaluation complète : résultats, sélection, rapport et runs MLflow.

    Args:
        tmp_path: dossier temporaire fourni par pytest.
    """
    prepared, models, uri = _trained(tmp_path)

    summary = run_evaluation(
        tmp_path / "controlled",
        prepared,
        models,
        write_evaluation_config(tmp_path),
        tmp_path / "evaluation.md",
        tracking_uri=uri,
        verbose=False,
    )

    assert set(summary["results"]) == set(SMALL_CONFIG["algorithms"])
    result = summary["results"]["isolation_forest"]
    assert result["injection"]["test_youden_pic"]["mean"] == pytest.approx(
        sum(result["youden_values"]["injection"]) / 2
    )
    assert len(result["youden_values"]["retrain"]) == 2
    selection = read_json(models / SELECTION_FILE)
    assert selection["algorithm"] == summary["selected"]
    assert selection["model_uri"].startswith("runs:/")
    assert read_json(models / EVALUATION_FILE)["selected"] == summary["selected"]
    report = (tmp_path / "evaluation.md").read_text(encoding="utf-8")
    assert "Modèle retenu" in report and "Stabilité" in report
    assert (tmp_path / "figures" / "evaluation_stability.png").exists()

    mlflow.set_tracking_uri(uri)
    runs = mlflow.search_runs(
        search_all_experiments=True, filter_string="tags.step = 'evaluation'"
    )
    assert len(runs) == 1 + 3
    assert (runs["tags.selected"] == "true").sum() == 1
    children = runs[runs["tags.algorithm"].notna()]
    assert children["tags.git_sha"].notna().all()
    assert set(children["params.algorithm"]) == set(SMALL_CONFIG["algorithms"])
    for run_id in runs["run_id"]:
        names = {i.dataset.name for i in mlflow.get_run(run_id).inputs.dataset_inputs}
        assert names == {"test_controle_seed0", "test_controle_seed1"}


def test_tampered_model_is_rejected(tmp_path) -> None:
    """Vérifie qu'un modèle modifié après l'entraînement est refusé (empreinte invalide).

    Args:
        tmp_path: dossier temporaire fourni par pytest.
    """
    _, models, _ = _trained(tmp_path)
    expected = read_json(models / "comparison.json")["selected"]["one_class_svm"][
        "model_sha256"
    ]
    model_file = next((models / "one_class_svm").glob("model.*"))
    model_file.write_bytes(model_file.read_bytes() + b"x")

    with pytest.raises(ValueError, match="Empreinte"):
        load_selected_model(models, "one_class_svm", expected)


def test_tampered_controlled_set_is_rejected(tmp_path) -> None:
    """Vérifie qu'un jeu contrôlé modifié après l'injection est refusé.

    Args:
        tmp_path: dossier temporaire fourni par pytest.
    """
    prepared = _prepared_dir(tmp_path)
    write_controlled_sets(
        prepared, tmp_path / "controlled", write_evaluation_config(tmp_path)
    )
    path = tmp_path / "controlled" / "test_seed0.parquet"
    df = pd.read_parquet(path)
    df.loc[0, "pm25"] = 1.0
    df.to_parquet(path, index=False)

    with pytest.raises(ValueError, match="modifié"):
        load_controlled_sets(tmp_path / "controlled", pd.Series(dtype=float))


def test_run_evaluation_merges_parallel_trainings(tmp_path) -> None:
    """Vérifie l'évaluation de 3 entraînements séparés (une étape Kubeflow par algorithme).

    Les modèles et les comparaisons sont fusionnés dans le dossier de sortie,
    qui sert ensuite à la promotion ; les dossiers d'entrée ne sont pas modifiés.

    Args:
        tmp_path: dossier temporaire fourni par pytest.
    """
    prepared = _prepared_dir(tmp_path)
    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    config = _write_config(tmp_path, SMALL_CONFIG)
    dirs = []
    for algorithm in SMALL_CONFIG["algorithms"]:
        out = tmp_path / f"trained_{algorithm}"
        run_training(
            prepared, out, config, out / "comparison.md", uri, False, [algorithm]
        )
        dirs.append(out)
    write_controlled_sets(
        prepared, tmp_path / "controlled", write_evaluation_config(tmp_path)
    )
    merged = tmp_path / "evaluated"

    summary = run_evaluation(
        tmp_path / "controlled",
        prepared,
        dirs,
        write_evaluation_config(tmp_path),
        merged / "evaluation.md",
        tracking_uri=uri,
        verbose=False,
        output_dir=merged,
    )

    comparison = read_json(merged / "comparison.json")
    assert set(comparison["selected"]) == set(SMALL_CONFIG["algorithms"])
    assert len(comparison["parent_run_ids"]) == 3
    assert read_json(merged / SELECTION_FILE)["algorithm"] == summary["selected"]
    for algorithm in SMALL_CONFIG["algorithms"]:
        assert (merged / algorithm / "MLmodel").exists()
    assert not (dirs[0] / SELECTION_FILE).exists()


def test_merge_rejects_incompatible_trainings(tmp_path) -> None:
    """Vérifie qu'on ne fusionne pas des entraînements faits sur des données différentes.

    Un algorithme présent dans deux dossiers est aussi refusé.

    Args:
        tmp_path: dossier temporaire fourni par pytest.
    """

    def write(name: str, algorithm: str, dataset: str) -> object:
        """Écrit une sortie d'entraînement minimale.

        Args:
            name: nom du dossier.
            algorithm: algorithme entraîné.
            dataset: empreinte du jeu utilisé.

        Returns:
            Le dossier écrit.
        """
        out = tmp_path / name
        (out / algorithm).mkdir(parents=True)
        write_json(
            {
                "trained_at": "2026-01-01T00:00:00+00:00",
                "parent_run_id": name,
                "dataset_sha256": dataset,
                "train_period": "p",
                "contamination": 0.01,
                "selection_metric": "m",
                "runs": {algorithm: []},
                "selected": {algorithm: {"run_id": name}},
            },
            out / "comparison.json",
        )
        return out

    a = write("a", "isolation_forest", "x")
    b = write("b", "one_class_svm", "y")
    c = write("c", "isolation_forest", "x")
    with pytest.raises(ValueError, match="dataset_sha256"):
        merge_training_outputs([a, b], tmp_path / "out1")
    with pytest.raises(ValueError, match="plusieurs"):
        merge_training_outputs([a, c], tmp_path / "out2")
