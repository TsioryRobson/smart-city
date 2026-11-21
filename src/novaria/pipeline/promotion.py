"""Étape 5 du pipeline : enregistrement dans le Model Registry et promotion.

Le modèle candidat (sortie de l'entraînement, choisi par l'évaluation) est
enregistré comme nouvelle version de ``air-anomaly-detector``. Il devient
``@champion`` seulement s'il respecte la règle de CLAUDE.md (section 7), sur
les mêmes jeux de test contrôlés que le champion actuel :
- détection des pics au moins égale à celle du champion, à 2 points près ;
- fausses alertes inférieures ou égales ;
- latence unitaire sous le seuil.

Si la version est promue, l'ancien champion devient ``@previous`` (rollback).
Sinon, elle reste enregistrée avec la décision ``refuse``. L'option ``--force``
promeut malgré un refus ; c'est tracé (tag ``forced``).

Exécution :
``python -m novaria.pipeline.promotion --models models``
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import mlflow
import yaml
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException
from sklearn.pipeline import Pipeline

from novaria.common.config import (
    CONTROLLED_DATA_DIR,
    EVALUATION_CONFIG_PATH,
    MLFLOW_MODEL_NAME,
    MODELS_DIR,
    POLLUTANTS,
    PREPARED_DATA_DIR,
)
from novaria.common.io import read_json, sha256_file, write_json
from novaria.common.tracking import git_sha, setup_mlflow
from novaria.models.metrics import single_row_latency_ms
from novaria.pipeline.evaluation import (
    SELECTION_FILE,
    ControlledSet,
    evaluate_on_controlled_sets,
    load_controlled_sets,
    load_selected_model,
    summarize_runs,
)
from novaria.pipeline.training import (
    COMPARISON_FILE,
    read_prepared,
    serialized_model_path,
)

CHAMPION = "champion"
PREVIOUS = "previous"
PROMOTION_FILE = "promotion.json"
PROMOTED = "promu"
REFUSED = "refuse"


# --- Candidat et champion ----------------------------------------------------
def candidate_entry(
    models_dir: Path, algorithm: str | None
) -> tuple[str, dict[str, Any]]:
    """Trouve le modèle candidat dans les sorties de l'entraînement.

    Args:
        models_dir: dossier des modèles produits par l'entraînement.
        algorithm: algorithme à promouvoir ; par défaut celui de
            ``selection.json`` (choisi par l'évaluation).

    Returns:
        Le couple ``(algorithme, entrée du modèle retenu dans comparison.json)``.

    Raises:
        ValueError: si aucun algorithme n'est donné ni sélectionné, ou s'il n'a
            pas été entraîné dans ce dossier.
    """
    if algorithm is None:
        selection_path = models_dir / SELECTION_FILE
        if not selection_path.exists():
            raise ValueError(
                "Aucun algorithme donné et pas de selection.json : utiliser --algorithm"
            )
        algorithm = read_json(selection_path)["algorithm"]
    comparison = read_json(models_dir / COMPARISON_FILE)
    if algorithm not in comparison["selected"]:
        raise ValueError(f"{algorithm} n'a pas été entraîné dans {models_dir}")
    return algorithm, {**comparison["selected"][algorithm], "summary": comparison}


def current_alias_version(client: MlflowClient, name: str, alias: str) -> Any | None:
    """Renvoie la version portant un alias, ou ``None`` si l'alias n'existe pas.

    Args:
        client: client MLflow.
        name: nom du modèle enregistré.
        alias: alias recherché (``champion`` ou ``previous``).

    Returns:
        La version du modèle (objet MLflow) ou ``None``.
    """
    try:
        return client.get_model_version_by_alias(name, alias)
    except MlflowException:
        return None


def load_registered_model(name: str, version: Any) -> Pipeline:
    """Charge une version du Registry après avoir vérifié son empreinte.

    C'est le contrôle que fera l'API au chargement : le fichier sérialisé doit
    avoir l'empreinte du tag ``model_sha256`` de la version.

    Args:
        name: nom du modèle enregistré.
        version: version du modèle (objet MLflow).

    Returns:
        Le pipeline scikit-learn.

    Raises:
        ValueError: si l'empreinte ne correspond pas.
    """
    uri = f"models:/{name}/{version.version}"
    local = Path(mlflow.artifacts.download_artifacts(uri))
    expected = version.tags.get("model_sha256")
    actual = sha256_file(serialized_model_path(local))
    if actual != expected:
        raise ValueError(f"Empreinte invalide pour {name} v{version.version}")
    return mlflow.sklearn.load_model(str(local))


def alias_version_number(client: MlflowClient, alias: str) -> str | None:
    """Donne le numéro de la version qui porte un alias, sous forme de texte.

    Args:
        client: client MLflow.
        alias: alias recherché.

    Returns:
        Le numéro de version (ex. ``"2"``), ou ``None`` si l'alias n'existe pas.
    """
    version = current_alias_version(client, MLFLOW_MODEL_NAME, alias)
    return str(version.version) if version is not None else None


# --- Décision ----------------------------------------------------------------
def promotion_metrics(model: Pipeline, sets: list[ControlledSet]) -> dict[str, float]:
    """Mesure les critères de promotion d'un modèle sur les jeux contrôlés.

    Args:
        model: pipeline ajusté.
        sets: jeux de test contrôlés.

    Returns:
        Moyennes sur les tirages de la détection des pics et des fausses
        alertes (heures ordinaires), Youden des pics, et latence unitaire.
    """
    runs, _ = evaluate_on_controlled_sets(model, sets)
    summary = summarize_runs(runs)
    return {
        "detection_rate_pic": summary["test_detection_rate_pic"]["mean"],
        "false_alarm_rate": summary["test_false_alarm_rate"]["mean"],
        "youden_pic": summary["test_youden_pic"]["mean"],
        "latency_ms_single_row": single_row_latency_ms(model, sets[0].data),
    }


def promotion_decision(
    candidate: dict[str, float],
    champion: dict[str, float] | None,
    tolerance: float,
    max_latency_ms: float,
) -> tuple[bool, list[str]]:
    """Applique la règle de promotion de CLAUDE.md (section 7).

    Args:
        candidate: métriques du candidat (voir ``promotion_metrics``).
        champion: métriques du champion actuel, ou ``None`` s'il n'y en a pas.
        tolerance: tolérance sur la détection (0,02 = 2 points).
        max_latency_ms: latence unitaire maximale.

    Returns:
        Le couple ``(promu ou non, justification ligne par ligne)``.
    """
    reasons = []
    latency_ok = candidate["latency_ms_single_row"] <= max_latency_ms
    reasons.append(
        f"Latence {candidate['latency_ms_single_row']:.1f} ms "
        f"{'≤' if latency_ok else '>'} {max_latency_ms} ms."
    )
    if champion is None:
        reasons.append(
            "Aucun champion : la première version est promue si la latence est respectée."
        )
        return latency_ok, reasons

    detection_ok = (
        candidate["detection_rate_pic"] >= champion["detection_rate_pic"] - tolerance
    )
    alarms_ok = candidate["false_alarm_rate"] <= champion["false_alarm_rate"]
    reasons.append(
        f"Détection des pics {candidate['detection_rate_pic']:.1%} contre "
        f"{champion['detection_rate_pic']:.1%} pour le champion "
        f"({'respecté' if detection_ok else 'non respecté'}, tolérance {tolerance:.0%})."
    )
    reasons.append(
        f"Fausses alertes {candidate['false_alarm_rate']:.2%} contre "
        f"{champion['false_alarm_rate']:.2%} ({'respecté' if alarms_ok else 'non respecté'})."
    )
    return latency_ok and detection_ok and alarms_ok, reasons


# --- Orchestration -----------------------------------------------------------
def register_candidate(
    client: MlflowClient, algorithm: str, entry: dict[str, Any]
) -> Any:
    """Enregistre le candidat comme nouvelle version et vérifie l'empreinte enregistrée.

    Args:
        client: client MLflow.
        algorithm: nom de l'algorithme.
        entry: entrée du modèle retenu dans ``comparison.json``.

    Returns:
        La nouvelle version du modèle (objet MLflow), avec ses tags.

    Raises:
        ValueError: si le modèle enregistré n'a pas l'empreinte attendue.
    """
    summary = entry["summary"]
    tags = {
        "algorithm": algorithm,
        "training_run_id": entry["run_id"],
        "model_sha256": entry["model_sha256"],
        "git_sha": summary["git_sha"],
        "dataset_sha256": summary["dataset_sha256"],
        "train_period": summary.get("train_period", "inconnue"),
        "params": ", ".join(f"{k}={v}" for k, v in entry["params"].items()),
    }
    registered = mlflow.register_model(entry["model_uri"], MLFLOW_MODEL_NAME, tags=tags)
    version = client.get_model_version(MLFLOW_MODEL_NAME, registered.version)
    load_registered_model(MLFLOW_MODEL_NAME, version)  # contrôle d'intégrité
    return version


def run_promotion(
    models_dir: str | Path,
    controlled_dir: str | Path,
    prepared_dir: str | Path,
    config_path: str | Path,
    algorithm: str | None = None,
    force: bool = False,
    tracking_uri: str | None = None,
) -> dict[str, Any]:
    """Enregistre le candidat, décide de sa promotion et met à jour les alias.

    Args:
        models_dir: dossier de l'entraînement qui a produit le candidat.
        controlled_dir: dossier des jeux de test contrôlés.
        prepared_dir: dossier de la préparation (train de référence pour
            repérer les épisodes réels de pollution).
        config_path: configuration d'évaluation YAML (latence et tolérance).
        algorithm: algorithme candidat (défaut : celui de ``selection.json``).
        force: promeut même si la règle refuse la version (tracé).
        tracking_uri: URI MLflow explicite.

    Returns:
        Le compte rendu de la promotion (aussi écrit dans ``promotion.json``).
    """
    models_dir = Path(models_dir)
    with Path(config_path).open(encoding="utf-8") as f:
        config = yaml.safe_load(f)
    tolerance = config["promotion"]["detection_tolerance"]
    max_latency = config["selection"]["max_latency_ms"]

    setup_mlflow(tracking_uri)
    client = MlflowClient()
    algorithm, entry = candidate_entry(models_dir, algorithm)
    candidate_model = load_selected_model(models_dir, algorithm, entry["model_sha256"])
    train = read_prepared(Path(prepared_dir) / "train.parquet")
    _, sets = load_controlled_sets(
        Path(controlled_dir), train[POLLUTANTS].quantile(0.99)
    )

    champion = current_alias_version(client, MLFLOW_MODEL_NAME, CHAMPION)
    version = register_candidate(client, algorithm, entry)
    candidate_metrics = promotion_metrics(candidate_model, sets)
    champion_metrics = (
        promotion_metrics(load_registered_model(MLFLOW_MODEL_NAME, champion), sets)
        if champion is not None
        else None
    )
    accepted, reasons = promotion_decision(
        candidate_metrics, champion_metrics, tolerance, max_latency
    )
    promoted = accepted or force
    if force and not accepted:
        reasons.append("Promotion FORCÉE malgré le refus de la règle (option --force).")

    if promoted:
        if champion is not None:
            client.set_registered_model_alias(
                MLFLOW_MODEL_NAME, PREVIOUS, champion.version
            )
        client.set_registered_model_alias(MLFLOW_MODEL_NAME, CHAMPION, version.version)
    decision = PROMOTED if promoted else REFUSED
    for key, value in {
        "decision": decision,
        "forced": str(force and not accepted).lower(),
        "decision_reasons": " ".join(reasons),
        "compared_to_version": str(champion.version)
        if champion is not None
        else "aucune",
    }.items():
        client.set_model_version_tag(MLFLOW_MODEL_NAME, version.version, key, value)

    report = {
        "promoted_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "model_name": MLFLOW_MODEL_NAME,
        "algorithm": algorithm,
        "version": str(version.version),
        "decision": decision,
        "forced": force and not accepted,
        "previous_champion": str(champion.version) if champion is not None else None,
        "candidate_metrics": candidate_metrics,
        "champion_metrics": champion_metrics,
        "reasons": reasons,
        "aliases": {
            alias: alias_version_number(client, alias) for alias in (CHAMPION, PREVIOUS)
        },
    }

    with mlflow.start_run(run_name="promotion"):
        mlflow.set_tags(
            {
                "step": "promotion",
                "git_sha": git_sha(),
                "algorithm": algorithm,
                "training_run_id": entry["run_id"],
                "model_version": str(version.version),
                "decision": decision,
                "forced": str(report["forced"]).lower(),
            }
        )
        mlflow.log_params(
            {"detection_tolerance": tolerance, "max_latency_ms": max_latency}
        )
        mlflow.log_metrics({f"candidate_{k}": v for k, v in candidate_metrics.items()})
        if champion_metrics is not None:
            mlflow.log_metrics(
                {f"champion_{k}": v for k, v in champion_metrics.items()}
            )
        write_json(report, models_dir / PROMOTION_FILE)
        mlflow.log_artifact(str(models_dir / PROMOTION_FILE))
    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Lit les arguments de la ligne de commande.

    Args:
        argv: arguments à analyser (défaut : ceux du processus).

    Returns:
        Les arguments : dossiers des modèles, des jeux contrôlés et de la
        préparation, configuration, algorithme, ``force`` et URI MLflow.
    """
    parser = argparse.ArgumentParser(
        description="Enregistrement et promotion dans le Model Registry."
    )
    parser.add_argument(
        "--models", type=Path, default=MODELS_DIR, help="sortie de l'entraînement"
    )
    parser.add_argument(
        "--controlled", type=Path, default=CONTROLLED_DATA_DIR, help="jeux contrôlés"
    )
    parser.add_argument(
        "--prepared",
        type=Path,
        default=PREPARED_DATA_DIR,
        help="sortie de la préparation",
    )
    parser.add_argument(
        "--config", type=Path, default=EVALUATION_CONFIG_PATH, help="configuration YAML"
    )
    parser.add_argument(
        "--algorithm", default=None, help="défaut : algorithme de selection.json"
    )
    parser.add_argument(
        "--force", action="store_true", help="promouvoir malgré un refus (tracé)"
    )
    parser.add_argument(
        "--tracking-uri",
        default=None,
        help="URI MLflow (défaut : MLFLOW_TRACKING_URI ou local)",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Point d'entrée CLI : enregistre, décide et affiche les alias.

    Args:
        argv: arguments de la ligne de commande (défaut : ceux du processus).
    """
    args = parse_args(argv)
    report = run_promotion(
        args.models,
        args.controlled,
        args.prepared,
        args.config,
        args.algorithm,
        args.force,
        args.tracking_uri,
    )
    print(
        f"{report['model_name']} v{report['version']} ({report['algorithm']}) : {report['decision']}"
    )
    for reason in report["reasons"]:
        print(f"  - {reason}")
    print(f"Alias : {report['aliases']}")


if __name__ == "__main__":
    main()
