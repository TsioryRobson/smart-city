"""Étape 3 du pipeline : entraînement et comparaison des 3 algorithmes.

Pour chaque algorithme (Isolation Forest, LOF, One-Class SVM) et chaque
combinaison d'hyperparamètres de ``configs/training.yaml`` :
1. entraîne le pipeline préprocesseur + détecteur sur le train uniquement ;
2. mesure, sur une validation contrôlée (anomalies injectées), le taux de
   détection par type d'anomalie et le taux de fausses alertes ;
3. mesure le temps d'entraînement et d'inférence ;
4. enregistre le tout dans un run MLflow.

Le meilleur réglage de chaque algorithme est sauvegardé (dossier local et
artefact MLflow), avec son empreinte SHA-256. Le tableau de comparaison est
écrit en JSON et en Markdown. L'enregistrement dans le Registry et le choix du
champion sont faits plus tard (``evaluation.py``, ``promotion.py``).

Exécution :
``python -m novaria.pipeline.training --input data/prepared --output models``
"""

from __future__ import annotations

import argparse
import itertools
import shutil
import time
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import pandas as pd
import sklearn
import skops
import yaml
from mlflow.models import ModelSignature
from mlflow.types import ColSpec, Schema, TensorSpec
from sklearn.pipeline import Pipeline

import novaria
from novaria.common.config import (
    MODEL_FEATURES,
    MODELS_DIR,
    PREPARED_DATA_DIR,
    REPORTS_DIR,
    STATION_COLUMN,
    TIMESTAMP_COLUMN,
    TRAINING_CONFIG_PATH,
)
from novaria.common.injection import (
    INJECTION_TYPES,
    build_controlled_set,
    injection_config_from_dict,
)
from novaria.common.io import read_json, sha256_file, write_json
from novaria.common.tracking import git_sha, log_dataset, setup_mlflow
from novaria.models.detectors import ALGORITHMS, SKOPS_TRUSTED_TYPES, build_model
from novaria.models.metrics import detection_metrics, single_row_latency_ms

COMPARISON_FILE = "comparison.json"


# --- Configuration -----------------------------------------------------------
def load_training_config(path: str | Path) -> dict[str, Any]:
    """Lit et vérifie la configuration d'entraînement YAML.

    Args:
        path: chemin du fichier YAML.

    Returns:
        La configuration, sous forme de dictionnaire.

    Raises:
        ValueError: si un algorithme est inconnu, si une grille est vide ou si
            la contamination est hors de ]0 ; 0,5].
    """
    with Path(path).open(encoding="utf-8") as f:
        config = yaml.safe_load(f)
    unknown = set(config["algorithms"]) - set(ALGORITHMS)
    if unknown:
        raise ValueError(f"Algorithmes inconnus : {sorted(unknown)}")
    for name, algo in config["algorithms"].items():
        if not expand_grid(algo.get("grid", {})):
            raise ValueError(f"Grille vide pour {name}")
    if not 0 < config["contamination"] <= 0.5:
        raise ValueError("contamination doit être dans ]0 ; 0,5]")
    return config


def expand_grid(grid: dict[str, list[Any]]) -> list[dict[str, Any]]:
    """Développe une grille d'hyperparamètres en liste de combinaisons.

    Args:
        grid: ``{hyperparamètre: [valeurs possibles]}``.

    Returns:
        Une combinaison par élément du produit cartésien, dans un ordre
        déterministe (clés triées). Une grille vide donne une liste vide.
    """
    if not grid:
        return []
    keys = sorted(grid)
    return [
        dict(zip(keys, values, strict=True))
        for values in itertools.product(*(grid[k] for k in keys))
    ]


# --- Données -----------------------------------------------------------------
def read_prepared(path: str | Path) -> pd.DataFrame:
    """Lit un jeu préparé (Parquet) et remet l'horodatage en index.

    Args:
        path: chemin du fichier Parquet produit par la préparation.

    Returns:
        La trame préparée indexée par ``timestamp``.
    """
    return pd.read_parquet(path).set_index(TIMESTAMP_COLUMN)


def sample_training_rows(
    df: pd.DataFrame, max_rows: int | None, seed: int
) -> pd.DataFrame:
    """Échantillonne le train en gardant la répartition par station et par mois.

    Args:
        df: train préparé indexé par ``timestamp``.
        max_rows: nombre de lignes visé ; ``None`` ou une valeur supérieure à la
            taille du train garde tout le train.
        seed: graine de l'échantillonnage.

    Returns:
        L'échantillon (environ ``max_rows`` lignes), trié par date.
    """
    if max_rows is None or max_rows >= len(df):
        return df
    frac = max_rows / len(df)
    month = pd.DatetimeIndex(df.index).month
    sample = df.groupby([df[STATION_COLUMN], month], group_keys=False).sample(
        frac=frac, random_state=seed
    )
    return sample.sort_index(kind="stable")


# --- Mesures -----------------------------------------------------------------
def train_candidate(
    algorithm: str,
    params: dict[str, Any],
    train: pd.DataFrame,
    validation: pd.DataFrame,
    labels: pd.Series,
    config: dict[str, Any],
) -> tuple[Pipeline, dict[str, float]]:
    """Entraîne un réglage d'un algorithme et calcule toutes ses métriques.

    Args:
        algorithm: nom de l'algorithme.
        params: hyperparamètres (dont ``scaler``).
        train: train préparé, déjà échantillonné si besoin.
        validation: validation contrôlée.
        labels: étiquettes d'injection de la validation.
        config: configuration d'entraînement (contamination, graine, latence max).

    Returns:
        Le couple ``(pipeline ajusté, métriques)``.
    """
    model = build_model(
        algorithm, params, config["contamination"], config["random_state"]
    )
    start = time.perf_counter()
    model.fit(train)
    fit_seconds = time.perf_counter() - start

    start = time.perf_counter()
    scores = model.decision_function(validation)
    predict_seconds = time.perf_counter() - start

    latency = single_row_latency_ms(model, validation)
    metrics = {
        "fit_time_s": fit_seconds,
        "inference_time_ms_per_row": predict_seconds * 1000 / len(validation),
        "latency_ms_single_row": latency,
        "latency_ok": float(latency <= config["max_latency_ms"]),
        "n_train_rows": float(len(train)),
        **detection_metrics(scores, labels, prefix="val"),
    }
    return model, metrics


# --- Sauvegarde --------------------------------------------------------------
def save_model(model: Pipeline, path: Path, example: pd.DataFrame) -> str:
    """Sauvegarde un pipeline au format MLflow, avec le code ``novaria`` embarqué.

    Le code du paquet est copié dans le modèle (``code_paths``), pour qu'il se
    recharge sans installer ``novaria`` (API, service IA). Le modèle est
    sérialisé au format skops (sûr) ; seuls les types de
    ``SKOPS_TRUSTED_TYPES`` sont acceptés au chargement.

    Args:
        model: pipeline ajusté.
        path: dossier de destination ; remplacé s'il existe.
        example: quelques lignes préparées, pour la signature et l'exemple.

    Returns:
        L'empreinte SHA-256 du fichier sérialisé du modèle (tag ``model_sha256``).
    """
    if path.exists():
        shutil.rmtree(path)
    inputs = example[MODEL_FEATURES]
    signature = model_signature()
    mlflow.sklearn.save_model(
        model,
        path,
        signature=signature,
        input_example=inputs.head(3),
        code_paths=[str(Path(novaria.__file__).parent)],
        pip_requirements=[
            f"scikit-learn=={sklearn.__version__}",
            f"pandas=={pd.__version__}",
            f"numpy=={np.__version__}",
            f"skops=={skops.__version__}",
        ],
        skops_trusted_types=SKOPS_TRUSTED_TYPES,
    )
    return sha256_file(serialized_model_path(path))


def model_signature() -> ModelSignature:
    """Construit la signature MLflow du modèle : 17 variables optionnelles, un score.

    Toutes les variables sont optionnelles : une mesure manquante (NaN) est
    imputée par le préprocesseur du modèle et ne doit pas être refusée.

    Returns:
        La signature : entrées ``MODEL_FEATURES`` (réels), sortie un score réel
        par ligne.
    """
    inputs = Schema(
        [ColSpec("double", name, required=False) for name in MODEL_FEATURES]
    )
    outputs = Schema([TensorSpec(np.dtype("float64"), (-1,))])
    return ModelSignature(inputs=inputs, outputs=outputs)


def serialized_model_path(model_dir: Path) -> Path:
    """Trouve le fichier sérialisé d'un modèle MLflow scikit-learn.

    Le nom dépend du format (``model.skops``, ``model.pkl``...) : il est lu
    dans le fichier ``MLmodel``.

    Args:
        model_dir: dossier du modèle MLflow.

    Returns:
        Le chemin du fichier sérialisé.
    """
    with (model_dir / "MLmodel").open(encoding="utf-8") as f:
        mlmodel = yaml.safe_load(f)
    return model_dir / mlmodel["flavors"]["sklearn"]["pickled_model"]


def select_best(runs: list[dict[str, Any]], metric: str) -> dict[str, Any]:
    """Choisit le meilleur réglage d'un algorithme.

    Args:
        runs: résultats des réglages (clés ``metrics`` et ``params``).
        metric: métrique à maximiser.

    Returns:
        Le réglage retenu ; à égalité, celui qui a le moins de fausses alertes.
    """
    return max(
        runs,
        key=lambda r: (r["metrics"][metric], -r["metrics"]["val_false_alarm_rate"]),
    )


def comparison_markdown(summary: dict[str, Any]) -> str:
    """Met en forme le tableau de comparaison des algorithmes en Markdown.

    Args:
        summary: résumé produit par ``run_training``.

    Returns:
        Le contenu Markdown : modèles retenus, puis tous les réglages testés.
    """
    columns = [
        ("val_detection_rate_pic", "Détection pics", "{:.1%}"),
        ("val_false_alarm_rate", "Fausses alertes", "{:.2%}"),
        ("val_youden_pic", "Youden pics", "{:.3f}"),
        ("val_roc_auc_pic", "AUC pics", "{:.3f}"),
        ("val_detection_rate_capteur_fige", "Détection figé", "{:.1%}"),
        ("val_detection_rate_valeur_impossible", "Détection impossible", "{:.1%}"),
        ("fit_time_s", "Entraînement (s)", "{:.1f}"),
        ("inference_time_ms_per_row", "Inférence (ms/ligne)", "{:.3f}"),
        ("latency_ms_single_row", "Latence unitaire (ms)", "{:.1f}"),
    ]

    def row(name: str, run: dict[str, Any]) -> str:
        """Formate une ligne du tableau.

        Args:
            name: libellé de la première colonne.
            run: résultat d'un réglage.

        Returns:
            La ligne Markdown.
        """
        cells = [
            fmt.format(run["metrics"].get(key, float("nan"))) for key, _, fmt in columns
        ]
        return f"| {name} | " + " | ".join(cells) + " |"

    header = "| Modèle | " + " | ".join(label for _, label, _ in columns) + " |"
    separator = "|---" * (len(columns) + 1) + "|"
    lines = [
        "# Comparaison des algorithmes (validation contrôlée)",
        "",
        (
            f"Généré le {summary['trained_at']} · commit `{summary['git_sha']}` · "
            f"jeu `{str(summary['dataset_sha256'])[:12]}` · contamination {summary['contamination']:.0%}"
        ),
        "",
        f"Validation : {summary['validation']['rows']} lignes, dont "
        + ", ".join(f"{n} `{k}`" for k, n in summary["validation"]["injected"].items())
        + ". Critère de choix du réglage : `"
        + summary["selection_metric"]
        + "`.",
        "",
        "## Modèles retenus (meilleur réglage par algorithme)",
        "",
        header,
        separator,
    ]
    for name, best in summary["selected"].items():
        lines.append(row(name, best))
    lines += [
        "",
        "## Tous les réglages testés",
        "",
        header.replace("Modèle", "Réglage"),
        separator,
    ]
    for name, runs in summary["runs"].items():
        for run in runs:
            params = ", ".join(f"{k}={v}" for k, v in run["params"].items())
            lines.append(row(f"{name} ({params})", run))
    lines += [
        "",
        (
            "Le seuil de chaque modèle est calibré pour 1 alerte sur 100 mesures du train. "
            "Le capteur figé et les valeurs impossibles relèvent surtout des règles "
            "(nettoyage, contrôle « capteur figé ») : leur détection par le modèle est "
            "indiquée pour information. Résultats sur la validation seulement : le test "
            "contrôlé est réservé à `evaluation.py`."
        ),
        "",
    ]
    return "\n".join(lines)


# --- Orchestration -----------------------------------------------------------
def log_training_inputs(
    train_rows: pd.DataFrame,
    full_train: pd.DataFrame,
    validation: pd.DataFrame,
    labels: pd.Series,
    input_dir: Path,
    metadata: dict[str, Any],
    seed: int,
) -> None:
    """Enregistre dans le run MLflow actif le train utilisé et la validation contrôlée.

    Le train garde l'empreinte de son fichier Parquet s'il est utilisé en
    entier ; un échantillon reçoit l'identifiant calculé par MLflow.

    Args:
        train_rows: lignes du train réellement utilisées.
        full_train: train complet.
        validation: validation contrôlée (anomalies injectées).
        labels: étiquettes d'injection de la validation.
        input_dir: dossier de la préparation (sources des jeux).
        metadata: métadonnées de la préparation (empreintes des fichiers).
        seed: graine de l'injection dans la validation.
    """
    files = metadata["files"]
    full = len(train_rows) == len(full_train)
    log_dataset(
        train_rows,
        name="train" if full else "train_echantillon",
        context="entrainement",
        source=input_dir / files["train"]["path"],
        sha256=files["train"]["sha256"] if full else None,
    )
    log_dataset(
        validation.assign(injection=labels.to_numpy()),
        name="validation_controlee",
        context="validation",
        source=input_dir / files["val"]["path"],
        tags={
            "source_sha256": files["val"]["sha256"],
            "injection_seed": seed,
            **{f"injected_{k}": int((labels == k).sum()) for k in INJECTION_TYPES},
        },
    )


def run_training(
    input_dir: str | Path,
    output_dir: str | Path,
    config_path: str | Path,
    report_path: str | Path,
    tracking_uri: str | None = None,
    verbose: bool = True,
) -> dict[str, Any]:
    """Entraîne et compare tous les réglages, sauvegarde le meilleur par algorithme.

    Args:
        input_dir: dossier de sortie de la préparation.
        output_dir: dossier des modèles retenus et du fichier de comparaison.
        config_path: configuration d'entraînement YAML.
        report_path: chemin du tableau de comparaison Markdown.
        tracking_uri: URI MLflow explicite (sinon variable d'environnement,
            sinon suivi local).
        verbose: affiche la progression si True.

    Returns:
        Le résumé de l'entraînement (aussi écrit dans ``comparison.json``).
    """
    input_dir, output_dir = Path(input_dir), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = load_training_config(config_path)
    metadata = read_json(input_dir / "metadata.json")
    seed = config["random_state"]

    train = read_prepared(input_dir / "train.parquet")
    injection = injection_config_from_dict(config["validation_injection"])
    validation, labels = build_controlled_set(
        read_prepared(input_dir / "val.parquet"), injection, seed
    )

    sha = git_sha()
    train_period = f"{train.index.min():%Y-%m} -> {train.index.max():%Y-%m}"
    common_tags = {
        "git_sha": sha,
        "dataset_sha256": metadata["dataset_sha256"],
        "train_period": train_period,
    }
    setup_mlflow(tracking_uri)

    summary: dict[str, Any] = {
        "trained_at": datetime.now(UTC).isoformat(timespec="seconds"),
        **common_tags,
        "contamination": config["contamination"],
        "selection_metric": config["selection_metric"],
        "validation": {
            "rows": len(validation),
            "injected": {k: int((labels == k).sum()) for k in INJECTION_TYPES},
        },
        "runs": {},
        "selected": {},
    }

    with mlflow.start_run(run_name="training") as parent:
        mlflow.set_tags({**common_tags, "step": "training"})
        mlflow.log_params(
            {"contamination": config["contamination"], "random_state": seed}
        )
        mlflow.log_dict(config, "training_config.json")
        mlflow.log_artifact(str(input_dir / "metadata.json"), artifact_path="data")
        log_training_inputs(train, train, validation, labels, input_dir, metadata, seed)
        summary["parent_run_id"] = parent.info.run_id

        for algorithm, algo_config in config["algorithms"].items():
            train_rows = sample_training_rows(
                train, algo_config.get("max_train_samples"), seed
            )
            runs, models = [], {}
            for i, params in enumerate(expand_grid(algo_config["grid"])):
                with mlflow.start_run(run_name=f"{algorithm}-{i}", nested=True) as run:
                    model, metrics = train_candidate(
                        algorithm, params, train_rows, validation, labels, config
                    )
                    mlflow.set_tags(
                        {**common_tags, "algorithm": algorithm, "step": "training"}
                    )
                    mlflow.log_params(
                        {
                            "algorithm": algorithm,
                            "contamination": config["contamination"],
                            "max_train_samples": algo_config.get("max_train_samples"),
                            **params,
                        }
                    )
                    mlflow.log_metrics(metrics)
                    log_training_inputs(
                        train_rows, train, validation, labels, input_dir, metadata, seed
                    )
                runs.append(
                    {"run_id": run.info.run_id, "params": params, "metrics": metrics}
                )
                models[run.info.run_id] = model
                if verbose:
                    print(
                        f"{algorithm:<22} {params} -> pics {metrics['val_detection_rate_pic']:.1%}, "
                        f"fausses alertes {metrics['val_false_alarm_rate']:.2%}, "
                        f"{metrics['fit_time_s']:.1f} s"
                    )

            best = select_best(runs, config["selection_metric"])
            model_dir = output_dir / algorithm
            best["model_sha256"] = save_model(
                models[best["run_id"]], model_dir, validation
            )
            best["model_path"] = str(model_dir)
            with mlflow.start_run(run_id=best["run_id"], nested=True):
                mlflow.log_artifacts(str(model_dir), artifact_path="model")
                mlflow.set_tags(
                    {"selected": "true", "model_sha256": best["model_sha256"]}
                )
            best["model_uri"] = f"runs:/{best['run_id']}/model"
            summary["runs"][algorithm] = runs
            summary["selected"][algorithm] = best

        write_json(summary, output_dir / COMPARISON_FILE)
        report_path = Path(report_path)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(comparison_markdown(summary), encoding="utf-8")
        mlflow.log_artifact(str(output_dir / COMPARISON_FILE))
        mlflow.log_artifact(str(report_path))
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Lit les arguments de la ligne de commande.

    Args:
        argv: arguments à analyser (défaut : ceux du processus).

    Returns:
        Les arguments : dossiers d'entrée et de sortie, configuration, rapport
        et URI MLflow.
    """
    parser = argparse.ArgumentParser(
        description="Entraînement et comparaison des 3 algorithmes."
    )
    parser.add_argument(
        "--input", type=Path, default=PREPARED_DATA_DIR, help="sortie de la préparation"
    )
    parser.add_argument(
        "--output", type=Path, default=MODELS_DIR, help="dossier des modèles retenus"
    )
    parser.add_argument(
        "--config", type=Path, default=TRAINING_CONFIG_PATH, help="configuration YAML"
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=REPORTS_DIR / "training_comparison.md",
        help="tableau Markdown",
    )
    parser.add_argument(
        "--tracking-uri",
        default=None,
        help="URI MLflow (défaut : MLFLOW_TRACKING_URI ou local)",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Point d'entrée CLI : entraîne, compare et affiche les modèles retenus.

    Args:
        argv: arguments de la ligne de commande (défaut : ceux du processus).
    """
    args = parse_args(argv)
    summary = run_training(
        args.input, args.output, args.config, args.report, args.tracking_uri
    )
    print("\nModèles retenus :")
    for algorithm, best in summary["selected"].items():
        m = best["metrics"]
        print(
            f"  {algorithm:<22} pics {m['val_detection_rate_pic']:.1%} | "
            f"fausses alertes {m['val_false_alarm_rate']:.2%} | {best['params']}"
        )
    print(f"Comparaison : {args.report}")
    print(f"Modèles : {args.output}")


if __name__ == "__main__":
    main()
