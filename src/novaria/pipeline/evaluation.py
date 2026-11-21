"""Étape 4 du pipeline : évaluation des 3 modèles retenus et sélection du meilleur.

Pour chaque algorithme, le modèle retenu à l'entraînement (``models/<algo>/``)
est évalué sur les MÊMES jeux de test contrôlés (``novaria.pipeline.injection``) :
1. **performance** : détection par type d'anomalie, fausses alertes, AUC,
   moyennées sur plusieurs tirages d'anomalies ;
2. **stabilité** : écart-type entre tirages d'anomalies, entre réentraînements
   sur des sous-échantillons du train, et variation des fausses alertes par
   mois et par station ;
3. **coût** : latence unitaire et temps d'inférence.

Le meilleur modèle est choisi par la règle de ``configs/evaluation.yaml`` et
écrit dans ``selection.json`` pour ``promotion.py``. Tout est tracé dans MLflow.

Exécution :
``python -m novaria.pipeline.evaluation --controlled data/controlled --models models``
"""

from __future__ import annotations

import argparse
import time
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import yaml
from sklearn.pipeline import Pipeline

from novaria.common.config import (
    CONTROLLED_DATA_DIR,
    EVALUATION_CONFIG_PATH,
    MODELS_DIR,
    POLLUTANTS,
    PREPARED_DATA_DIR,
    REPORTS_DIR,
    STATION_COLUMN,
    WEATHER,
)
from novaria.common.injection import SPIKE
from novaria.common.io import read_json, sha256_file, write_json
from novaria.common.tracking import git_sha, log_dataset, setup_mlflow
from novaria.models.detectors import build_model
from novaria.models.metrics import (
    detection_metrics,
    real_episode_mask,
    single_row_latency_ms,
)
from novaria.pipeline.injection import MANIFEST_FILE, read_controlled
from novaria.pipeline.training import (
    COMPARISON_FILE,
    read_prepared,
    sample_training_rows,
    serialized_model_path,
)

EVALUATION_FILE = "evaluation.json"
SELECTION_FILE = "selection.json"
KEY_METRICS = (
    "test_detection_rate_pic",
    "test_false_alarm_rate",
    "test_youden_pic",
    "test_roc_auc_pic",
    "test_detection_rate_capteur_fige",
    "test_detection_rate_valeur_impossible",
)


@dataclass(frozen=True)
class ControlledSet:
    """Un jeu de test contrôlé prêt pour l'évaluation.

    Attributes:
        seed: graine de l'injection.
        data: données contrôlées indexées par ``timestamp``.
        labels: type d'anomalie injectée par ligne (``None`` = référence).
        episodes: masque des épisodes réels de pollution extrême (heures de
            référence où un polluant dépasse le 99e centile du train).
    """

    seed: int
    data: pd.DataFrame
    labels: pd.Series
    episodes: np.ndarray


# --- Chargement --------------------------------------------------------------
def load_selected_model(
    models_dir: Path, algorithm: str, expected_sha256: str
) -> Pipeline:
    """Charge le modèle retenu d'un algorithme après avoir vérifié son empreinte.

    Args:
        models_dir: dossier des modèles produits par l'entraînement.
        algorithm: nom de l'algorithme (sous-dossier du modèle).
        expected_sha256: empreinte enregistrée à l'entraînement (``model_sha256``).

    Returns:
        Le pipeline scikit-learn ajusté.

    Raises:
        ValueError: si l'empreinte du fichier ne correspond pas (modèle modifié).
    """
    model_dir = models_dir / algorithm
    actual = sha256_file(serialized_model_path(model_dir))
    if actual != expected_sha256:
        raise ValueError(
            f"Empreinte du modèle {algorithm} invalide : {actual[:12]} au lieu de {expected_sha256[:12]}"
        )
    return mlflow.sklearn.load_model(str(model_dir))


def load_controlled_sets(
    controlled_dir: Path, train_p99: pd.Series
) -> tuple[dict[str, Any], list[ControlledSet]]:
    """Charge le manifeste et tous les jeux contrôlés, en vérifiant leurs empreintes.

    Args:
        controlled_dir: dossier produit par ``novaria.pipeline.injection``.
        train_p99: 99e centile de chaque polluant sur le train, pour repérer
            les épisodes réels de pollution extrême.

    Returns:
        Le couple ``(manifeste, jeux contrôlés)``.

    Raises:
        ValueError: si un fichier a été modifié depuis l'injection.
    """
    manifest = read_json(controlled_dir / MANIFEST_FILE)
    sets = []
    for entry in manifest["files"]:
        path = controlled_dir / entry["path"]
        if sha256_file(path) != entry["sha256"]:
            raise ValueError(f"Jeu contrôlé modifié depuis l'injection : {path.name}")
        df, labels = read_controlled(path)
        sets.append(
            ControlledSet(entry["seed"], df, labels, real_episode_mask(df, train_p99))
        )
    return manifest, sets


# --- Mesures -----------------------------------------------------------------
def summarize(values: Sequence[float]) -> dict[str, float]:
    """Résume une série de mesures : moyenne, écart-type, minimum, maximum.

    Args:
        values: mesures (une par tirage ou par réentraînement).

    Returns:
        ``{"mean", "std", "min", "max"}`` ; l'écart-type est celui de
        l'échantillon (0 pour une seule mesure).
    """
    array = np.asarray(values, dtype=float)
    return {
        "mean": float(array.mean()),
        "std": float(array.std(ddof=1)) if len(array) > 1 else 0.0,
        "min": float(array.min()),
        "max": float(array.max()),
    }


def summarize_runs(runs: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    """Résume chaque métrique sur plusieurs exécutions.

    Args:
        runs: métriques de chaque exécution (mêmes clés).

    Returns:
        ``{métrique: résumé}`` (voir ``summarize``), pour les métriques
        présentes dans toutes les exécutions.
    """
    keys = set.intersection(*(set(r) for r in runs))
    return {key: summarize([r[key] for r in runs]) for key in sorted(keys)}


def false_alarm_breakdown(
    scores: np.ndarray, controlled: ControlledSet, weather_range: pd.DataFrame
) -> dict[str, Any]:
    """Détaille les fausses alertes (heures ordinaires) par mois, par station et par météo.

    Args:
        scores: score d'anomalie par ligne du jeu contrôlé.
        controlled: jeu contrôlé, avec le masque des épisodes réels.
        weather_range: minimum et maximum de chaque variable météo sur le
            train (lignes ``min`` et ``max``).

    Returns:
        ``{"by_month": {"2015-12": taux, ...}, "by_station": {station: taux},
        "weather_out_of_range_share": part}`` ; la dernière valeur est la part
        des fausses alertes tombant sur une météo jamais vue au train.
    """
    df = controlled.data
    ordinary = controlled.labels.isna().to_numpy() & ~controlled.episodes
    alerts = pd.Series(scores < 0, index=df.index)[ordinary]
    months = pd.DatetimeIndex(alerts.index).strftime("%Y-%m")
    stations = df[STATION_COLUMN].to_numpy()[ordinary]
    weather = df.loc[ordinary, WEATHER]
    outside = (
        (weather < weather_range.loc["min"]) | (weather > weather_range.loc["max"])
    ).any(axis=1)
    false_alarms = alerts.to_numpy()
    return {
        "by_month": alerts.groupby(months).mean().to_dict(),
        "by_station": alerts.groupby(stations).mean().to_dict(),
        "weather_out_of_range_share": float(outside.to_numpy()[false_alarms].mean())
        if false_alarms.any()
        else 0.0,
    }


def evaluate_on_controlled_sets(
    model: Pipeline, sets: list[ControlledSet]
) -> tuple[list[dict[str, float]], np.ndarray]:
    """Évalue un modèle sur chaque jeu contrôlé.

    Args:
        model: pipeline ajusté.
        sets: jeux contrôlés.

    Returns:
        Le couple ``(métriques par jeu, scores du premier jeu)``. Chaque
        dictionnaire de métriques contient aussi ``inference_time_ms_per_row``.
    """
    runs, first_scores = [], None
    for controlled in sets:
        start = time.perf_counter()
        scores = model.decision_function(controlled.data)
        elapsed = time.perf_counter() - start
        metrics = detection_metrics(
            scores, controlled.labels, prefix="test", episodes=controlled.episodes
        )
        metrics["inference_time_ms_per_row"] = elapsed * 1000 / len(controlled.data)
        runs.append(metrics)
        if first_scores is None:
            first_scores = scores
    return runs, first_scores


def retrain_stability(
    algorithm: str,
    params: dict[str, Any],
    train: pd.DataFrame,
    controlled: ControlledSet,
    contamination: float,
    runs: int,
    fraction: float,
) -> list[dict[str, float]]:
    """Réentraîne un modèle sur plusieurs sous-échantillons du train et l'évalue.

    Chaque réentraînement utilise une graine différente, à la fois pour le
    sous-échantillon (stratifié par station et par mois) et pour l'algorithme.

    Args:
        algorithm: nom de l'algorithme.
        params: hyperparamètres retenus à l'entraînement.
        train: train préparé complet.
        controlled: jeu contrôlé de référence.
        contamination: part attendue d'anomalies (seuil).
        runs: nombre de réentraînements.
        fraction: part du train utilisée à chaque réentraînement.

    Returns:
        Les métriques de test de chaque réentraînement.
    """
    results = []
    for seed in range(runs):
        sample = sample_training_rows(train, round(fraction * len(train)), seed)
        model = build_model(algorithm, params, contamination, random_state=seed).fit(
            sample
        )
        scores = model.decision_function(controlled.data)
        results.append(
            detection_metrics(
                scores, controlled.labels, prefix="test", episodes=controlled.episodes
            )
        )
    return results


# --- Sélection ---------------------------------------------------------------
def select_model(
    results: dict[str, dict[str, Any]], selection: dict[str, Any]
) -> tuple[str, list[str]]:
    """Choisit le meilleur modèle selon la règle de la configuration.

    Règle : éliminer les modèles trop lents ; classer par la métrique moyenne ;
    parmi ceux à moins de ``tolerance`` du premier, préférer le moins de
    fausses alertes, puis la plus faible variabilité entre tirages.

    Args:
        results: résultats d'évaluation par algorithme (clés ``injection`` et
            ``latency_ms_single_row``).
        selection: section ``selection`` de la configuration.

    Returns:
        Le couple ``(algorithme retenu, justification ligne par ligne)``.

    Raises:
        ValueError: si aucun modèle ne respecte la latence maximale.
    """
    metric = selection["metric"]
    reasons = []
    eligible = {}
    for name, result in results.items():
        latency = result["latency_ms_single_row"]
        if latency > selection["max_latency_ms"]:
            reasons.append(
                f"{name} écarté : latence {latency:.1f} ms > {selection['max_latency_ms']} ms."
            )
        else:
            eligible[name] = result
    if not eligible:
        raise ValueError("Aucun modèle ne respecte la latence maximale")

    def score(name: str) -> float:
        """Donne la métrique moyenne de sélection d'un modèle.

        Args:
            name: nom de l'algorithme.

        Returns:
            La moyenne de la métrique sur les tirages d'anomalies.
        """
        return eligible[name]["injection"][metric]["mean"]

    best_score = max(score(n) for n in eligible)
    contenders = [
        n for n in eligible if best_score - score(n) <= selection["tolerance"]
    ]
    for name in sorted(eligible, key=score, reverse=True):
        status = "retenu dans le groupe de tête" if name in contenders else "écarté"
        reasons.append(f"{name} : {metric} moyen {score(name):.3f} ({status}).")
    chosen = min(
        contenders,
        key=lambda n: (
            eligible[n]["injection"]["test_false_alarm_rate"]["mean"],
            eligible[n]["injection"][metric]["std"],
        ),
    )
    if len(contenders) > 1:
        reasons.append(
            f"Écart inférieur à {selection['tolerance']} entre {', '.join(sorted(contenders))} : "
            f"{chosen} a le moins de fausses alertes."
        )
    return chosen, reasons


# --- Rapport -----------------------------------------------------------------
def pct(value: float) -> str:
    """Formate une proportion en pourcentage à une décimale.

    Args:
        value: proportion entre 0 et 1.

    Returns:
        Le pourcentage, par exemple ``96.4 %``.
    """
    return f"{100 * value:.1f} %"


def evaluation_markdown(summary: dict[str, Any]) -> str:
    """Met en forme les résultats de l'évaluation en Markdown.

    Args:
        summary: résumé produit par ``run_evaluation``.

    Returns:
        Le contenu Markdown : performance, stabilité, coût et sélection.
    """
    results = summary["results"]
    lines = [
        "# Évaluation des modèles sur le test contrôlé",
        "",
        (
            f"Généré le {summary['evaluated_at']} · commit `{summary['git_sha']}` · "
            f"test 2015-12 → 2016-02 · {summary['injection_seeds']} tirages d'anomalies · "
            f"{summary['retrain_runs']} réentraînements à {summary['train_fraction']:.0%} du train"
        ),
        "",
        "## Performance (moyenne ± écart-type sur les tirages d'anomalies)",
        "",
        (
            "Fausses alertes : alertes sur les heures de référence **ordinaires**. Les heures "
            "où un polluant dépasse le 99e centile du train sont de vrais épisodes de pollution "
            "extrême (non étiquetés) : une alerte y est légitime, elle est comptée à part."
        ),
        "",
        "| Modèle | Détection pics | Fausses alertes | Youden pics | AUC pics | Détection figé | Détection impossible |",
        "|---|---|---|---|---|---|---|",
    ]
    for name, r in results.items():
        inj = r["injection"]
        cells = [
            f"{pct(inj[k]['mean'])} ± {100 * inj[k]['std']:.1f}"
            if k != "test_roc_auc_pic" and k != "test_youden_pic"
            else f"{inj[k]['mean']:.3f} ± {inj[k]['std']:.3f}"
            for k in KEY_METRICS
        ]
        lines.append(f"| {name} | " + " | ".join(cells) + " |")

    first = next(iter(results.values()))["injection"]
    lines += [
        "",
        "## Alertes sur les données réelles de référence",
        "",
        (
            f"Épisodes réels de pollution extrême : {pct(first['test_real_episode_share']['mean'])} "
            "des heures de référence du test."
        ),
        "",
        (
            "| Modèle | Alertes sur toute la référence | Alertes sur épisodes réels | "
            "Fausses alertes sur heures ordinaires | Part des fausses alertes avec météo hors plage du train |"
        ),
        "|---|---|---|---|---|",
    ]
    for name, r in results.items():
        inj = r["injection"]
        lines.append(
            f"| {name} | {pct(inj['test_alert_rate_reference']['mean'])} | "
            f"{pct(inj['test_alert_rate_real_episodes']['mean'])} | "
            f"{pct(inj['test_false_alarm_rate']['mean'])} | "
            f"{pct(r['false_alarms']['weather_out_of_range_share'])} |"
        )

    lines += [
        "",
        "## Stabilité",
        "",
        (
            "| Modèle | Youden : écart-type entre tirages | Youden : écart-type entre réentraînements | "
            "Youden : min – max réentraînements | Fausses alertes par mois | Fausses alertes : station max |"
        ),
        "|---|---|---|---|---|---|",
    ]
    for name, r in results.items():
        retrain = r["retrain"]["test_youden_pic"]
        months = ", ".join(
            f"{m} {pct(v)}" for m, v in r["false_alarms"]["by_month"].items()
        )
        station, worst = max(
            r["false_alarms"]["by_station"].items(), key=lambda kv: kv[1]
        )
        lines.append(
            f"| {name} | {r['injection']['test_youden_pic']['std']:.4f} | {retrain['std']:.4f} | "
            f"{retrain['min']:.3f} – {retrain['max']:.3f} | {months} | {station} {pct(worst)} |"
        )

    lines += [
        "",
        "## Coût",
        "",
        "| Modèle | Latence unitaire (ms) | Inférence (ms / ligne) | Taille du modèle |",
        "|---|---|---|---|",
    ]
    for name, r in results.items():
        inference = r["injection"]["inference_time_ms_per_row"]["mean"]
        lines.append(
            f"| {name} | {r['latency_ms_single_row']:.1f} | {inference:.3f} | {r['model_size_mb']:.1f} Mo |"
        )

    lines += [
        "",
        "## Sélection",
        "",
        f"**Modèle retenu : `{summary['selected']}`**",
        "",
    ]
    lines += [f"- {reason}" for reason in summary["selection_reasons"]]
    lines += [
        "",
        (
            "Les anomalies sont synthétiques : ces chiffres ne valident pas une performance réelle "
            "(CLAUDE.md, règle 7). Le test ne couvre que l'hiver."
        ),
        "",
    ]
    return "\n".join(lines)


def stability_figure(summary: dict[str, Any], path: Path) -> Path:
    """Trace la stabilité : Youden des pics par tirage et par réentraînement, fausses alertes par mois.

    Args:
        summary: résumé produit par ``run_evaluation``.
        path: chemin de l'image PNG.

    Returns:
        Le chemin de l'image écrite.
    """
    results = summary["results"]
    names = list(results)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    data = [r["youden_values"]["injection"] for r in results.values()]
    data_retrain = [r["youden_values"]["retrain"] for r in results.values()]
    positions = np.arange(len(names))
    axes[0].boxplot(data, positions=positions - 0.18, widths=0.3)
    axes[0].boxplot(data_retrain, positions=positions + 0.18, widths=0.3)
    axes[0].set_xticks(positions, names)
    axes[0].set_title("Youden des pics (gauche : tirages, droite : réentraînements)")
    months = sorted(
        {m for r in results.values() for m in r["false_alarms"]["by_month"]}
    )
    width = 0.8 / len(names)
    for i, (name, r) in enumerate(results.items()):
        values = [100 * r["false_alarms"]["by_month"].get(m, np.nan) for m in months]
        axes[1].bar(np.arange(len(months)) + i * width, values, width, label=name)
    axes[1].axhline(
        100 * summary["contamination"],
        color="0.3",
        ls="--",
        lw=1,
        label="contamination",
    )
    axes[1].set_xticks(np.arange(len(months)) + width, months)
    axes[1].set(title="Fausses alertes par mois (%)", ylabel="%")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


# --- Orchestration -----------------------------------------------------------
def log_controlled_inputs(
    sets: list[ControlledSet], manifest: dict[str, Any], controlled_dir: Path
) -> None:
    """Enregistre les jeux de test contrôlés comme entrées du run MLflow actif.

    Args:
        sets: jeux contrôlés évalués.
        manifest: manifeste de l'injection (empreinte et contenu de chaque fichier).
        controlled_dir: dossier des jeux contrôlés.
    """
    entries = {entry["seed"]: entry for entry in manifest["files"]}
    for controlled in sets:
        entry = entries[controlled.seed]
        log_dataset(
            controlled.data.assign(injection=controlled.labels.to_numpy()),
            name=f"test_controle_seed{controlled.seed}",
            context="test",
            source=controlled_dir / entry["path"],
            sha256=entry["sha256"],
            tags={f"injected_{k}": n for k, n in entry["injected"].items()},
        )


def evaluate_algorithm(
    algorithm: str,
    selected: dict[str, Any],
    models_dir: Path,
    sets: list[ControlledSet],
    train: pd.DataFrame,
    config: dict[str, Any],
    contamination: float,
) -> dict[str, Any]:
    """Évalue un algorithme : performance, stabilité et coût.

    Args:
        algorithm: nom de l'algorithme.
        selected: entrée du modèle retenu dans ``comparison.json``.
        models_dir: dossier des modèles.
        sets: jeux contrôlés.
        train: train préparé (pour les réentraînements).
        config: configuration d'évaluation.
        contamination: part attendue d'anomalies utilisée à l'entraînement.

    Returns:
        Les résultats de l'algorithme.
    """
    model = load_selected_model(models_dir, algorithm, selected["model_sha256"])
    injection_runs, scores = evaluate_on_controlled_sets(model, sets)
    retrain_runs = retrain_stability(
        algorithm,
        selected["params"],
        train,
        sets[0],
        contamination,
        config["stability"]["retrain_runs"],
        config["stability"]["train_fraction"],
    )
    model_file = serialized_model_path(models_dir / algorithm)
    return {
        "training_run_id": selected["run_id"],
        "params": selected["params"],
        "model_sha256": selected["model_sha256"],
        "model_size_mb": model_file.stat().st_size / 1e6,
        "injection": summarize_runs(injection_runs),
        "retrain": summarize_runs(retrain_runs),
        "youden_values": {
            "injection": [r["test_youden_pic"] for r in injection_runs],
            "retrain": [r["test_youden_pic"] for r in retrain_runs],
        },
        "false_alarms": false_alarm_breakdown(
            scores, sets[0], train[WEATHER].agg(["min", "max"])
        ),
        "latency_ms_single_row": single_row_latency_ms(model, sets[0].data),
    }


def run_evaluation(
    controlled_dir: str | Path,
    prepared_dir: str | Path,
    models_dir: str | Path,
    config_path: str | Path,
    report_path: str | Path,
    tracking_uri: str | None = None,
    verbose: bool = True,
) -> dict[str, Any]:
    """Évalue les 3 modèles retenus, choisit le meilleur et écrit les résultats.

    Args:
        controlled_dir: dossier des jeux contrôlés (injection).
        prepared_dir: dossier de la préparation (train pour les réentraînements).
        models_dir: dossier des modèles et de ``comparison.json``.
        config_path: configuration d'évaluation YAML.
        report_path: chemin du rapport Markdown.
        tracking_uri: URI MLflow explicite (sinon variable d'environnement,
            sinon suivi local).
        verbose: affiche la progression si True.

    Returns:
        Le résumé de l'évaluation (aussi écrit dans ``evaluation.json``).
    """
    controlled_dir, models_dir, report_path = (
        Path(controlled_dir),
        Path(models_dir),
        Path(report_path),
    )
    with Path(config_path).open(encoding="utf-8") as f:
        config = yaml.safe_load(f)
    comparison = read_json(models_dir / COMPARISON_FILE)
    train = read_prepared(Path(prepared_dir) / "train.parquet")
    manifest, sets = load_controlled_sets(
        controlled_dir, train[POLLUTANTS].quantile(0.99)
    )

    results = {}
    for algorithm, selected in comparison["selected"].items():
        if verbose:
            print(f"Évaluation de {algorithm}…")
        results[algorithm] = evaluate_algorithm(
            algorithm,
            selected,
            models_dir,
            sets,
            train,
            config,
            comparison["contamination"],
        )
    chosen, reasons = select_model(results, config["selection"])

    summary = {
        "evaluated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "git_sha": git_sha(),
        "dataset_sha256": comparison["dataset_sha256"],
        "controlled_files": {e["path"]: e["sha256"] for e in manifest["files"]},
        "contamination": comparison["contamination"],
        "injection_seeds": len(sets),
        "retrain_runs": config["stability"]["retrain_runs"],
        "train_fraction": config["stability"]["train_fraction"],
        "results": results,
        "selected": chosen,
        "selection_reasons": reasons,
    }
    best = comparison["selected"][chosen]
    selection = {
        "algorithm": chosen,
        "training_run_id": best["run_id"],
        "model_uri": best["model_uri"],
        "model_sha256": best["model_sha256"],
        "params": best["params"],
        "test_metrics": {k: v["mean"] for k, v in results[chosen]["injection"].items()},
        "latency_ms_single_row": results[chosen]["latency_ms_single_row"],
        "reasons": reasons,
    }

    write_json(summary, models_dir / EVALUATION_FILE)
    write_json(selection, models_dir / SELECTION_FILE)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(evaluation_markdown(summary), encoding="utf-8")
    figure = stability_figure(
        summary, report_path.parent / "figures" / "evaluation_stability.png"
    )

    setup_mlflow(tracking_uri)
    with mlflow.start_run(run_name="evaluation"):
        mlflow.set_tags(
            {
                "step": "evaluation",
                "git_sha": summary["git_sha"],
                "dataset_sha256": summary["dataset_sha256"],
                "selected_algorithm": chosen,
                "selected_training_run_id": best["run_id"],
            }
        )
        mlflow.log_params(
            {"injection_seeds": len(sets), **config["stability"], **config["selection"]}
        )
        mlflow.log_dict(config, "evaluation_config.json")
        log_controlled_inputs(sets, manifest, controlled_dir)
        for algorithm, result in results.items():
            with mlflow.start_run(run_name=f"evaluation-{algorithm}", nested=True):
                mlflow.set_tags(
                    {
                        "step": "evaluation",
                        "algorithm": algorithm,
                        "git_sha": summary["git_sha"],
                        "dataset_sha256": summary["dataset_sha256"],
                        "training_run_id": result["training_run_id"],
                        "model_uri": comparison["selected"][algorithm]["model_uri"],
                        "model_sha256": result["model_sha256"],
                        "selected": str(algorithm == chosen).lower(),
                    }
                )
                mlflow.log_params(
                    {
                        "algorithm": algorithm,
                        "contamination": summary["contamination"],
                        **result["params"],
                    }
                )
                log_controlled_inputs(sets, manifest, controlled_dir)
                metrics = {
                    f"{k}_mean": v["mean"] for k, v in result["injection"].items()
                }
                metrics |= {
                    f"{k}_std": v["std"] for k, v in result["injection"].items()
                }
                metrics["retrain_youden_pic_std"] = result["retrain"][
                    "test_youden_pic"
                ]["std"]
                metrics["latency_ms_single_row"] = result["latency_ms_single_row"]
                metrics["model_size_mb"] = result["model_size_mb"]
                mlflow.log_metrics(metrics)
        for path in (
            models_dir / EVALUATION_FILE,
            models_dir / SELECTION_FILE,
            report_path,
            figure,
        ):
            mlflow.log_artifact(str(path))
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Lit les arguments de la ligne de commande.

    Args:
        argv: arguments à analyser (défaut : ceux du processus).

    Returns:
        Les arguments : dossiers des jeux contrôlés, de la préparation et des
        modèles, configuration, rapport et URI MLflow.
    """
    parser = argparse.ArgumentParser(
        description="Évaluation et sélection du meilleur modèle."
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
        "--models",
        type=Path,
        default=MODELS_DIR,
        help="modèles retenus à l'entraînement",
    )
    parser.add_argument(
        "--config", type=Path, default=EVALUATION_CONFIG_PATH, help="configuration YAML"
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=REPORTS_DIR / "evaluation.md",
        help="rapport Markdown",
    )
    parser.add_argument(
        "--tracking-uri",
        default=None,
        help="URI MLflow (défaut : MLFLOW_TRACKING_URI ou local)",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Point d'entrée CLI : évalue, sélectionne et affiche le modèle retenu.

    Args:
        argv: arguments de la ligne de commande (défaut : ceux du processus).
    """
    args = parse_args(argv)
    summary = run_evaluation(
        args.controlled,
        args.prepared,
        args.models,
        args.config,
        args.report,
        args.tracking_uri,
    )
    for name, r in summary["results"].items():
        inj = r["injection"]
        print(
            f"  {name:<22} pics {pct(inj['test_detection_rate_' + SPIKE]['mean'])} | "
            f"fausses alertes {pct(inj['test_false_alarm_rate']['mean'])} | "
            f"Youden {inj['test_youden_pic']['mean']:.3f} ± {inj['test_youden_pic']['std']:.3f}"
        )
    print(f"\nModèle retenu : {summary['selected']}")
    for reason in summary["selection_reasons"]:
        print(f"  - {reason}")
    print(f"Rapport : {args.report}")


if __name__ == "__main__":
    main()
