"""Métriques communes à l'entraînement (validation) et à l'évaluation (test).

Un seul calcul pour les deux étapes : les chiffres de la validation et du
test contrôlés sont donc directement comparables.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline

from novaria.common.config import POLLUTANTS
from novaria.common.injection import INJECTION_TYPES, SPIKE

LATENCY_CALLS = 30


def real_episode_mask(df: pd.DataFrame, train_p99: pd.Series) -> np.ndarray:
    """Repère les heures de pollution réellement extrême dans les données de référence.

    Une heure est un épisode réel si au moins un polluant dépasse le 99e
    centile du train. Ces heures ne sont pas étiquetées, mais une alerte y est
    légitime : elle ne doit pas être comptée comme fausse alerte.

    Args:
        df: jeu préparé ou contrôlé, avec les colonnes des polluants.
        train_p99: 99e centile de chaque polluant sur le train.

    Returns:
        Un masque booléen, True pour les heures d'épisode réel.
    """
    return (df[POLLUTANTS] > train_p99[POLLUTANTS]).any(axis=1).to_numpy()


def detection_metrics(
    scores: np.ndarray,
    labels: pd.Series,
    prefix: str = "val",
    episodes: np.ndarray | None = None,
) -> dict[str, float]:
    """Calcule les métriques d'un jeu contrôlé à partir des scores d'anomalie.

    Args:
        scores: score d'anomalie par ligne (négatif = alerte).
        labels: type d'anomalie injectée par ligne, ``None`` pour la référence.
        prefix: préfixe des noms de métriques (``val`` ou ``test``).
        episodes: masque des épisodes réels de pollution extrême (voir
            ``real_episode_mask``). S'il est fourni, les fausses alertes ne sont
            comptées que sur les heures de référence ordinaires, et le taux
            d'alerte sur les épisodes réels est calculé à part.

    Returns:
        Taux de fausses alertes, taux de détection global et par type, AUC ROC
        et précision moyenne pour les pics (indépendantes du seuil), indice de
        Youden des pics (détection - fausses alertes) et, si ``episodes`` est
        fourni, taux d'alerte sur toute la référence et sur les épisodes réels.
    """
    scores = np.asarray(scores)
    alerts = scores < 0
    all_reference = labels.isna().to_numpy()
    reference = all_reference
    metrics: dict[str, float] = {}
    if episodes is not None:
        episodes = np.asarray(episodes, dtype=bool)
        reference = all_reference & ~episodes
        metrics[f"{prefix}_alert_rate_reference"] = float(alerts[all_reference].mean())
        real = all_reference & episodes
        metrics[f"{prefix}_alert_rate_real_episodes"] = (
            float(alerts[real].mean()) if real.any() else 0.0
        )
        metrics[f"{prefix}_real_episode_share"] = float(
            real.sum() / all_reference.sum()
        )
    metrics |= {
        f"{prefix}_false_alarm_rate": float(alerts[reference].mean()),
        f"{prefix}_detection_rate": float(alerts[~all_reference].mean())
        if (~all_reference).any()
        else 0.0,
    }
    for kind in INJECTION_TYPES:
        mask = (labels == kind).to_numpy()
        metrics[f"{prefix}_detection_rate_{kind}"] = (
            float(alerts[mask].mean()) if mask.any() else 0.0
        )

    spikes = (labels == SPIKE).to_numpy()
    subset = reference | spikes
    if spikes.any() and reference.any():
        metrics[f"{prefix}_roc_auc_pic"] = float(
            roc_auc_score(spikes[subset], -scores[subset])
        )
        metrics[f"{prefix}_average_precision_pic"] = float(
            average_precision_score(spikes[subset], -scores[subset])
        )
    metrics[f"{prefix}_youden_pic"] = (
        metrics[f"{prefix}_detection_rate_{SPIKE}"]
        - metrics[f"{prefix}_false_alarm_rate"]
    )
    return metrics


def single_row_latency_ms(
    model: Pipeline, df: pd.DataFrame, calls: int = LATENCY_CALLS
) -> float:
    """Mesure le temps de réponse médian pour une seule mesure, comme dans l'API.

    Args:
        model: pipeline ajusté.
        df: trame préparée dont on prend une ligne.
        calls: nombre d'appels mesurés.

    Returns:
        La latence médiane en millisecondes.
    """
    row = df.iloc[[0]]
    model.decision_function(row)  # préchauffage
    durations = []
    for _ in range(calls):
        start = time.perf_counter()
        model.decision_function(row)
        durations.append(time.perf_counter() - start)
    return float(np.median(durations) * 1000)
