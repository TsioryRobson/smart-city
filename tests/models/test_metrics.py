"""Tests unitaires de novaria.models.metrics."""

import numpy as np
import pandas as pd

from novaria.models.metrics import detection_metrics, real_episode_mask


def test_real_episode_mask() -> None:
    """Vérifie qu'une heure est un épisode réel dès qu'un polluant dépasse le 99e centile du train."""
    p99 = pd.Series(
        {
            "pm25": 300.0,
            "pm10": 400.0,
            "no2": 150.0,
            "co": 6000.0,
            "so2": 100.0,
            "o3": 250.0,
        }
    )
    df = pd.DataFrame(
        {
            "pm25": [50.0, 350.0, 50.0],
            "pm10": [80.0, 380.0, 80.0],
            "no2": [40.0, 40.0, 40.0],
            "co": [800.0, 800.0, 7000.0],
            "so2": [5.0, 5.0, 5.0],
            "o3": [40.0, 40.0, np.nan],
        }
    )

    assert real_episode_mask(df, p99).tolist() == [False, True, True]


def test_episodes_are_not_counted_as_false_alarms() -> None:
    """Vérifie qu'une alerte sur un épisode réel n'est pas une fausse alerte.

    6 lignes de référence dont 2 épisodes réels : 1 alerte sur un épisode,
    1 alerte sur 4 heures ordinaires (25 %). 2 pics, 1 détecté.
    """
    scores = np.array([-1.0, 1.0, -0.5, 1.0, 1.0, 1.0, -2.0, 0.3])
    labels = pd.Series([None] * 6 + ["pic", "pic"], dtype="object")
    episodes = np.array([True, True, False, False, False, False, False, False])

    metrics = detection_metrics(scores, labels, prefix="test", episodes=episodes)

    assert metrics["test_false_alarm_rate"] == 0.25
    assert metrics["test_alert_rate_real_episodes"] == 0.5
    assert metrics["test_alert_rate_reference"] == 2 / 6
    assert metrics["test_real_episode_share"] == 2 / 6
    assert metrics["test_youden_pic"] == 0.25


def test_without_episodes_all_reference_counts() -> None:
    """Vérifie que, sans masque d'épisodes, toute alerte sur la référence est une fausse alerte."""
    scores = np.array([-1.0, 1.0, -2.0])
    labels = pd.Series([None, None, "pic"], dtype="object")

    metrics = detection_metrics(scores, labels, prefix="val")

    assert metrics["val_false_alarm_rate"] == 0.5
    assert "val_alert_rate_real_episodes" not in metrics
