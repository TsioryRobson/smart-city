"""Tests unitaires de novaria.models.detectors."""

import numpy as np
import pytest

from novaria.models.detectors import ALGORITHMS, build_estimator, build_model
from novaria.preparation.features import add_features
from tests.factories import measurements

PARAMS = {
    "isolation_forest": {"n_estimators": 50},
    "local_outlier_factor": {"n_neighbors": 10},
    "one_class_svm": {"nu": 0.05},
}


def _prepared(n: int = 200, seed: int = 0):
    """Construit une trame préparée synthétique (mesures + variables dérivées).

    Args:
        n: nombre d'heures par station.
        seed: graine des mesures.

    Returns:
        La trame préparée, avec ``station_id`` et ``wd`` en plus des variables.
    """
    return add_features(measurements(n=n, seed=seed))


@pytest.mark.parametrize("algorithm", ALGORITHMS)
def test_threshold_gives_contamination_on_train(algorithm: str) -> None:
    """Vérifie que le seuil calibré donne la part d'alertes attendue sur le train.

    C'est le protocole commun : à contamination égale, les 3 algorithmes
    alertent sur la même part du train.

    Args:
        algorithm: algorithme testé.
    """
    train = _prepared()
    model = build_model(
        algorithm, PARAMS[algorithm], contamination=0.05, random_state=0
    ).fit(train)

    if algorithm == "local_outlier_factor":
        scores = model[-1].estimator_.negative_outlier_factor_ - model[-1].threshold_
    else:
        scores = model.decision_function(train)

    assert np.mean(scores < 0) == pytest.approx(0.05, abs=0.01)


@pytest.mark.parametrize("algorithm", ALGORITHMS)
def test_extreme_point_is_flagged(algorithm: str) -> None:
    """Vérifie qu'un point très éloigné du train a un score négatif et ``predict`` = -1.

    Args:
        algorithm: algorithme testé.
    """
    train = _prepared()
    model = build_model(
        algorithm, PARAMS[algorithm], contamination=0.05, random_state=0
    ).fit(train)
    outlier = train.iloc[[0]].copy()
    outlier[["pm25", "pm10", "co", "so2"]] = [900.0, 950.0, 15000.0, 800.0]

    assert model.decision_function(outlier)[0] < 0
    assert model.predict(outlier)[0] == -1


def test_model_accepts_missing_values_and_extra_columns() -> None:
    """Vérifie que le pipeline impute les NaN et ignore les colonnes hors variables."""
    train = _prepared()
    model = build_model("isolation_forest", {"n_estimators": 20}, 0.01, 0).fit(train)
    row = train.iloc[[0]].copy()
    row[["temp", "pm25"]] = np.nan

    assert np.isfinite(model.decision_function(row)).all()


def test_scaler_choice() -> None:
    """Vérifie que la clé ``scaler`` choisit la mise à l'échelle et n'atteint pas le détecteur."""
    model = build_model("one_class_svm", {"scaler": "robust", "nu": 0.1}, 0.01, 0)
    scaler = model.named_steps["preprocessor"].transformers[0][1].named_steps["scaler"]

    assert type(scaler).__name__ == "RobustScaler"
    assert model.named_steps["detector"].estimator.nu == 0.1


def test_invalid_inputs_are_rejected() -> None:
    """Vérifie les erreurs : algorithme inconnu, scaler inconnu, contamination hors bornes."""
    with pytest.raises(ValueError):
        build_estimator("kmeans", {}, 0)
    with pytest.raises(ValueError):
        build_model("isolation_forest", {"scaler": "minmax"}, 0.01, 0)
    with pytest.raises(ValueError):
        build_model("isolation_forest", {}, 0.9, 0).fit(_prepared(n=20))
