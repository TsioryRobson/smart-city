"""Tests unitaires de novaria.preparation.preprocessor sur des données synthétiques."""

import numpy as np
import pandas as pd

from novaria.common.config import MODEL_FEATURES
from novaria.preparation.features import add_features
from novaria.preparation.preprocessor import build_preprocessor
from novaria.preparation.split import split_chrono
from tests.factories import measurements

SPLIT = {
    "train_start": pd.Timestamp("2020-01-01"),
    "val_start": pd.Timestamp("2020-01-06"),
    "test_start": pd.Timestamp("2020-01-07 12:00"),
    "test_end": pd.Timestamp("2020-01-10"),
}


def _prepared(n: int = 200) -> pd.DataFrame:
    """Construit une trame préparée avec une tendance et des valeurs manquantes.

    La tendance sur pm25 rend les statistiques du train différentes de celles
    du jeu complet, ce qui permet de détecter une fuite. Des NaN sont placés à
    intervalles réguliers sur deux variables pour tester l'imputation.

    Args:
        n: nombre d'heures par station.

    Returns:
        Une trame avec ``station_id``, les mesures et ``MODEL_FEATURES``.
    """
    df = add_features(measurements(n=n))
    df["pm25"] = np.linspace(0, 100, len(df))
    df.iloc[::7, df.columns.get_loc("pm25")] = np.nan
    df.iloc[::11, df.columns.get_loc("co")] = np.nan
    return df


def test_output_is_model_features_without_nan() -> None:
    """Vérifie que la sortie contient exactement ``MODEL_FEATURES``, sans NaN.

    La trame d'entrée contient aussi ``station_id`` et ``wd`` (texte), qui
    doivent être ignorés par le préprocesseur.
    """
    train, val, test = split_chrono(_prepared(), **SPLIT)
    pre = build_preprocessor().fit(train)

    for part in (train, val, test):
        out = pre.transform(part)
        assert isinstance(out, pd.DataFrame)
        assert list(out.columns) == MODEL_FEATURES
        assert out.isna().sum().sum() == 0


def test_scaler_fitted_on_train_only() -> None:
    """Vérifie que la normalisation est ajustée sur le train uniquement (pas de fuite).

    La moyenne apprise pour pm25 doit être celle du train imputé, et non
    celle du jeu complet.
    """
    df = _prepared()
    train, _, _ = split_chrono(df, **SPLIT)
    pre = build_preprocessor().fit(train)
    scaler = pre.named_transformers_["features"].named_steps["scaler"]
    i = MODEL_FEATURES.index("pm25")

    train_pm25 = train["pm25"].fillna(train["pm25"].median())
    full_pm25 = df["pm25"].fillna(df["pm25"].median())
    np.testing.assert_allclose(scaler.mean_[i], train_pm25.mean())
    assert not np.isclose(scaler.mean_[i], full_pm25.mean())


def test_empty_feature_is_kept() -> None:
    """Vérifie qu'une variable entièrement vide au train reste en sortie.

    Le modèle doit toujours recevoir toutes les colonnes de ``MODEL_FEATURES``.
    """
    df = _prepared()
    df["rain"] = np.nan

    out = build_preprocessor().fit_transform(df)

    assert list(out.columns) == MODEL_FEATURES
