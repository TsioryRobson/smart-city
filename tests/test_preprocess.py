"""Tests unitaires de src.preprocess sur des données synthétiques."""

import numpy as np
import pandas as pd

from src.config import FEATURES
from src.data import split_chrono
from src.preprocess import build_preprocessor

SPLIT = {
    "train_start": pd.Timestamp("2020-01-01"),
    "val_start": pd.Timestamp("2020-01-06"),
    "test_start": pd.Timestamp("2020-01-07 12:00"),
    "test_end": pd.Timestamp("2020-01-10"),
}


def _synthetic_df(n: int = 200) -> pd.DataFrame:
    """Construit une série horaire avec une tendance et des valeurs manquantes.

    La tendance rend les statistiques du train différentes de celles du jeu
    complet, ce qui permet de détecter une fuite. Des NaN sont placés à
    intervalles réguliers sur deux variables pour tester l'imputation.

    Args:
        n: nombre d'heures.

    Returns:
        Un DataFrame indexé par ``timestamp`` avec les colonnes ``FEATURES``.
    """
    rng = np.random.default_rng(0)
    index = pd.date_range("2020-01-01", periods=n, freq="h", name="timestamp")
    trend = np.linspace(0, 100, n)[:, None]
    values = trend + rng.normal(size=(n, len(FEATURES)))
    df = pd.DataFrame(values, index=index, columns=FEATURES)
    df.iloc[::7, 0] = np.nan
    df.iloc[::11, 3] = np.nan
    return df


def test_no_nan_after_transform() -> None:
    """Vérifie qu'il ne reste aucun NaN après transformation, sur les trois jeux.

    Contrôle aussi que la sortie est un DataFrame avec les colonnes ``FEATURES``.
    """
    train, val, test = split_chrono(_synthetic_df(), **SPLIT)
    pre = build_preprocessor().fit(train)

    for part in (train, val, test):
        out = pre.transform(part)
        assert isinstance(out, pd.DataFrame)
        assert list(out.columns) == FEATURES
        assert out.isna().sum().sum() == 0


def test_scaler_fitted_on_train_only() -> None:
    """Vérifie que le scaler est ajusté sur le train uniquement (pas de fuite).

    La moyenne apprise doit être celle du train imputé, et non celle du jeu complet.
    """
    df = _synthetic_df()
    train, _, _ = split_chrono(df, **SPLIT)
    pre = build_preprocessor().fit(train)

    imputed_train = train.fillna(train.median())
    scaler_mean = pre.named_steps["scaler"].mean_

    np.testing.assert_allclose(scaler_mean, imputed_train.mean().to_numpy())

    imputed_full = df.fillna(df.median())
    assert not np.allclose(scaler_mean, imputed_full.mean().to_numpy())
