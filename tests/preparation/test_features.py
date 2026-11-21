"""Tests unitaires de novaria.preparation.features."""

import numpy as np
import pandas as pd

from novaria.common.config import MODEL_FEATURES
from novaria.preparation.features import (
    add_features,
    add_time_features,
    cyclic_encoding,
    encode_wind_direction,
)
from tests.factories import measurements


def test_cyclic_encoding_wraps_around() -> None:
    """Vérifie que l'encodage cyclique rapproche les deux bouts du cycle.

    23 h doit être plus proche de 0 h que 12 h ne l'est, et 0 h et 24 h doivent
    être identiques.
    """
    sin, cos = cyclic_encoding(pd.Series([0, 23, 12, 24]), 24)
    points = np.column_stack([sin, cos])

    def dist(a: int, b: int) -> float:
        """Calcule la distance euclidienne entre deux points encodés.

        Args:
            a: position du premier point.
            b: position du second point.

        Returns:
            La distance entre les deux points.
        """
        return float(np.linalg.norm(points[a] - points[b]))

    assert dist(0, 1) < dist(0, 2)
    np.testing.assert_allclose(points[0], points[3], atol=1e-12)


def test_encode_wind_direction() -> None:
    """Vérifie l'angle des directions et le NaN pour une direction absente.

    N vaut (sin 0, cos 1), E vaut (sin 1, cos 0), et une direction manquante
    ou inconnue donne NaN.
    """
    df = pd.DataFrame({"wd": pd.array(["N", "E", None, "XYZ"], dtype="string")})

    out = encode_wind_direction(df)

    np.testing.assert_allclose(out["wd_sin"].iloc[:2], [0.0, 1.0], atol=1e-12)
    np.testing.assert_allclose(out["wd_cos"].iloc[:2], [1.0, 0.0], atol=1e-12)
    assert out["wd_sin"].iloc[2:].isna().all()


def test_add_time_features_uses_index() -> None:
    """Vérifie l'heure et le mois encodés à partir de l'index.

    6 h donne hour_sin = 1 ; janvier donne month_sin = 0 et month_cos = 1.
    """
    index = pd.DatetimeIndex(["2020-01-15 06:00"], name="timestamp")
    out = add_time_features(pd.DataFrame({"x": [1]}, index=index))

    np.testing.assert_allclose(out["hour_sin"], [1.0], atol=1e-12)
    np.testing.assert_allclose(
        out[["month_sin", "month_cos"]].iloc[0], [0.0, 1.0], atol=1e-12
    )


def test_add_features_produces_all_model_features() -> None:
    """Vérifie qu'après ``add_features`` toutes les colonnes du modèle existent."""
    out = add_features(measurements(n=5))

    assert set(MODEL_FEATURES) <= set(out.columns)
    assert out[MODEL_FEATURES].isna().sum().sum() == 0
