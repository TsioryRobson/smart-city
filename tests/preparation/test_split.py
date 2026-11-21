"""Tests unitaires de novaria.preparation.split."""

import pandas as pd
import pytest

from novaria.common.config import STATION_COLUMN
from novaria.preparation.split import split_chrono
from tests.factories import measurements


def _bounds() -> dict[str, pd.Timestamp]:
    """Construit des bornes de découpage adaptées aux données synthétiques.

    Train sur 3 jours, validation et test sur 1 jour chacun, puis une période
    « démo » qui doit être exclue.

    Returns:
        Les arguments ``train_start``, ``val_start``, ``test_start`` et
        ``test_end`` de ``split_chrono``.
    """
    return {
        "train_start": pd.Timestamp("2020-01-02"),
        "val_start": pd.Timestamp("2020-01-05"),
        "test_start": pd.Timestamp("2020-01-06"),
        "test_end": pd.Timestamp("2020-01-07"),
    }


def test_split_chrono_by_dates_no_overlap() -> None:
    """Vérifie le découpage par dates : tailles, ordre et absence de chevauchement.

    Chaque jeu doit être trié et contenir toutes les stations.
    """
    train, val, test = split_chrono(measurements(n=24 * 8), **_bounds())

    assert (len(train), len(val), len(test)) == (3 * 48, 48, 48)
    assert train.index.min() == pd.Timestamp("2020-01-02")
    assert train.index.max() < val.index.min()
    assert val.index.max() < test.index.min()
    for part in (train, val, test):
        assert part.index.is_monotonic_increasing
        assert set(part[STATION_COLUMN]) == {"A", "B"}


def test_split_chrono_excludes_demo_period() -> None:
    """Vérifie que le test ne contient aucune donnée de la période de démo."""
    _, _, test = split_chrono(measurements(n=24 * 8), **_bounds())

    assert not (test.index >= pd.Timestamp("2020-01-07")).any()


def test_split_chrono_rejects_unordered_bounds() -> None:
    """Vérifie que des bornes de découpage dans le désordre lèvent une erreur."""
    bounds = _bounds() | {"val_start": pd.Timestamp("2020-01-01")}
    with pytest.raises(AssertionError):
        split_chrono(measurements(), **bounds)


def test_split_chrono_rejects_empty_split() -> None:
    """Vérifie qu'un jeu vide (données trop courtes) lève une erreur."""
    with pytest.raises(AssertionError):
        split_chrono(measurements(n=24), **_bounds())
