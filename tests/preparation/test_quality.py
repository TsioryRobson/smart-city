"""Tests unitaires de novaria.preparation.quality."""

import numpy as np
import pandas as pd

from novaria.common.config import STATION_COLUMN
from novaria.preparation.quality import (
    count_missing_hours,
    missing_hours_by_station,
    missing_summary,
    quality_report,
)
from tests.factories import measurements


def test_count_missing_hours() -> None:
    """Vérifie le comptage des heures absentes, y compris sur un index vide."""
    index = pd.DatetimeIndex(["2020-01-01 00:00", "2020-01-01 03:00"])
    assert count_missing_hours(index) == 2
    assert count_missing_hours(pd.DatetimeIndex([])) == 0


def test_missing_hours_by_station_is_per_station() -> None:
    """Vérifie que les heures manquantes sont comptées séparément pour chaque station.

    Une heure retirée à la station B ne doit pas être comptée pour la station A.
    """
    df = measurements(n=10)
    df = df[~((df[STATION_COLUMN] == "B") & (df.index == "2020-01-01 05:00"))]

    assert missing_hours_by_station(df) == {"A": 0, "B": 1}


def test_missing_summary() -> None:
    """Vérifie le nombre et le pourcentage de valeurs manquantes par variable."""
    df = pd.DataFrame({"x": [1.0, np.nan, np.nan, 4.0], "y": [1.0, 2.0, 3.0, 4.0]})

    assert missing_summary(df) == {
        "x": {"count": 2, "pct": 50.0},
        "y": {"count": 0, "pct": 0.0},
    }


def test_quality_report_counts() -> None:
    """Vérifie les compteurs du rapport qualité.

    Contrôle les stations, le doublon, la valeur manquante, la concentration
    négative, la direction inconnue et l'absence d'heure manquante.
    """
    df = measurements(n=30)
    df.iloc[0, df.columns.get_loc("pm25")] = np.nan
    df.iloc[1, df.columns.get_loc("so2")] = -1.0
    df.iloc[2, df.columns.get_loc("wd")] = "XYZ"
    df = pd.concat([df, df.iloc[[3]]])

    report = quality_report(df, verbose=False)

    assert report["stations"] == ["A", "B"]
    assert report["duplicates"] == 1
    assert report["missing"]["pm25"]["count"] == 1
    assert report["inconsistencies"]["valeur_negative"]["so2"] == 1
    assert report["inconsistencies"]["direction_inconnue"]["wd"] == 1
    assert report["missing_hours"] == {"A": 0, "B": 0}
