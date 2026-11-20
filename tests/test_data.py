"""Tests unitaires de src.data sur des données synthétiques."""

import numpy as np
import pandas as pd

from src.config import FEATURES
from src.data import (
    build_datetime_index,
    remove_duplicate_timestamps,
    select_features,
    split_chrono,
)


def _hourly_df(n: int = 100) -> pd.DataFrame:
    index = pd.date_range("2020-01-01", periods=n, freq="h", name="timestamp")
    return pd.DataFrame({"x": np.arange(n, dtype=float)}, index=index)


def test_split_chrono_order_proportions_no_overlap() -> None:
    df = _hourly_df(100)
    train, val, test = split_chrono(df)

    assert (len(train), len(val), len(test)) == (70, 15, 15)
    assert len(train) + len(val) + len(test) == len(df)
    assert train.index.max() < val.index.min()
    assert val.index.max() < test.index.min()
    for part in (train, val, test):
        assert part.index.is_monotonic_increasing
    assert train.index.intersection(val.index).empty
    assert val.index.intersection(test.index).empty
    # Aucun mélange : la concaténation redonne exactement le jeu d'origine.
    pd.testing.assert_frame_equal(pd.concat([train, val, test]), df)


def test_remove_duplicate_timestamps_keeps_first() -> None:
    index = pd.DatetimeIndex(
        [
            "2020-01-01 00:00",
            "2020-01-01 01:00",
            "2020-01-01 01:00",
            "2020-01-01 02:00",
        ],
        name="timestamp",
    )
    df = pd.DataFrame({"x": [1.0, 2.0, 99.0, 3.0]}, index=index)

    out = remove_duplicate_timestamps(df)

    assert len(out) == 3
    assert not out.index.duplicated().any()
    assert out.loc["2020-01-01 01:00", "x"] == 2.0


def test_select_features_returns_exactly_features() -> None:
    raw = pd.DataFrame(
        {
            "No": [1, 2],
            "year": [2013, 2013],
            "month": [3, 3],
            "day": [1, 1],
            "hour": [1, 0],
            "PM2.5": [4, None],
            "PM10": [4, 8],
            "SO2": [4, 4],
            "NO2": [7, 7],
            "CO": [300, 300],
            "O3": [77, 77],
            "TEMP": [-0.7, "bad"],
            "PRES": [1023, 1023],
            "wd": ["N", "NNW"],
            "station": ["S", "S"],
        }
    )

    out = select_features(build_datetime_index(raw))

    assert list(out.columns) == FEATURES
    assert out.index.name == "timestamp"
    assert out.index.is_monotonic_increasing
    assert all(pd.api.types.is_numeric_dtype(out[c]) for c in FEATURES)
    # La valeur non numérique devient NaN, les négatives sont conservées.
    assert out["temp"].isna().sum() == 1
    assert (out["temp"] < 0).sum() == 1
