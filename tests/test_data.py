"""Tests unitaires de src.data sur des données synthétiques."""

import logging

import numpy as np
import pandas as pd
import pytest

from src.config import FEATURES, STATION_COLUMN
from src.data import (
    build_datetime_index,
    clean,
    count_missing_hours,
    fix_inconsistencies,
    frozen_sensor_runs,
    load_all_stations,
    missing_hours_by_station,
    missing_summary,
    quality_report,
    remove_duplicates,
    select_features,
    setup_data_checks_log,
    split_chrono,
)

LOGGER = "smart_city.data_checks"


def _clean_df(
    n: int = 48,
    stations: tuple[str, ...] = ("A", "B"),
    start: str = "2020-01-01",
) -> pd.DataFrame:
    """Construit des données horaires valides (aucune incohérence) sur plusieurs stations.

    Les valeurs sont tirées au hasard (graine fixe) dans des plages réalistes,
    avec PM2.5 toujours inférieur à PM10, pour que chaque test n'introduise que
    les problèmes qu'il veut vérifier.

    Args:
        n: nombre d'heures par station.
        stations: noms des stations à générer.
        start: premier horodatage de chaque station.

    Returns:
        Un DataFrame indexé par ``timestamp``, trié par date, avec les colonnes
        ``station`` et ``FEATURES``.
    """
    rng = np.random.default_rng(0)
    parts = []
    for station in stations:
        index = pd.date_range(start, periods=n, freq="h", name="timestamp")
        values = {
            "pm25": rng.uniform(10, 50, n),
            "pm10": rng.uniform(60, 100, n),
            "no2": rng.uniform(10, 50, n),
            "co": rng.uniform(300, 900, n),
            "so2": rng.uniform(1, 20, n),
            "o3": rng.uniform(10, 80, n),
            "temp": rng.uniform(-10, 30, n),
        }
        parts.append(pd.DataFrame({STATION_COLUMN: station, **values}, index=index))
    return pd.concat(parts).sort_index(kind="stable")


def _raw_df() -> pd.DataFrame:
    """Construit un petit extrait de CSV brut, avec les noms de colonnes d'origine.

    Il contient une valeur manquante (PM2.5), une valeur non numérique (TEMP),
    une température négative et deux stations à la même heure.

    Returns:
        Un DataFrame brut de 3 lignes, au format des CSV du dataset.
    """
    return pd.DataFrame(
        {
            "No": [1, 2, 3],
            "year": [2013, 2013, 2013],
            "month": [3, 3, 3],
            "day": [1, 1, 1],
            "hour": [1, 0, 0],
            "PM2.5": [4, None, 5],
            "PM10": [4, 8, 9],
            "SO2": [4, 4, 4],
            "NO2": [7, 7, 7],
            "CO": [300, 300, 300],
            "O3": [77, 77, 77],
            "TEMP": [-0.7, "bad", 1.0],
            "PRES": [1023, 1023, 1023],
            "wd": ["N", "NNW", "N"],
            "station": ["S", "S", "R"],
        }
    )


# --- Chargement --------------------------------------------------------------
def test_select_features_keeps_station_and_features() -> None:
    """Vérifie que ``select_features`` garde la station et les variables, sans rien corriger.

    Contrôle l'ordre des colonnes, le tri par date puis station, la conversion
    numérique, et le fait que la température négative reste inchangée.
    """
    out = select_features(build_datetime_index(_raw_df()))

    assert list(out.columns) == [STATION_COLUMN, *FEATURES]
    assert out.index.name == "timestamp"
    assert out.index.is_monotonic_increasing
    # Même horodatage : trié par station.
    assert list(out[STATION_COLUMN]) == ["R", "S", "S"]
    assert all(pd.api.types.is_numeric_dtype(out[c]) for c in FEATURES)
    # La valeur non numérique devient NaN ; select_features ne corrige rien d'autre.
    assert out["temp"].isna().sum() == 1
    assert (out["temp"] < 0).sum() == 1


def test_load_all_stations_concatenates_csvs(tmp_path) -> None:
    """Vérifie que ``load_all_stations`` concatène tous les CSV de station du dossier.

    Un fichier qui ne correspond pas au motif des CSV doit être ignoré.

    Args:
        tmp_path: dossier temporaire fourni par pytest.
    """
    raw = _raw_df()
    for station in ("A", "B"):
        raw.assign(station=station).to_csv(
            tmp_path / f"PRSA_Data_{station}_20130301-20170228.csv", index=False
        )
    (tmp_path / "other.csv").write_text("ignored")

    out = load_all_stations(tmp_path)

    assert len(out) == 2 * len(raw)
    assert set(out["station"]) == {"A", "B"}


def test_load_all_stations_raises_without_csv(tmp_path) -> None:
    """Vérifie qu'un dossier sans CSV de station lève ``FileNotFoundError``.

    Args:
        tmp_path: dossier temporaire vide fourni par pytest.
    """
    with pytest.raises(FileNotFoundError):
        load_all_stations(tmp_path)


# --- Contrôle qualité --------------------------------------------------------
def test_count_missing_hours() -> None:
    """Vérifie le comptage des heures absentes, y compris sur un index vide."""
    index = pd.DatetimeIndex(["2020-01-01 00:00", "2020-01-01 03:00"])
    assert count_missing_hours(index) == 2
    assert count_missing_hours(pd.DatetimeIndex([])) == 0


def test_missing_hours_by_station_is_per_station() -> None:
    """Vérifie que les heures manquantes sont comptées séparément pour chaque station.

    Une heure retirée à la station B ne doit pas être comptée pour la station A.
    """
    df = _clean_df(n=10)
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
    négative et l'absence d'heure manquante.
    """
    df = _clean_df(n=30)
    df.iloc[0, df.columns.get_loc("pm25")] = np.nan
    df.iloc[1, df.columns.get_loc("so2")] = -1.0
    df = pd.concat([df, df.iloc[[2]]])

    report = quality_report(df, verbose=False)

    assert report["stations"] == ["A", "B"]
    assert report["duplicates"] == 1
    assert report["missing"]["pm25"]["count"] == 1
    assert report["inconsistencies"]["valeur_negative"]["so2"] == 1
    assert report["missing_hours"] == {"A": 0, "B": 0}


# --- Vérification et correction ----------------------------------------------
def test_remove_duplicates_is_per_station() -> None:
    """Vérifie que les doublons sont détectés par couple (station, horodatage).

    Deux stations à la même heure ne sont pas un doublon. Pour un vrai doublon,
    la première occurrence est gardée.
    """
    index = pd.DatetimeIndex(
        ["2020-01-01 00:00", "2020-01-01 00:00", "2020-01-01 00:00", "2020-01-01 01:00"],
        name="timestamp",
    )
    df = pd.DataFrame(
        {STATION_COLUMN: ["A", "B", "A", "A"], "x": [1.0, 2.0, 99.0, 3.0]}, index=index
    )

    out = remove_duplicates(df)

    # Même heure, stations différentes : pas un doublon.
    assert len(out) == 3
    assert list(out[STATION_COLUMN]) == ["A", "B", "A"]
    # La première occurrence est gardée.
    assert 99.0 not in out["x"].to_numpy()


def test_fix_inconsistencies_sets_impossible_values_to_nan() -> None:
    """Vérifie que seules les valeurs impossibles sont mises à NaN.

    Sont mises à NaN : une concentration négative, des valeurs hors bornes et
    un couple PM2.5 > PM10 (les deux valeurs). Restent inchangés : un pic
    extrême mais possible et une température négative. L'entrée n'est pas modifiée.
    """
    df = _clean_df(n=10, stations=("A",))
    col = df.columns.get_loc
    df.iloc[0, col("no2")] = -5.0  # concentration négative
    df.iloc[1, col("o3")] = 5000.0  # hors bornes
    df.iloc[2, col("temp")] = 80.0  # hors bornes
    df.iloc[3, col("pm25")] = 120.0  # PM2.5 > PM10
    df.iloc[3, col("pm10")] = 100.0
    df.iloc[4, col("pm25")] = 900.0  # pic extrême mais possible
    df.iloc[4, col("pm10")] = 950.0
    df.iloc[5, col("temp")] = -15.0  # température négative : valide

    out = fix_inconsistencies(df)

    assert np.isnan(out.iloc[0]["no2"])
    assert np.isnan(out.iloc[1]["o3"])
    assert np.isnan(out.iloc[2]["temp"])
    assert np.isnan(out.iloc[3]["pm25"]) and np.isnan(out.iloc[3]["pm10"])
    assert out.iloc[4]["pm25"] == 900.0 and out.iloc[4]["pm10"] == 950.0
    assert out.iloc[5]["temp"] == -15.0
    assert out[FEATURES].isna().sum().sum() == 5
    # L'entrée n'est pas modifiée.
    assert df.iloc[0]["no2"] == -5.0


def test_frozen_sensor_runs_detects_long_repeats_only() -> None:
    """Vérifie qu'une valeur répétée est signalée seulement au-delà du seuil d'heures.

    Une série de 25 h est signalée avec la bonne station, variable, valeur et
    durée ; une série de 10 h ne l'est pas.
    """
    df = _clean_df(n=40, stations=("A",))
    df.iloc[5:30, df.columns.get_loc("so2")] = 3.0  # 25 h figé
    df.iloc[0:10, df.columns.get_loc("no2")] = 7.0  # 10 h : trop court

    runs = frozen_sensor_runs(df, min_hours=24)

    assert len(runs) == 1
    run = runs.iloc[0]
    assert (run["station"], run["column"], run["value"], run["hours"]) == (
        "A",
        "so2",
        3.0,
        25,
    )


def test_frozen_sensor_runs_broken_by_missing_hour() -> None:
    """Vérifie qu'une heure absente coupe une série de capteur figé.

    30 h de valeur constante avec un trou au milieu donnent deux séries trop
    courtes, donc aucun signalement.
    """
    df = _clean_df(n=30, stations=("A",))
    df.iloc[:, df.columns.get_loc("co")] = 500.0
    df = df.drop(df.index[15])

    assert frozen_sensor_runs(df, min_hours=24).empty


def test_clean_writes_every_issue_to_log(tmp_path) -> None:
    """Vérifie que ``clean`` écrit chaque type de problème dans le journal.

    Contrôle la présence des entrées valeur manquante, heure manquante,
    doublon et valeur négative, et la suppression du doublon.

    Args:
        tmp_path: dossier temporaire fourni par pytest, qui reçoit le journal.
    """
    df = _clean_df(n=10, stations=("A",))
    df.iloc[0, df.columns.get_loc("pm25")] = np.nan
    df.iloc[1, df.columns.get_loc("so2")] = -1.0
    df = df.drop(df.index[6])
    df = pd.concat([df, df.iloc[[2]]]).sort_index(kind="stable")

    log_path = setup_data_checks_log(tmp_path / "checks.log")
    try:
        out = clean(df)
    finally:
        logger = logging.getLogger(LOGGER)
        for handler in list(logger.handlers):
            handler.close()
            logger.removeHandler(handler)

    content = log_path.read_text(encoding="utf-8")
    assert "error=valeur_manquante | action=signale" in content
    assert "error=heure_manquante | action=signale" in content
    assert "error=doublon | action=supprime" in content
    assert "column=so2 | value=-1.0 | error=valeur_negative | action=impute" in content
    assert "station=A" in content
    assert len(out) == 9


# --- Découpage ---------------------------------------------------------------
def _split_bounds() -> dict[str, pd.Timestamp]:
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
    df = _clean_df(n=24 * 8)
    train, val, test = split_chrono(df, **_split_bounds())

    # 3 jours, 1 jour, 1 jour, sur 2 stations.
    assert (len(train), len(val), len(test)) == (3 * 48, 48, 48)
    assert train.index.min() == pd.Timestamp("2020-01-02")
    assert train.index.max() < val.index.min()
    assert val.index.max() < test.index.min()
    for part in (train, val, test):
        assert part.index.is_monotonic_increasing
        assert set(part[STATION_COLUMN]) == {"A", "B"}


def test_split_chrono_excludes_demo_period() -> None:
    """Vérifie que le test ne contient aucune donnée de la période de démo."""
    df = _clean_df(n=24 * 8)
    _, _, test = split_chrono(df, **_split_bounds())

    assert not (test.index >= pd.Timestamp("2020-01-07")).any()


def test_split_chrono_rejects_unordered_bounds() -> None:
    """Vérifie que des bornes de découpage dans le désordre lèvent une erreur."""
    bounds = _split_bounds() | {"val_start": pd.Timestamp("2020-01-01")}
    with pytest.raises(AssertionError):
        split_chrono(_clean_df(), **bounds)
