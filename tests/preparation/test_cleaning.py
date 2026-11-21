"""Tests unitaires de novaria.preparation.cleaning sur des données synthétiques."""

import numpy as np
import pandas as pd
import pytest

from novaria.common.config import MEASURES, STATION_COLUMN
from novaria.preparation.cleaning import (
    clean,
    fix_inconsistencies,
    from_raw,
    frozen_sensor_runs,
    remove_duplicates,
    standardize,
    to_local_time,
)
from novaria.preparation.journal import close_data_checks_log, setup_data_checks_log
from tests.factories import measurements, raw_csv_frame


# --- Mise au format ----------------------------------------------------------
def test_from_raw_builds_contract_frame() -> None:
    """Vérifie que ``from_raw`` produit la trame commune, sans rien corriger.

    Contrôle les noms de colonnes du contrat, l'index horaire trié, la
    conversion numérique (« bad » devient NaN) et le fait qu'une valeur
    négative reste inchangée à ce stade.
    """
    raw = raw_csv_frame("S", n=3)
    raw["TEMP"] = raw["TEMP"].astype(object)
    raw.loc[1, "TEMP"] = "bad"
    raw.loc[2, "SO2"] = -4.0

    out = from_raw(raw)

    assert list(out.columns) == [STATION_COLUMN, *MEASURES]
    assert out.index.name == "timestamp"
    assert out.index.is_monotonic_increasing
    assert out["temp"].isna().sum() == 1
    assert (out["so2"] < 0).sum() == 1


def test_standardize_sorts_by_time_then_station_and_keeps_columns() -> None:
    """Vérifie le tri (date puis station), les mesures absentes et ``keep_columns``.

    Une mesure absente du message devient NaN, la direction du vent passe en
    majuscules, et une colonne demandée dans ``keep_columns`` est conservée.
    """
    frame = pd.DataFrame(
        {
            "timestamp": ["2020-01-01T01:00", "2020-01-01T00:00", "2020-01-01T00:00"],
            "station_id": ["A", "B", "A"],
            "pm25": [1.0, 2.0, 3.0],
            "wd": [" ne", "N", None],
            "extra": [10, 20, 30],
        }
    )

    out = standardize(frame, keep_columns=["extra"])

    assert list(out[STATION_COLUMN]) == ["A", "B", "A"]
    assert list(out["extra"]) == [30, 20, 10]
    assert out["pm10"].isna().all()
    assert out["wd"].iloc[2] == "NE"


def test_to_local_time_converts_aware_timestamps() -> None:
    """Vérifie la conversion vers l'heure de Pékin (UTC+8) et l'arrondi à l'heure.

    Un horodatage sans fuseau est gardé tel quel ; un horodatage en UTC ou
    avec un autre fuseau est converti.
    """
    naive = to_local_time(pd.Series(["2016-05-01T14:30:00"]))
    utc = to_local_time(pd.Series(["2016-05-01T06:00:00Z"]))
    mixed = to_local_time(
        pd.Series(["2016-05-01T06:00:00Z", "2016-05-01T09:00:00+03:00"])
    )

    assert naive.iloc[0] == pd.Timestamp("2016-05-01 14:00")
    assert utc.iloc[0] == pd.Timestamp("2016-05-01 14:00")
    assert list(mixed) == [pd.Timestamp("2016-05-01 14:00")] * 2


# --- Correction --------------------------------------------------------------
def test_remove_duplicates_is_per_station() -> None:
    """Vérifie que les doublons sont détectés par couple (station, horodatage).

    Deux stations à la même heure ne sont pas un doublon. Pour un vrai doublon,
    la première occurrence est gardée.
    """
    df = measurements(n=2)
    first_a = df.iloc[[0]].copy()
    first_a["pm25"] = 99.0
    df = pd.concat([df, first_a]).sort_index(kind="stable")

    out = remove_duplicates(df)

    assert len(out) == 4
    assert 99.0 not in out["pm25"].to_numpy()


def test_fix_inconsistencies_sets_impossible_values_to_nan() -> None:
    """Vérifie que seules les valeurs impossibles sont mises à NaN.

    Sont mises à NaN : une concentration négative, une pluie négative, des
    valeurs hors bornes, un couple PM2.5 > PM10 (les deux valeurs) et une
    direction de vent inconnue. Restent inchangés : un pic extrême mais
    possible et une température négative. L'entrée n'est pas modifiée.
    """
    df = measurements(n=10, stations=("A",))
    col = df.columns.get_loc
    df.iloc[0, col("no2")] = -5.0  # concentration négative
    df.iloc[0, col("rain")] = -1.0  # pluie négative
    df.iloc[1, col("o3")] = 5000.0  # hors bornes
    df.iloc[2, col("pres")] = 500.0  # hors bornes
    df.iloc[3, col("pm25")] = 120.0  # PM2.5 > PM10
    df.iloc[3, col("pm10")] = 100.0
    df.iloc[4, col("pm25")] = 900.0  # pic extrême mais possible
    df.iloc[4, col("pm10")] = 950.0
    df.iloc[5, col("temp")] = -15.0  # température négative : valide
    df.iloc[6, col("wd")] = "XYZ"  # direction inconnue

    out = fix_inconsistencies(df)

    assert np.isnan(out.iloc[0]["no2"]) and np.isnan(out.iloc[0]["rain"])
    assert np.isnan(out.iloc[1]["o3"])
    assert np.isnan(out.iloc[2]["pres"])
    assert np.isnan(out.iloc[3]["pm25"]) and np.isnan(out.iloc[3]["pm10"])
    assert pd.isna(out.iloc[6]["wd"])
    assert out.iloc[4]["pm25"] == 900.0 and out.iloc[4]["pm10"] == 950.0
    assert out.iloc[5]["temp"] == -15.0
    assert int(out[MEASURES].isna().sum().sum()) == 7
    assert df.iloc[0]["no2"] == -5.0


def test_frozen_sensor_runs_detects_long_repeats_only() -> None:
    """Vérifie qu'une valeur répétée est signalée seulement au-delà du seuil d'heures.

    Une série de 25 h est signalée avec la bonne station, variable, valeur et
    durée ; une série de 10 h ne l'est pas.
    """
    df = measurements(n=40, stations=("A",))
    df.iloc[5:30, df.columns.get_loc("so2")] = 3.0
    df.iloc[0:10, df.columns.get_loc("no2")] = 7.0

    runs = frozen_sensor_runs(df, min_hours=24)

    assert len(runs) == 1
    run = runs.iloc[0]
    assert (run[STATION_COLUMN], run["column"], run["value"], run["hours"]) == (
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
    df = measurements(n=30, stations=("A",))
    df.iloc[:, df.columns.get_loc("co")] = 500.0
    df = df.drop(df.index[15])

    assert frozen_sensor_runs(df, min_hours=24).empty


def test_clean_writes_every_issue_to_log(tmp_path) -> None:
    """Vérifie que ``clean`` écrit chaque type de problème dans le journal.

    Contrôle les entrées valeur manquante, heure manquante, doublon et
    valeur négative, ainsi que la suppression du doublon.

    Args:
        tmp_path: dossier temporaire fourni par pytest, qui reçoit le journal.
    """
    df = measurements(n=10, stations=("A",))
    df.iloc[0, df.columns.get_loc("pm25")] = np.nan
    df.iloc[1, df.columns.get_loc("so2")] = -1.0
    df = df.drop(df.index[6])
    df = pd.concat([df, df.iloc[[2]]]).sort_index(kind="stable")

    log_path = setup_data_checks_log(tmp_path / "checks.log")
    try:
        out = clean(df)
    finally:
        close_data_checks_log()

    content = log_path.read_text(encoding="utf-8")
    assert "error=valeur_manquante | action=signale" in content
    assert "error=heure_manquante | action=signale" in content
    assert "error=doublon | action=supprime" in content
    assert "column=so2 | value=-1.0 | error=valeur_negative | action=impute" in content
    assert "station=A" in content
    assert len(out) == 9


def test_journal_is_silent_when_not_configured(tmp_path) -> None:
    """Vérifie qu'aucun fichier n'est écrit quand le journal n'est pas configuré.

    En streaming, le journal est optionnel : ``clean`` doit fonctionner sans.

    Args:
        tmp_path: dossier temporaire fourni par pytest.
    """
    df = measurements(n=5, stations=("A",))
    df.iloc[0, df.columns.get_loc("so2")] = -1.0

    out = fix_inconsistencies(df)

    assert np.isnan(out.iloc[0]["so2"])
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("bad", ["not-a-date"])
def test_standardize_rejects_unreadable_timestamp(bad: str) -> None:
    """Vérifie qu'un horodatage illisible lève une erreur au lieu d'être ignoré.

    Args:
        bad: horodatage invalide.
    """
    frame = pd.DataFrame({"timestamp": [bad], "station_id": ["A"]})
    with pytest.raises(ValueError):
        standardize(frame)
