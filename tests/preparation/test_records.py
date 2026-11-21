"""Tests unitaires de novaria.preparation.records (préparation pour le streaming)."""

import math

import pandas as pd
import pytest

from novaria.common.config import MODEL_FEATURES
from novaria.preparation.cleaning import fix_inconsistencies
from novaria.preparation.features import add_features
from novaria.preparation.records import prepare_record, prepare_records
from tests.factories import contract_message, measurements


def test_prepare_record_keeps_other_fields_and_adds_features() -> None:
    """Vérifie qu'un message valide garde ses champs et reçoit les variables dérivées.

    ``sensor_id``, ``timestamp`` d'origine et ``injection`` sont recopiés tels
    quels ; les valeurs sont des types JSON natifs.
    """
    message = contract_message(injection="pic")

    out = prepare_record(message)

    assert out["sensor_id"] == "IoT-003"
    assert out["timestamp"] == message["timestamp"]
    assert out["injection"] == "pic"
    assert out["pm25"] == 85.0
    assert set(MODEL_FEATURES) <= set(out)
    assert all(isinstance(out[f], float) for f in MODEL_FEATURES)


def test_prepare_record_sets_impossible_values_to_none() -> None:
    """Vérifie que les valeurs impossibles deviennent ``None`` dans le message préparé.

    PM2.5 > PM10 invalide les deux valeurs ; une direction inconnue donne
    ``wd``, ``wd_sin`` et ``wd_cos`` à ``None``.
    """
    out = prepare_record(contract_message(pm25=130.0, pm10=120.0, wd="XYZ"))

    assert out["pm25"] is None and out["pm10"] is None
    assert out["wd"] is None and out["wd_sin"] is None
    assert out["no2"] == 60.0


def test_prepare_record_converts_utc_to_local_hour() -> None:
    """Vérifie que l'heure encodée est l'heure locale de Pékin.

    06:00 UTC correspond à 14:00 à Pékin : les variables horaires doivent
    être celles de 14 h.
    """
    utc = prepare_record(contract_message(timestamp="2016-05-01T06:00:00Z"))
    local = prepare_record(contract_message(timestamp="2016-05-01T14:00:00"))

    assert math.isclose(utc["hour_sin"], local["hour_sin"])
    assert math.isclose(utc["hour_cos"], local["hour_cos"])


def test_prepare_records_keeps_order_and_handles_empty() -> None:
    """Vérifie que l'ordre des messages est conservé, même s'ils ne sont pas triés.

    Une liste vide donne une liste vide.
    """
    messages = [
        contract_message(sensor_id="late", timestamp="2016-05-01T15:00:00"),
        contract_message(sensor_id="early", timestamp="2016-05-01T13:00:00"),
    ]

    out = prepare_records(messages)

    assert [m["sensor_id"] for m in out] == ["late", "early"]
    assert prepare_records([]) == []


def test_prepare_record_requires_timestamp() -> None:
    """Vérifie qu'un message sans horodatage est refusé."""
    message = contract_message()
    del message["timestamp"]
    with pytest.raises(KeyError):
        prepare_record(message)


def test_streaming_matches_batch() -> None:
    """Vérifie qu'un message est préparé exactement comme la même ligne en batch.

    C'est la garantie de la règle 9 (un seul module de préparation) : aucun
    écart entre entraînement et streaming.
    """
    df = measurements(n=6)
    df.iloc[0, df.columns.get_loc("so2")] = -3.0
    df.iloc[1, df.columns.get_loc("pm25")] = 500.0
    batch = add_features(fix_inconsistencies(df))

    messages = df.reset_index().assign(
        timestamp=lambda d: d["timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%S")
    )
    streamed = pd.DataFrame(prepare_records(messages.to_dict("records")))

    for feature in MODEL_FEATURES:
        expected = batch[feature].to_numpy(dtype=float)
        actual = streamed[feature].to_numpy(dtype=float)
        pd.testing.assert_series_equal(
            pd.Series(expected), pd.Series(actual), check_names=False
        )
