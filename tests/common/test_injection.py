"""Tests unitaires de novaria.common.injection (anomalies synthétiques)."""

import numpy as np
import pandas as pd

from novaria.common.config import (
    NUMERIC_MEASURES,
    PHYSICAL_BOUNDS,
    POLLUTANTS,
    STATION_COLUMN,
)
from novaria.common.injection import (
    FROZEN,
    IMPOSSIBLE,
    SPIKE,
    InjectionConfig,
    inject_anomalies,
)
from novaria.preparation.cleaning import fix_inconsistencies
from tests.factories import measurements

CONFIG = InjectionConfig(
    spike_rate=0.05, impossible_rate=0.02, frozen_runs=3, frozen_hours=(6, 10)
)


def _injected(seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Injecte des anomalies dans des mesures synthétiques de 3 stations sur 10 jours.

    Args:
        seed: graine de l'injection.

    Returns:
        Le triplet ``(données d'origine, données modifiées, étiquettes)``.
    """
    df = measurements(n=240, stations=("A", "B", "C"))
    out, labels = inject_anomalies(df, CONFIG, seed=seed)
    return df, out, labels


def test_counts_and_input_untouched() -> None:
    """Vérifie le nombre de lignes de chaque type, et que l'entrée n'est pas modifiée.

    Les lignes de référence (étiquette ``None``) gardent leurs valeurs d'origine.
    """
    df, out, labels = _injected()
    original = df.copy()

    assert (labels == SPIKE).sum() == round(0.05 * len(df))
    assert (labels == IMPOSSIBLE).sum() == round(0.02 * len(df))
    assert (labels == FROZEN).sum() >= 3 * 5
    pd.testing.assert_frame_equal(df, original)
    reference = labels.isna().to_numpy()
    pd.testing.assert_frame_equal(out[reference], df[reference])


def test_reproducible_with_same_seed() -> None:
    """Vérifie que la même graine donne exactement la même injection, et une autre graine non."""
    _, out1, labels1 = _injected(seed=1)
    _, out2, labels2 = _injected(seed=1)
    _, _, labels3 = _injected(seed=2)

    pd.testing.assert_frame_equal(out1, out2)
    pd.testing.assert_series_equal(labels1, labels2)
    assert not labels1.equals(labels3)


def test_spikes_are_possible_and_high() -> None:
    """Vérifie que les pics restent physiquement possibles mais nettement plus hauts.

    Aucun pic ne dépasse la borne physique ni ne crée PM2.5 > PM10 : le
    nettoyage ne doit pas les effacer. Au moins un polluant augmente.
    """
    df, out, labels = _injected()
    spikes = (labels == SPIKE).to_numpy()

    for col in POLLUTANTS:
        assert (out.loc[spikes, col] <= PHYSICAL_BOUNDS[col][1]).all()
    assert (out.loc[spikes, "pm25"] <= out.loc[spikes, "pm10"]).all()
    increased = (out.loc[spikes, POLLUTANTS] > df.loc[spikes, POLLUTANTS] * 2).any(
        axis=1
    )
    assert increased.all()
    cleaned = fix_inconsistencies(out)
    assert not cleaned.loc[spikes, POLLUTANTS].isna().any().any()


def test_impossible_values_are_removed_by_cleaning() -> None:
    """Vérifie que chaque valeur impossible injectée est mise à NaN par le nettoyage."""
    _, out, labels = _injected()
    impossible = (labels == IMPOSSIBLE).to_numpy()

    cleaned = fix_inconsistencies(out)

    assert cleaned.loc[impossible, POLLUTANTS].isna().any(axis=1).all()


def test_frozen_runs_repeat_one_value_per_station() -> None:
    """Vérifie qu'une série figée répète la valeur de l'heure qui la précède, sur une station.

    Chaque ligne étiquetée ``capteur_fige`` a au moins une mesure égale à celle
    de l'heure précédente de la même station.
    """
    _, out, labels = _injected()

    for station, group in out.groupby(STATION_COLUMN):
        frozen = (labels[out[STATION_COLUMN] == station] == FROZEN).to_numpy()
        if not frozen.any():
            continue
        repeated = (group[NUMERIC_MEASURES].diff(1) == 0).any(axis=1)
        assert repeated.to_numpy()[frozen].all()


def test_each_row_has_at_most_one_anomaly() -> None:
    """Vérifie qu'aucune ligne ne cumule deux types d'anomalies.

    Les étiquettes sont uniques par ligne par construction ; on vérifie que
    leurs effectifs s'additionnent bien.
    """
    _, _, labels = _injected()
    counts = labels.value_counts()

    assert counts.sum() == labels.notna().sum()
    assert set(counts.index) <= {SPIKE, IMPOSSIBLE, FROZEN}
    assert np.isin(labels.dropna().unique(), [SPIKE, IMPOSSIBLE, FROZEN]).all()
