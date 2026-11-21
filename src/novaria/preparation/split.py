"""Découpage chronologique train / validation / test (jamais aléatoire)."""

from __future__ import annotations

import pandas as pd

from novaria.common.config import DEMO_START, TEST_START, TRAIN_START, VAL_START


def split_chrono(
    df: pd.DataFrame,
    train_start: pd.Timestamp = TRAIN_START,
    val_start: pd.Timestamp = VAL_START,
    test_start: pd.Timestamp = TEST_START,
    test_end: pd.Timestamp = DEMO_START,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Découpe le jeu en train / val / test par dates, sans mélange.

    Chaque jeu couvre ``[début, fin[``. Les données avant ``train_start`` et à
    partir de ``test_end`` (période rejouée par le simulateur) sont écartées.

    Args:
        df: DataFrame indexé par ``timestamp``.
        train_start: début du train.
        val_start: début de la validation (fin du train).
        test_start: début du test (fin de la validation).
        test_end: fin du test, exclue.

    Returns:
        Le triplet ``(train, val, test)``.

    Raises:
        AssertionError: si les bornes ne sont pas ordonnées, si un jeu est
            vide ou si deux jeux se chevauchent.
    """
    assert train_start < val_start < test_start < test_end, "Bornes non ordonnées"
    index = df.index
    train = df[(index >= train_start) & (index < val_start)]
    val = df[(index >= val_start) & (index < test_start)]
    test = df[(index >= test_start) & (index < test_end)]

    assert len(train) and len(val) and len(test), "Un des jeux est vide"
    assert train.index.max() < val.index.min(), "Chevauchement train / val"
    assert val.index.max() < test.index.min(), "Chevauchement val / test"
    assert test.index.max() < test_end, "Le test empiète sur la période de démo"
    return train, val, test
