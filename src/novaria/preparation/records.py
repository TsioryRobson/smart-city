"""Préparation de messages unitaires, pour le streaming (job PyFlink, sprint 2).

Un message suit le contrat ``mesures.brutes`` (mêmes champs que la requête
``/predict``). La préparation appelle exactement les fonctions sans état du
batch (``standardize``, ``fix_inconsistencies``, ``add_features``) : aucune
logique n'est dupliquée, donc un message est préparé comme la même ligne
l'aurait été à l'entraînement.

Ce qui n'est PAS fait ici, car il faut l'historique de la station :
suppression des doublons, heures manquantes et capteur figé. En streaming,
ces contrôles demanderont un état par station (``keyed state`` Flink).
L'imputation et la normalisation sont faites par le pipeline du modèle.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from typing import Any

import pandas as pd

from novaria.common.config import MEASURES, MODEL_FEATURES, TIMESTAMP_COLUMN
from novaria.preparation.cleaning import fix_inconsistencies, standardize
from novaria.preparation.features import add_features

_ROW = "_row"


def _to_json_value(value: Any) -> Any:
    """Convertit une valeur pandas / numpy en valeur JSON native.

    Args:
        value: valeur issue d'un DataFrame.

    Returns:
        ``None`` pour une valeur manquante, un ``float`` pour un nombre numpy,
        la valeur inchangée sinon.
    """
    if value is None or value is pd.NA:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    if hasattr(value, "item"):
        return value.item()
    return value


def prepare_records(records: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Nettoie un lot de messages et ajoute les variables dérivées.

    Pour chaque message : les mesures impossibles deviennent ``None`` (elles
    seront imputées par le modèle), les variables dérivées de
    ``MODEL_FEATURES`` sont ajoutées, et tous les autres champs (``sensor_id``,
    ``timestamp`` d'origine, ``injection``...) sont recopiés tels quels.
    L'ordre des messages est conservé. Les corrections sont journalisées si
    le journal de vérification est configuré.

    Args:
        records: messages au format du contrat ``mesures.brutes``.

    Returns:
        Les messages préparés (contrat ``mesures.propres``), un dict par message.
    """
    originals = [dict(record) for record in records]
    if not originals:
        return []
    frame = pd.DataFrame(originals)
    frame[_ROW] = range(len(originals))

    prepared = add_features(
        fix_inconsistencies(standardize(frame, keep_columns=[_ROW]))
    )
    prepared = prepared.sort_values(_ROW, kind="stable")

    output_columns = [*MEASURES, *[f for f in MODEL_FEATURES if f not in MEASURES]]
    results = []
    for original, (_, row) in zip(originals, prepared.iterrows(), strict=True):
        result = {**original}
        for col in output_columns:
            result[col] = _to_json_value(row[col])
        results.append(result)
    return results


def prepare_record(record: Mapping[str, Any]) -> dict[str, Any]:
    """Nettoie un seul message et ajoute les variables dérivées.

    Raccourci de ``prepare_records`` pour un message isolé.

    Args:
        record: message au format du contrat ``mesures.brutes``, avec au
            moins ``timestamp`` et ``station_id``.

    Returns:
        Le message préparé (contrat ``mesures.propres``).

    Raises:
        KeyError: si ``timestamp`` est absent.
    """
    if TIMESTAMP_COLUMN not in record:
        raise KeyError(f"Champ obligatoire absent : {TIMESTAMP_COLUMN}")
    return prepare_records([record])[0]
