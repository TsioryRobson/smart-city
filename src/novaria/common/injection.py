"""Générateur d'anomalies synthétiques injectées et identifiées.

Les anomalies réelles ne sont pas étiquetées (CLAUDE.md, règle 3) : on évalue
les modèles sur des données réelles de référence dans lesquelles on injecte
des anomalies connues. Trois types, dont les noms servent aussi de valeur au
champ ``injection`` des messages du simulateur :

- ``pic`` : pic de pollution, physiquement possible mais atypique
  (rôle du modèle) ;
- ``valeur_impossible`` : valeur négative, hors bornes ou PM2.5 > PM10
  (rôle du nettoyage, qui la met à NaN) ;
- ``capteur_fige`` : même valeur répétée plusieurs heures (rôle de la règle
  « capteur figé »).

Règle 4 : ces données ne servent JAMAIS à l'entraînement, seulement à la
validation et au test contrôlés.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from novaria.common.config import (
    FROZEN_SENSOR_MEASURES,
    PHYSICAL_BOUNDS,
    POLLUTANTS,
    STATION_COLUMN,
)
from novaria.preparation.cleaning import fix_inconsistencies

INJECTION_COLUMN = "injection"
SPIKE = "pic"
IMPOSSIBLE = "valeur_impossible"
FROZEN = "capteur_fige"
INJECTION_TYPES = (SPIKE, IMPOSSIBLE, FROZEN)


@dataclass(frozen=True)
class InjectionConfig:
    """Paramètres de l'injection.

    Attributes:
        spike_rate: part des lignes recevant un pic de pollution.
        impossible_rate: part des lignes recevant une valeur impossible.
        frozen_runs: nombre de séries de capteur figé.
        frozen_hours: durée minimale et maximale d'une série, en heures.
    """

    spike_rate: float = 0.02
    impossible_rate: float = 0.005
    frozen_runs: int = 12
    frozen_hours: tuple[int, int] = (24, 48)


def _spike_value(
    value: float, column: str, reference_p99: float, rng: np.random.Generator
) -> float:
    """Calcule la valeur d'un pic : nettement au-dessus de l'habituel, mais possible.

    Le pic vaut le maximum entre 3 à 6 fois la valeur d'origine et 1 à 1,5 fois
    le 99e centile de la variable, plafonné à 99 % de la borne physique haute
    pour rester une valeur possible (donc non corrigée par le nettoyage).

    Args:
        value: valeur d'origine (NaN accepté, remplacé par le centile).
        column: variable concernée.
        reference_p99: 99e centile de la variable sur les données de référence.
        rng: générateur aléatoire.

    Returns:
        La valeur du pic.
    """
    base = 0.0 if np.isnan(value) else value
    spike = max(base * rng.uniform(3, 6), reference_p99 * rng.uniform(1.0, 1.5))
    return min(spike, 0.99 * PHYSICAL_BOUNDS[column][1])


def inject_spikes(
    df: pd.DataFrame, labels: pd.Series, n: int, rng: np.random.Generator
) -> None:
    """Injecte ``n`` pics de pollution sur des lignes encore non modifiées (sur place).

    Chaque pic touche 1 à 3 polluants. Si PM2.5 est touché, PM10 l'est aussi,
    avec un pic au moins aussi grand, pour ne pas créer de PM2.5 > PM10.

    Args:
        df: trame de mesures à modifier (positions entières).
        labels: étiquettes par ligne, mises à jour avec ``pic``.
        n: nombre de lignes à modifier.
        rng: générateur aléatoire.
    """
    p99 = df[POLLUTANTS].quantile(0.99)
    free = np.flatnonzero(labels.isna().to_numpy())
    for row in rng.choice(free, size=min(n, len(free)), replace=False):
        columns = list(rng.choice(POLLUTANTS, size=rng.integers(1, 4), replace=False))
        if "pm25" in columns and "pm10" not in columns:
            columns.append("pm10")
        for col in columns:
            j = df.columns.get_loc(col)
            df.iat[row, j] = _spike_value(df.iat[row, j], col, p99[col], rng)
        if "pm25" in columns:
            i25, i10 = df.columns.get_loc("pm25"), df.columns.get_loc("pm10")
            df.iat[row, i10] = max(df.iat[row, i10], df.iat[row, i25])
        labels.iat[row] = SPIKE


def inject_impossible_values(
    df: pd.DataFrame, labels: pd.Series, n: int, rng: np.random.Generator
) -> None:
    """Injecte ``n`` valeurs impossibles sur des lignes encore non modifiées (sur place).

    Une valeur impossible est, au choix : une concentration négative, une
    valeur 1,5 fois au-dessus de la borne physique haute, ou PM2.5 > PM10.

    Args:
        df: trame de mesures à modifier (positions entières).
        labels: étiquettes par ligne, mises à jour avec ``valeur_impossible``.
        n: nombre de lignes à modifier.
        rng: générateur aléatoire.
    """
    free = np.flatnonzero(labels.isna().to_numpy())
    for row in rng.choice(free, size=min(n, len(free)), replace=False):
        kind = rng.integers(3)
        col = str(rng.choice(POLLUTANTS))
        j = df.columns.get_loc(col)
        if kind == 0:
            df.iat[row, j] = -rng.uniform(1, 50)
        elif kind == 1:
            df.iat[row, j] = PHYSICAL_BOUNDS[col][1] * 1.5
        else:
            i25, i10 = df.columns.get_loc("pm25"), df.columns.get_loc("pm10")
            pm10 = df.iat[row, i10] if not np.isnan(df.iat[row, i10]) else 50.0
            df.iat[row, i25] = pm10 * rng.uniform(1.5, 3)
        labels.iat[row] = IMPOSSIBLE


def inject_frozen_runs(
    df: pd.DataFrame,
    labels: pd.Series,
    n_runs: int,
    hours: tuple[int, int],
    rng: np.random.Generator,
) -> None:
    """Injecte ``n_runs`` séries de capteur figé (sur place).

    Pour une station et une variable tirées au hasard, la valeur d'une heure
    est recopiée sur les heures suivantes. La première heure est normale ; les
    suivantes sont étiquetées ``capteur_fige``. Une série ne chevauche aucune
    autre ligne déjà modifiée.

    Args:
        df: trame de mesures triée par date (positions entières).
        labels: étiquettes par ligne, mises à jour avec ``capteur_fige``.
        n_runs: nombre de séries à injecter.
        hours: durée minimale et maximale d'une série, en heures.
        rng: générateur aléatoire.
    """
    positions = {
        station: np.flatnonzero((df[STATION_COLUMN] == station).to_numpy())
        for station in df[STATION_COLUMN].dropna().unique()
    }
    stations = list(positions)
    for _ in range(n_runs):
        for _attempt in range(20):
            rows = positions[stations[rng.integers(len(stations))]]
            length = int(rng.integers(hours[0], hours[1] + 1))
            if len(rows) <= length:
                continue
            start = int(rng.integers(0, len(rows) - length))
            run = rows[start : start + length]
            col = str(rng.choice(FROZEN_SENSOR_MEASURES))
            j = df.columns.get_loc(col)
            value = df.iat[run[0], j]
            if labels.iloc[run].notna().any() or np.isnan(value):
                continue
            for row in run[1:]:
                df.iat[row, j] = value
                labels.iat[row] = FROZEN
            break


def inject_anomalies(
    df: pd.DataFrame, config: InjectionConfig, seed: int = 0
) -> tuple[pd.DataFrame, pd.Series]:
    """Injecte des anomalies identifiées dans une copie des données de référence.

    Les séries de capteur figé sont injectées en premier (elles ont besoin de
    lignes consécutives libres), puis les valeurs impossibles et les pics.
    Une ligne ne reçoit qu'une seule anomalie. Le résultat est reproductible
    pour une même graine.

    Args:
        df: trame de mesures nettoyée, triée par date (``station_id`` + mesures).
        config: taux et durées d'injection.
        seed: graine du générateur aléatoire.

    Returns:
        Le couple ``(données modifiées, étiquettes)`` : les étiquettes valent
        ``None`` pour une ligne de référence, sinon le type d'anomalie. L'entrée
        n'est pas modifiée.
    """
    rng = np.random.default_rng(seed)
    out = df.copy()
    labels = pd.Series(
        [None] * len(out), index=out.index, dtype="object", name=INJECTION_COLUMN
    )
    inject_frozen_runs(out, labels, config.frozen_runs, config.frozen_hours, rng)
    inject_impossible_values(out, labels, round(config.impossible_rate * len(out)), rng)
    inject_spikes(out, labels, round(config.spike_rate * len(out)), rng)
    return out, labels


def injection_config_from_dict(values: dict) -> InjectionConfig:
    """Construit les paramètres d'injection à partir d'une section de configuration YAML.

    Args:
        values: clés ``spike_rate``, ``impossible_rate``, ``frozen_runs`` et
            ``frozen_hours`` (liste de deux entiers).

    Returns:
        Les paramètres d'injection.
    """
    return InjectionConfig(**{**values, "frozen_hours": tuple(values["frozen_hours"])})


def build_controlled_set(
    df: pd.DataFrame, config: InjectionConfig, seed: int
) -> tuple[pd.DataFrame, pd.Series]:
    """Construit un jeu contrôlé : données réelles préparées + anomalies injectées.

    Les anomalies sont injectées sur les mesures, puis le même nettoyage
    qu'en production est réappliqué (``fix_inconsistencies``) : une valeur
    impossible injectée devient NaN, comme elle le deviendrait en streaming.
    Les variables dérivées (vent, heure, mois) ne dépendent pas des mesures
    modifiées et restent valables.

    Args:
        df: jeu préparé (validation ou test) indexé par ``timestamp``.
        config: taux et durées d'injection.
        seed: graine de l'injection.

    Returns:
        Le couple ``(jeu contrôlé, étiquettes)`` ; une étiquette vaut ``None``
        pour une ligne de référence, sinon le type d'anomalie.
    """
    injected, labels = inject_anomalies(df, config, seed=seed)
    return fix_inconsistencies(injected), labels
