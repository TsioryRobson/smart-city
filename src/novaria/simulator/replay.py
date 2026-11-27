"""Rejeu chronologique de la période Beijing réservée à la démonstration."""

from collections import deque
from collections.abc import Iterable, Iterator
from itertools import islice
from pathlib import Path
from threading import Event

import pandas as pd

from novaria.common.config import DEMO_END, DEMO_START, MEASURES, STATION_COLUMN
from novaria.common.injection import IMPOSSIBLE, SPIKE
from novaria.common.schemas import Injection, RawMeasurement
from novaria.preparation.cleaning import from_raw
from novaria.simulator.transport import Transport

# Profils fixes pour reproduire une démo. Les profils plausibles restent
# dans les bornes physiques et gardent PM2.5 <= PM10.
SCENARIOS: dict[Injection, dict[str, float | None]] = {
    SPIKE: {"pm25": 850, "pm10": 950},
    IMPOSSIBLE: {"pm25": -5},
    "combinaison_inhabituelle": {
        "pm25": 5,
        "pm10": 10,
        "no2": 300,
        "co": 8000,
        "so2": 100,
        "o3": 250,
    },
    "valeur_manquante": {"pm25": None},
}


def read_measurements(
    csv_path: Path, station: str, sensor: str
) -> Iterator[RawMeasurement]:
    """Lit le CSV et produit les mesures de la station dans la fenêtre de démonstration."""
    frame = from_raw(pd.read_csv(csv_path))
    rows = frame.loc[
        (frame.index >= DEMO_START)
        & (frame.index < DEMO_END)
        & (frame[STATION_COLUMN] == station)
    ]
    if rows.empty:
        raise ValueError(
            f"Aucune mesure de démonstration pour {station} dans {csv_path}"
        )

    for timestamp, row in rows.iterrows():
        values = {name: None if pd.isna(row[name]) else row[name] for name in MEASURES}
        yield RawMeasurement(
            station_id=station, sensor_id=sensor, timestamp=timestamp, **values
        )


class Replay:
    """Émet des mesures via un transport et applique les injections en attente."""

    def __init__(
        self, measurements: Iterable[RawMeasurement], transport: Transport
    ) -> None:
        """Associe la source de mesures et le transport d'émission."""
        self.measurements = measurements
        self.transport = transport
        self.pending: deque[Injection] = deque()
        self.stopped = Event()

    def inject(self, scenario: Injection) -> None:
        """Programme une injection sur la prochaine mesure disponible."""
        if scenario not in SCENARIOS:
            raise ValueError(f"Scénario inconnu : {scenario}")
        self.pending.append(scenario)

    def stop(self) -> None:
        """Interrompt l'attente et arrête l'émission avant la prochaine mesure."""
        self.stopped.set()

    def run(self, interval_seconds: float, limit: int | None = None) -> None:
        """Émet immédiatement la première mesure, puis attend entre les envois."""
        rows = islice(self.measurements, limit)
        for index, event in enumerate(rows):
            if self.stopped.is_set() or (
                index > 0 and self.stopped.wait(interval_seconds)
            ):
                break
            if self.pending:
                scenario = self.pending.popleft()
                event = event.model_copy(
                    update={**SCENARIOS[scenario], "injection": scenario}
                )
            self.transport.send(event)
