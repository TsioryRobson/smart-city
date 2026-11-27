"""Transports des événements IoT."""

import logging
from typing import Protocol

import requests

from novaria.common.schemas import RawMeasurement

LOGGER = logging.getLogger(__name__)


class Transport(Protocol):
    """Interface d'émission réutilisable pour le futur topic mesures.brutes."""

    def send(self, event: RawMeasurement) -> None:
        """Publie un événement issu du rejeu."""
        ...


class ConsoleTransport:
    """Écrit un événement JSON par ligne."""

    def send(self, event: RawMeasurement) -> None:
        """Affiche la mesure avec son étiquette de démonstration."""
        print(event.model_dump_json(), flush=True)


class HttpTransport:
    """Envoie la mesure brute au point d'entrée HTTP /predict."""

    def __init__(self, url: str, timeout_seconds: float = 5) -> None:
        """Configure l'URL et le délai d'attente de chaque requête."""
        self.url = url
        self.timeout_seconds = timeout_seconds

    def send(self, event: RawMeasurement) -> None:
        """Poste les champs du contrat API et affiche sa réponse."""
        if event.injection is not None:
            LOGGER.info("Événement %s : %s", event.event_id, event.injection)
        response = requests.post(
            self.url,
            json=event.model_dump(mode="json", exclude={"injection"}),
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        print(response.text, flush=True)
