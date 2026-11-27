"""Point d'entrée du simulateur Beijing en console ou via HTTP."""

import argparse
import logging
import math
import signal
from pathlib import Path
from types import FrameType

import requests

from novaria.common.config import DEMO_STATION, RAW_DATA_DIR, RAW_FILE_PATTERN
from novaria.simulator.replay import SCENARIOS, Replay, read_measurements
from novaria.simulator.transport import ConsoleTransport, HttpTransport

DEFAULT_URL = "http://127.0.0.1:8000/predict"


def interrupt(_signum: int, _frame: FrameType | None) -> None:
    """Traite SIGTERM comme Ctrl+C."""
    raise KeyboardInterrupt


def main() -> None:
    """Configure le rejeu, envoie les mesures et gère l'arrêt du processus."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, help="CSV brut de la station sélectionnée")
    parser.add_argument("--station", default=DEMO_STATION)
    parser.add_argument("--sensor", default="IoT-003")
    parser.add_argument(
        "--interval", type=float, default=1, help="Secondes entre les envois"
    )
    parser.add_argument("--limit", type=int, help="Nombre maximum de mesures")
    parser.add_argument("--transport", choices=("console", "http"), default="console")
    parser.add_argument("--url", default=DEFAULT_URL, help="URL complète de /predict")
    parser.add_argument(
        "--scenario",
        choices=("reference", *SCENARIOS),
        default="reference",
        help="Scénario à appliquer à la première mesure",
    )
    args = parser.parse_args()
    if not math.isfinite(args.interval) or args.interval < 0:
        parser.error("--interval doit être un nombre fini positif ou nul")
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit doit être strictement positif")

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    csv_path = args.csv or RAW_DATA_DIR / RAW_FILE_PATTERN.replace("*", args.station)
    transport = (
        HttpTransport(args.url) if args.transport == "http" else ConsoleTransport()
    )
    replay = Replay(read_measurements(csv_path, args.station, args.sensor), transport)
    scenario = "pic" if args.scenario == "reference" else args.scenario
    if args.scenario != "reference":
        replay.inject(scenario)

    signal.signal(signal.SIGTERM, interrupt)
    if hasattr(signal, "SIGUSR1"):
        signal.signal(signal.SIGUSR1, lambda *_: replay.inject(scenario))
    try:
        replay.run(args.interval, args.limit)
    except KeyboardInterrupt:
        replay.stop()
    except (OSError, ValueError, requests.RequestException) as exc:
        parser.exit(1, f"Échec du rejeu : {exc}\n")


if __name__ == "__main__":
    main()
