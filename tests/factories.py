"""Générateurs de données synthétiques partagés par les tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from novaria.common.config import STATION_COLUMN, TIMESTAMP_COLUMN


def measurements(
    n: int = 48,
    stations: tuple[str, ...] = ("A", "B"),
    start: str = "2020-01-01",
    seed: int = 0,
) -> pd.DataFrame:
    """Construit une trame de mesures valides (aucune incohérence) sur plusieurs stations.

    Les valeurs sont tirées au hasard (graine fixe) dans des plages réalistes,
    avec PM2.5 toujours inférieur à PM10, pour que chaque test n'introduise que
    les problèmes qu'il veut vérifier.

    Args:
        n: nombre d'heures par station.
        stations: noms des stations à générer.
        start: premier horodatage de chaque station.
        seed: graine du générateur aléatoire.

    Returns:
        Une trame indexée par ``timestamp``, triée par date puis station, avec
        ``station_id`` et les 12 mesures du contrat.
    """
    rng = np.random.default_rng(seed)
    parts = []
    for station in stations:
        index = pd.date_range(start, periods=n, freq="h", name=TIMESTAMP_COLUMN)
        values = {
            "pm25": rng.uniform(10, 50, n),
            "pm10": rng.uniform(60, 100, n),
            "no2": rng.uniform(10, 50, n),
            "co": rng.uniform(300, 900, n),
            "so2": rng.uniform(1, 20, n),
            "o3": rng.uniform(10, 80, n),
            "temp": rng.uniform(-10, 30, n),
            "pres": rng.uniform(1000, 1030, n),
            "dewp": rng.uniform(-20, 20, n),
            "rain": rng.uniform(0, 2, n),
            "wspm": rng.uniform(0, 5, n),
            "wd": rng.choice(["N", "NE", "SW", "W"], n),
        }
        frame = pd.DataFrame({STATION_COLUMN: station, **values}, index=index)
        frame[STATION_COLUMN] = frame[STATION_COLUMN].astype("string")
        frame["wd"] = frame["wd"].astype("string")
        parts.append(frame)
    return pd.concat(parts).sort_index(kind="stable")


def raw_csv_frame(
    station: str, n: int = 24, start: str = "2020-01-01", seed: int = 0
) -> pd.DataFrame:
    """Construit des lignes au format d'un CSV brut du dataset (noms d'origine).

    Args:
        station: nom de la station.
        n: nombre d'heures.
        start: premier horodatage.
        seed: graine du générateur aléatoire.

    Returns:
        Un DataFrame avec les colonnes ``No, year, month, day, hour, PM2.5...``.
    """
    frame = measurements(n=n, stations=(station,), start=start, seed=seed)
    index = pd.DatetimeIndex(frame.index)
    return pd.DataFrame(
        {
            "No": range(1, n + 1),
            "year": index.year,
            "month": index.month,
            "day": index.day,
            "hour": index.hour,
            "PM2.5": frame["pm25"].to_numpy(),
            "PM10": frame["pm10"].to_numpy(),
            "SO2": frame["so2"].to_numpy(),
            "NO2": frame["no2"].to_numpy(),
            "CO": frame["co"].to_numpy(),
            "O3": frame["o3"].to_numpy(),
            "TEMP": frame["temp"].to_numpy(),
            "PRES": frame["pres"].to_numpy(),
            "DEWP": frame["dewp"].to_numpy(),
            "RAIN": frame["rain"].to_numpy(),
            "wd": frame["wd"].to_numpy(),
            "WSPM": frame["wspm"].to_numpy(),
            "station": station,
        }
    )


def write_raw_csvs(
    directory: Path, stations: tuple[str, ...] = ("A", "B"), **kwargs: object
) -> list[Path]:
    """Écrit un CSV brut par station, nommé comme ceux du dataset.

    Args:
        directory: dossier de destination.
        stations: stations à écrire.
        **kwargs: arguments passés à ``raw_csv_frame`` (``n``, ``start``...).

    Returns:
        Les chemins des CSV écrits.
    """
    paths = []
    for i, station in enumerate(stations):
        path = directory / f"PRSA_Data_{station}_20130301-20170228.csv"
        raw_csv_frame(station, seed=i, **kwargs).to_csv(path, index=False)
        paths.append(path)
    return paths


def contract_message(**overrides: object) -> dict[str, object]:
    """Construit un message valide au format du contrat ``mesures.brutes``.

    Reprend l'exemple de la requête ``/predict`` de CLAUDE.md.

    Args:
        **overrides: champs à remplacer ou ajouter.

    Returns:
        Le message sous forme de dictionnaire.
    """
    message: dict[str, object] = {
        "sensor_id": "IoT-003",
        "station_id": "Aotizhongxin",
        "timestamp": "2016-05-01T14:00:00",
        "pm25": 85.0,
        "pm10": 120.0,
        "no2": 60.0,
        "co": 900.0,
        "so2": 8.0,
        "o3": 70.0,
        "temp": 22.1,
        "pres": 1008.0,
        "dewp": 10.2,
        "rain": 0.0,
        "wd": "NE",
        "wspm": 2.1,
        "injection": None,
    }
    message.update(overrides)
    return message


class FakeMinio:
    """Client MinIO en mémoire : mêmes méthodes que ``minio.Minio`` utilisées par le projet.

    Attributes:
        objects: contenu des objets, par ``(bucket, nom)``.
    """

    class _Object:
        """Objet listé, avec les attributs lus par ``novaria.common.storage``.

        Attributes:
            object_name: nom complet de l'objet.
            is_dir: True pour un préfixe (sous-dossier).
        """

        def __init__(self, object_name: str, is_dir: bool = False) -> None:
            """Crée l'objet listé.

            Args:
                object_name: nom complet de l'objet.
                is_dir: True pour un préfixe (sous-dossier).
            """
            self.object_name = object_name
            self.is_dir = is_dir

    def __init__(self) -> None:
        """Crée un stockage vide."""
        self.objects: dict[tuple[str, str], bytes] = {}

    def list_objects(self, bucket: str, prefix: str = "") -> list[_Object]:
        """Liste les objets directement sous un préfixe, comme MinIO (non récursif).

        Args:
            bucket: bucket listé.
            prefix: préfixe des objets.

        Returns:
            Les objets du préfixe, puis ses sous-dossiers.
        """
        files, dirs = [], set()
        for b, name in sorted(self.objects):
            if b != bucket or not name.startswith(prefix):
                continue
            rest = name[len(prefix) :]
            if "/" in rest:
                dirs.add(prefix + rest.split("/")[0] + "/")
            else:
                files.append(self._Object(name))
        return files + [self._Object(d, is_dir=True) for d in sorted(dirs)]

    def fget_object(self, bucket: str, name: str, path: str) -> None:
        """Copie un objet dans un fichier local.

        Args:
            bucket: bucket source.
            name: nom de l'objet.
            path: fichier local de destination.
        """
        Path(path).write_bytes(self.objects[(bucket, name)])

    def fput_object(self, bucket: str, name: str, path: str) -> None:
        """Copie un fichier local dans un objet.

        Args:
            bucket: bucket de destination.
            name: nom de l'objet.
            path: fichier local source.
        """
        self.objects[(bucket, name)] = Path(path).read_bytes()
