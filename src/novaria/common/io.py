"""Utilitaires de fichiers partagés : empreintes SHA-256 et JSON."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def sha256_file(path: str | Path, chunk_size: int = 1 << 20) -> str:
    """Calcule l'empreinte SHA-256 d'un fichier, lu par blocs.

    Args:
        path: chemin du fichier.
        chunk_size: taille des blocs lus, en octets.

    Returns:
        L'empreinte en hexadécimal.
    """
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        while chunk := f.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def combined_sha256(hashes: Iterable[str]) -> str:
    """Combine plusieurs empreintes en une seule, indépendante de leur ordre.

    Sert d'empreinte d'un jeu de données composé de plusieurs fichiers : elle
    change dès qu'un fichier change, est ajouté ou retiré.

    Args:
        hashes: empreintes SHA-256 en hexadécimal.

    Returns:
        L'empreinte combinée en hexadécimal.
    """
    return hashlib.sha256("\n".join(sorted(hashes)).encode("ascii")).hexdigest()


def to_serializable(obj: Any) -> Any:
    """Convertit récursivement les types numpy / pandas en types JSON natifs.

    Args:
        obj: objet à convertir (dict, liste, scalaire...).

    Returns:
        Un objet sérialisable par ``json`` ; NaN devient ``None``, les dates
        deviennent des chaînes ISO 8601 et les chemins des chaînes.
    """
    if isinstance(obj, dict):
        return {str(k): to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_serializable(v) for v in obj]
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return None if np.isnan(obj) else float(obj)
    if isinstance(obj, float) and np.isnan(obj):
        return None
    return obj


def write_json(data: Any, path: str | Path) -> Path:
    """Écrit un objet en JSON lisible (UTF-8, indenté), en créant le dossier parent.

    Args:
        data: objet à écrire ; converti avec ``to_serializable``.
        path: chemin du fichier JSON.

    Returns:
        Le chemin du fichier écrit.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(to_serializable(data), f, indent=2, ensure_ascii=False)
    return path


def read_json(path: str | Path) -> Any:
    """Lit un fichier JSON.

    Args:
        path: chemin du fichier JSON.

    Returns:
        Le contenu décodé.
    """
    with Path(path).open(encoding="utf-8") as f:
        return json.load(f)
