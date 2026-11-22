"""Accès au stockage objet MinIO (bucket des données du projet).

Utilisé par les étapes du pipeline Kubeflow : l'ingestion lit les CSV bruts
sous ``raw/``, la préparation publie les jeux versionnés sous ``datasets/``.
Configuration par variables d'environnement (ConfigMap
``novaria-pipeline-config`` et Secret ``novaria-minio-pipeline``) :
``MINIO_ENDPOINT``, ``MINIO_ACCESS_KEY``, ``MINIO_SECRET_KEY`` et, en option,
``MINIO_SECURE`` (``true`` pour HTTPS).
"""

from __future__ import annotations

import fnmatch
import os
from pathlib import Path, PurePosixPath

from minio import Minio


def client_from_env() -> Minio:
    """Crée un client MinIO à partir des variables d'environnement.

    Returns:
        Le client MinIO.

    Raises:
        KeyError: si ``MINIO_ENDPOINT``, ``MINIO_ACCESS_KEY`` ou
            ``MINIO_SECRET_KEY`` manque.
    """
    return Minio(
        os.environ["MINIO_ENDPOINT"],
        access_key=os.environ["MINIO_ACCESS_KEY"],
        secret_key=os.environ["MINIO_SECRET_KEY"],
        secure=os.environ.get("MINIO_SECURE", "false").lower() == "true",
    )


def download_prefix(
    client: Minio, bucket: str, prefix: str, destination: str | Path, pattern: str = "*"
) -> list[Path]:
    """Télécharge les objets d'un préfixe dont le nom correspond à un motif.

    Seuls les objets directement sous le préfixe sont lus (pas les
    sous-dossiers).

    Args:
        client: client MinIO.
        bucket: bucket source.
        prefix: préfixe des objets (ex. ``raw/``).
        destination: dossier local (créé si besoin).
        pattern: motif du nom de fichier (ex. ``PRSA_Data_*.csv``).

    Returns:
        Les fichiers téléchargés, triés par nom.

    Raises:
        FileNotFoundError: si aucun objet ne correspond.
    """
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    paths = []
    for obj in client.list_objects(bucket, prefix=prefix):
        name = PurePosixPath(obj.object_name).name
        if obj.is_dir or not fnmatch.fnmatch(name, pattern):
            continue
        path = destination / name
        client.fget_object(bucket, obj.object_name, str(path))
        paths.append(path)
    if not paths:
        raise FileNotFoundError(f"Aucun objet {pattern} sous {bucket}/{prefix}")
    return sorted(paths)


def upload_directory(
    client: Minio, source: str | Path, bucket: str, prefix: str
) -> list[str]:
    """Envoie tous les fichiers d'un dossier local sous un préfixe du bucket.

    Args:
        client: client MinIO.
        source: dossier local.
        bucket: bucket de destination.
        prefix: préfixe de destination (ex. ``datasets/prepared/<empreinte>``).

    Returns:
        Les noms des objets écrits, triés.
    """
    source = Path(source)
    names = []
    for path in sorted(p for p in source.rglob("*") if p.is_file()):
        name = str(
            PurePosixPath(prefix.rstrip("/")) / path.relative_to(source).as_posix()
        )
        client.fput_object(bucket, name, str(path))
        names.append(name)
    return names
