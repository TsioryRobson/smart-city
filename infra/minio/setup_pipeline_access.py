"""Prépare l'accès du pipeline Kubeflow aux données dans MinIO (moindre privilège).

Crée, s'ils n'existent pas :
- le bucket des données du projet (``novaria-data``) ;
- la politique ``novaria-pipeline`` : lecture du bucket, écriture seulement
  sous ``datasets/`` (jeux versionnés produits par le pipeline) ;
- le compte MinIO du pipeline, rattaché à cette seule politique.

Puis dépose les CSV bruts de ``data/raw/`` sous ``raw/`` (un fichier déjà présent
avec la même empreinte SHA-256 n'est pas renvoyé).

Les pods n'utilisent jamais le compte administrateur MinIO. Les identifiants
viennent de ``.env`` (voir ``.env.example``).

Exécution (stack Docker Compose démarrée) :
``set -a; . ./.env; set +a; python infra/minio/setup_pipeline_access.py``
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

from minio import Minio, MinioAdmin
from minio.credentials import StaticProvider
from minio.error import S3Error

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
RAW_PREFIX = "raw/"
DATASETS_PREFIX = "datasets/"
POLICY_NAME = "novaria-pipeline"


def pipeline_policy(bucket: str) -> dict:
    """Construit la politique MinIO du pipeline : lecture du bucket, écriture sous ``datasets/``.

    Args:
        bucket: nom du bucket des données.

    Returns:
        Le document de politique IAM (format S3).
    """
    arn = f"arn:aws:s3:::{bucket}"
    return {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Action": ["s3:GetBucketLocation", "s3:ListBucket"],
                "Resource": [arn],
            },
            {"Effect": "Allow", "Action": ["s3:GetObject"], "Resource": [f"{arn}/*"]},
            {
                "Effect": "Allow",
                "Action": ["s3:PutObject", "s3:DeleteObject"],
                "Resource": [f"{arn}/{DATASETS_PREFIX}*"],
            },
        ],
    }


def sha256_of(path: Path) -> str:
    """Calcule l'empreinte SHA-256 d'un fichier.

    Args:
        path: chemin du fichier.

    Returns:
        L'empreinte en hexadécimal.
    """
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def required_env(name: str) -> str:
    """Lit une variable d'environnement obligatoire.

    Args:
        name: nom de la variable.

    Returns:
        Sa valeur.

    Raises:
        SystemExit: si la variable est absente ou vide.
    """
    value = os.environ.get(name, "")
    if not value or value == "changeme":
        raise SystemExit(f"Variable {name} absente ou non modifiée : voir .env.example")
    return value


def ensure_access(
    admin: MinioAdmin, client: Minio, bucket: str, user: str, password: str
) -> None:
    """Crée le bucket, la politique et le compte du pipeline s'ils n'existent pas.

    Args:
        admin: client d'administration MinIO (compte administrateur).
        client: client S3 MinIO (compte administrateur).
        bucket: nom du bucket des données.
        user: identifiant du compte du pipeline.
        password: mot de passe du compte du pipeline.
    """
    if not client.bucket_exists(bucket):
        client.make_bucket(bucket)
        print(f"Bucket créé : {bucket}")
    with tempfile.TemporaryDirectory() as tmp:
        policy_path = Path(tmp) / f"{POLICY_NAME}.json"
        policy_path.write_text(json.dumps(pipeline_policy(bucket)), encoding="utf-8")
        admin.policy_add(POLICY_NAME, str(policy_path))
    admin.user_add(user, password)
    existing = json.loads(admin.user_info(user)).get("policyName", "")
    if POLICY_NAME not in existing.split(","):
        admin.attach_policy([POLICY_NAME], user=user)
    print(f"Compte {user} : politique {POLICY_NAME} seulement")


def upload_raw_files(client: Minio, bucket: str, raw_dir: Path = RAW_DIR) -> int:
    """Dépose les CSV bruts sous ``raw/``, sans renvoyer un fichier déjà présent à l'identique.

    L'empreinte SHA-256 est gardée en métadonnée de l'objet (``sha256``).

    Args:
        client: client S3 MinIO.
        bucket: nom du bucket des données.
        raw_dir: dossier des CSV bruts.

    Returns:
        Le nombre de fichiers envoyés.

    Raises:
        SystemExit: si aucun CSV n'est trouvé.
    """
    paths = sorted(raw_dir.glob("*.csv"))
    if not paths:
        raise SystemExit(f"Aucun CSV dans {raw_dir}")
    sent = 0
    for path in paths:
        name = RAW_PREFIX + path.name
        sha = sha256_of(path)
        try:
            current = client.stat_object(bucket, name).metadata.get("x-amz-meta-sha256")
        except S3Error:  # objet absent
            current = None
        if current == sha:
            continue
        client.fput_object(
            bucket, name, str(path), content_type="text/csv", metadata={"sha256": sha}
        )
        sent += 1
    print(f"{len(paths)} CSV dans {bucket}/{RAW_PREFIX} ({sent} envoyé(s))")
    return sent


def main() -> int:
    """Point d'entrée : prépare le bucket, le compte du pipeline et dépose les CSV.

    Returns:
        0 en cas de succès.
    """
    endpoint = os.environ.get("MINIO_ENDPOINT", "localhost:9000")
    root = StaticProvider(
        required_env("MINIO_ROOT_USER"), required_env("MINIO_ROOT_PASSWORD")
    )
    admin = MinioAdmin(endpoint=endpoint, credentials=root, secure=False)
    client = Minio(endpoint, credentials=root, secure=False)
    bucket = os.environ.get("NOVARIA_DATA_BUCKET", "novaria-data")
    ensure_access(
        admin,
        client,
        bucket,
        required_env("MINIO_PIPELINE_USER"),
        required_env("MINIO_PIPELINE_PASSWORD"),
    )
    upload_raw_files(client, bucket)
    return 0


if __name__ == "__main__":
    sys.exit(main())
