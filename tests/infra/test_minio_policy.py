"""Tests de la politique MinIO du pipeline Kubeflow (moindre privilège)."""

import importlib.util
from pathlib import Path

SCRIPT = (
    Path(__file__).resolve().parents[2] / "infra" / "minio" / "setup_pipeline_access.py"
)


def _load_module():
    """Charge le script d'administration MinIO comme module (il n'est pas dans un paquet).

    Returns:
        Le module chargé.
    """
    spec = importlib.util.spec_from_file_location("setup_pipeline_access", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_policy_reads_bucket_and_writes_only_datasets() -> None:
    """Vérifie que la politique ne permet d'écrire que sous ``datasets/``, dans un seul bucket.

    Aucune action d'administration ni accès à un autre bucket (comme ``mlflow``)
    ne doit apparaître.
    """
    policy = _load_module().pipeline_policy("novaria-data")
    statements = policy["Statement"]

    write = [s for s in statements if "s3:PutObject" in s["Action"]]
    assert len(write) == 1
    assert write[0]["Resource"] == ["arn:aws:s3:::novaria-data/datasets/*"]

    resources = {r for s in statements for r in s["Resource"]}
    assert all(r.startswith("arn:aws:s3:::novaria-data") for r in resources)
    actions = {a for s in statements for a in s["Action"]}
    assert not any(a.startswith("admin:") or a == "s3:*" for a in actions)
    assert all(s["Effect"] == "Allow" for s in statements)
