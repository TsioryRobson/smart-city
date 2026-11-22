"""Vérifie depuis un pod Kubeflow l'accès aux services du sprint 1.

Deux vérifications, lancées comme deux étapes du pipeline de validation :
- ``minio`` : lit un CSV brut de ``novaria-data/raw/``, écrit puis supprime un
  objet sous ``datasets/``, et vérifie que l'écriture sous ``raw/`` est refusée ;
- ``mlflow`` : lit l'expérience du projet et la version ``@champion`` du
  Registry, puis enregistre un run de contrôle dans l'expérience
  ``kfp-validation``.

Configuration par variables d'environnement (ConfigMap ``novaria-pipeline-config``
et Secret ``novaria-minio-pipeline``). Code de sortie non nul en cas d'échec.
"""

from __future__ import annotations

import argparse
import io
import os
import sys
from datetime import UTC, datetime

EXPERIMENT = "novaria-air-anomaly"
MODEL_NAME = "air-anomaly-detector"
CHECK_EXPERIMENT = "kfp-validation"


def check_minio() -> None:
    """Vérifie les droits du compte MinIO du pipeline sur le bucket des données.

    Raises:
        RuntimeError: si un droit attendu manque, ou si un droit interdit est accordé.
    """
    from minio import Minio
    from minio.error import S3Error

    bucket = os.environ["NOVARIA_DATA_BUCKET"]
    client = Minio(
        os.environ["MINIO_ENDPOINT"],
        access_key=os.environ["MINIO_ACCESS_KEY"],
        secret_key=os.environ["MINIO_SECRET_KEY"],
        secure=False,
    )
    raws = [o.object_name for o in client.list_objects(bucket, prefix="raw/")]
    if not raws:
        raise RuntimeError(f"Aucun CSV sous {bucket}/raw/")
    head = client.get_object(bucket, raws[0]).read(120).decode()
    print(
        f"{len(raws)} CSV lisibles ; début de {raws[0]} : {head.splitlines()[0][:60]}…"
    )

    name = f"datasets/_kfp_check/{datetime.now(UTC):%Y%m%dT%H%M%S}.txt"
    client.put_object(bucket, name, io.BytesIO(b"ok"), 2)
    client.remove_object(bucket, name)
    print("Écriture et suppression sous datasets/ : autorisées")

    try:
        client.put_object(bucket, "raw/_interdit.txt", io.BytesIO(b"x"), 1)
    except S3Error as error:
        print(f"Écriture sous raw/ : refusée comme attendu ({error.code})")
    else:
        client.remove_object(bucket, "raw/_interdit.txt")
        raise RuntimeError(
            "Le compte du pipeline peut écrire sous raw/ : politique trop large"
        )


def check_mlflow() -> None:
    """Vérifie l'accès au suivi et au Model Registry de MLflow, puis trace un run de contrôle.

    Raises:
        RuntimeError: si l'expérience du projet ou l'alias ``@champion`` est introuvable.
    """
    import mlflow
    from mlflow import MlflowClient

    client = MlflowClient()
    experiment = client.get_experiment_by_name(EXPERIMENT)
    if experiment is None:
        raise RuntimeError(f"Expérience {EXPERIMENT} introuvable")
    champion = client.get_model_version_by_alias(MODEL_NAME, "champion")
    print(
        f"Expérience {EXPERIMENT} ({experiment.experiment_id}) ; "
        f"{MODEL_NAME}@champion = v{champion.version} "
        f"({champion.tags.get('train_period', '?')})"
    )
    mlflow.set_experiment(CHECK_EXPERIMENT)
    with mlflow.start_run(run_name="kfp-validation") as run:
        mlflow.set_tags(
            {"step": "kfp-validation", "pod": os.environ.get("HOSTNAME", "?")}
        )
        mlflow.log_metric("champion_version", float(champion.version))
    print(f"Run de contrôle enregistré : {run.info.run_id}")


def main() -> int:
    """Point d'entrée : lance la vérification demandée.

    Returns:
        0 si la vérification réussit, 1 sinon.
    """
    parser = argparse.ArgumentParser(
        description="Vérification d'accès depuis Kubeflow."
    )
    parser.add_argument("target", choices=["minio", "mlflow"])
    target = parser.parse_args().target
    try:
        check_minio() if target == "minio" else check_mlflow()
    except Exception as error:  # noqa: BLE001 - toute erreur doit finir en échec lisible de l'étape
        print(f"ÉCHEC {target} : {error}", file=sys.stderr)
        return 1
    print(f"OK {target}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
