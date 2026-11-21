"""Crée le bucket des artefacts MLflow dans MinIO s'il n'existe pas.

Lancé au démarrage du conteneur MLflow, avant le serveur. MinIO peut mettre
quelques secondes à démarrer : on réessaie pendant une minute.
"""

import os
import sys
import time

import boto3
from botocore.exceptions import ClientError, EndpointConnectionError

ATTEMPTS = 30
DELAY_SECONDS = 2


def main() -> int:
    """Crée le bucket ``MLFLOW_BUCKET`` dans MinIO, en attendant que MinIO réponde.

    Returns:
        0 si le bucket existe ou a été créé, 1 si MinIO reste injoignable.
    """
    bucket = os.environ.get("MLFLOW_BUCKET", "mlflow")
    client = boto3.client("s3", endpoint_url=os.environ["MLFLOW_S3_ENDPOINT_URL"])
    for attempt in range(1, ATTEMPTS + 1):
        try:
            existing = {b["Name"] for b in client.list_buckets().get("Buckets", [])}
            if bucket not in existing:
                client.create_bucket(Bucket=bucket)
                print(f"Bucket créé : {bucket}")
            else:
                print(f"Bucket déjà présent : {bucket}")
            return 0
        except (EndpointConnectionError, ClientError) as error:
            print(f"MinIO pas prêt ({attempt}/{ATTEMPTS}) : {error}", file=sys.stderr)
            time.sleep(DELAY_SECONDS)
    return 1


if __name__ == "__main__":
    sys.exit(main())
