"""Soumet un pipeline compilé à Kubeflow Pipelines et attend la fin du run.

L'API de Kubeflow est jointe par l'interface, exposée en local avec :
``kubectl -n kubeflow port-forward svc/ml-pipeline-ui 8080:80``

Usage :
``.venv-kfp/bin/python kubeflow/pipelines/run.py kubeflow/pipelines/validation_pipeline.yaml``
"""

import argparse
import sys
from pathlib import Path

import kfp

SUCCESS = "SUCCEEDED"


def parse_args() -> argparse.Namespace:
    """Lit les arguments de la ligne de commande.

    Returns:
        Les arguments : fichier compilé, adresse de Kubeflow, expérience et délai.
    """
    parser = argparse.ArgumentParser(description="Soumission d'un pipeline Kubeflow.")
    parser.add_argument("package", type=Path, help="pipeline compilé (.yaml)")
    parser.add_argument(
        "--host",
        default="http://localhost:8080",
        help="interface Kubeflow (port-forward)",
    )
    parser.add_argument("--experiment", default="novaria", help="expérience Kubeflow")
    parser.add_argument(
        "--timeout", type=int, default=1800, help="attente maximale, en secondes"
    )
    return parser.parse_args()


def main() -> int:
    """Point d'entrée : crée le run, attend sa fin et affiche son statut.

    Returns:
        0 si le run a réussi, 1 sinon.
    """
    args = parse_args()
    client = kfp.Client(host=args.host)
    run = client.create_run_from_pipeline_package(
        str(args.package), arguments={}, experiment_name=args.experiment
    )
    print(f"Run lancé : {run.run_id}")
    print(f"Interface : {args.host}/#/runs/details/{run.run_id}")
    result = client.wait_for_run_completion(run.run_id, timeout=args.timeout)
    print(f"Statut : {result.state}")
    return 0 if result.state == SUCCESS else 1


if __name__ == "__main__":
    sys.exit(main())
