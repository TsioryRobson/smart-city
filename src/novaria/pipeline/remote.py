"""Lance un script du pipeline dans une étape Kubeflow, ses dossiers étant dans MinIO.

Les étapes du pipeline Kubeflow ne s'échangent pas d'artefacts Kubeflow (qui
apparaîtraient comme des dossiers dans le graphe) : chaque dossier produit est
envoyé dans MinIO, sous ``<bucket>/<préfixe>/<run Kubeflow>/<nom>/``, et
l'étape suivante le télécharge. Le script lui-même ne change pas : il lit et
écrit des dossiers locaux.

Dans les arguments du script, ``{nom}`` est remplacé par le dossier local
d'une entrée ou d'une sortie. Exemple :

``python -m novaria.pipeline.remote --inputs prepared --outputs trained --
training --input {prepared} --output {trained} --algorithms one_class_svm``

Les étapes parallèles qui produisent le même dossier (les 3 entraînements)
donnent ``--group`` : leur sortie va dans un sous-dossier (``trained/<algo>/``).

L'identifiant du run Kubeflow vient de la variable ``KFP_RUN_ID`` ; l'accès à
MinIO, de ``novaria.common.storage.client_from_env``.
"""

from __future__ import annotations

import argparse
import importlib
import os
import tempfile
from collections.abc import Sequence
from pathlib import Path

from novaria.common.tracking import KFP_RUN_ID_ENV

DEFAULT_BUCKET = "novaria-data"
DEFAULT_PREFIX = "datasets/runs"
# Scripts que le lanceur accepte (modules de novaria.pipeline).
STEPS = ("ingestion", "preparation", "training", "injection", "evaluation", "promotion")


def run_prefix(prefix: str, run_id: str | None = None) -> str:
    """Construit le préfixe MinIO des dossiers d'un run Kubeflow.

    Args:
        prefix: préfixe commun (ex. ``datasets/runs``).
        run_id: identifiant du run ; défaut : variable ``KFP_RUN_ID``.

    Returns:
        ``<prefix>/<run_id>``.

    Raises:
        ValueError: si aucun identifiant de run n'est disponible.
    """
    run_id = run_id or os.environ.get(KFP_RUN_ID_ENV)
    if not run_id:
        raise ValueError(f"Identifiant du run Kubeflow absent ({KFP_RUN_ID_ENV})")
    return f"{prefix.rstrip('/')}/{run_id}"


def fill_arguments(arguments: Sequence[str], folders: dict[str, Path]) -> list[str]:
    """Remplace ``{nom}`` par le dossier local correspondant dans les arguments du script.

    Args:
        arguments: arguments du script.
        folders: dossier local de chaque entrée et sortie, par nom.

    Returns:
        Les arguments complétés.

    Raises:
        ValueError: si un argument contient un ``{nom}`` inconnu.
    """
    filled = []
    for argument in arguments:
        value = argument
        for name, folder in folders.items():
            value = value.replace(f"{{{name}}}", str(folder))
        if "{" in value and "}" in value:
            raise ValueError(f"Dossier inconnu dans l'argument : {argument}")
        filled.append(value)
    return filled


def run_step(
    step: str,
    arguments: Sequence[str],
    inputs: Sequence[str],
    outputs: Sequence[str],
    bucket: str = DEFAULT_BUCKET,
    prefix: str = DEFAULT_PREFIX,
    run_id: str | None = None,
    client=None,
    group: str | None = None,
) -> list[str]:
    """Télécharge les entrées, lance le script, puis envoie ses sorties dans MinIO.

    Args:
        step: script de ``novaria.pipeline`` (ex. ``training``).
        arguments: arguments du script, avec ``{nom}`` pour les dossiers.
        inputs: noms des dossiers produits par les étapes précédentes.
        outputs: noms des dossiers produits par cette étape.
        bucket: bucket MinIO des données.
        prefix: préfixe commun des runs.
        run_id: identifiant du run (défaut : ``KFP_RUN_ID``).
        client: client MinIO (défaut : créé depuis l'environnement).
        group: sous-dossier MinIO des sorties, pour les étapes parallèles qui
            produisent le même dossier (ex. l'algorithme entraîné).

    Returns:
        Les noms des objets envoyés dans MinIO.

    Raises:
        ValueError: si le script n'est pas une étape du pipeline.
    """
    from novaria.common.storage import client_from_env, download_tree, upload_directory

    if step not in STEPS:
        raise ValueError(f"Étape inconnue : {step} (attendu : {', '.join(STEPS)})")
    client = client or client_from_env()
    base = run_prefix(prefix, run_id)
    with tempfile.TemporaryDirectory(prefix=f"novaria-{step}-") as tmp:
        folders = {name: Path(tmp) / name for name in (*inputs, *outputs)}
        for name in inputs:
            download_tree(client, bucket, f"{base}/{name}", folders[name])
            print(f"Entrée {name} : minio://{bucket}/{base}/{name}/")
        module = importlib.import_module(f"novaria.pipeline.{step}")
        module.main(fill_arguments(arguments, folders))
        sent = []
        for name in outputs:
            target = f"{base}/{name}/{group}" if group else f"{base}/{name}"
            sent += upload_directory(client, folders[name], bucket, target)
            print(f"Sortie {name} : minio://{bucket}/{target}/")
    return sent


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Lit les arguments du lanceur ; ce qui suit ``--`` est le script et ses arguments.

    Args:
        argv: arguments à analyser (défaut : ceux du processus).

    Returns:
        Les arguments : entrées, sorties, bucket, préfixe, groupe, puis ``step`` et
        ``arguments`` (le script et ses arguments).
    """
    parser = argparse.ArgumentParser(
        description="Étape Kubeflow : dossiers échangés par MinIO."
    )
    parser.add_argument("--inputs", nargs="*", default=[], help="dossiers à lire")
    parser.add_argument("--outputs", nargs="*", default=[], help="dossiers produits")
    parser.add_argument("--bucket", default=DEFAULT_BUCKET, help="bucket MinIO")
    parser.add_argument("--prefix", default=DEFAULT_PREFIX, help="préfixe des runs")
    parser.add_argument(
        "--group", default=None, help="sous-dossier des sorties (étapes parallèles)"
    )
    parser.add_argument("step", choices=STEPS, help="script de novaria.pipeline")
    parser.add_argument("arguments", nargs=argparse.REMAINDER, help="ses arguments")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Point d'entrée CLI : exécute une étape avec ses dossiers dans MinIO.

    Args:
        argv: arguments de la ligne de commande (défaut : ceux du processus).
    """
    args = parse_args(argv)
    run_step(
        args.step,
        args.arguments,
        args.inputs,
        args.outputs,
        args.bucket,
        args.prefix,
        group=args.group,
    )


if __name__ == "__main__":
    main()
