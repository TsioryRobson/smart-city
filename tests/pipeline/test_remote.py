"""Tests de novaria.pipeline.remote (étapes Kubeflow dont les dossiers passent par MinIO)."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from novaria.pipeline.remote import fill_arguments, parse_args, run_prefix, run_step
from tests.factories import FakeMinio


def test_run_prefix_needs_a_run_id(monkeypatch) -> None:
    """Vérifie que le préfixe du run vient de ``KFP_RUN_ID`` et qu'il est obligatoire.

    Args:
        monkeypatch: outil pytest pour modifier l'environnement.
    """
    monkeypatch.delenv("KFP_RUN_ID", raising=False)
    with pytest.raises(ValueError, match="KFP_RUN_ID"):
        run_prefix("datasets/runs")
    monkeypatch.setenv("KFP_RUN_ID", "run-1")
    assert run_prefix("datasets/runs/") == "datasets/runs/run-1"


def test_fill_arguments() -> None:
    """Vérifie le remplacement de ``{nom}`` par le dossier local, y compris dans un chemin."""
    folders = {"prepared": Path("/tmp/p"), "checks": Path("/tmp/c")}
    assert fill_arguments(
        ["--input", "{prepared}", "--log", "{checks}/data.log", "--n", "3"], folders
    ) == ["--input", "/tmp/p", "--log", "/tmp/c/data.log", "--n", "3"]
    with pytest.raises(ValueError, match="inconnu"):
        fill_arguments(["{trained}"], folders)


def test_parse_args_separates_runner_and_script() -> None:
    """Vérifie que tout ce qui suit ``--`` revient au script, options comprises."""
    args = parse_args(
        ["--inputs", "prepared", "--outputs", "trained", "--group", "lof", "--",
         "training", "--input", "{prepared}", "--algorithms", "lof"]
    )  # fmt: skip
    assert args.inputs == ["prepared"] and args.outputs == ["trained"]
    assert args.group == "lof" and args.step == "training"
    assert args.arguments == ["--input", "{prepared}", "--algorithms", "lof"]


def test_run_step_moves_folders_through_minio(monkeypatch) -> None:
    """Vérifie le cycle d'une étape : téléchargement des entrées, script, envoi des sorties.

    Le script est remplacé par une fonction qui copie le fichier d'entrée dans
    la sortie ; la sortie d'une étape parallèle va dans son sous-dossier.

    Args:
        monkeypatch: outil pytest pour remplacer le module du script.
    """
    client = FakeMinio()
    client.objects[("data", "datasets/runs/r1/prepared/sub/train.parquet")] = b"t"
    calls = []

    def fake_main(argv: list[str]) -> None:
        """Copie ``train.parquet`` de l'entrée vers la sortie.

        Args:
            argv: arguments reçus par le script.
        """
        calls.append(argv)
        source, target = Path(argv[1]), Path(argv[3])
        target.mkdir(parents=True)
        (target / "model.bin").write_bytes(
            (source / "sub" / "train.parquet").read_bytes()
        )

    monkeypatch.setattr(
        "novaria.pipeline.remote.importlib.import_module",
        lambda name: SimpleNamespace(main=fake_main),
    )

    sent = run_step(
        "training",
        ["--input", "{prepared}", "--output", "{trained}"],
        inputs=["prepared"],
        outputs=["trained"],
        bucket="data",
        run_id="r1",
        client=client,
        group="one_class_svm",
    )

    assert sent == ["datasets/runs/r1/trained/one_class_svm/model.bin"]
    assert client.objects[("data", sent[0])] == b"t"
    assert calls[0][0] == "--input" and calls[0][1].endswith("/prepared")
    with pytest.raises(ValueError, match="inconnue"):
        run_step("deploy", [], [], [], run_id="r1", client=client)
