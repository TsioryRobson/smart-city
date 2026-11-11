"""Validation de l'étape 2 : vérifie que l'infrastructure MLflow est opérationnelle.

Lancement (dans le conteneur dev) :
    python scripts/check_stack.py

Ce script vérifie de bout en bout ce dont le projet aura besoin ensuite :
  1. le serveur MLflow répond (variable MLFLOW_TRACKING_URI) ;
  2. un run enregistre paramètres et métriques (base PostgreSQL) ;
  3. un artefact est envoyé et relu via le serveur (proxied artifacts) ;
  4. un modèle est enregistré dans le Model Registry avec un alias ;
  5. le modèle est rechargé par son alias et produit des prédictions.

Il utilise une expérience et un modèle dédiés ("smoke-test"), séparés des
vrais travaux du projet.
"""

from __future__ import annotations

import os
import sys
import tempfile
import urllib.request
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
from mlflow import MlflowClient
from sklearn.ensemble import IsolationForest

EXPERIMENT = "00-smoke-test"
MODEL_NAME = "smoke-test-model"
ALIAS = "ok"


def step(message: str) -> None:
    print(f"[OK] {message}")


def main() -> int:
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
    if not tracking_uri:
        print("[ERREUR] MLFLOW_TRACKING_URI n'est pas définie.")
        return 1

    # 1. Serveur joignable
    with urllib.request.urlopen(f"{tracking_uri}/health", timeout=5) as resp:
        if resp.status != 200:
            print(f"[ERREUR] /health a répondu {resp.status}")
            return 1
    with urllib.request.urlopen(f"{tracking_uri}/version", timeout=5) as resp:
        server_version = resp.read().decode().strip()
    step(f"Serveur MLflow joignable sur {tracking_uri} (version {server_version})")
    if server_version != mlflow.__version__:
        print(f"[ATTENTION] Client {mlflow.__version__} != serveur {server_version}")

    mlflow.set_experiment(EXPERIMENT)

    # 2 + 3. Run avec paramètres, métriques et artefact
    rng = np.random.default_rng(42)
    x_train = rng.normal(size=(500, 3))
    model = IsolationForest(n_estimators=50, random_state=42).fit(x_train)

    with mlflow.start_run(run_name="check-stack") as run:
        mlflow.log_param("n_estimators", 50)
        mlflow.log_metric("score_moyen", float(model.score_samples(x_train).mean()))

        with tempfile.TemporaryDirectory() as tmp:
            note = Path(tmp) / "note.txt"
            note.write_text("Test d'envoi d'artefact via le serveur MLflow.\n", encoding="utf-8")
            mlflow.log_artifact(str(note))

        # 4. Enregistrement dans le Model Registry
        info = mlflow.sklearn.log_model(
            model,
            name="model",
            registered_model_name=MODEL_NAME,
            input_example=x_train[:2],
            # MLflow sérialise les modèles scikit-learn au format sécurisé skops
            # (pas de pickle). Les modèles à base d'arbres (Isolation Forest)
            # exigent de déclarer explicitement ce type comme fiable.
            skops_trusted_types=["sklearn.tree._tree.Tree"],
        )
    step(f"Run enregistré (run_id={run.info.run_id})")

    artifact_uri = run.info.artifact_uri
    if not artifact_uri.startswith("mlflow-artifacts:"):
        print(f"[ERREUR] Artefacts hors du proxy MLflow : {artifact_uri}")
        return 1
    downloaded = mlflow.artifacts.download_artifacts(run_id=run.info.run_id, artifact_path="note.txt")
    if "Test d'envoi" not in Path(downloaded).read_text(encoding="utf-8"):
        print("[ERREUR] Artefact relu incorrect")
        return 1
    step("Artefact envoyé puis relu via le serveur (proxied artifacts)")

    client = MlflowClient()
    version = info.registered_model_version
    client.set_registered_model_alias(MODEL_NAME, ALIAS, version)
    step(f"Modèle '{MODEL_NAME}' version {version} enregistré, alias @{ALIAS}")

    # 5. Rechargement par alias, comme le fera l'API FastAPI
    loaded = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}@{ALIAS}")
    predictions = loaded.predict(rng.normal(size=(5, 3)))
    if len(predictions) != 5:
        print("[ERREUR] Prédictions inattendues")
        return 1
    step(f"Modèle rechargé via models:/{MODEL_NAME}@{ALIAS} et prédictions OK")

    print("\nÉtape 2 validée : l'infrastructure MLflow est prête.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
