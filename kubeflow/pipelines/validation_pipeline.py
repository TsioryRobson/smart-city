"""Pipeline de validation de l'installation Kubeflow (branche chore/s2-kubeflow-setup).

Deux étapes en parallèle, dans l'image ``novaria/kfp-check`` :
- ``check-minio`` : droits du compte MinIO du pipeline sur ``novaria-data`` ;
- ``check-mlflow`` : suivi et Model Registry de MLflow (``@champion``).

Chaque étape reçoit les adresses par la ConfigMap ``novaria-pipeline-config`` ;
seule l'étape MinIO reçoit le Secret ``novaria-minio-pipeline`` (moindre privilège).

Compiler :   .venv-kfp/bin/python kubeflow/pipelines/validation_pipeline.py
Soumettre :  .venv-kfp/bin/python kubeflow/pipelines/run.py kubeflow/pipelines/validation_pipeline.yaml
"""

# Pas de « from __future__ import annotations » : KFP lit les annotations des
# composants (dsl.ContainerSpec) et ne les reconnaît pas sous forme de texte.
from pathlib import Path

from kfp import compiler, dsl, kubernetes

CHECK_IMAGE = "novaria/kfp-check:0.1.0"
# Le lanceur KFP ignore l'ENTRYPOINT de l'image : la commande doit être explicite.
CHECK_COMMAND = ["python", "/app/check_access.py"]
CONFIG_MAP = "novaria-pipeline-config"
MINIO_SECRET = "novaria-minio-pipeline"


@dsl.container_component
def check_minio() -> dsl.ContainerSpec:
    """Décrit l'étape qui vérifie l'accès du pipeline aux données dans MinIO.

    Returns:
        La spécification du conteneur à lancer.
    """
    return dsl.ContainerSpec(image=CHECK_IMAGE, command=CHECK_COMMAND, args=["minio"])


@dsl.container_component
def check_mlflow() -> dsl.ContainerSpec:
    """Décrit l'étape qui vérifie l'accès au suivi et au Registry de MLflow.

    Returns:
        La spécification du conteneur à lancer.
    """
    return dsl.ContainerSpec(image=CHECK_IMAGE, command=CHECK_COMMAND, args=["mlflow"])


def use_pipeline_config(task: dsl.PipelineTask) -> dsl.PipelineTask:
    """Injecte les adresses des services (ConfigMap) dans une étape, et fixe sa politique d'image.

    ``IfNotPresent`` : l'image est construite en local, sans registre.

    Args:
        task: étape du pipeline.

    Returns:
        La même étape, configurée.
    """
    kubernetes.use_config_map_as_env(
        task,
        config_map_name=CONFIG_MAP,
        config_map_key_to_env={
            "MLFLOW_TRACKING_URI": "MLFLOW_TRACKING_URI",
            "MINIO_ENDPOINT": "MINIO_ENDPOINT",
            "NOVARIA_DATA_BUCKET": "NOVARIA_DATA_BUCKET",
        },
    )
    kubernetes.set_image_pull_policy(task, "IfNotPresent")
    task.set_caching_options(False)
    return task


@dsl.pipeline(
    name="novaria-validation",
    description="Valide l'accès des pods Kubeflow à MinIO et MLflow (sprint 1).",
)
def validation_pipeline() -> None:
    """Définit le pipeline de validation : deux vérifications indépendantes, en parallèle."""
    minio_task = use_pipeline_config(check_minio().set_display_name("check-minio"))
    kubernetes.use_secret_as_env(
        minio_task,
        secret_name=MINIO_SECRET,
        secret_key_to_env={
            "MINIO_ACCESS_KEY": "MINIO_ACCESS_KEY",
            "MINIO_SECRET_KEY": "MINIO_SECRET_KEY",
        },
    )
    use_pipeline_config(check_mlflow().set_display_name("check-mlflow"))


if __name__ == "__main__":
    output = Path(__file__).with_suffix(".yaml")
    compiler.Compiler().compile(validation_pipeline, str(output))
    print(f"Pipeline compilé : {output}")
