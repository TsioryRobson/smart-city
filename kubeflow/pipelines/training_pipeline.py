"""Pipeline Kubeflow d'entraînement : les scripts de ``novaria.pipeline`` en composants.

Graphe (les 3 entraînements et l'injection tournent en parallèle) :

    ingestion -> preparation -+-> training (Isolation Forest) -+
                              +-> training (LOF)               +-> evaluation -> promotion
                              +-> training (One-Class SVM)     |
                              +-> injection (tests contrôlés) -+

Chaque étape lance un script dans l'image ``novaria/pipeline`` (construite en
local par ``kubeflow/components/build.sh``). Les dossiers échangés entre
étapes sont des artefacts Kubeflow (stockage interne du cluster) ; les CSV
bruts sont lus dans MinIO (``novaria-data/raw/``) et les jeux préparés y sont
publiés (``datasets/prepared/<empreinte>/``). Runs et modèles vont dans
MLflow ; la promotion met à jour ``@champion`` / ``@previous`` selon la règle
de CLAUDE.md (section 7).

Compilation (venv du SDK ``kfp``) :
``.venv-kfp/bin/python kubeflow/pipelines/training_pipeline.py``
"""

from kfp import compiler, dsl, kubernetes

PIPELINE_IMAGE = "novaria/pipeline:0.1.0"
CONFIG_MAP = "novaria-pipeline-config"
MINIO_SECRET = "novaria-minio-pipeline"
DATA_BUCKET = "novaria-data"
ALGORITHMS = {
    "isolation_forest": "Isolation Forest",
    "local_outlier_factor": "Local Outlier Factor",
    "one_class_svm": "One-Class SVM",
}
OUTPUT = "kubeflow/pipelines/training_pipeline.yaml"


def script(module: str) -> list[str]:
    """Construit la commande qui lance un script du pipeline Novaria.

    Le lanceur de Kubeflow ignore l'``ENTRYPOINT`` de l'image : la commande est
    toujours donnée explicitement.

    Args:
        module: nom du script dans ``novaria.pipeline`` (ex. ``training``).

    Returns:
        La commande ``python -m novaria.pipeline.<module>``.
    """
    return ["python", "-m", f"novaria.pipeline.{module}"]


# --- Composants ---------------------------------------------------------------
@dsl.container_component
def ingestion(data_bucket: str, ingested: dsl.Output[dsl.Dataset]) -> dsl.ContainerSpec:
    """Étape 1 : lit les CSV bruts dans MinIO et écrit le Parquet et son manifeste.

    Args:
        data_bucket: bucket MinIO des données (CSV sous ``raw/``).
        ingested: dossier de sortie (``measurements.parquet``, ``manifest.json``).

    Returns:
        La spécification du conteneur.
    """
    return dsl.ContainerSpec(
        image=PIPELINE_IMAGE,
        command=script("ingestion"),
        args=["--source-bucket", data_bucket, "--output", ingested.path],
    )


@dsl.container_component
def preparation(
    data_bucket: str,
    ingested: dsl.Input[dsl.Dataset],
    prepared: dsl.Output[dsl.Dataset],
    quality: dsl.Output[dsl.Artifact],
    checks_log: dsl.Output[dsl.Artifact],
) -> dsl.ContainerSpec:
    """Étape 2 : nettoyage, variables et découpage chronologique ; publication dans MinIO.

    Args:
        data_bucket: bucket MinIO où publier les jeux préparés (``datasets/``).
        ingested: sortie de l'ingestion.
        prepared: jeux ``train``, ``val``, ``test`` et ``metadata.json``.
        quality: rapport qualité JSON.
        checks_log: journal de vérification des données (``.log``).

    Returns:
        La spécification du conteneur.
    """
    return dsl.ContainerSpec(
        image=PIPELINE_IMAGE,
        command=script("preparation"),
        args=[
            "--input", ingested.path,
            "--output", prepared.path,
            "--report", quality.path,
            "--log", checks_log.path,
            "--publish-bucket", data_bucket,
        ],
    )  # fmt: skip


@dsl.container_component
def training(
    algorithm: str,
    prepared: dsl.Input[dsl.Dataset],
    trained: dsl.Output[dsl.Model],
    report: dsl.Output[dsl.Markdown],
) -> dsl.ContainerSpec:
    """Étape 3 : entraîne la grille d'un algorithme et garde son meilleur réglage.

    Args:
        algorithm: algorithme de ``configs/training.yaml``.
        prepared: sortie de la préparation.
        trained: modèle retenu et ``comparison.json``.
        report: tableau de comparaison des réglages (affiché par l'interface).

    Returns:
        La spécification du conteneur.
    """
    return dsl.ContainerSpec(
        image=PIPELINE_IMAGE,
        command=script("training"),
        args=[
            "--input", prepared.path,
            "--output", trained.path,
            "--report", report.path,
            "--algorithms", algorithm,
        ],
    )  # fmt: skip


@dsl.container_component
def injection(
    prepared: dsl.Input[dsl.Dataset], controlled: dsl.Output[dsl.Dataset]
) -> dsl.ContainerSpec:
    """Étape 4a : fige les jeux de test contrôlés (anomalies injectées dans le test).

    Args:
        prepared: sortie de la préparation (jamais le train).
        controlled: jeux contrôlés et leur manifeste.

    Returns:
        La spécification du conteneur.
    """
    return dsl.ContainerSpec(
        image=PIPELINE_IMAGE,
        command=script("injection"),
        args=["--input", prepared.path, "--output", controlled.path],
    )


@dsl.container_component
def evaluation(
    prepared: dsl.Input[dsl.Dataset],
    controlled: dsl.Input[dsl.Dataset],
    isolation_forest: dsl.Input[dsl.Model],
    local_outlier_factor: dsl.Input[dsl.Model],
    one_class_svm: dsl.Input[dsl.Model],
    evaluated: dsl.Output[dsl.Model],
    report: dsl.Output[dsl.Markdown],
) -> dsl.ContainerSpec:
    """Étape 4b : évalue les 3 modèles sur les mêmes tests contrôlés et choisit le meilleur.

    Args:
        prepared: sortie de la préparation (train pour la stabilité).
        controlled: jeux de test contrôlés.
        isolation_forest: sortie de l'entraînement Isolation Forest.
        local_outlier_factor: sortie de l'entraînement LOF.
        one_class_svm: sortie de l'entraînement One-Class SVM.
        evaluated: modèles fusionnés, ``evaluation.json`` et ``selection.json``.
        report: rapport d'évaluation (affiché par l'interface ; la figure de
            stabilité est dans MLflow).

    Returns:
        La spécification du conteneur.
    """
    return dsl.ContainerSpec(
        image=PIPELINE_IMAGE,
        command=script("evaluation"),
        args=[
            "--controlled", controlled.path,
            "--prepared", prepared.path,
            "--models", isolation_forest.path, local_outlier_factor.path, one_class_svm.path,
            "--output", evaluated.path,
            "--report", report.path,
        ],
    )  # fmt: skip


@dsl.container_component
def promotion(
    prepared: dsl.Input[dsl.Dataset],
    controlled: dsl.Input[dsl.Dataset],
    evaluated: dsl.Input[dsl.Model],
    decision: dsl.Output[dsl.Artifact],
) -> dsl.ContainerSpec:
    """Étape 5 : enregistre le modèle choisi et le compare à ``@champion``.

    Args:
        prepared: sortie de la préparation (épisodes réels de pollution).
        controlled: jeux de test contrôlés (mêmes que l'évaluation).
        evaluated: sortie de l'évaluation (modèle choisi).
        decision: ``promotion.json`` (décision, raisons, alias).

    Returns:
        La spécification du conteneur.
    """
    return dsl.ContainerSpec(
        image=PIPELINE_IMAGE,
        command=script("promotion"),
        args=[
            "--models", evaluated.path,
            "--controlled", controlled.path,
            "--prepared", prepared.path,
            "--output", decision.path,
        ],
    )  # fmt: skip


# --- Réglages Kubernetes des étapes --------------------------------------------
def configure(
    task: dsl.PipelineTask, name: str, memory: str, cpu: str, minio: bool = False
) -> dsl.PipelineTask:
    """Applique les réglages communs à une étape du pipeline.

    - adresses de MLflow et MinIO (ConfigMap) ; identifiants MinIO (Secret)
      seulement si l'étape lit ou écrit le bucket des données ;
    - identifiant du run Kubeflow (``KFP_RUN_ID``), mis en tag des runs MLflow ;
    - image locale (``IfNotPresent``, pas de registre) ;
    - limites de mémoire et de CPU ;
    - pas de cache : les CSV de MinIO et le Registry changent sans que les
      paramètres du pipeline changent.

    Args:
        task: étape du pipeline.
        name: nom affiché dans l'interface.
        memory: limite de mémoire (ex. ``3G``).
        cpu: limite de CPU (ex. ``2``).
        minio: True si l'étape a besoin des identifiants MinIO.

    Returns:
        L'étape configurée.
    """
    task.set_display_name(name)
    task.set_caching_options(False)
    task.set_memory_request("512M").set_memory_limit(memory)
    task.set_cpu_request("500m").set_cpu_limit(cpu)
    kubernetes.use_config_map_as_env(
        task,
        config_map_name=CONFIG_MAP,
        config_map_key_to_env={
            "MLFLOW_TRACKING_URI": "MLFLOW_TRACKING_URI",
            "MINIO_ENDPOINT": "MINIO_ENDPOINT",
        },
    )
    if minio:
        kubernetes.use_secret_as_env(
            task,
            secret_name=MINIO_SECRET,
            secret_key_to_env={
                "MINIO_ACCESS_KEY": "MINIO_ACCESS_KEY",
                "MINIO_SECRET_KEY": "MINIO_SECRET_KEY",
            },
        )
    kubernetes.use_field_path_as_env(
        task, env_name="KFP_RUN_ID", field_path="metadata.labels['pipeline/runid']"
    )
    kubernetes.set_image_pull_policy(task, "IfNotPresent")
    return task


@dsl.pipeline(
    name="novaria-training",
    description=(
        "Entraînement des 3 détecteurs d'anomalies (en parallèle), évaluation sur "
        "les tests contrôlés et promotion dans le Model Registry MLflow."
    ),
)
def training_pipeline(data_bucket: str = DATA_BUCKET) -> None:
    """Enchaîne ingestion, préparation, 3 entraînements, injection, évaluation et promotion.

    Args:
        data_bucket: bucket MinIO des données (CSV bruts et jeux publiés).
    """
    ingested = configure(
        ingestion(data_bucket=data_bucket), "1. ingestion", "2G", "1", minio=True
    )
    prepared = configure(
        preparation(data_bucket=data_bucket, ingested=ingested.outputs["ingested"]),
        "2. preparation",
        "3G",
        "1",
        minio=True,
    )
    datasets = prepared.outputs["prepared"]

    trained = {
        algorithm: configure(
            training(algorithm=algorithm, prepared=datasets),
            f"3. training · {label}",
            "3G",
            "2",
        )
        for algorithm, label in ALGORITHMS.items()
    }
    controlled = configure(
        injection(prepared=datasets), "4a. injection (tests contrôlés)", "2G", "1"
    )
    evaluated = configure(
        evaluation(
            prepared=datasets,
            controlled=controlled.outputs["controlled"],
            **{a: t.outputs["trained"] for a, t in trained.items()},
        ),
        "4b. evaluation",
        "4G",
        "2",
    )
    configure(
        promotion(
            prepared=datasets,
            controlled=controlled.outputs["controlled"],
            evaluated=evaluated.outputs["evaluated"],
        ),
        "5. promotion",
        "3G",
        "1",
    )


if __name__ == "__main__":
    compiler.Compiler().compile(training_pipeline, OUTPUT)
    print(f"Pipeline compilé : {OUTPUT}")
