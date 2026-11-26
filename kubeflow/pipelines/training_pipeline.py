"""Pipeline Kubeflow d'entraînement : les scripts de ``novaria.pipeline`` en composants.

Graphe (les 3 entraînements et l'injection tournent en parallèle) :

    ingestion -> preparation -+-> training (Isolation Forest) -+
                              +-> training (LOF)               +-> evaluation -> promotion -> decision
                              +-> training (One-Class SVM)     |
                              +-> injection (tests contrôlés) -+

Chaque étape lance un script dans l'image ``novaria/pipeline`` (construite en
local par ``kubeflow/components/build.sh``), par le lanceur
``novaria.pipeline.remote`` : les dossiers échangés entre étapes passent par
MinIO (``novaria-data/datasets/runs/<run Kubeflow>/``), pas par des artefacts
Kubeflow. Le graphe ne montre donc que les étapes, plus un seul artefact :
``decision`` (``promotion.json``). L'ordre des étapes est donné par
``.after()``.

Les CSV bruts sont lus dans ``novaria-data/raw/`` et les jeux préparés publiés
sous ``datasets/prepared/<empreinte>/``. Runs, rapports et modèles vont dans
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
# Lanceur commun : télécharge les entrées depuis MinIO, lance le script, envoie
# les sorties. Le lanceur Kubeflow ignore l'ENTRYPOINT : commande explicite.
REMOTE = ["python", "-m", "novaria.pipeline.remote"]
OUTPUT = "kubeflow/pipelines/training_pipeline.yaml"


def remote_args(
    bucket: str,
    step: str,
    arguments: list,
    inputs: tuple[str, ...] = (),
    outputs: tuple[str, ...] = (),
    group: str | None = None,
) -> list:
    """Construit les arguments du lanceur ``novaria.pipeline.remote``.

    Args:
        bucket: bucket MinIO des données (paramètre du pipeline).
        step: script de ``novaria.pipeline``.
        arguments: arguments du script ; ``{nom}`` désigne un dossier.
        inputs: dossiers produits par les étapes précédentes.
        outputs: dossiers produits par cette étape.
        group: sous-dossier des sorties (étapes parallèles).

    Returns:
        La liste d'arguments du conteneur.
    """
    args = ["--bucket", bucket, "--inputs", *inputs, "--outputs", *outputs]
    if group is not None:
        args += ["--group", group]
    return [*args, "--", step, *arguments]


# --- Composants ---------------------------------------------------------------
@dsl.container_component
def ingestion(data_bucket: str) -> dsl.ContainerSpec:
    """Étape 1 : lit les CSV bruts dans MinIO et écrit le Parquet et son manifeste.

    Sortie MinIO ``ingested/`` (``measurements.parquet``, ``manifest.json``).

    Args:
        data_bucket: bucket MinIO des données (CSV sous ``raw/``).

    Returns:
        La spécification du conteneur.
    """
    return dsl.ContainerSpec(
        image=PIPELINE_IMAGE,
        command=REMOTE,
        args=remote_args(
            data_bucket,
            "ingestion",
            ["--source-bucket", data_bucket, "--output", "{ingested}"],
            outputs=("ingested",),
        ),
    )


@dsl.container_component
def preparation(data_bucket: str) -> dsl.ContainerSpec:
    """Étape 2 : nettoyage, variables et découpage chronologique ; publication des jeux.

    Sorties MinIO ``prepared/`` (train, val, test, métadonnées) et ``checks/``
    (rapport qualité, journal de vérification).

    Args:
        data_bucket: bucket MinIO des données.

    Returns:
        La spécification du conteneur.
    """
    return dsl.ContainerSpec(
        image=PIPELINE_IMAGE,
        command=REMOTE,
        args=remote_args(
            data_bucket,
            "preparation",
            [
                "--input", "{ingested}",
                "--output", "{prepared}",
                "--report", "{checks}/data_quality.json",
                "--log", "{checks}/data_checks.log",
                "--publish-bucket", data_bucket,
            ],
            inputs=("ingested",),
            outputs=("prepared", "checks"),
        ),
    )  # fmt: skip


@dsl.container_component
def training(data_bucket: str, algorithm: str) -> dsl.ContainerSpec:
    """Étape 3 : entraîne la grille d'un algorithme et garde son meilleur réglage.

    Sortie MinIO ``trained/<algorithm>/`` (modèle retenu, ``comparison.json``,
    tableau Markdown).

    Args:
        data_bucket: bucket MinIO des données.
        algorithm: algorithme de ``configs/training.yaml``.

    Returns:
        La spécification du conteneur.
    """
    return dsl.ContainerSpec(
        image=PIPELINE_IMAGE,
        command=REMOTE,
        args=remote_args(
            data_bucket,
            "training",
            [
                "--input", "{prepared}",
                "--output", "{trained}",
                "--report", "{trained}/training_comparison.md",
                "--algorithms", algorithm,
            ],
            inputs=("prepared",),
            outputs=("trained",),
            group=algorithm,
        ),
    )  # fmt: skip


@dsl.container_component
def injection(data_bucket: str) -> dsl.ContainerSpec:
    """Étape 4a : fige les jeux de test contrôlés (anomalies injectées dans le test).

    Sortie MinIO ``controlled/`` (5 jeux et leur manifeste).

    Args:
        data_bucket: bucket MinIO des données.

    Returns:
        La spécification du conteneur.
    """
    return dsl.ContainerSpec(
        image=PIPELINE_IMAGE,
        command=REMOTE,
        args=remote_args(
            data_bucket,
            "injection",
            ["--input", "{prepared}", "--output", "{controlled}"],
            inputs=("prepared",),
            outputs=("controlled",),
        ),
    )


@dsl.container_component
def evaluation(data_bucket: str) -> dsl.ContainerSpec:
    """Étape 4b : évalue les 3 modèles sur les mêmes tests contrôlés et choisit le meilleur.

    Sortie MinIO ``evaluated/`` (modèles fusionnés, ``evaluation.json``,
    ``selection.json``, rapport).

    Args:
        data_bucket: bucket MinIO des données.

    Returns:
        La spécification du conteneur.
    """
    return dsl.ContainerSpec(
        image=PIPELINE_IMAGE,
        command=REMOTE,
        args=remote_args(
            data_bucket,
            "evaluation",
            [
                "--controlled", "{controlled}",
                "--prepared", "{prepared}",
                "--models", *(f"{{trained}}/{a}" for a in ALGORITHMS),
                "--output", "{evaluated}",
                "--report", "{evaluated}/evaluation.md",
            ],
            inputs=("prepared", "controlled", "trained"),
            outputs=("evaluated",),
        ),
    )  # fmt: skip


@dsl.container_component
def promotion(
    data_bucket: str, decision: dsl.Output[dsl.Artifact]
) -> dsl.ContainerSpec:
    """Étape 5 : enregistre le modèle choisi et le compare à ``@champion``.

    Seule sortie visible dans le graphe Kubeflow : l'artefact ``decision``.

    Args:
        data_bucket: bucket MinIO des données.
        decision: ``promotion.json`` (décision, raisons, métriques, alias).

    Returns:
        La spécification du conteneur.
    """
    return dsl.ContainerSpec(
        image=PIPELINE_IMAGE,
        command=REMOTE,
        args=remote_args(
            data_bucket,
            "promotion",
            [
                "--models", "{evaluated}",
                "--controlled", "{controlled}",
                "--prepared", "{prepared}",
                "--output", decision.path,
            ],
            inputs=("prepared", "controlled", "evaluated"),
        ),
    )  # fmt: skip


# --- Réglages Kubernetes des étapes --------------------------------------------
def configure(
    task: dsl.PipelineTask, name: str, memory: str, cpu: str
) -> dsl.PipelineTask:
    """Applique les réglages communs à une étape du pipeline.

    - adresses de MLflow et MinIO (ConfigMap) et identifiants MinIO (Secret) :
      chaque étape lit ou écrit ses dossiers dans ``novaria-data`` ;
    - identifiant du run Kubeflow (``KFP_RUN_ID``) : préfixe MinIO des dossiers
      du run et tag des runs MLflow ;
    - image locale (``IfNotPresent``, pas de registre) ;
    - limites de mémoire et de CPU ;
    - pas de cache : les CSV de MinIO et le Registry changent sans que les
      paramètres du pipeline changent.

    Args:
        task: étape du pipeline.
        name: nom affiché dans l'interface.
        memory: limite de mémoire (ex. ``3G``).
        cpu: limite de CPU (ex. ``2``).

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
        data_bucket: bucket MinIO des données (CSV bruts, jeux publiés, dossiers
            des runs).
    """
    ingested = configure(ingestion(data_bucket=data_bucket), "1. ingestion", "2G", "1")
    prepared = configure(
        preparation(data_bucket=data_bucket), "2. preparation", "3G", "1"
    ).after(ingested)
    trained = [
        configure(
            training(data_bucket=data_bucket, algorithm=algorithm),
            f"3. training · {label}",
            "3G",
            "2",
        ).after(prepared)
        for algorithm, label in ALGORITHMS.items()
    ]
    controlled = configure(
        injection(data_bucket=data_bucket), "4a. injection (tests contrôlés)", "2G", "1"
    ).after(prepared)
    evaluated = configure(
        evaluation(data_bucket=data_bucket), "4b. evaluation", "4G", "2"
    ).after(*trained, controlled)
    configure(promotion(data_bucket=data_bucket), "5. promotion", "3G", "1").after(
        evaluated
    )


if __name__ == "__main__":
    compiler.Compiler().compile(training_pipeline, OUTPUT)
    print(f"Pipeline compilé : {OUTPUT}")
