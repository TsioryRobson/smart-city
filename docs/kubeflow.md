# Kubernetes et Kubeflow Pipelines (sprint 2)

Kubeflow Pipelines **2.18.0 standalone** sur le Kubernetes de Docker Desktop. Les pods utilisent les services du
sprint 1 (MLflow, MinIO) lancés par Docker Compose.

## Architecture

| Élément | Où | Rôle |
|---|---|---|
| Kubernetes 1.34 | Docker Desktop, mode **kubeadm**, 1 nœud | Exécute Kubeflow et les étapes des pipelines |
| Kubeflow Pipelines | namespace `kubeflow` | API, interface, orchestrateur Argo, MySQL et SeaweedFS **internes** (métadonnées et fichiers échangés entre étapes) |
| MLflow | Docker Compose, `host.docker.internal:5000` | Suivi des runs et Model Registry (comme au sprint 1) |
| MinIO | Docker Compose, `host.docker.internal:9000` | Bucket `novaria-data` : CSV bruts (`raw/`), jeux produits par le pipeline (`datasets/`) |
| PostgreSQL | Docker Compose, **aucun port publié** | Utilisé seulement par MLflow ; inaccessible aux pods |

## Installation

```bash
./kubeflow/pull-images.sh        # télécharge les 17 images à l'avance (≈ 1,4 Go)
./kubeflow/install.sh            # Kubeflow Pipelines 2.18.0 (≈ 1 min si les images sont là)
set -a; . ./.env; set +a
python infra/minio/setup_pipeline_access.py   # bucket novaria-data, compte du pipeline, CSV
./kubeflow/configure-access.sh   # ConfigMap + Secret dans le namespace kubeflow
kubectl -n kubeflow port-forward svc/ml-pipeline-ui 8080:80   # interface : http://localhost:8080
```

Validation (pipeline à 2 étapes en parallèle, image `novaria/kfp-check`) :

```bash
docker build -t novaria/kfp-check:0.1.0 kubeflow/check
.venv-kfp/bin/python kubeflow/pipelines/validation_pipeline.py
.venv-kfp/bin/python kubeflow/pipelines/run.py kubeflow/pipelines/validation_pipeline.yaml
```

Résultat (2026-11-22) : run `SUCCEEDED`. `check-minio` lit les 12 CSV, écrit sous `datasets/`, se voit refuser
l'écriture sous `raw/` ; `check-mlflow` lit `air-anomaly-detector@champion` et enregistre un run dans l'expérience
MLflow `kfp-validation`.

## Choix

| Décision | Justification |
|---|---|
| Kubernetes de Docker Desktop en mode **kubeadm** | Le cluster utilise directement les images de Docker : nos images de composants sont construites en local, **sans registre**. Le mode kind aurait imposé le *containerd image store* et le retéléchargement de toutes les images. |
| KFP **2.18.0** (pas 2.18.1) | Version publiée depuis plus de deux semaines (2026-10-28) ; la 2.18.1 n'avait que 4 jours. |
| Manifestes **rendus et figés** dans `kubeflow/platform/upstream-2.18.0/` | Les manifestes officiels vont chercher ceux d'Argo sur GitHub au rendu, et kustomize abandonne après 27 s. Figés : installation sans réseau, reproductible, et manifestes relisibles (revue de sécurité). Régénération : `./kubeflow/render-manifests.sh`. |
| Déploiement `kfp-visualization-server` retiré | Image de 881 Mo, utilisée seulement par les visualisations de Kubeflow v1. Son **Service** est gardé (sans pod) : le serveur d'API refuse de démarrer sans la variable `ML_PIPELINE_VISUALIZATIONSERVER_SERVICE_HOST`. |
| Images téléchargées à l'avance (`pull-images.sh`) | Connexion lente : téléchargement reprenable, image par image, avant l'installation. |
| Pods → services Compose par `host.docker.internal` | Nom fourni par Docker Desktop pour joindre la machine ; ajouté à `--allowed-hosts` de MLflow (sinon 403). |
| **PostgreSQL non publié** (`docker-compose.sql-access.yml` pour un accès ponctuel) | Un port publié est joignable depuis les pods, et le réseau de Docker Desktop n'applique pas les NetworkPolicy. Seul MLflow doit parler à la base. |
| Compte MinIO **dédié** au pipeline (`novaria-pipeline`) | Lecture du bucket `novaria-data`, écriture seulement sous `datasets/` ; ni `raw/`, ni bucket `mlflow`, ni administration (vérifié par un test et par le pipeline de validation). |
| Secret injecté **seulement** dans les étapes qui en ont besoin | `check-mlflow` ne reçoit pas les identifiants MinIO (`kfp-kubernetes`, `use_secret_as_env`). MLflow sert lui-même les artefacts : les étapes n'ont besoin que de son URL. |
| SDK `kfp` dans un venv séparé (`.venv-kfp`) | `kfp` 2.18 impose `protobuf<7`, alors que MLflow utilise protobuf 7 dans `.venv`. Les pipelines lancent nos scripts dans des conteneurs : le SDK ne sert qu'à compiler et soumettre. |
| Composants « conteneur » avec **commande explicite** | Le lanceur KFP ignore l'`ENTRYPOINT` de l'image et exécute directement `command` + `args`. |
| Pas de mise en cache pour la validation | Chaque run de validation doit vraiment tester les accès. |

## Pipeline d'entraînement

Branche `feature/s2-kubeflow-pipeline`. Les scripts de `novaria.pipeline` deviennent des étapes, sans réécrire
leur logique : chaque étape lance le script avec ses arguments dans l'image `novaria/pipeline`.

```
1. ingestion -> 2. preparation -+-> 3. training · Isolation Forest -+
                                +-> 3. training · LOF               +-> 4b. evaluation -> 5. promotion
                                +-> 3. training · One-Class SVM     |
                                +-> 4a. injection (tests contrôlés) +
```

| Étape | Script et arguments | Entrées | Sorties (artefacts Kubeflow) |
|---|---|---|---|
| 1. ingestion | `ingestion --source-bucket novaria-data` | CSV de `novaria-data/raw/` (MinIO) | `ingested` (Parquet + manifeste) |
| 2. preparation | `preparation --publish-bucket novaria-data` | `ingested` | `prepared` (train/val/test), `quality` (JSON), `checks_log` (.log) ; copie dans `datasets/prepared/<empreinte>/` |
| 3. training ×3 | `training --algorithms <algo>` | `prepared` | `trained` (modèle retenu + `comparison.json`), `report` (Markdown) |
| 4a. injection | `injection` | `prepared` (test seulement) | `controlled` (5 jeux + manifeste) |
| 4b. evaluation | `evaluation --models <3 dossiers> --output` | `prepared`, `controlled`, les 3 `trained` | `evaluated` (modèles fusionnés, `selection.json`), `report` (Markdown) |
| 5. promotion | `promotion --output` | `prepared`, `controlled`, `evaluated` | `decision` (`promotion.json`) ; version et alias dans le Registry |

```bash
./kubeflow/components/build.sh                                   # image novaria/pipeline:0.1.0
.venv-kfp/bin/python kubeflow/pipelines/training_pipeline.py     # -> training_pipeline.yaml
.venv-kfp/bin/python kubeflow/pipelines/run.py kubeflow/pipelines/training_pipeline.yaml --timeout 3600
```

### Choix

| Décision | Justification |
|---|---|
| Une étape par algorithme (`--algorithms`) | Les 3 entraînements tournent en parallèle et apparaissent séparément dans le graphe ; un échec n'invalide que son algorithme. Même configuration, même train, même validation contrôlée : le protocole commun est conservé. |
| Fusion dans `evaluation.py` (`--models` multiple) | `promotion.py` et l'API continuent de lire un seul dossier (`comparison.json` + un sous-dossier par modèle). La fusion refuse des entraînements faits sur des données ou un protocole différents (`dataset_sha256`, `train_period`, contamination, métrique). |
| Injection = étape à part | Les jeux contrôlés ne dépendent que du test : ils sont construits pendant les entraînements et visibles dans le graphe. Toujours jamais sur le train. |
| Données d'entrée lues dans MinIO | Règle du projet : les pods ne lisent pas les fichiers locaux. Empreintes calculées sur les fichiers téléchargés : même `dataset_sha256` qu'en local (vérifié : `439a2ed197c6629a`). |
| Jeux préparés publiés sous `datasets/prepared/<empreinte>/` | Version du jeu = son empreinte : republier le même jeu ne change rien, un nouveau jeu a un nouveau dossier. Seul préfixe où le compte du pipeline peut écrire. |
| Une image pour toutes les étapes (`novaria/pipeline`) | Même code et mêmes versions de bibliothèques partout ; `PYTHONPATH=/app/src`, configurations dans `/app/configs`, utilisateur non-root 10001. Wheels montés le temps de l'installation (pas dans l'image). |
| Commit embarqué (`NOVARIA_GIT_SHA`) | Pas de dépôt Git dans l'image : `build.sh` passe le commit (suffixe `-dirty` si modifications), repris dans le tag `git_sha` des runs. |
| Tag MLflow `kfp_run_id` | Chaque run MLflow (entraînement, évaluation, promotion) pointe vers le run Kubeflow qui l'a produit (label `pipeline/runid` du pod, par l'API descendante de Kubernetes). |
| Identifiants MinIO dans 2 étapes seulement | Ingestion et préparation. Les autres n'ont besoin que de l'URL de MLflow, qui sert lui-même les artefacts. |
| Pas de cache Kubeflow | Les entrées réelles (CSV dans MinIO, `@champion` du Registry) changent sans que les paramètres du pipeline changent : un résultat en cache serait faux. |
| Limites mémoire / CPU par étape | 3 Go et 2 CPU par entraînement, 4 Go pour l'évaluation : les étapes parallèles tiennent dans les 15 Go de la machine virtuelle de Docker. |
| Rapports en `dsl.Markdown` | Affichés dans l'onglet « Visualization » de l'étape. Kubeflow 2.18 ne résout pas `ConcatPlaceholder` (chemin calculé dans un dossier d'artefact) : chaque fichier est un artefact à part. |

### Run de référence (2026-11-22)

Run `28a79e85-c321-4116-bae8-97037d269aac` : **SUCCEEDED** en 21 min 30 s, fenêtre de référence (train 2014-09 → 2015-08).

| Étape | Durée |
|---|---|
| ingestion | 15 s |
| preparation | 27 s |
| training · Isolation Forest / LOF / One-Class SVM (en parallèle) | 3 min 30 / 9 min 50 / 10 min 30 |
| injection | 45 s |
| evaluation | 7 min 30 |
| promotion | 55 s |

- Modèle choisi : One-Class SVM (mêmes hyperparamètres qu'au sprint 1).
- `air-anomaly-detector` **v3 promue** par la règle (sans `--force`) : détection des pics 95,9 % contre 95,9 %,
  fausses alertes 2,94 % contre 2,94 %, latence 12,5 ms. Mêmes données et même protocole que la v2 : résultats
  identiques, ce qui confirme la reproductibilité entre le local et Kubeflow. Alias : `@champion` = v3,
  `@previous` = v2 (v1 n'a plus d'alias mais reste dans le Registry).
- 28 runs MLflow portent `kfp_run_id` (23 entraînement, 4 évaluation, 1 promotion).
- Un premier run (`63a71da4…`) a échoué à la préparation à cause de `ConcatPlaceholder` ; il reste visible dans l'interface.

### Voir le pipeline

1. `kubectl -n kubeflow port-forward svc/ml-pipeline-ui 8080:80`, puis http://localhost:8080.
2. **Runs** (ou **Experiments → novaria**) → run `novaria-training` : graphe des 8 étapes.
3. Cliquer une étape : **Logs** (sortie du script), **Input/Output** (artefacts), **Visualization** pour les
   rapports Markdown (entraînement, évaluation).
4. Côté MLflow (http://localhost:5000) : filtre `tags.kfp_run_id = "<id du run>"` ; Registry → `air-anomaly-detector`.

## Ressources mesurées

| Ensemble | Mémoire |
|---|---|
| Kubeflow Pipelines (13 pods) | ≈ 0,85 Go |
| Image `novaria/pipeline` | 1,3 Go (disque) |
| Stack MLflow (PostgreSQL, MinIO, MLflow) | ≈ 0,47 Go |

## Sécurité : points relevés dans les manifestes officiels (à traiter au sprint 3)

- Stockage interne SeaweedFS : identifiants par défaut `minio` / `minio123` (Secret `mlpipeline-minio-artifact`).
- MySQL interne : compte `root` **sans mot de passe** (Secret `mysql-secret`).
- Ces services ne sont joignables que depuis le cluster (pas de port publié) ; l'interface n'est exposée que par
  `port-forward` sur la machine. Ce sont de bons exemples pour l'« avant / après » DevSecOps.
- Hors projet : un PostgreSQL installé sur Windows écoute sur toutes les interfaces (port 5432) et reste joignable
  depuis les pods.

## Reprendre après un arrêt

```bash
docker compose up -d                                         # MLflow, MinIO, PostgreSQL
kubectl -n kubeflow get pods                                 # Kubeflow redémarre avec Docker Desktop
kubectl -n kubeflow port-forward svc/ml-pipeline-ui 8080:80  # interface
```

Désinstaller : `kubectl delete -k kubeflow/platform && kubectl delete -f kubeflow/platform/upstream-2.18.0/cluster-scoped.yaml`.
