# NOVARIA Smart City AI — Détection d'anomalies de qualité de l'air

POC de Master 2 : détection non supervisée de mesures IoT atypiques (dataset
Beijing Multi-Site Air Quality), avec MLflow, Kubeflow Pipelines, FastAPI et
une chaîne DevSecOps (GitHub Actions, SonarCloud, Snyk).

> Ce README est complété au fil des étapes du projet.

---

## Prérequis

- Windows 10/11 avec **WSL2** (Ubuntu) et **Docker Desktop** (moteur WSL2)
- **VS Code** avec l'extension **Dev Containers**
- Git

Le projet doit être placé **dans le système de fichiers WSL** (par exemple
`~/smart-city-ai` dans Ubuntu), pas sur `C:\` : les volumes Docker y sont
beaucoup plus rapides et les droits sur les fichiers sont corrects.

---

## Étape 2 — Infrastructure de base

### Ce qui est lancé

| Service    | Rôle                                                        | Accès depuis Windows     |
|------------|-------------------------------------------------------------|--------------------------|
| `postgres` | PostgreSQL 16 : bases `mlflow` et `urbanhub` (une par usage) | `localhost:5432`         |
| `mlflow`   | Tracking, Model Registry, stockage des artefacts             | http://localhost:5000    |
| `dev`      | Environnement Python 3.12 + bibliothèques ML figées          | via VS Code              |

```
Conteneur dev ──HTTP──▶ MLflow ──SQL──▶ PostgreSQL (base mlflow)
                          │
                          └──▶ volume mlartifacts (fichiers des modèles)
```

Les artefacts (modèles, graphiques) transitent **par le serveur MLflow**
(mode *proxied artifacts*) : tout client — conteneur dev, pod Kubeflow, API —
n'a besoin que de l'adresse de MLflow.

### Démarrage

1. Créer le fichier de configuration et **changer les mots de passe** :
   ```bash
   cp .env.example .env
   ```
2. Ouvrir le dossier dans VS Code, puis : `F1` → **Dev Containers: Reopen in Container**.
   Le premier lancement construit les images (quelques minutes).

   *Sans VS Code :* `docker compose up -d --build`, puis
   `docker compose exec dev bash`.
3. Ouvrir http://localhost:5000 : l'interface MLflow doit s'afficher.

### Validation

Dans un terminal du conteneur dev :

```bash
python scripts/check_stack.py
```

Résultat attendu :

```
[OK] Serveur MLflow joignable sur http://mlflow:5000 (version 3.17.2)
[OK] Run enregistré (run_id=...)
[OK] Artefact envoyé puis relu via le serveur (proxied artifacts)
[OK] Modèle 'smoke-test-model' version 1 enregistré, alias @ok
[OK] Modèle rechargé via models:/smoke-test-model@ok et prédictions OK

Étape 2 validée : l'infrastructure MLflow est prête.
```

Dans l'interface MLflow, on retrouve l'expérience `00-smoke-test` et le
modèle `smoke-test-model` dans l'onglet *Models*.

Vérifier les deux bases PostgreSQL :

```bash
docker compose exec postgres psql -U postgres -c "\l"
```

### Commandes utiles

| Action                                   | Commande                                   |
|------------------------------------------|--------------------------------------------|
| Démarrer                                 | `docker compose up -d`                     |
| Arrêter (données conservées)             | `docker compose down`                      |
| Tout réinitialiser (**efface les données**) | `docker compose down -v`                |
| Logs de MLflow                           | `docker compose logs -f mlflow`            |
| Consommation RAM/CPU                     | `docker stats`                             |

### Choix techniques

- **PostgreSQL unique, deux bases, deux utilisateurs** : isolation entre
  MLflow et l'application (moindre privilège), un seul conteneur à gérer.
- **Image MLflow construite localement** : l'image officielle n'inclut pas le
  pilote PostgreSQL ; les versions sont figées.
- **`MLFLOW_SERVER_ALLOWED_HOSTS`** : depuis MLflow 3.5, le serveur refuse les
  noms d'hôte inconnus (protection DNS rebinding). La liste autorise le réseau
  Compose (`mlflow`), la machine locale et `host.docker.internal` (Kubeflow).
- **Sérialisation skops** : MLflow enregistre les modèles scikit-learn au
  format skops, plus sûr que pickle. Les modèles à base d'arbres (Isolation
  Forest) déclarent `skops_trusted_types=["sklearn.tree._tree.Tree"]`.
- **Limites mémoire** (512 Mo / 1 Go / 3 Go) et **un seul worker MLflow** :
  compatibilité avec la contrainte d'un laptop 8 Go.
- **Ports liés à `127.0.0.1`** : services non exposés sur le réseau.
- **Utilisateurs non-root** dans les images.

### Dépendances Python

Les versions sont figées dans des fichiers `.txt` générés à partir des `.in` :

| Fichier source                | Fichier figé                     | Utilisé par                         |
|-------------------------------|----------------------------------|-------------------------------------|
| `requirements/ml.in`          | `requirements/ml.txt`            | dev, futur training Kubeflow, API   |
| `requirements/dev.in`         | `requirements/dev.txt`           | dev uniquement (pytest, ruff…)      |
| `infra/mlflow/requirements.in`| `infra/mlflow/requirements.txt`  | serveur MLflow                      |

Pour ajouter une bibliothèque : modifier le `.in`, puis régénérer (dans le conteneur dev) :

```bash
pip install uv
cd requirements
uv pip compile ml.in --python-version 3.12 --no-header --annotation-style line -o ml.txt
uv pip compile dev.in --python-version 3.12 --no-header --annotation-style line -o dev.txt
```

puis reconstruire : `docker compose build dev`.

### Dépannage

| Symptôme | Cause probable | Solution |
|----------|----------------|----------|
| `Invalid Host header` / erreur 403 de MLflow | Nom d'hôte absent de `MLFLOW_ALLOWED_HOSTS` | Ajouter le nom dans `.env`, puis `docker compose up -d mlflow` |
| Bases `mlflow`/`urbanhub` absentes | Le script d'init ne s'exécute qu'au **premier** démarrage | `docker compose down -v` puis relancer |
| `set: Illegal option -` au démarrage de postgres | Fin de ligne Windows (CRLF) dans le script `.sh` | Vérifier `.gitattributes`, re-cloner, ou `dos2unix` |
| Erreurs de droits sur les fichiers | Projet sur `C:\` au lieu de WSL | Déplacer le projet dans `~/` sous Ubuntu |
| `password authentication failed` | `.env` modifié après la création du volume | `docker compose down -v` puis relancer |
