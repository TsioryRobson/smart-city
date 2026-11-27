# Inférence et démonstration HTTP

## Démonstration locale

Depuis la racine du dépôt sous WSL, installez les dépendances et placez les CSV Beijing dans `data/raw/`. Le [guide du simulateur](simulator.md) détaille les scénarios et les options de rejeu.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.lock.txt
python -m pip install -e .
NOVARIA_MODEL_PROVIDER=mock python -m uvicorn novaria.api.main:app --host 127.0.0.1 --port 8000
```

Dans un deuxième terminal, depuis la racine du dépôt :

```bash
source .venv/bin/activate
curl -fsS http://127.0.0.1:8000/health
python -m novaria.simulator --transport http --limit 1
python -m novaria.simulator --transport http --scenario pic --limit 1
python -m novaria.simulator --transport http --scenario valeur_impossible --limit 1
python -m novaria.simulator --transport http --scenario combinaison_inhabituelle --limit 1
python -m novaria.simulator --transport http --scenario valeur_manquante --limit 1
python -m pytest -q
ruff check .
```

Swagger est accessible sur [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs). En mode `mock`, une règle déterministe fournit un score de démonstration. Les réponses portent `is_mock: true`, `model_provider: "mock"` et `model_version: "mock-v1"`.

## Fournisseurs de modèles

| `NOVARIA_MODEL_PROVIDER` | Chargement |
|---|---|
| `mlflow` (défaut) | Résout `air-anomaly-detector@champion` au démarrage, puis charge la version résolue après vérification du tag `model_sha256`. `MLFLOW_TRACKING_URI` sélectionne le serveur ou la base locale. `MLFLOW_REGISTRY_URI` peut cibler un registre distinct. |
| `local` | Charge les sorties d'entraînement dans `NOVARIA_MODELS_DIR` (défaut `models/`). Utilise l'algorithme de `selection.json` ou celui de `NOVARIA_MODEL_ALGORITHM`. Le fichier `comparison.json` fournit l'empreinte de l'artefact. |
| `mock` | Mode explicite pour le développement et les tests, avec des scores fixes. |

```bash
MLFLOW_TRACKING_URI=http://127.0.0.1:5000 \
MLFLOW_HTTP_REQUEST_TIMEOUT=5 MLFLOW_HTTP_REQUEST_MAX_RETRIES=0 \
python -m uvicorn novaria.api.main:app

NOVARIA_MODEL_PROVIDER=local NOVARIA_MODELS_DIR="$PWD/models" \
python -m uvicorn novaria.api.main:app
```

Chaque processus charge le pipeline une fois au démarrage. Redémarrez le service après avoir modifié l'alias ou changé l'artefact. Si le chargement échoue, `/health` indique `model_loaded: false` et les deux routes d'inférence et de santé renvoient HTTP 503 pour les requêtes valides.

## Contrat d'inférence

`POST /predict` accepte les sept mesures `pm25`, `pm10`, `no2`, `co`, `so2`, `o3`, `temp` et les champs météo optionnels `pres`, `dewp`, `rain`, `wd`, `wspm`. Les sept clés principales sont obligatoires et acceptent `null` pour une valeur manquante. Un identifiant est généré pour les anciens clients qui omettent `event_id`. Un horodatage sans fuseau est interprété en heure locale de Pékin.

L'API appelle `prepare_record` et transmet les 17 colonnes dans l'ordre de `MODEL_FEATURES`. Le pipeline enregistré effectue l'imputation et la mise à l'échelle. Le score calibré est négatif lorsque `is_anomaly` vaut `true`. Chaque algorithme et chaque version conserve sa propre échelle.

- Les erreurs de format, de type et les champs inconnus produisent une réponse HTTP 422.
- Les mesures explicitement manquantes ou physiquement incohérentes donnent une réponse HTTP 200 avec `data_quality_issues`. La préparation partagée écarte les valeurs incohérentes avant l'imputation. Les mesures météo facultatives omises sont imputées.
- Les dépassements métier figurent dans `health_exceedances`. `NOVARIA_HOURLY_LIMITS` accepte un objet JSON associant les polluants à des seuils horaires en µg/m³. Par défaut, `health_exceedances` vaut `null`. Le paramétrage doit correspondre aux seuils horaires retenus pour la démo.
- `anomaly_type` et son équivalent historique `alert_type` donnent la priorité à `defaut_capteur`, puis à `seuil_sanitaire`, puis à `pollution_atypique`. `is_anomaly` reste le verdict du détecteur et les dépassements sont reportés indépendamment.

## Docker Compose

Pour la démo sans le reste de la stack :

```bash
NOVARIA_MODEL_PROVIDER=mock docker compose -f docker-compose.api.yml up -d --build api
curl -fsS http://127.0.0.1:8000/health
docker compose -f docker-compose.api.yml stop api
```

Pour utiliser le registre intégré, préparez le `.env` de la stack existante et démarrez les services sur le même réseau Compose :

```bash
docker compose up -d --build --wait
docker compose -f docker-compose.yml -f docker-compose.api.yml up -d --build api
```

Le conteneur utilise `http://mlflow:5000` par défaut. `NOVARIA_API_MLFLOW_URI` permet de changer l'adresse côté conteneur. Pour monter un dossier de modèles locaux en lecture seule :

```bash
NOVARIA_MODEL_PROVIDER=local docker compose -f docker-compose.api.yml run --rm --build \
  --service-ports -v "$PWD/models:/app/models:ro" api
```
