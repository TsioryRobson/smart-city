#!/bin/sh
# =============================================================================
# Initialisation PostgreSQL : crée deux bases isolées, chacune avec son propre
# utilisateur (principe du moindre privilège).
#
#   - MLFLOW_DB_NAME : réservée au serveur MLflow (il crée lui-même ses tables)
#   - APP_DB_NAME    : réservée à l'application Urban Hub (API FastAPI)
#
# ATTENTION : exécuté UNE SEULE FOIS, au premier démarrage, quand le volume
# pgdata est vide. Pour le rejouer : docker compose down -v (efface les données).
# =============================================================================
set -eu

psql -v ON_ERROR_STOP=1 \
     --username "$POSTGRES_USER" --dbname postgres \
     -v mlflow_db="$MLFLOW_DB_NAME" -v mlflow_user="$MLFLOW_DB_USER" -v mlflow_pwd="$MLFLOW_DB_PASSWORD" \
     -v app_db="$APP_DB_NAME" -v app_user="$APP_DB_USER" -v app_pwd="$APP_DB_PASSWORD" <<'EOSQL'
-- Base MLflow
CREATE ROLE :"mlflow_user" WITH LOGIN PASSWORD :'mlflow_pwd';
CREATE DATABASE :"mlflow_db" OWNER :"mlflow_user";
REVOKE ALL ON DATABASE :"mlflow_db" FROM PUBLIC;

-- Base applicative Urban Hub
CREATE ROLE :"app_user" WITH LOGIN PASSWORD :'app_pwd';
CREATE DATABASE :"app_db" OWNER :"app_user";
REVOKE ALL ON DATABASE :"app_db" FROM PUBLIC;
EOSQL

echo "Bases '$MLFLOW_DB_NAME' et '$APP_DB_NAME' créées."
