#!/usr/bin/env bash
# Installe Kubeflow Pipelines 2.18.0 standalone sur le Kubernetes local.
#
# Prérequis : Kubernetes de Docker Desktop (mode kubeadm) actif, contexte
# kubectl « docker-desktop ». Avec une connexion lente, lancer d'abord
# ./kubeflow/pull-images.sh : le cluster réutilise les images de Docker.
# Les manifestes sont figés dans kubeflow/platform/ : aucun accès à GitHub.
#
# Usage : ./kubeflow/install.sh   (depuis la racine du dépôt)
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
PLATFORM="$HERE/platform"
MANIFESTS="$PLATFORM/upstream-2.18.0"
CONTEXT="docker-desktop"

# Sécurité : ne jamais installer sur un autre cluster par erreur.
current="$(kubectl config current-context)"
if [ "$current" != "$CONTEXT" ]; then
  echo "Contexte kubectl « $current » : attendu « $CONTEXT ». Arrêt." >&2
  exit 1
fi

echo "1/3 Ressources de niveau cluster (CRD, namespace kubeflow)"
kubectl apply -f "$MANIFESTS/cluster-scoped.yaml"
kubectl wait --for condition=established --timeout=120s crd/applications.app.k8s.io

echo "2/3 Kubeflow Pipelines (sans le serveur de visualisation)"
kubectl apply -k "$PLATFORM"

echo "3/3 Attente des déploiements (jusqu'à 20 min si des images manquent)"
kubectl -n kubeflow wait --for=condition=Available --timeout=1200s deployment --all

echo "Kubeflow Pipelines prêt. Interface :"
echo "  kubectl -n kubeflow port-forward svc/ml-pipeline-ui 8080:80"
echo "  puis http://localhost:8080"
