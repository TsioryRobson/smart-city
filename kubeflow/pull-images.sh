#!/usr/bin/env bash
# Télécharge à l'avance les images de Kubeflow Pipelines 2.18.0 dans Docker.
# Avec Kubernetes de Docker Desktop (kubeadm), le cluster utilise directement
# ces images : l'installation ne télécharge plus rien ensuite.
# Les plus petites d'abord ; une image déjà présente est sautée ; 3 essais.
# kfp-visualization-server (881 Mo) est volontairement absent : non installé.
set -u
IMAGES=(
  docker.io/alpine:3.23
  ghcr.io/kubeflow/kfp-viewer-crd-controller:2.18.0
  ghcr.io/kubeflow/kfp-cache-deployer:2.18.0
  ghcr.io/kubeflow/kfp-persistence-agent:2.18.0
  ghcr.io/kubeflow/kfp-cache-server:2.18.0
  ghcr.io/kubeflow/kfp-scheduled-workflow-controller:2.18.0
  quay.io/argoproj/workflow-controller:v4.1.6
  ghcr.io/kubeflow/kfp-launcher:2.18.0
  ghcr.io/kubeflow/kfp-driver:2.18.0
  gcr.io/tfx-oss-public/ml_metadata_store_server:1.14.0
  quay.io/argoproj/argoexec:v4.1.6
  ghcr.io/kubeflow/kfp-metadata-writer:2.18.0
  ghcr.io/kubeflow/kfp-api-server:2.18.0
  ghcr.io/kubeflow/kfp-metadata-envoy:2.18.0
  ghcr.io/kubeflow/kfp-frontend:2.18.0
  ghcr.io/chrislusf/seaweedfs:4.51@sha256:f7a49e3f3972ebed5be2ed4befa4a0c68eb7bdddba6e692d42729ceb545134b8
  mysql:8.4
)
failed=0
for image in "${IMAGES[@]}"; do
  if docker image inspect "$image" >/dev/null 2>&1; then
    echo "$(date +%T) déjà présente : $image"; continue
  fi
  ok=0
  for attempt in 1 2 3; do
    echo "$(date +%T) téléchargement ($attempt/3) : $image"
    if docker pull -q "$image" >/dev/null; then ok=1; break; fi
    sleep 10
  done
  if [ "$ok" = 1 ]; then echo "$(date +%T) OK : $image"; else echo "$(date +%T) ÉCHEC : $image"; failed=$((failed+1)); fi
done
echo "$(date +%T) terminé, échecs : $failed"
exit "$failed"
