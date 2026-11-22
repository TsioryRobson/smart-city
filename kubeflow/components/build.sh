#!/usr/bin/env bash
# Construit l'image des composants du pipeline d'entraînement (novaria/pipeline).
# Le commit embarqué (tag MLflow git_sha) porte le suffixe -dirty si le dépôt
# a des modifications non commitées, comme dans novaria.common.tracking.git_sha.
#   ./kubeflow/components/build.sh [tag]
set -euo pipefail
cd "$(dirname "$0")/../.."

TAG="${1:-novaria/pipeline:0.1.0}"
SHA="$(git rev-parse HEAD)"
if [ -n "$(git status --porcelain)" ]; then
  SHA="${SHA}-dirty"
fi

docker build -f kubeflow/components/Dockerfile --build-arg GIT_SHA="${SHA}" -t "${TAG}" .
echo "Image ${TAG} construite (git_sha ${SHA})"
