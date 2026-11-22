#!/usr/bin/env bash
# Régénère les manifestes figés de Kubeflow Pipelines (kubeflow/platform/upstream-<version>/).
#
# À lancer seulement pour changer de version de KFP. Les manifestes officiels
# vont chercher ceux d'Argo sur GitHub pendant le rendu, et kustomize abandonne
# au bout de 27 s : on réessaie jusqu'à 5 fois.
#
# Usage : ./kubeflow/render-manifests.sh [version]   (défaut 2.18.0)
set -euo pipefail

KFP_VERSION="${1:-2.18.0}"
HERE="$(cd "$(dirname "$0")" && pwd)"
UPSTREAM="$HERE/upstream"
OUT="$HERE/platform/upstream-$KFP_VERSION"

rm -rf "$UPSTREAM"
git clone -q --depth 1 --branch "$KFP_VERSION" --filter=blob:none --sparse \
  https://github.com/kubeflow/pipelines.git "$UPSTREAM"
git -C "$UPSTREAM" sparse-checkout set manifests/kustomize
mkdir -p "$OUT"

render() {
  # Rend un dossier kustomize des manifestes officiels dans un fichier figé.
  local source="$1" target="$2"
  for attempt in 1 2 3 4 5; do
    if kubectl kustomize "$UPSTREAM/manifests/kustomize/$source" > "$target.tmp"; then
      {
        echo "# Rendu figé de kubeflow/pipelines $KFP_VERSION : manifests/kustomize/$source"
        echo "# Généré par kubeflow/render-manifests.sh — ne pas modifier à la main."
        cat "$target.tmp"
      } > "$target"
      rm -f "$target.tmp"
      echo "OK : $source ($(grep -c '^kind:' "$target") objets)"
      return 0
    fi
    echo "Échec du rendu de $source ($attempt/5), nouvel essai…" >&2
    sleep 5
  done
  return 1
}

render cluster-scoped-resources "$OUT/cluster-scoped.yaml"
render env/platform-agnostic "$OUT/platform-agnostic.yaml"
echo "Penser à mettre à jour la version dans platform/kustomization.yaml et install.sh."
