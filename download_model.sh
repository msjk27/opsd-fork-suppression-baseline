#!/usr/bin/env bash
# Download a trained checkpoint published as a GitHub Release asset set and verify its checksums.
# Usage: bash download_model.sh [DEST_DIR] [TAG]
#   default TAG  = model-opsd-vanilla-s42 (Qwen3-1.7B after vanilla OPSD, seed 42; see PLAN.md)
#   default DEST = models/qwen3-1.7b-opsd-vanilla-s42
set -euo pipefail
TAG=${2:-model-opsd-vanilla-s42}
DEST=${1:-models/qwen3-1.7b-opsd-vanilla-s42}
BASE=https://github.com/msjk27/opsd-fork-suppression-baseline/releases/download/$TAG

mkdir -p "$DEST"
cd "$DEST"
curl -fL --retry 3 -o SHA256SUMS "$BASE/SHA256SUMS"
for f in $(awk '{print $2}' SHA256SUMS); do
  if [ -f "$f" ] && grep " $f\$" SHA256SUMS | sha256sum -c --status - ; then
    echo "ok (cached): $f"; continue
  fi
  curl -fL --retry 3 -C - -o "$f" "$BASE/$f"
done
sha256sum -c SHA256SUMS
echo "Model ready at $(pwd)"
