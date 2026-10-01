#!/usr/bin/env bash
# Download the checkpoints SceneRig opens by path or from the Hugging Face cache.
#
# facebook/sam3 and facebook/sam-3d-objects are manually gated. A token that is
# not on the authorized list saves only the public README and license.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

KEY_FILE="${SCENERIG_KEYS_FILE:-$HOME/.zshrc}"
if [ -f "$KEY_FILE" ]; then
  # shellcheck disable=SC2046
  eval "$(grep -E '^[[:space:]]*export[[:space:]]+(HF_TOKEN|HUGGING_FACE_HUB_TOKEN)=' "$KEY_FILE" 2>/dev/null)" || true
fi
export HF_TOKEN="${HF_TOKEN:-${HUGGING_FACE_HUB_TOKEN:-}}"
if [ -z "${HF_TOKEN:-}" ]; then
  echo "Export HF_TOKEN before downloading weights." >&2
  exit 1
fi
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"

HF="$ROOT/.venv/bin/hf"
if [ ! -x "$HF" ]; then
  echo "Missing $HF. Run: bash scripts/install.sh --only main" >&2
  exit 1
fi

download_public() {
  local repo="$1"
  echo "Downloading $repo into the Hugging Face cache"
  "$HF" download "$repo"
}

say_gated() {
  local repo="$1"
  echo "Could not download $repo." >&2
  echo "Request access at https://huggingface.co/$repo while logged in as this token's user, then rerun." >&2
}

SAM3D_HF="lib/utils/third_party/sam3d/checkpoints/hf"
if [ -s "$SAM3D_HF/pipeline.yaml" ]; then
  echo "SAM 3D checkpoints already at $SAM3D_HF/pipeline.yaml"
else
  echo "Downloading facebook/sam-3d-objects"
  tmp="lib/utils/third_party/sam3d/checkpoints/hf-download"
  if "$HF" download --repo-type model --local-dir "$tmp" facebook/sam-3d-objects; then
    if [ -d "$tmp/checkpoints" ]; then
      rm -rf "$SAM3D_HF"
      mv "$tmp/checkpoints" "$SAM3D_HF"
      rm -rf "$tmp"
    elif [ -s "$tmp/pipeline.yaml" ]; then
      rm -rf "$SAM3D_HF"
      mv "$tmp" "$SAM3D_HF"
    fi
    if [ ! -s "$SAM3D_HF/pipeline.yaml" ]; then
      echo "Downloaded facebook/sam-3d-objects but $SAM3D_HF/pipeline.yaml is missing." >&2
      exit 1
    fi
  else
    say_gated facebook/sam-3d-objects
    exit 1
  fi
fi

SAM3_PT="lib/utils/third_party/sam3/checkpoints/sam3.pt"
if [ -s "$SAM3_PT" ]; then
  echo "SAM 3 checkpoint already at $SAM3_PT"
else
  echo "Downloading facebook/sam3"
  mkdir -p "$(dirname "$SAM3_PT")"
  if ! "$HF" download facebook/sam3 sam3.pt --local-dir "$(dirname "$SAM3_PT")"; then
    say_gated facebook/sam3
    exit 1
  fi
fi

download_public allenai/MolmoPoint-8B
download_public Qwen/Qwen-Image-Edit-2509

echo
echo "Weights ready. scripts/run_e2e.sh uses $SAM3_PT unless SAM3_CHECKPOINT is already set."
