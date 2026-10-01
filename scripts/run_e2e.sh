#!/usr/bin/env bash
# Usage:
#   scripts/run_e2e.sh <image_path> <output_dir> <gpu> [static_scene.py args...]
#
# Examples:
#   scripts/run_e2e.sh dataset_selected/abc/1.png output/abc_1 0
#   scripts/run_e2e.sh input.png output/run 0 --depth depth.npy --camera intrinsics.json
#   scripts/run_e2e.sh input.png output/run_gpt6 0 --gpt6-harness

set -euo pipefail

if [ "$#" -lt 3 ]; then
    echo "Usage: scripts/run_e2e.sh <image_path> <output_dir> <gpu> [args...]" >&2
    exit 2
fi

IMAGE="$1"
OUTPUT_DIR="$2"
GPU="$3"
shift 3

SCENERIG_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SCENERIG_ROOT"

if [ ! -f "$IMAGE" ]; then
    echo "[SceneRig] image not found: $IMAGE" >&2
    exit 1
fi

KEY_FILE="${SCENERIG_KEYS_FILE:-${GRASE_KEYS_FILE:-$HOME/.zshrc}}"
eval "$(grep -E '^[[:space:]]*export[[:space:]]+(HF_TOKEN|HUGGING_FACE_HUB_TOKEN|CLAUDE_API_KEY|CLAUDE_BASE_URL|ANTHROPIC_API_KEY|OPENAI_API_KEY|OPENAI_BASE_URL|FIREWORKS_API_KEY|FIREWORKS_BASE_URL|GEMINI_API_KEY|GEMINI_BASE_URL|QWEN_API_KEY|QWEN_BASE_URL)=' "$KEY_FILE" 2>/dev/null)" || true

export HF_TOKEN="${HF_TOKEN:-${HUGGING_FACE_HUB_TOKEN:-}}"
export HUGGING_FACE_HUB_TOKEN="${HUGGING_FACE_HUB_TOKEN:-$HF_TOKEN}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-}"
export CLAUDE_API_KEY="${CLAUDE_API_KEY:-${ANTHROPIC_API_KEY:-}}"

if [ -z "${SAM3_CHECKPOINT:-}" ] && [ -s lib/utils/third_party/sam3/checkpoints/sam3.pt ]; then
    export SAM3_CHECKPOINT="$SCENERIG_ROOT/lib/utils/third_party/sam3/checkpoints/sam3.pt"
fi

export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-0}"
unset HUGGINGFACE_HUB_CACHE
export HF_HOME="${HF_HOME:-${XDG_CACHE_HOME:-$HOME/.cache}/huggingface}"

export CUDA_CACHE_PATH="${CUDA_CACHE_PATH:-$HOME/.nv/ComputeCache}"
export CUDA_CACHE_MAXSIZE="${CUDA_CACHE_MAXSIZE:-4294967296}"

if [ -d .venv ]; then
    # shellcheck disable=SC1091
    source .venv/bin/activate
fi

exec python lib/runners/static_scene.py \
    --image "$IMAGE" \
    --output-dir "${OUTPUT_DIR%/}" \
    --gpu "$GPU" \
    --model "${SCENERIG_MODEL:-claude-opus-5}" \
    --blender-command "${SCENERIG_BLENDER_COMMAND:-${GRASE_BLENDER_COMMAND:-lib/utils/third_party/blender-4.5/blender}}" \
    "$@"
