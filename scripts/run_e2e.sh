#!/usr/bin/env bash
# End-to-end SceneRig run for one image.
#
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

# Bash does not source zsh startup files. Load common exports from the user's key
# file for local runs; explicit environment variables still win.
KEY_FILE="${SCENERIG_KEYS_FILE:-${GRASE_KEYS_FILE:-$HOME/.zshrc}}"
eval "$(grep -E '^[[:space:]]*export[[:space:]]+(HF_TOKEN|HUGGING_FACE_HUB_TOKEN|CLAUDE_API_KEY|CLAUDE_BASE_URL|ANTHROPIC_API_KEY|OPENAI_API_KEY|OPENAI_BASE_URL|FIREWORKS_API_KEY|FIREWORKS_BASE_URL|GEMINI_API_KEY|GEMINI_BASE_URL|QWEN_API_KEY|QWEN_BASE_URL)=' "$KEY_FILE" 2>/dev/null)" || true

export HF_TOKEN="${HF_TOKEN:-${HUGGING_FACE_HUB_TOKEN:-}}"
export HUGGING_FACE_HUB_TOKEN="${HUGGING_FACE_HUB_TOKEN:-$HF_TOKEN}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-}"
export CLAUDE_API_KEY="${CLAUDE_API_KEY:-${ANTHROPIC_API_KEY:-}}"

# Default to cached checkpoints for repeatable cluster runs. Set these to 0 when
# preparing a fresh machine so Hugging Face can download missing weights.
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
unset HUGGINGFACE_HUB_CACHE

HF_HOME="$(bash "$SCENERIG_ROOT/scripts/prewarm_models.sh" || true)"
if [ -z "${HF_HOME:-}" ]; then
    if [ -d "$HOME/.cache/huggingface" ]; then
        HF_HOME="$HOME/.cache/huggingface"
    else
        HF_HOME="/fsx/rundongluo/.cache/huggingface"
    fi
fi
export HF_HOME

export CUDA_CACHE_PATH="${CUDA_CACHE_PATH:-$HOME/.nv/ComputeCache}"
export CUDA_CACHE_MAXSIZE="${CUDA_CACHE_MAXSIZE:-4294967296}"
bash "$SCENERIG_ROOT/scripts/prewarm_cycles_kernels.sh" "$GPU" || true

SAM3D_LIB="${SAM3D_LIB:-$HOME/.local/micromamba/envs/sam3d-objects/lib}"
[ -d "$SAM3D_LIB" ] || SAM3D_LIB="/fsx/rundongluo/.local/micromamba/envs/sam3d-objects/lib"
export LD_LIBRARY_PATH="$SAM3D_LIB:${LD_LIBRARY_PATH:-}"

export GRASE_ISAAC_IDLE_TIMEOUT="${GRASE_ISAAC_IDLE_TIMEOUT:-5400}"
bash "$SCENERIG_ROOT/scripts/prewarm_isaac.sh" || true

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
