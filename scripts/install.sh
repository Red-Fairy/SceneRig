#!/usr/bin/env bash
# Install SceneRig and the isolated backends it shells out to.
#
# The runtime calls a separate interpreter per model (lib/utils/_path.py), and
# those need different Python and PyTorch builds, so one uv environment cannot
# hold them. This script reads the NVIDIA driver, picks the PyTorch CUDA build
# it supports, and creates each interpreter at the path the code expects.
#
#   bash scripts/install.sh                # everything except weights
#   bash scripts/install.sh --detect       # print the detected configuration
#   bash scripts/install.sh --verify       # check what is installed
#   bash scripts/install.sh --only sam3,molmo
#   bash scripts/install.sh --skip isaac,blender
#   bash scripts/install.sh --weights      # also run scripts/download_weights.sh
#
# Components: main blender isaac sam3 sam3d molmo sharp lanpaint
#
# Environment overrides:
#   SCENERIG_TORCH_BACKEND=cu126|cu128|cu129|cu130   PyTorch CUDA build
#   TORCH_CUDA_ARCH_LIST="8.6;9.0"                   GPU archs for compiled extensions
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

TP="lib/utils/third_party"
SAM3_REPO="https://github.com/facebookresearch/sam3.git"
SAM3_REV="2345a4ad109ac29c569da749c91d84f10dc08c40"
SHARP_REPO="https://github.com/apple/ml-sharp.git"
SHARP_REV="aed6527499ef91cba3b54c18d49a870f25947190"
LANPAINT_PIPELINE_REPO="https://github.com/charrywhite/LanPaint-diffusers.git"
LANPAINT_PIPELINE_REV="a2f6483acb678ecb0b6268a311d55d6d51f37c4f"
LANPAINT_PKG="git+https://github.com/scraed/LanPaint.git@2d7912f9a5efe5ece8de334c7ca18317b8288c39"
NVDIFFRAST_PKG="git+https://github.com/NVlabs/nvdiffrast.git@253ac4fcea7de5f396371124af597e6cc957bfae"
KAOLIN_LINKS="https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.5.1_cu121.html"
BLENDER_URL="https://download.blender.org/release/Blender4.5/blender-4.5.14-linux-x64.tar.xz"
MICROMAMBA_URL="https://github.com/mamba-org/micromamba-releases/releases/latest/download/micromamba-linux-64"

SAM3D_ENV="$ROOT/.micromamba/envs/sam3d-objects"
SHARP_SRC="$TP/ml-sharp"
SHARP_CUDA="$ROOT/$TP/sharp/cuda"

# A user pip.conf with an unreachable extra index stalls every install.
export PIP_CONFIG_FILE="${PIP_CONFIG_FILE:-/dev/null}"
export PIP_DISABLE_PIP_VERSION_CHECK=1
unset PIP_EXTRA_INDEX_URL PIP_FIND_LINKS PIP_INDEX_URL || true

ONLY=""
SKIP=""
MODE="install"
WITH_WEIGHTS=0
while [ "$#" -gt 0 ]; do
  case "$1" in
    --only) ONLY="$2"; shift 2 ;;
    --skip) SKIP="$2"; shift 2 ;;
    --weights) WITH_WEIGHTS=1; shift ;;
    --detect) MODE="detect"; shift ;;
    --verify) MODE="verify"; shift ;;
    -h|--help) sed -n '2,21p' "$0"; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

say() { printf '\n==> %s\n' "$*"; }
warn() { printf 'WARNING: %s\n' "$*" >&2; }
die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }

want() {
  local name="$1"
  if [ -n "$ONLY" ]; then
    case ",$ONLY," in *,"$name",*) ;; *) return 1 ;; esac
  fi
  case ",$SKIP," in *,"$name",*) return 1 ;; esac
  return 0
}

# ver_ge 12.8 12.6 -> true
ver_ge() {
  awk -v a="$1" -v b="$2" 'BEGIN {
    split(a, x, "."); split(b, y, ".");
    exit !((x[1] > y[1]) || (x[1] == y[1] && x[2] + 0 >= y[2] + 0))
  }'
}

# ---------------------------------------------------------------------------
# Platform detection
# ---------------------------------------------------------------------------

detect_platform() {
  ARCH="$(uname -m)"
  GLIBC="$(getconf GNU_LIBC_VERSION 2>/dev/null | awk '{print $2}')"
  GLIBC="${GLIBC:-0.0}"
  DRIVER=""
  DRIVER_CUDA=""
  GPU_NAMES=""
  GPU_CAPS=""
  if command -v nvidia-smi >/dev/null 2>&1; then
    DRIVER="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 || true)"
    DRIVER_CUDA="$(nvidia-smi 2>/dev/null | sed -n 's/.*CUDA Version: *\([0-9][0-9]*\.[0-9][0-9]*\).*/\1/p' | head -1 || true)"
    GPU_NAMES="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | sort -u | paste -sd, - || true)"
    GPU_CAPS="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | sort -u | paste -sd';' - || true)"
  fi

  if [ -n "${SCENERIG_TORCH_BACKEND:-}" ]; then
    TORCH_BACKEND="$SCENERIG_TORCH_BACKEND"
    case "$TORCH_BACKEND" in
      cu126|cu128|cu129|cu130) ;;
      *) die "SCENERIG_TORCH_BACKEND must be cu126, cu128, cu129, or cu130 (got $TORCH_BACKEND)." ;;
    esac
  elif [ -z "$DRIVER_CUDA" ]; then
    die "nvidia-smi not found. Install the NVIDIA driver, or set SCENERIG_TORCH_BACKEND (e.g. cu128) to install on a machine without a GPU."
  elif ver_ge "$DRIVER_CUDA" 12.8; then
    # cu128 is the tested build; drivers for CUDA 12.9 and 13.x run it too.
    TORCH_BACKEND="cu128"
  elif ver_ge "$DRIVER_CUDA" 12.6; then
    TORCH_BACKEND="cu126"
  else
    die "Driver $DRIVER supports CUDA $DRIVER_CUDA. SceneRig needs CUDA 12.6 or newer (driver 560+)."
  fi

  if [ -n "$DRIVER_CUDA" ]; then
    local needed="${TORCH_BACKEND#cu}"
    needed="${needed:0:2}.${needed:2}"
    if ! ver_ge "$DRIVER_CUDA" "$needed"; then
      warn "SCENERIG_TORCH_BACKEND=$TORCH_BACKEND needs CUDA $needed but the driver supports $DRIVER_CUDA."
    fi
  fi

  # PyTorch 2.8.0 (SHARP) has no cu130 wheels.
  SHARP_BACKEND="$TORCH_BACKEND"
  [ "$SHARP_BACKEND" = "cu130" ] && SHARP_BACKEND="cu129"
  SHARP_TOOLKIT="${SHARP_BACKEND#cu}"
  SHARP_TOOLKIT="${SHARP_TOOLKIT:0:2}.${SHARP_TOOLKIT:2}"

  if [ -z "${TORCH_CUDA_ARCH_LIST:-}" ] && [ -n "$GPU_CAPS" ]; then
    export TORCH_CUDA_ARCH_LIST="$GPU_CAPS"
  fi

  BLACKWELL=0
  local cap
  for cap in ${GPU_CAPS//;/ }; do
    if ver_ge "$cap" 10.0; then BLACKWELL=1; fi
  done
  return 0
}

print_platform() {
  cat <<EOF
Platform
  arch            $ARCH
  glibc           $GLIBC
  GPU             ${GPU_NAMES:-none detected}
  compute cap     ${GPU_CAPS:-unknown}
  driver          ${DRIVER:-none} (CUDA ${DRIVER_CUDA:-n/a})
Selected builds
  main, SAM 3, Molmo, LanPaint   torch 2.11.0+$TORCH_BACKEND
  SHARP                          torch 2.8.0+$SHARP_BACKEND, CUDA $SHARP_TOOLKIT toolkit for gsplat
  SAM 3D                         torch 2.5.1+cu121, CUDA 12.1 toolkit (conda)
  extension archs                ${TORCH_CUDA_ARCH_LIST:-all supported by each torch build}
EOF
  if [ "$BLACKWELL" -eq 1 ]; then
    warn "Blackwell GPU detected. SAM 3D pins torch 2.5.1+cu121, which has no kernels for compute capability 10.x/12.x; SAM 3D reconstruction will not run on this GPU."
  fi
  if ! ver_ge "$GLIBC" 2.35; then
    warn "glibc $GLIBC is older than 2.35. Isaac Sim 5.1 wheels require 2.35 (Ubuntu 22.04 or newer); the isaac step will be skipped."
  fi
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

require_uv() {
  command -v uv >/dev/null 2>&1 || die "uv is required: curl -LsSf https://astral.sh/uv/install.sh | sh"
}

ensure_micromamba() {
  if [ -n "${MICROMAMBA:-}" ]; then
    return 0
  fi
  if command -v micromamba >/dev/null 2>&1; then
    MICROMAMBA="$(command -v micromamba)"
  else
    MICROMAMBA="$ROOT/.tools/micromamba"
    if [ ! -x "$MICROMAMBA" ]; then
      mkdir -p "$ROOT/.tools"
      curl -fsSL "$MICROMAMBA_URL" -o "$MICROMAMBA"
      chmod +x "$MICROMAMBA"
    fi
  fi
  export MAMBA_ROOT_PREFIX="$ROOT/.micromamba"
}

# uv venvs have no bin/pip; Isaac's documented install calls venv/bin/pip.
make_venv() {
  local python="$1" dest="$2"
  [ -x "$dest/bin/python" ] || uv venv --python "$python" "$dest"
  [ -x "$dest/bin/pip" ] || "$dest/bin/python" -m ensurepip --upgrade >/dev/null
  if [ ! -e "$dest/bin/pip" ] && [ -x "$dest/bin/pip3" ]; then
    ln -s pip3 "$dest/bin/pip"
  fi
}

clone_at() {
  local repo="$1" rev="$2" dest="$3"
  if [ ! -d "$dest/.git" ]; then
    git clone "$repo" "$dest"
    git -C "$dest" checkout -q "$rev"
  fi
}

pip_torch() {
  local py="$1" torch="$2" vision="$3" backend="$4"
  "$py" -m pip install "torch==$torch" "torchvision==$vision" \
    --index-url "https://download.pytorch.org/whl/$backend"
}

# Mirror of lib/utils/cuda_toolkit.py for build steps run from bash.
use_toolkit() {
  local prefix="$1" targets="$1/targets/x86_64-linux"
  export CUDA_HOME="$prefix"
  export PATH="$prefix/bin:$PATH"
  export CPATH="$targets/include${CPATH:+:$CPATH}"
  export LIBRARY_PATH="$targets/lib:$prefix/lib${LIBRARY_PATH:+:$LIBRARY_PATH}"
  [ -x "$prefix/bin/x86_64-conda-linux-gnu-gcc" ] && export CC="$prefix/bin/x86_64-conda-linux-gnu-gcc"
  [ -x "$prefix/bin/x86_64-conda-linux-gnu-g++" ] && export CXX="$prefix/bin/x86_64-conda-linux-gnu-g++"
  return 0
}

# Compile gsplat's CUDA kernels now so the first pipeline run does not.
warm_gsplat() {
  local py="$1" toolkit="$2"
  PYTHONPATH="$ROOT" "$py" - "$toolkit" <<'PY' || warn "gsplat did not compile with $2; it will retry on first use."
import sys
from lib.utils.cuda_toolkit import use_cuda_toolkit
use_cuda_toolkit(sys.argv[1])
import torch  # noqa: F401
from gsplat.cuda._backend import _C
assert _C is not None, "gsplat CUDA extension unavailable"
print("gsplat CUDA extension ready")
PY
}

# ---------------------------------------------------------------------------
# Checks (used to skip finished steps and by --verify)
# ---------------------------------------------------------------------------

py_ok() {
  local py="$1"; shift
  [ -x "$py" ] && "$py" -c "$*" >/dev/null 2>&1
}

check_main() {
  py_ok .venv/bin/python "import torch, moge; assert torch.__version__ == '2.11.0+$TORCH_BACKEND'"
}
check_blender() { [ -x "$TP/blender-4.5/blender" ]; }
check_isaac() {
  py_ok "$TP/isaac/venv/bin/python" "import importlib.util as u; assert u.find_spec('isaacsim')"
}
check_sam3() {
  py_ok "$TP/sam3/.venv/bin/python" "import torch, sam3, einops, pycocotools, psutil; assert torch.__version__ == '2.11.0+$TORCH_BACKEND'"
}
check_sam3d() {
  py_ok "$TP/sam3d/.venv/bin/python" "import torch, pytorch3d, kaolin, nvdiffrast.torch; assert torch.__version__.startswith('2.5.1')"
}
check_molmo() {
  py_ok "$TP/molmo/.venv/bin/python" "import torch, transformers; assert torch.__version__ == '2.11.0+$TORCH_BACKEND'; assert transformers.__version__ == '4.57.1'"
}
check_sharp() {
  py_ok "$TP/sharp/.venv/bin/python" "import torch, sharp; assert torch.__version__ == '2.8.0+$SHARP_BACKEND'" \
    && [ -x "$SHARP_CUDA/bin/nvcc" ] \
    && "$SHARP_CUDA/bin/nvcc" --version | grep -q "release $SHARP_TOOLKIT"
}
check_lanpaint() {
  py_ok "$TP/lanpaint-qwen/.venv/bin/python" "import torch, diffusers; from diffusers import QwenImageEditPlusPipeline; from lanpaint_pipeline.registry import create_adapter; assert torch.__version__ == '2.11.0+$TORCH_BACKEND'; assert diffusers.__version__ == '0.37.1'"
}

COMPONENTS="main blender isaac sam3 sam3d molmo sharp lanpaint"

verify_all() {
  local name status failed=0
  printf '\n%-10s %s\n' component status
  for name in $COMPONENTS; do
    want "$name" || continue
    if "check_$name"; then status="ok"; else status="MISSING"; failed=1; fi
    printf '%-10s %s\n' "$name" "$status"
  done
  if [ -s "$TP/sam3d/checkpoints/hf/pipeline.yaml" ]; then status="ok"; else status="MISSING (scripts/download_weights.sh)"; fi
  printf '%-10s %s\n' "sam3d wts" "$status"
  if [ -s "$TP/sam3/checkpoints/sam3.pt" ] || [ -n "${SAM3_CHECKPOINT:-}" ]; then status="ok"; else status="not local (downloaded from facebook/sam3 on first use)"; fi
  printf '%-10s %s\n' "sam3 wts" "$status"
  return "$failed"
}

# ---------------------------------------------------------------------------
# Components
# ---------------------------------------------------------------------------

install_main() {
  say "Main environment: Python 3.11, torch 2.11.0+$TORCH_BACKEND"
  uv sync --no-default-groups --group dev --group "$TORCH_BACKEND"
}

install_blender() {
  if check_blender; then
    say "Blender already present"
  else
    say "Blender 4.5.14 LTS"
    mkdir -p "$TP"
    curl -fL "$BLENDER_URL" | tar -xJ -C "$TP"
    ln -sfn blender-4.5.14-linux-x64 "$TP/blender-4.5"
  fi
  if ! command -v apt-get >/dev/null 2>&1; then
    warn "No apt-get. Install the X11/GL/Vulkan libraries listed in scripts/install_system_libs.sh with your package manager."
  elif [ "$(id -u)" -ne 0 ] && ! command -v sudo >/dev/null 2>&1; then
    warn "No root or sudo; skipping system libraries. Blender needs libSM, libXi, libGL and friends."
  else
    bash scripts/install_system_libs.sh
  fi
}

install_isaac() {
  local venv="$TP/isaac/venv"
  say "Isaac Sim 5.1: Python 3.11"
  if ! ver_ge "$GLIBC" 2.35; then
    warn "Skipping Isaac Sim: glibc $GLIBC < 2.35."
    return 0
  fi
  make_venv 3.11 "$venv"
  if check_isaac; then echo "already installed"; return 0; fi
  "$venv/bin/pip" install 'isaacsim[all,extscache]==5.1.0' \
    --extra-index-url https://pypi.nvidia.com
}

install_sam3() {
  local src="$TP/sam3" venv="$TP/sam3/.venv"
  say "SAM 3: Python 3.12, torch 2.11.0+$TORCH_BACKEND"
  clone_at "$SAM3_REPO" "$SAM3_REV" "$src"
  make_venv 3.12 "$venv"
  if check_sam3; then echo "already installed"; return 0; fi
  # torch 2.10.0+cu128 fails bfloat16 GEMM on CUDA 13 drivers; stay on 2.11.
  pip_torch "$venv/bin/python" 2.11.0 0.26.0 "$TORCH_BACKEND"
  "$venv/bin/python" -m pip install -e "$src"
  "$venv/bin/python" -m pip install -r scripts/constraints/sam3.txt
}

install_sam3d() {
  local src="$TP/sam3d" link="$TP/sam3d/.venv"
  say "SAM 3D: conda env sam3d-objects, torch 2.5.1+cu121"
  if [ "$BLACKWELL" -eq 1 ] && [ -z "${SCENERIG_FORCE_SAM3D:-}" ]; then
    warn "Skipping SAM 3D on Blackwell (CUDA 12.1 cannot target this GPU). Set SCENERIG_FORCE_SAM3D=1 to try anyway."
    return 0
  fi
  ensure_micromamba
  if [ ! -x "$SAM3D_ENV/bin/python" ]; then
    "$MICROMAMBA" create -y -p "$SAM3D_ENV" -f "$src/environments/default.yml"
  fi
  [ -e "$link" ] || ln -s "$SAM3D_ENV" "$link"
  if check_sam3d; then echo "already installed"; return 0; fi
  (
    # Build pytorch3d and nvdiffrast with the env's CUDA 12.1 and GCC 12, not
    # whatever /usr/local/cuda happens to be.
    use_toolkit "$SAM3D_ENV"
    cd "$src"
    local py="$SAM3D_ENV/bin/python"
    export PIP_EXTRA_INDEX_URL="https://download.pytorch.org/whl/cu121"
    "$py" -m pip install -e '.[dev]'
    "$py" -m pip install -e '.[p3d]'
    PIP_FIND_LINKS="$KAOLIN_LINKS" "$py" -m pip install -e '.[inference]'
    ./patching/hydra
    unset PIP_EXTRA_INDEX_URL
    # nvdiffrast's build imports torch, so it cannot use an isolated build env.
    "$py" -m pip install --no-build-isolation "$NVDIFFRAST_PKG"
  )
  TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-$HOME/.cache/torch_extensions_scenerig}" \
    warm_gsplat "$SAM3D_ENV/bin/python" "$SAM3D_ENV"
}

install_molmo() {
  local venv="$TP/molmo/.venv"
  say "MolmoPoint: Python 3.11, torch 2.11.0+$TORCH_BACKEND, transformers 4.57.1"
  make_venv 3.11 "$venv"
  if check_molmo; then echo "already installed"; return 0; fi
  pip_torch "$venv/bin/python" 2.11.0 0.26.0 "$TORCH_BACKEND"
  "$venv/bin/python" -m pip install -r scripts/constraints/molmo.txt
}

install_sharp() {
  local venv="$TP/sharp/.venv"
  say "SHARP: Python 3.13, torch 2.8.0+$SHARP_BACKEND, CUDA $SHARP_TOOLKIT toolkit"
  clone_at "$SHARP_REPO" "$SHARP_REV" "$SHARP_SRC"
  make_venv 3.13 "$venv"
  if check_sharp; then echo "already installed"; return 0; fi
  local py="$ROOT/$venv/bin/python"
  pip_torch "$py" 2.8.0 0.23.0 "$SHARP_BACKEND"
  # Upstream's lockfile pins the cu128 CUDA wheels; let torch bring its own.
  local reqs
  reqs="$(mktemp)"
  grep -v -E '^(nvidia-|torch==|torchvision==|triton==)' "$SHARP_SRC/requirements.txt" > "$reqs"
  # The lockfile contains `-e .`, so it must run from the SHARP checkout.
  (cd "$SHARP_SRC" && "$py" -m pip install -r "$reqs")
  rm -f "$reqs"
  # gsplat compiles its kernels on first use and needs nvcc matching torch.
  if ! { [ -x "$SHARP_CUDA/bin/nvcc" ] && "$SHARP_CUDA/bin/nvcc" --version | grep -q "release $SHARP_TOOLKIT"; }; then
    ensure_micromamba
    rm -rf "$SHARP_CUDA"
    "$MICROMAMBA" create -y -p "$SHARP_CUDA" -c conda-forge \
      "cuda-version=$SHARP_TOOLKIT" cuda-nvcc cuda-cudart-dev cuda-cccl \
      "gcc_linux-64=13" "gxx_linux-64=13"
  fi
  warm_gsplat "$py" "$SHARP_CUDA"
}

install_lanpaint() {
  local src="$TP/lanpaint-qwen" venv="$TP/lanpaint-qwen/.venv"
  say "LanPaint + Qwen: Python 3.12, torch 2.11.0+$TORCH_BACKEND, diffusers 0.37.1"
  clone_at "$LANPAINT_PIPELINE_REPO" "$LANPAINT_PIPELINE_REV" "$src"
  make_venv 3.12 "$venv"
  if check_lanpaint; then echo "already installed"; return 0; fi
  pip_torch "$venv/bin/python" 2.11.0 0.26.0 "$TORCH_BACKEND"
  "$venv/bin/python" -m pip install -r scripts/constraints/lanpaint.txt
  "$venv/bin/python" -m pip install "$LANPAINT_PKG"
  "$venv/bin/python" -m pip install --no-deps -e "$src"
}

# ---------------------------------------------------------------------------

[ "$(uname -s)" = "Linux" ] || die "SceneRig supports Linux only."
detect_platform
[ "$ARCH" = "x86_64" ] || die "SceneRig supports x86_64 only (got $ARCH)."
print_platform

case "$MODE" in
  detect) exit 0 ;;
  verify) verify_all; exit $? ;;
esac

require_uv
for component in $COMPONENTS; do
  if want "$component"; then
    "install_$component"
  fi
done

if [ "$WITH_WEIGHTS" -eq 1 ]; then
  bash scripts/download_weights.sh
fi

verify_all || warn "Some components are missing; rerun this script or check the output above."

cat <<'EOF'

Next:
  export HF_TOKEN=...          # approved for facebook/sam3 and facebook/sam-3d-objects
  export CLAUDE_API_KEY=...    # required, including for --preprocess-only
  bash scripts/download_weights.sh
EOF
