# SceneRig

SceneRig reconstructs a physically grounded 3D scene from a single RGB image.
The public pipeline is intentionally narrow:

- input image
- optional metric depth `.npy` plus optional camera intrinsics JSON
- optional `--gpt6-harness`
- final Blender/result artifacts under `<output_dir>/scene/`

Generative re-segmentation is enabled by default. Policy-eval, stage-rerun
utilities, remote submission, and dormant mesh/depth backends are not part of
this release. A read-only local result viewer is included.

## Setup

SceneRig uses one main Python environment plus isolated environments for heavy
model servers.

```bash
git clone https://github.com/Red-Fairy/SceneRig.git
cd SceneRig

uv sync
source .venv/bin/activate
```

The lockfile uses PyTorch CUDA 12.8. MoGE is pinned to a MoGE-2 source revision
and is installed by `uv sync`. SceneRig does not require another depth backend.

Install the headless runtime libraries, Blender 4.5 LTS, and Isaac Sim 5.1:

```bash
bash scripts/install_system_libs.sh

mkdir -p lib/utils/third_party
curl -L https://download.blender.org/release/Blender4.5/blender-4.5.14-linux-x64.tar.xz \
  | tar -xJ -C lib/utils/third_party
ln -sfn blender-4.5.14-linux-x64 lib/utils/third_party/blender-4.5

uv venv --python 3.11 lib/utils/third_party/isaac/venv
lib/utils/third_party/isaac/venv/bin/pip install \
  'isaacsim[all,extscache]==5.1.0' \
  --extra-index-url https://pypi.nvidia.com
```

If `SCENERIG_BLENDER_COMMAND` is unset, the launcher uses
`lib/utils/third_party/blender-4.5/blender`. Isaac Sim requires Python 3.11;
the default interpreter is `lib/utils/third_party/isaac/venv/bin/python`.
Installing Isaac Sim confirms acceptance of NVIDIA's license terms shown by pip.

Verify both runtimes before downloading model weights:

```bash
lib/utils/third_party/blender-4.5/blender --version
OMNI_KIT_ACCEPT_EULA=YES lib/utils/third_party/isaac/venv/bin/python -c \
  'from isaacsim import SimulationApp; app=SimulationApp({"headless": True}); app.close()'
```

## Model Environments

Configure these interpreter paths directly, or create venvs at the defaults in
[lib/utils/_path.py](lib/utils/_path.py):

```bash
export SAM3_PYTHON=/path/to/sam3/.venv/bin/python
export SAM3D_PYTHON=/path/to/sam3d/.venv/bin/python
export MOLMO_PYTHON=/path/to/molmo/.venv/bin/python
export LANPAINT_QWEN_PYTHON=/path/to/lanpaint-qwen/.venv/bin/python
export SHARP_PYTHON=/path/to/sharp/.venv/bin/python
export SCENERIG_ISAAC_PYTHON=/path/to/isaac/venv/bin/python
```

Model/checkpoint sources used by the default pipeline:

- MoGE-2: `Ruicheng/moge-2-vitl-normal`, loaded by the MoGE package.
- MolmoPoint: `allenai/MolmoPoint-8B`, loaded by `lib/tools/geometry/molmo_server.py`.
- SAM3: clone/install the SAM3 image segmentation repo into `lib/utils/third_party/sam3`.
- SAM3D Objects: this repo keeps the SAM3D source under `lib/utils/third_party/sam3d`; download its checkpoints into `lib/utils/third_party/sam3d/checkpoints/`.
- Blender 4.5 and Isaac Sim are used for rendering and physical settling.

Install each model backend with its upstream instructions in the corresponding
directory. The tested source repositories are:

```bash
git clone https://github.com/facebookresearch/sam3.git lib/utils/third_party/sam3
git clone https://github.com/charrywhite/LanPaint-diffusers.git \
  lib/utils/third_party/lanpaint-qwen
```

SAM3D is vendored in `lib/utils/third_party/sam3d`; follow its
`doc/setup.md` using Python 3.11, PyTorch 2.5.1 + CUDA 12.1, PyTorch3D, and
Kaolin 0.17. Download gated checkpoints after accepting each model license:

```bash
huggingface-cli download facebook/sam-3d-objects \
  --local-dir lib/utils/third_party/sam3d/checkpoints/hf
huggingface-cli download facebook/sam3
huggingface-cli download allenai/MolmoPoint-8B
huggingface-cli download Qwen/Qwen-Image-Edit-2509
```

SAM3D texture baking also needs the CUDA toolkit and `nvdiffrast`:

```bash
CUDA_HOME=/usr/local/cuda \
  lib/utils/third_party/sam3d/.venv/bin/pip install \
  'git+https://github.com/NVlabs/nvdiffrast.git@v0.3.3'
```

MoGE-2 downloads `Ruicheng/moge-2-vitl-normal` automatically on first use.
The default interpreter paths above are discovered automatically, so the
environment variables are only needed when environments live elsewhere.

To allow Hugging Face downloads on a fresh machine:

```bash
export HF_TOKEN=...
export HF_HUB_OFFLINE=0
export TRANSFORMERS_OFFLINE=0
```

The launcher downloads missing weights by default. After the cache is populated,
set both variables to `1` for offline use.

## API Keys

Set keys in your shell, or put `export ...` lines in `~/.zshrc`. The launcher
loads `SCENERIG_KEYS_FILE`, then `GRASE_KEYS_FILE`, then `~/.zshrc`.

```bash
export OPENAI_API_KEY=...
export CLAUDE_API_KEY=...        # or ANTHROPIC_API_KEY
export CLAUDE_BASE_URL=...       # only if using an OpenAI-compatible proxy
```

Long agent turns default to a 600-second request timeout. Override it when a
provider has a stricter limit:

```bash
export SCENERIG_MODEL_TIMEOUT_SECONDS=600
export SCENERIG_MODEL_BUDGET_SECONDS=3600
```

Do not commit keys. [lib/utils/_api_keys.py](lib/utils/_api_keys.py) only reads
environment variables.

## Run

Basic image-to-scene:

```bash
bash scripts/run_e2e.sh path/to/image.png output/example 0
```

With provided metric depth and intrinsics:

```bash
bash scripts/run_e2e.sh path/to/image.png output/example_depth 0 \
  --depth path/to/depth.npy \
  --camera path/to/intrinsics.json
```

Depth must be metric Z-depth aligned pixel-for-pixel with the RGB image.
`intrinsics.json` should be:

```json
{"fx": 1000.0, "fy": 1000.0, "cx": 512.0, "cy": 384.0, "w": 1024, "h": 768}
```

Enable the GPT-6 harness:

```bash
bash scripts/run_e2e.sh path/to/image.png output/example_gpt6 0 --gpt6-harness
```

A quick preprocessing smoke test:

```bash
python lib/runners/static_scene.py \
  --image dataset_selected/misc/still_life.jpg \
  --output-dir output/smoke \
  --gpu 0 \
  --preprocess-only
```

`dataset_selected/` is a local smoke-test set and is gitignored.

## Outputs

For `output/example`, the scene run lives at `output/example/scene/`.
Important artifacts include:

- `input.png`: normalized RGB input
- `moge/`: depth, point map, intrinsics, normals
- `masks/masks.json`: objects, root surfaces, and generative re-segmentation records
- `meshes/`: per-object SAM3D meshes
- `physics/`: settle results
- `preprocess_manifest.json`: preprocessing artifact manifest
- `final/pipeline_result.json`: final pipeline summary when the agent completes
- `blender_file.blend`: latest Blender scene

## Evaluation

Run the lightweight evaluator on a run directory:

```bash
python scripts/evaluate_run.py output/example
```

It prints JSON with missing required artifacts, object counts, generative
re-segmentation count, and final-result status. It does not render images or
start viewers.

## Detailed Local Demo

Start the read-only web demo after one or more runs have written to
`output/`:

```bash
python site/demo_server.py --output-dir output --port 8502
```

Open `http://127.0.0.1:8502`. The demo lists completed and active scenes and
shows preprocessing details, object poses, scene-graph relationships, stage
renders, timing, logs, and generator/verifier conversations. Its 3D/Render
control exports the current `.blend` to a cached GLB on first use, then provides
an orbitable browser view alongside the final Cycles render. Active runs update
automatically while the pipeline writes new artifacts.

You can point `--output-dir` at the whole `output/` tree, one run directory, or
one `scene/` directory. The default bind address is local-only; use
`--host 0.0.0.0` only on a trusted network or behind your own authenticated
tunnel.

For a lighter status page without the preprocessing and 3D panels, run:

```bash
python site/dashboard.py --output-dir output --port 8765
```
