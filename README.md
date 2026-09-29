# SceneRig

SceneRig reconstructs a physically grounded 3D scene from a single RGB image.
The public pipeline is intentionally narrow:

- input image
- optional metric depth `.npy` plus optional camera intrinsics JSON
- final Blender/result artifacts under `<output_dir>/scene/`

Generative re-segmentation is enabled by default. A read-only local result
viewer is included.

## Setup

SceneRig requires Linux, Python 3.11, CUDA 12.8, Blender 4.5 LTS, and Isaac Sim
5.1.

### Python

```bash
git clone https://github.com/Red-Fairy/SceneRig.git
cd SceneRig

uv sync
source .venv/bin/activate
```

`uv sync` installs the main environment, PyTorch, and MoGE-2.

### Blender And Isaac Sim

Install Blender under `lib/utils/third_party/blender-4.5`, then install its
headless system libraries:

```bash
mkdir -p lib/utils/third_party
curl -L https://download.blender.org/release/Blender4.5/blender-4.5.14-linux-x64.tar.xz \
  | tar -xJ -C lib/utils/third_party
ln -sfn blender-4.5.14-linux-x64 lib/utils/third_party/blender-4.5

bash scripts/install_system_libs.sh
```

Install Isaac Sim in its own Python 3.11 environment:

```bash
uv venv --python 3.11 lib/utils/third_party/isaac/venv
lib/utils/third_party/isaac/venv/bin/pip install \
  'isaacsim[all,extscache]==5.1.0' \
  --extra-index-url https://pypi.nvidia.com
```

Verify both installations:

```bash
lib/utils/third_party/blender-4.5/blender --version
OMNI_KIT_ACCEPT_EULA=YES lib/utils/third_party/isaac/venv/bin/python -c \
  'from isaacsim import SimulationApp; app=SimulationApp({"headless": True}); app.close()'
```

### Models

Install model backends in these default locations:

- SAM3: `lib/utils/third_party/sam3/.venv`
- SAM3D Objects: `lib/utils/third_party/sam3d/.venv`
- MolmoPoint: `lib/utils/third_party/molmo/.venv`
- LanPaint/Qwen: `lib/utils/third_party/lanpaint-qwen/.venv`
- SHARP: `lib/utils/third_party/sharp/.venv`

Clone SAM3 and LanPaint, then follow their upstream installation instructions:

```bash
git clone https://github.com/facebookresearch/sam3.git lib/utils/third_party/sam3
git clone https://github.com/charrywhite/LanPaint-diffusers.git \
  lib/utils/third_party/lanpaint-qwen
```

SAM3D is included in `lib/utils/third_party/sam3d`. Follow
[`doc/setup.md`](lib/utils/third_party/sam3d/doc/setup.md), including its CUDA,
PyTorch3D, Kaolin, and `nvdiffrast` steps.

Log in to Hugging Face and download the model weights:

```bash
export HF_TOKEN=...

huggingface-cli download facebook/sam-3d-objects \
  --local-dir lib/utils/third_party/sam3d/checkpoints/hf
huggingface-cli download facebook/sam3
huggingface-cli download allenai/MolmoPoint-8B
huggingface-cli download Qwen/Qwen-Image-Edit-2509
```

MoGE-2 downloads `Ruicheng/moge-2-vitl-normal` automatically on first use.

### API Keys

Export the provider keys in your shell or add them to `~/.zshrc`:

```bash
export OPENAI_API_KEY=...
export CLAUDE_API_KEY=...        # or ANTHROPIC_API_KEY
export CLAUDE_BASE_URL=...       # only if using an OpenAI-compatible proxy
```

Do not commit API keys.

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
