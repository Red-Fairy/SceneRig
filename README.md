<div align="center">

# SceneRig

**An Agentic System for Simulation-Ready 3D Scene Reconstruction from Single Images**

Rundong Luo<sup>1,2</sup>, Yunong Liu<sup>1</sup>, Di Cao<sup>1</sup>,
Matthew Tancik<sup>1</sup>, Shyamal Buch<sup>1</sup>, Colton Stearns<sup>1</sup>,
Wenqi Xian<sup>1</sup>

<sup>1</sup> Luma AI &nbsp;&nbsp; <sup>2</sup> Cornell University

![Paper](https://img.shields.io/badge/Paper-Coming_Soon-c94c4c)
[![Demo](https://img.shields.io/badge/Demo-Project_Page-1687c9)](https://red-fairy.github.io/SceneRig/)
[![Code](https://img.shields.io/badge/Code-GitHub-444444)](https://github.com/Red-Fairy/SceneRig)

</div>

## 🎯 Overview

**Input:** one RGB image.

**Optional inputs:** metric Z-depth as a pixel-aligned `.npy` file and camera
intrinsics as JSON.

**Output:** a reconstructed Blender scene, final render, scene geometry,
materials, and physical-settling results under `<output_dir>/scene/`.

Generative re-segmentation is enabled by default. Completed and active runs can
be inspected in the local web viewer.

## 🛠️ Setup

Requirements: Linux x86_64, an NVIDIA GPU with a driver for CUDA 12.6 or newer
(driver 560+), [uv](https://docs.astral.sh/uv/), and about 150 GB of disk for
environments and weights. A system CUDA toolkit is not needed. Isaac Sim needs
glibc 2.35+ (Ubuntu 22.04 or newer).

```bash
git clone https://github.com/Red-Fairy/SceneRig.git
cd SceneRig
bash scripts/install.sh --detect   # optional: show what will be installed
bash scripts/install.sh
```

The script reads the driver and installs the newest tested PyTorch CUDA build
it supports: cu128 on CUDA 12.8+ drivers, cu126 on CUDA 12.6 drivers. Override
with `SCENERIG_TORCH_BACKEND=cu126|cu128|cu129|cu130`. Re-running skips
finished components; `--only sam3,molmo` and `--skip isaac` limit the work, and
`--verify` reports what is installed.

The model backends cannot share one environment: they need different Python
and PyTorch builds, and the runtime calls each one at a fixed path.

| Backend | Interpreter | Build |
| --- | --- | --- |
| SceneRig | `.venv` | Python 3.11, torch 2.11.0 |
| SAM 3 | `lib/utils/third_party/sam3/.venv` | Python 3.12, torch 2.11.0 |
| SAM 3D | `lib/utils/third_party/sam3d/.venv` | conda env, torch 2.5.1+cu121, CUDA 12.1 toolkit |
| MolmoPoint | `lib/utils/third_party/molmo/.venv` | Python 3.11, torch 2.11.0, transformers 4.57.1 |
| SHARP | `lib/utils/third_party/sharp/.venv` | Python 3.13, torch 2.8.0, CUDA toolkit for gsplat |
| LanPaint | `lib/utils/third_party/lanpaint-qwen/.venv` | Python 3.12, torch 2.11.0, diffusers 0.36.0 |
| Isaac Sim | `lib/utils/third_party/isaac/venv` | Python 3.11, isaacsim 5.1.0 |

SAM 3D and SHARP compile CUDA extensions, so each gets a conda CUDA toolkit
that matches its torch build. SAM 3D's torch 2.5.1 has no Blackwell
(B200, RTX 50-series) kernels, so the script skips it on those GPUs.

To install only the main environment by hand, `uv sync` uses cu128. For another
build:

```bash
uv sync --no-default-groups --group dev --group cu126
```

`facebook/sam3` and `facebook/sam-3d-objects` are gated. Request access on both
model pages, then:

```bash
export HF_TOKEN=...
bash scripts/download_weights.sh
```

SAM 3D weights land in `lib/utils/third_party/sam3d/checkpoints/hf/` and SAM 3
in `lib/utils/third_party/sam3/checkpoints/sam3.pt`. MolmoPoint-8B and
Qwen-Image-Edit-2509 go to the Hugging Face cache (`HF_HOME`). MoGE-2 downloads
on first use.

Check Blender and Isaac:

```bash
lib/utils/third_party/blender-4.5/blender --version
OMNI_KIT_ACCEPT_EULA=YES lib/utils/third_party/isaac/venv/bin/python -c \
  'from isaacsim import SimulationApp; app=SimulationApp({"headless": True}); app.close()'
```

### 🔑 API Keys

The default agent is Claude, including for `--preprocess-only`. OpenAI is used
only for novel-view polish (`gpt-image-2`); without it the raw SHARP renders
are kept.

```bash
export CLAUDE_API_KEY=...        # or ANTHROPIC_API_KEY
export OPENAI_API_KEY=...        # optional; novel-view polish only
export CLAUDE_BASE_URL=...       # only if using an OpenAI-compatible proxy
```

`scripts/run_e2e.sh` reads these exports from `~/.zshrc`. Do not commit API keys.

## 🚀 Run

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

## 📦 Outputs

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

## 📊 Evaluation

Run the lightweight evaluator on a run directory:

```bash
python scripts/evaluate_run.py output/example
```

It prints JSON with missing required artifacts, object counts, generative
re-segmentation count, and final-result status. It does not render images or
start viewers.

## 🖥️ Detailed Local Demo

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
