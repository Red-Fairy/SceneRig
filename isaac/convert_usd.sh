# System libs for headless Blender + Isaac Sim. Ephemeral (wiped on pod restart) —
# comment out if already installed this boot. Same list as scripts/install_system_libs.sh.
apt-get update -qq
apt-get install -y -qq \
  libsm6 libice6 libxext6 libxrender1 libxi6 libxxf86vm1 libxfixes3 \
  libxkbcommon0 libgl1 libegl1 libxrandr2 libxinerama1 libxcursor1 \
  libxt6 libglu1-mesa libvulkan1 vulkan-tools

export OMNI_KIT_ACCEPT_EULA=YES OMNI_KIT_ALLOW_ROOT=1  # required by step 3 (Isaac Sim boot)

ISAAC_PY=/fsx/rundongluo/isaac/venv/bin/python
EXP=${1:?usage: bash isaac/convert_usd.sh output/static_scene/<experiment>}

mkdir -p $EXP/scene/isaac

# 1. blend → USD (runs inside the repo's Blender; bakes procedural materials)
lib/utils/third_party/blender-4.5/blender -b "$EXP/scene/final/final.blend" \
    --python isaac/isaac_export_usd.py -- "$EXP/scene/isaac/scene_visual.usdc"

# 2. resolve pipeline names to independent USD roots and publish object_identity.json.
#    This deliberately fails on ambiguous names or inter-object nesting.
$ISAAC_PY isaac/usd_dump_objects.py "$EXP/scene/isaac/scene_visual.usdc" \
    "$EXP/scene/isaac/visual_meshes.npz" "$EXP/scene/placement.json" \
    "$EXP/scene/isaac/object_identity.json"

# 3. stamp physics (pure pxr, fast, no simulator; runtime convex decomposition)
$ISAAC_PY isaac/isaac_add_physics.py "$EXP/scene/isaac/scene_visual.usdc" \
    "$EXP/scene/isaac/scene.usd"

# 4. settle-verify (boots headless Isaac Sim, ~2 min)
$ISAAC_PY isaac/isaac_verify_settle.py "$EXP/scene/isaac/scene.usd" \
    "$EXP/scene/isaac/verify_report.json"

# 5. package a self-contained usdz for sharing (scene + textures in one file)
$ISAAC_PY -c "from pxr import UsdUtils; UsdUtils.CreateNewUsdzPackage(
    '$EXP/scene/isaac/scene.usd', '$EXP/scene/isaac/scene.usdz')"
