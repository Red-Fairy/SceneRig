#!/bin/bash
# Headless Blender + Isaac Sim system libraries (X11/GL/Vulkan).
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
if [ "$(id -u)" -eq 0 ]; then
  APT=(apt-get)
elif command -v sudo >/dev/null 2>&1; then
  APT=(sudo apt-get)
else
  echo "Run this script as root, or install sudo." >&2
  exit 1
fi
"${APT[@]}" update -qq
"${APT[@]}" install -y -qq \
  libsm6 libice6 libxext6 libxrender1 libxi6 libxxf86vm1 libxfixes3 \
  libxkbcommon0 libgl1 libegl1 libxrandr2 libxinerama1 libxcursor1 \
  libxt6 libglu1-mesa libvulkan1 vulkan-tools rsync xserver-xorg-core xauth
echo "System libraries installed; verifying Blender..."
BLENDER="${SCENERIG_BLENDER_COMMAND:-lib/utils/third_party/blender-4.5/blender}"
if [ ! -x "$BLENDER" ]; then
  echo "Blender not found at $BLENDER. Install Blender 4.5 LTS or set SCENERIG_BLENDER_COMMAND." >&2
  exit 1
fi
"$BLENDER" --background --factory-startup \
  --python-expr "import bpy; print('blender OK', bpy.app.version_string)" 2>&1 | grep -i "blender OK"
