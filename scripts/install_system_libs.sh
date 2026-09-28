#!/bin/bash
# Headless-Blender + Isaac Sim system libraries (X11/GL/Vulkan). These live on the pod's ephemeral
# root fs and are WIPED on every pod restart (the /fsx project tree persists, but
# apt-installed libs do not). Re-run this after a restart if Blender fails with
# `libSM.so.6: cannot open shared object file` or similar.
set -e
# apt goes interactive under a tty (tmux): keyboard-configuration prompt hung a warm-up 09-14.
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq \
  libsm6 libice6 libxext6 libxrender1 libxi6 libxxf86vm1 libxfixes3 \
  libxkbcommon0 libgl1 libegl1 libxrandr2 libxinerama1 libxcursor1 \
  libxt6 libglu1-mesa libvulkan1 vulkan-tools rsync xserver-xorg-core xauth
echo "system libs installed; verifying Blender..."
# Blender 4.5.13 LTS is the pipeline default since 2026-09-15 (owner). The binary lives on the shared
# /fsx and is reached through a gitignored symlink inside third_party; recreate it after a fresh checkout.
[ -e lib/utils/third_party/blender-4.5 ] || ln -sfn /fsx/rundongluo/blender/blender-4.5.13-linux-x64 lib/utils/third_party/blender-4.5
BLENDER=lib/utils/third_party/blender-4.5/blender
"$BLENDER" --background --factory-startup \
  --python-expr "import bpy; print('blender OK', bpy.app.version_string)" 2>&1 | grep -i "blender OK"
