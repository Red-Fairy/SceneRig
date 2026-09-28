# Separate GRASE -> Isaac conversion: physics, verification, status, and USDZ.
# Run from the grase project root:
#     bash isaac/convert_usd_physics.sh [output/static_scene/<exp>]
# After a pod restart, first re-install the ephemeral system libs:
#     bash scripts/install_system_libs.sh

EXP=${1:-output/static_scene/0714_e2e_real8334}
python3 isaac/convert_scene.py "$EXP" --package-usdz
