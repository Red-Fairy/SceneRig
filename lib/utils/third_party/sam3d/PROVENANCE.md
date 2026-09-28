# Provenance

Vendored copy of SAM3D (sam-3d-objects), tracked directly in this repo since
2026-07-21 (git history of the checkout was dropped; this file records it).

Lineage:

- Upstream: `https://github.com/facebookresearch/sam-3d-objects` @ `e19b169`
  (merge of PR #109 `feature-human-object`; upstream has since moved on —
  the SA-3DAO/SS-encoder release and the reworked layout post-optimization
  are NOT in this copy).
- Fork: `https://github.com/Fugtemypt123/sam-3d-texture-baking` @ `af582ce`
  ("fix white texture bug when export glb") = upstream `e19b169` plus a small
  patch (+11/−7): hardcode `rendering_engine="nvdiffrast"` and force
  `with_mesh_postprocess=True, with_texture_baking=True` in the GLB export
  (`inference_pipeline.py`, `postprocessing_utils.py`), and comment out the
  `kernel_size`/`subpixel_offset` rasterizer kwargs in `gaussian_render.py`
  (compat shim for vanilla `diff-gaussian-rasterization`; upstream expects a
  mip-splatting variant).
- Local (GRASE): `layout_post_optimization` additionally returns
  `flag_icp`/`pose_pre_icp`/`pose_post_icp` snapshots
  (`inference_utils.py`, `inference_pipeline_pointmap.py`), consumed by
  `lib/tools/sam3d/sam3d_server.py`/`sam3d_worker.py` for the demo site's
  before/after-ICP pose readout (CHANGELOG 2026-07-08).

Pruned relative to the fork: `.git/`, `notebook/` (232 MB demo images),
`doc/` (gifs). `checkpoints/` (13 GB weights) and the `sam3d-objects`
micromamba env are provisioned separately and stay untracked.

EXCEPTION (restored 2026-07-22): `notebook/inference.py` — the prune took the
whole `notebook/` dir, but this one file is the `Inference`/`load_image`
wrapper that `lib/tools/sam3d/sam3d_worker.py` imports (it broke the SAM3D
server boot: "No module named 'inference'"). Re-fetched verbatim from the fork
@ `af582ce`; the demo images stay pruned.

License: Meta "SAM License" (see `LICENSE`).

To sync with upstream later: clone upstream fresh and diff against this tree,
minding the three patch layers above.
