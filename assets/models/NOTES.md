# Reconstruction viewer assets

Each GLB corresponds to the method and scene in `data/gallery.json`. The directory keys match the gallery's model groups and scene identifiers. These are the saved reconstructions used for the qualitative comparison, before its additional five-second stability test.

The edited white-mug example uses the refreshed scene for every method. REST3D and SimFoundry retain the white support used in their gallery renders, cropped to a finite patch around the objects to avoid browser clipping. SimFoundry exports use the saved object poses and visual meshes; the robot and collision proxies are omitted, as in the comparison images.

Procedural Blender base colours are baked into textures for glTF, including metallic materials. Browser exports use neutral lighting, so illumination, reflections, and procedural shading can differ from the offline renders. The source-view render remains visible beside the interactive model. REST3D exports include a rigid rotation of the whole scene to preserve the source camera's roll with the browser's orbit controls; relative object poses are unchanged.

For `main/scene-23/rest3d.glb`, the saved source camera misses the reconstructed objects. Its viewer starts with the complete scene framed instead, with an explanatory note beside the original source render.

The custom diffuse/emission paint shaders in `gpt6/scene-12/codex.glb` are approximated with baked colour textures and glTF materials, retaining their transparency. Its backface-only outline meshes use reversed winding and backface culling to reproduce the original contour effect.

Models are compressed using meshoptimizer/gltfpack v1.3 with 16-bit positions, 14-bit texture coordinates, 12-bit normals, and WebP textures limited to 1024 pixels. Models with broad coordinate ranges use floating-point position quantization to preserve small details; `encoding.json` records these exceptions. The GLBs contain visual geometry and materials, not the simulation's physics configuration.

One exceptionally dense scene, `gpt6/scene-13/viga.glb`, contains a broccoli mesh with about 40 million source vertices. Its browser copy additionally uses gltfpack's `-si 0.05 -se 0.0005` simplification settings, retaining about four million triangles in the full scene. The original layout and gallery render are unchanged. Other models do not request mesh simplification.

The page loads one scene at a time. Viewer code, decoder, models, and reference images are all served from this repository.
