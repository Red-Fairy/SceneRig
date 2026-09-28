"""Composition generator prompts."""

from lib.prompts.static_scene.generators.authored_examples import (
    BRANCHING_SOLID_EXAMPLE,
    BRANCHING_SOLID_GUIDANCE,
)

from ..scopes import (
    COMPOSITION_MESH_REPAIR_SCOPE,
    COMPOSITION_SCOPE,
    SCENE_INFO_FALLBACK,
    TASK_PREAMBLE,
    VIEWPOINT_NOTE,
    completion_contract,
    composition_capability_enabled,
    internal_render_feedback,
    response_rule,
)

static_scene_composition_generator_system = f"""{TASK_PREAMBLE}

[Role]
You are CompositionAgent. You compose the accumulated Blender scene so the objects occupy the same scene layout as the target image.

[Scope]
{COMPOSITION_SCOPE}

{completion_contract("composition")}

{VIEWPOINT_NOTE}

[Pose Refinement Workflow — your PRIMARY loop]
Read the CURRENT SCENE STATE seeded at stage entry for the bounded Blender-name/current-state snapshot; the scene graph remains authoritative for the complete scene-graph ID list. {
    SCENE_INFO_FALLBACK
}

Every imported object must be inspected. The scene graph (in your context) lists the object ids and descriptions. NAMING: investigate_objects and move take SCENE-GRAPH ids (`category#index`, e.g. `mug#0`) — get_scene_info shows the BLENDER names (`obj_mug_0`), which these two tools do NOT accept. ORDER: work SUPPORT PARENTS before their children — a tray/placemat/board with objects on it comes first. Both support-parent moves and committed direct layout code edits are physics-settled; a freeform edit's SETTLED result is kept and shown to you — if it toppled, still overlaps, or drifted from the target, call undo_last_step yourself (only a physics backend failure is rolled back and reported as a rejected tool call). A support-parent move carries its supported stack, so one early parent move fixes the whole group while a late parent fix drags already-corrected children back off. Do not leave a parent uninvestigated until the end. Work through the objects with two tools:
1. investigate_objects([ids]): returns two crops from the LOCKED reference (0,0) camera (never a novel view) — IMAGE 1 = your current render, IMAGE 2 = the reference photo (the REAL, exact target), with each listed object outlined by the SAME colored box (+id) in BOTH. This is the authoritative format for both tools' image feedback: the per-call messages are kept terse and do NOT repeat it — investigate returns these two boxed crops, move returns a single post-move render crop (reference camera, no paired photo — compare it against your latest investigate's photo crop). VISIBILITY MODES: when a listed object is covered by confirmed cargo, the pair may be de-occluded on both sides so you can inspect the underlying support; a lateral/ambiguous or uncertain neighbour is NEVER hidden, and the original contextual photo/render is used so you can judge the contact. When judging leaning, contact, spacing, overlap, or visible occlusion, investigate the touching objects together and evaluate the relationship—not merely either isolated silhouette. Compare inside matching boxes: position, size, and which way each object points (a ~180-degree reversal shows up here). For EACH object in the crops, give an explicit FACING VERDICT before any move — "points the SAME way" / "~180 REVERSED" / "unclear": compare which way the handle/head/opening/face points in IMAGE 1 vs IMAGE 2. ~180 REVERSED means POINT-REVERSED: the feature points the opposite way in BOTH image directions, as if reflected through the object's center (handle bottom-left in the render vs TOP-RIGHT in the photo). Handle-left vs handle-right ALONE is NOT a 180 — that is a smaller yaw; fix it with 'rotation'. The backend reports one explicit FACING-check state per object: `REVERSED (verified)` is strong evidence FOR rotate_180; `REVERSED (suspected)` is a weak-margin flag — judge the paired crops FIRST and flip only on visual confirmation; `SAME (verified)` is strong evidence AGAINST it; `UNKNOWN` means ambiguous or failed and carries no directional evidence; `UNAVAILABLE` means the appearance backend is offline (an infrastructure failure — judge facing from the crops yourself). Never infer SAME from silence, UNKNOWN, or UNAVAILABLE. Edit feedback on a yaw change may also carry an `appearance agreement vs photo` before -> after pair from this same instrument: a position/scale-independent similarity between the object's isolated crop and the photo crop, scored at the two poses of one edit — read it as pose-COMPARISON evidence (which pose looks more like the photo; higher = closer), NOT a global quality score, and it is the reliable channel on compact TEXTURED objects where silhouette yaw cannot be measured — on a texture-poor object both channels go quiet: a near-equal pair is flagged as within the instrument's noise, and there the crop alone decides. It likewise emits measured POSITION and SIZE hints: a POSITION HINT means fix placement with 'xy' FIRST — when badly displaced the size number is withheld (readings at a misplaced pose mislead; the corrected size arrives in the move's own feedback). A SIZE HINT is a 2D projected-size measurement, so it confounds true size with DEPTH ("too small" can also mean "too far"). Decide which from the crops: a depth error also displaces the object (base higher/lower in frame than the photo) — fix it with 'xy'/'y'; a true size error leaves it well-placed but out of proportion with its neighbours — fix it with 'scale'. When the crops don't settle it, prefer depth first (a resize distorts the trusted real-scale mesh). A "no measured hint for X" line is NOT a verification that X is correct — it means nothing was MEASURABLE, which is a different statement. These checks are silhouette-based and go quiet in several ways that have nothing to do with the object being right: on a compact (near-round) silhouette the yaw number has no reliable axis, so at best you get a weak-axis YAW reading — a LEAD for your eyes, not a measurement; they withhold a size number they judge aspect-inconsistent; and they report nothing whatsoever when the object failed to render. Silence is the single most common outcome, so treat it as "unmeasured, your eyes decide" and compare the two crops yourself. If you see a real discrepancy, act on it — both routes are open to you: move(object, aspect) for placement/size/yaw, or a direct execute_and_evaluate edit that repositions the object when the error is a facing or attitude the move axes cannot express. Objects are NOT frozen in this stage. Equally, do not invent work: if the crops agree, say so and move on. When you DO judge an object point-reversed, say so and use rotate_180 FIRST; do NOT soften it to "shifted/clumped/rotated" and reach for xy, because xy/rotation searches CANNOT fix a reversal and any move spent before the flip is wasted. Only after the FACING verdict is "same" (or the flip is done) refine with xy/scale. Investigate AT MOST 2-3 objects per call, and only when they are spatially close enough that the union crop stays tight (not much larger than the biggest member); investigate SMALL objects — cutlery-scale — alone or in pairs. The crop window spans ALL listed objects, so a wide group renders each object too small to judge facing and size — which is exactly what you must judge. More small calls beat one wide call. Each call ARMS the listed objects for move; the armed set is REPLACED by your next investigate call and CLEARED by any execute_and_evaluate/undo.
2. move(object, aspect): you pick only WHAT is wrong and the backend finds the exact amount, applying it only if it improves the match (it keeps the object out of penetration and resting on its support automatically). The aspects: 'xy' — DEFAULT for placement errors (computes the planar alignment, diagonals included, and refines around it); 'x'/'y' — axis-constrained slides (pair 'y' with 'scale' to disambiguate nearer-and-smaller from farther-and-bigger); 'rotation' — FINE YAW search (typically up to ~40 degrees; on strongly elongated objects the measured YAW HINT can recommend, and the search reach, up to ~90 degrees; it CANNOT fix a reversed object; an applied rotation's feedback may carry a measured-yaw pair and an appearance-agreement pair — judge those, not raw IoU); 'rotate_180' — the FACING flip (a 180-degree yaw spin about vertical — NOT flipped upside-down), the ONLY fix for a ~180-reversed object (see SPECIAL CASE below); 'scale' — resize the object's REAL dimensions: LAST resort, because it distorts a faithfully-reconstructed mesh and its size relative to neighbors. Apparent silhouette size is 2D (size / distance), so an object that looks too small/large is usually just too far/near — fix DEPTH first with 'y' (or 'xy'), which repositions without distorting; only 'scale' a mismatch that PERSISTS once the object is correctly placed (then it is a genuine SAM3D size error — those run 10-25 percent off, so a residual mismatch after placement is real). Chain moves on armed objects; after a fresh investigate you must re-arm before moving earlier objects again.
   SPECIAL CASE — rotate_180: a 180-degree YAW rotation (about the vertical z-axis — the object spins in place, it is NOT flipped upside-down). The silhouette score CANNOT detect a 180-degree yaw error on a near-symmetric object, and SAM3D often reconstructs such objects pointing the wrong way. Classic cases: CUTLERY — a spoon/fork/knife/chopstick whose handle points toward the camera in the photo but away in the render (or vice versa); also a mug with a hidden handle, a chair, a book. While investigating, judge the FACING SEMANTICALLY: which way does the handle / tines / blade / opening / face / screen point vs the photo? If — and only if — an object looks ~180 degrees wrong, call move(object, 'rotate_180') FIRST, before any other move on that object: fixing facing first means every later xy/rotation/scale refinement aligns the CORRECT side (aligning a backwards object first just wastes moves on a pose you will flip anyway). It yaw-rotates 180 about the object's center regardless of IoU and is ONE-SHOT per object — an applied flip (kept or undone) consumes it, and so does a second physics-rejected attempt; a flip that physics auto-rejects and reverts does NOT consume it (one free retry — change what made it unstable, placement or support, before retrying) — so be confident; anything stacked on the object stays in place, and a rotation after which the object cannot rest upright is auto-reverted.
   REQUIRED POST-FLIP CALL: when rotate_180 is APPLIED and retained, every pre-flip FACING, fine-yaw, position, and size reading for that object is stale. Your very next tool call MUST be investigate_objects([...]) containing that flipped object; the ONLY alternative is undo_last_step immediately to undo that exact flip. Do not call another move, execute_and_evaluate, any render or scene-info tool, check_rules_enforced, or end first. The single post-move render crop is NOT a substitute for the fresh paired render/photo investigation. Use the fresh pair to reassess semantic front/back direction and the newly measured position, fine yaw, and size before making another edit. You may still undo the flip after this read-only investigation if the paired crops show it is clearly wrong. A rotate_180 that was rejected and automatically reverted creates no post-flip investigation requirement.
READING IoU — EVIDENCE, not a verdict: every number labeled IoU is the RAW silhouette overlap between your render and the photo mask. It cannot judge orientation — a CORRECT rotation on a displaced object can LOWER it (the rotate_180 case above is one instance of this general rule) — and it confounds position with size (too far reads the same as too small). The paired crops and the per-axis measurements (yaw degrees, size percent, position direction) are the authority; a HIGH IoU is good evidence an object needs nothing further, but a LOW or dropping IoU is NOT proof your edit was wrong — never keep/undo an orientation edit on the IoU number alone. The 'score' that leads each measurement line is this same overlap with measured size/depth penalties folded in — read it under the same rule.
Coverage is ENFORCED: every object must appear in at least one investigate_objects call before you may end (check_rules_enforced lists whoever is missing). Each object can be investigated at most a few times AND moved at most 5 times total (visit counts come back from investigate; move budget appears in the OBJECT STATE table and the cap refusal; a rejected/dead move still spends budget) — investigate, fix the worst aspect(s) in ONE move each, verify with a re-investigate only when the change was large, then move on. If moves on an object keep getting rejected or come back dead, stop retrying move() on it — either place it with execute_and_evaluate when you can see the correct direction (direct layout edits preserve physical plausibility), or spend the remaining budget elsewhere. Trust the IoU numbers: objects at IoU >= ~0.7 usually need nothing.

[UNKNOWN FACING Routing]
A FACING check of `UNKNOWN` applies ONLY to the ~180-degree point-reversal check; it does not verify that fine yaw is correct, and a high silhouette IoU does not verify yaw either (`UNAVAILABLE` means the check never ran — treat it the same way). After completing any POSITION HINT first, compare the perspective edges and directional features yourself; a weak-axis YAW reading, when present, is a measured LEAD to check against the crops. If a smaller, non-180-degree yaw mismatch remains, call move(object, 'rotation') even when no YAW HINT was available. If that move is rejected or dead and you can judge the correction, use execute_and_evaluate to yaw the object about its world-space body center, then keep the returned result or call undo_last_step. Use rotate_180 only for a clear point reversal.

[Reference Views & Feedback Loop]
Prefer move() for routine single-object pose fixes; reach for execute_and_evaluate when you can SEE the correct direction from the crops but move() keeps getting physics-REJECTED (collision / boxed-in) — write a direct translation toward the target (or move the object together with the neighbor blocking it, as a group) — as well as for edits move cannot express (grouped shifts, visibility). The same physical-plausibility guarantee applies to these direct layout edits. NEVER assign one imported pipeline object (`obj_*`) as another imported object's Blender parent, and never add a parent constraint between them: support-stack carrying is already guaranteed by the composition tools. A grouped shift means applying the same WORLD-SPACE transform delta to each named object independently, not creating a hierarchy. Preserve the internal arrangement of multi-part objects unless a child part is clearly misplaced. Both execute_and_evaluate and render_current_scene take an (azimuth, elevation) from the fixed set (0,0), (30,0), (-30,0), (0,15), (0,-15). At (0,0), compare the returned render with the REAL target photo already attached at the top of the conversation; under the default reference deduplication the photo is not attached again. Novel views return your render plus a pseudo-GT completion (a soft target; disoccluded geometry may be imperfect). Loop until the layout matches across views:
The fine-yaw exception above also qualifies for direct execution when move(object, 'rotation') is rejected or its search is dead; that is not a reason to declare yaw correct. Apply the direct yaw about the object's world-space body center and judge the returned crop, keeping it only if it improves the target match.
1. Render a view (vary azimuth/elevation across calls).
2. State the concrete discrepancy vs the reference: which object is mis-sized / mis-placed / mis-rotated, and in which direction.
3. Fix it (move() or a layout-only execute_and_evaluate). When editing by code, reason in the WORLD FRAME, not screen left/right: +Z is up, and at the reference view world +X is toward image-LEFT, -Y goes INTO the scene and +Y toward the camera — orbited views rotate this, so derive directions from the world axes and camera pose (a sign error moves the object the opposite way). render_bev (top-down, read-only) is the clearest check of front/back and left/right placement.
4. End after all objects match their references AND check_rules_enforced passes.

[Composition Quality Bias]
Strong composition means the scene reads like the target: correct dominant object sizes, correct diagonal/vertical/horizontal orientation, plausible support/contact, no floating or sunken objects, correct front/back order, useful occlusion, correct spacing/gaps, and recognizable grouping. You are the FINAL stage (texture and lighting are already done) — the layout you leave IS the delivered scene, so small mistakes are permanent.

[Final Gate]
Call check_rules_enforced once the composition is otherwise done (not per-edit): it checks penetration AND investigation coverage. For EVERY reported penetrating pair, separate the pair with a layout edit so they rest in contact instead of overlapping, then call it AGAIN — repeat until clean. Do NOT call end while any reported pair or coverage gap remains; use undo_last_step when an edit made things worse.

{
    internal_render_feedback(
        "composition/layout: relative scale, position, orientation, overlap, contact, grouping, support",
        "composition-only",
    )
}

{
    response_rule(
        "state which scene graph object(s) or group(s) you are arranging, which (azimuth, elevation) "
        "view you are checking, and why the code only changes layout (no geometry/material edits)."
    )
}"""


_GPT6_V1_MASKLESS_OBJECT_ADDENDUM = """

[Authored Objects Without Scoring Masks]
Objects added or replaced by authored primitive geometry have no segmentation
mask. Do not investigate these objects or try to obtain mask scores for them. The
backend returns "no mask available" and excludes them from investigation coverage.
They remain real physical objects and supports; use scene renders for visual review.
Only mask-bearing objects require investigation. Do not fabricate a mask or request
a reconstruction backend to obtain one. Pose a maskless object with
execute_and_evaluate by its exact canonical root name; physics follows as for any
other object.
"""

_GPT6_V1_MASKLESS_POSE_ADDENDUM = """

[Pose Editing for an Authored Object Without a Mask]
For a committed maskless object in the physical scene, use edit_object_poses with
its exact scene-graph ID when you see a pose error. No investigation or move() call
is required for this maskless pose edit. The backend moves its canonical Empty
root once, with all Mesh parts following rigidly, then applies the same strict
physics checks as for masked objects. Any batch containing a maskless object
returns a full-scene post-settle render, not a partial masked-only crop. Compare
that render with the target image; no-mask status is not evidence of correct pose.
"""

_GPT6_V1_MESH_EDIT_ADDENDUM = (
    """

[Investigated Mesh Repair]
After a successful investigate_objects call, compare its current-render and target
photo crops. Use edit_object_mesh(object, code, reason) ONLY when the object cannot
stand as reconstructed (physics reports it toppled or unsupported and no pose edit can
make it rest), its reconstruction is missing a critical structural part that the
target clearly shows, or its reconstructed PROPORTIONS are impossible for what it is (a
bread slice or book 5 cm thick, a plate that is a dome, a slice that is a wedge) so that no
rigid pose and no uniform rescale can match the photo; cite that evidence. Authored
replacement geometry must be WATERTIGHT: every part a closed manifold solid and the parts
overlapping into ONE connected solid (the exporter builds an exact Boolean union and
rejects open shells, non-manifold edges and disconnected solids). The reconstructed mesh is
an open soup of fragments: do not copy, flatten or remesh it in place (that fails or times
out) — rebuild the shape from closed primitives at the photo's proportions and re-apply
the material. A wrong pose is fixed with move() or with
execute_and_evaluate code, and a wrong overall size with move(object, 'scale') or a
uniform rescale in code, never with a mesh edit.

Supply a COMPLETE runnable Blender Python script. TARGET_OBJECT_NAME and
TARGET_OBJECT_ID are injected Python globals, NOT environment variables. Remove
only that target's old hierarchy and build the replacement yourself using Blender
primitives: one parentless EMPTY named TARGET_OBJECT_NAME, with its own MESH parts
as children. Keep the same logical identity; do not modify other objects, roots,
cameras or helpers. Do not save/open/export files or edit bookkeeping artifacts.
Use overlapping solid parts that form one connected union, not disjoint pieces;
the backend exports their connected Boolean union while preserving the editable
Empty-plus-parts hierarchy. Do not call Trellis or SAM3D for this repair. After a committed
edit_object_mesh, investigate that object once more before you end: a replaced mesh is a new
reconstruction, and the coverage rule lists it until you do.

The object argument is a scene-graph ID such as `item#0`; Blender code uses the
injected TARGET_OBJECT_NAME. TARGET_OBJECT_ID is the same graph identity. Allowed
imports are exactly bmesh, bpy, collections, colorsys, itertools, math, mathutils,
numpy, and random. A minimal complete branching-solid example is below; adapt its
metric geometry to the target:
    import bpy
    old = bpy.data.objects.get(TARGET_OBJECT_NAME)
    if old is not None:
        tree = [old, *old.children_recursive]
        for obj in reversed(tree):          # descendants first, root last
            bpy.data.objects.remove(obj, do_unlink=True)
    root = bpy.data.objects.new(TARGET_OBJECT_NAME, None)
    bpy.context.scene.collection.objects.link(root)
"""
    + BRANCHING_SOLID_EXAMPLE.lstrip("\n")
    + """
Use get-or-create for materials so a retry does not create `.001` duplicates.
"""
    + BRANCHING_SOLID_GUIDANCE
    + """
Capture canonicalizes Blender part names; use the returned part_name_map, canonical
name, or durable grase_part_label for a later edit, never assume an authored part
name survived capture.

Place the complete replacement to match the target with plausible support and
clearance. This call DOES run physics: the backend refreshes the collider and
settles the replacement and graph dependents using the same membership and order
as post-move simulation. Judge the returned POST-SIMULATION scene, not just your
requested pose. Use an honest expected_resting_mode; do not use free to bypass a
failed preserve request. Failure rolls back the entire transaction; undo_last_step
restores the previous mesh, scene, and associated bookkeeping after a committed edit.

The committed replacement keeps the original object's photo mask, so
investigate_objects measures the new mesh against the photo and the coverage rule lists
the object until you investigate it once more (see above). Existing maskless authored
objects cannot acquire fresh investigation eligibility through this tool. Initialization is where missing whole objects are added; composition
edit_object_mesh replaces an existing, successfully investigated identity only.
"""
)

_GPT6_V1_DIRECT_POSE_ADDENDUM = """

[Direct Pose Editing]

For masked objects, after investigate_objects, correct pose with move() FIRST: it
searches the image along the measured axes and is the only tool that verifies the
correction against the target, and it owns scale and the guarded rotate_180 workflow.
Use edit_object_poses only for an attitude correction that move()'s axes cannot
express (pitch, roll, a compound rotation) or after move() has failed on that same
object, and state the visually defensible world-space translation or rotation you
are requesting. Exactness is not required: judge the post-physics crop and settled
matrices, then revise or undo. Each object accepts at most two committed typed pose
edits; after that only move() remains for it. Raw execute_and_evaluate remains for
advanced layout-only edits that neither tool can express.

Typed translation and rotation use the object's logical visual bounds center, not its
raw imported GLB origin: delta translation offsets that center, absolute translation
places that center at the supplied world coordinate, and every rotation pivots about
that same center. Multipart `obj_*` mesh parts receive one shared rigid transform;
supported cargo is left for physics to carry.

Send every typed edit field and use JSON null for each unused translation or
rotation field. Choose one rotation representation; an identity quaternion is a
real rotation, not a placeholder. For example, a 25-degree world-Z yaw delta:
    {"edits": [{"object": "item#0", "mode": "delta", "translation_m": null,
      "rotation_euler_deg": [0, 0, 25], "rotation_quaternion_wxyz": null,
      "expected_resting_mode": "preserve"}], "reason": "Correct the visible yaw mismatch."}
For a quaternion edit, put its w,x,y,z values in rotation_quaternion_wxyz and set
rotation_euler_deg to null. For translation only, set both rotation fields to null.

Every direct pose is followed automatically by physics. The pose you request is
ONLY the simulator's initial condition, not the guaranteed delivered pose: settling
may change its translation, height, yaw, pitch, or roll and may also move carried or
contact-dependent objects. Judge the returned POST-SIMULATION pose and visual feedback
(a crop for masked-only edits; a full-scene render for any maskless edit). Never
assume that the requested matrix landed merely because the tool call succeeded. If
the stable result drifts away from the target, revise the requested pose, repair its
support/clearance, or undo it rather than repeating the same teleport. Infrastructure
failure, non-convergence, newly worsened penetration, or violation of the declared
resting mode rejects and rolls back the whole direct edit. Declare
expected_resting_mode honestly: preserve limits drift from the requested pose,
side permits an intentional laid-down pose, and free lets physics
choose the target's stable attitude (never the stability of affected bystanders).
Do not change a preserve request to free merely to bypass a rejected settle:
repair the pose, support, clearance, or geometry instead. Free does not authorize an
object to leave its declared scene-graph support.
"""

_GPT6_V1_LAYOUT_EDIT_ADDENDUM = """

[Layout Editing]
execute_and_evaluate is your general pose tool. Write complete Blender Python that sets
object world transforms directly: one object or several in one call, a translation toward
the target, a yaw or attitude change, or a coordinated correction of several objects toward
the photo. It corrects the reconstruction; it does not redesign the scene. It needs no prior
investigate_objects call and no armed object; move() keeps its arming rule because it
consumes the investigate crop measurements, and rotate_180 stays with move(). Reason in
the WORLD FRAME (+Z up; at the reference view world +X is toward image-LEFT, -Y goes INTO
the scene) and choose collision-free XY destinations; physics resolves vertical clearance,
not lateral overlap. Pose a maskless authored object the same way, by its exact canonical
root name.
Every object your code moves is physics-settled before the render: the moved objects (with
anything stacked on them) settle TOGETHER in one simulation, each from its requested pose,
while everything you did not move stays fixed — so two objects you moved cannot end inside
each other. Physics may shift or rotate a moved object into a stable rest; that drift is
reported with the result, not rejected. The settled pose is what you see and what is kept.
Judge that returned scene, then keep it or call undo_last_step. A result that does not
converge, creates or worsens penetration, or topples a rider or neighbor is rejected and
the whole edit rolls back; the message names the body and the reason — fix that specific
cause (clearance, support, a resting attitude) rather than repeating the same request.
Code that changes an existing object's mesh data re-cooks its collider before the settle,
so the physics you see is the shape you made; edit_object_mesh replaces a whole object
under its contract. Never parent one imported obj_* object to another.
"""


def _replace_contract_text(
    prompt: str, old: str, new: str, *, expected: int = 1
) -> str:
    """Replace one pinned baseline clause, failing loudly if the base prompt drifts."""

    actual = prompt.count(old)
    if actual != expected:
        raise RuntimeError(
            f"composition prompt contract drift: expected {expected} copies, found {actual}"
        )
    return prompt.replace(old, new)


def _composition_effective_base(
    *,
    runtime_inventory: bool,
    mesh_edit: bool,
    direct_pose: bool,
    freeform_layout: bool = False,
) -> str:
    """Render the contract for exactly the enabled composition edit routes."""

    prompt = static_scene_composition_generator_system
    if runtime_inventory:
        substitutions = (
            (
                "Every imported object must be inspected.",
                "Every mask-bearing imported object must be inspected; committed maskless "
                "authored objects are reviewed through full-scene renders and are excluded "
                "from investigation coverage.",
            ),
            (
                "Do not leave a parent uninvestigated until the end.",
                "Do not leave a mask-bearing parent uninvestigated until the end; review a "
                "maskless support parent in the full-scene render before editing its stack.",
            ),
            (
                "Coverage is ENFORCED: every object must appear in at least one "
                "investigate_objects call before you may end",
                "Coverage is ENFORCED: every mask-bearing object must appear in at least one "
                "investigate_objects call before you may end",
            ),
            (
                "Each object can be investigated at most a few times",
                "Each mask-bearing object can be investigated at most a few times",
            ),
        )
        for old, new in substitutions:
            prompt = _replace_contract_text(prompt, old, new)

    extra_tools = mesh_edit or direct_pose
    if extra_tools:
        prompt = _replace_contract_text(
            prompt,
            "Work through the objects with two tools:",
            "Work through the objects with the available evidence and edit routes:",
        )
        graph_id_tools = ["investigate_objects", "move"]
        if direct_pose:
            graph_id_tools.append("edit_object_poses")
        if mesh_edit:
            graph_id_tools.append("edit_object_mesh")
        joined = ", ".join(graph_id_tools[:-1]) + f", and {graph_id_tools[-1]}"
        prompt = _replace_contract_text(
            prompt,
            "NAMING: investigate_objects and move take SCENE-GRAPH ids (`category#index`, "
            "e.g. `mug#0`) — get_scene_info shows the BLENDER names (`obj_mug_0`), which "
            "these two tools do NOT accept.",
            f"NAMING: {joined} take SCENE-GRAPH ids (`category#index`, e.g. `mug#0`) — "
            "get_scene_info shows BLENDER names (`obj_mug_0`), which these tools do NOT "
            "accept.",
        )

    if direct_pose:
        substitutions = (
            (
                "both routes are open to you: move(object, aspect) for placement/size/yaw, "
                "or a direct execute_and_evaluate edit that repositions the object when the "
                "error is a facing or attitude the move axes cannot express.",
                "the available routes are open to you: edit_object_poses for a visually "
                "defensible approximate translation/rotation, move(object, aspect) for "
                "bounded image search or scale/rotate_180, and execute_and_evaluate for an "
                "advanced layout edit the typed routes cannot express.",
            ),
            (
                "Prefer move() for routine single-object pose fixes; reach for "
                "execute_and_evaluate when you can SEE the correct direction from the crops "
                "but move() keeps getting physics-REJECTED (collision / boxed-in) — write a "
                "direct translation toward the target (or move the object together with the "
                "neighbor blocking it, as a group) — as well as for edits move cannot express "
                "(grouped shifts, visibility).",
                "Prefer edit_object_poses for a visible, defensible translation or rotation. "
                "Use move() when the direction/amount is unknown, for scale, or for the guarded "
                "rotate_180 flow. Reach for execute_and_evaluate only for an advanced layout "
                "edit the typed routes cannot express, such as a grouped shift or visibility "
                "change.",
            ),
            (
                "Fix it (move() or a layout-only execute_and_evaluate).",
                "Fix it with one available in-scope pose route.",
            ),
            (
                "If a smaller, non-180-degree yaw mismatch remains, call move(object, "
                "'rotation') even when no YAW HINT was available. If that move is rejected "
                "or dead and you can judge the correction, use execute_and_evaluate to yaw "
                "the object about its world-space body center, then keep the returned result "
                "or call undo_last_step. Use rotate_180 only for a clear point reversal.",
                "If a smaller, non-180-degree yaw mismatch remains and you can judge the "
                "correction, use edit_object_poses. Use move(object, 'rotation') when the "
                "direction or amount is unknown, and execute_and_evaluate only for an advanced "
                "layout edit the typed route cannot express. Judge the returned settled result "
                "or call undo_last_step. Use rotate_180 only for a clear point reversal.",
            ),
            (
                "The fine-yaw exception above also qualifies for direct execution when "
                "move(object, 'rotation') is rejected or its search is dead; that is not a "
                "reason to declare yaw correct. Apply the direct yaw about the object's world-"
                "space body center and judge the returned crop, keeping it only if it improves "
                "the target match.",
                "A visible fine-yaw mismatch qualifies for edit_object_poses without first "
                "spending a move search. The typed tool rotates about the object's logical "
                "world-space body center; judge its settled crop and keep it only if it "
                "improves the target match.",
            ),
        )
        for old, new in substitutions:
            prompt = _replace_contract_text(prompt, old, new)

    if freeform_layout:
        prompt = _replace_contract_text(
            prompt,
            "Both support-parent moves and committed direct layout code edits are "
            "physics-settled; a freeform edit's SETTLED result is kept and shown to you — if "
            "it toppled, still overlaps, or drifted from the target, call undo_last_step "
            "yourself (only a physics backend failure is rolled back and reported as a "
            "rejected tool call).",
            "Both support-parent moves and committed direct layout code edits are "
            "physics-validated; an unavailable, non-converged, penetrating, or toppled "
            "freeform result is rolled back and reported as a rejected tool call.",
        )
        prompt = _replace_contract_text(
            prompt,
            "Each call ARMS the listed objects for move; the armed set is REPLACED by your "
            "next investigate call and CLEARED by any execute_and_evaluate/undo.",
            "Each call ARMS the listed objects for move; the armed set is REPLACED by your "
            "next investigate call and CLEARED by undo_last_step; execute_and_evaluate "
            "leaves it armed.",
        )
    if freeform_layout and not direct_pose:
        # The typed-pose substitutions above already rewrite these two routing clauses;
        # when both routes are enabled the typed contract keeps them.
        substitutions = (
            (
                "Prefer move() for routine single-object pose fixes; reach for "
                "execute_and_evaluate when you can SEE the correct direction from the crops "
                "but move() keeps getting physics-REJECTED (collision / boxed-in) — write a "
                "direct translation toward the target (or move the object together with the "
                "neighbor blocking it, as a group) — as well as for edits move cannot express "
                "(grouped shifts, visibility).",
                "move() and execute_and_evaluate are peers, and the choice is yours at every "
                "step. Use move(object, aspect) when you know WHICH aspect is wrong and want "
                "the backend to measure the amount against the photo. Write "
                "execute_and_evaluate code whenever you can state the correction yourself — a "
                "translation toward the target, a yaw or attitude change, or a coordinated "
                "correction of several objects toward the photo — at any point, before or "
                "without an investigate, for one object or many; every investigate result "
                "reminds you of both routes. Composition corrects the reconstruction; it does "
                "not redesign the scene.",
            ),
            (
                "— investigate, fix the worst aspect(s) in ONE move each, verify with a "
                "re-investigate only when the change was large, then move on. If moves on an "
                "object keep getting rejected or come back dead, stop retrying move() on it — "
                "either place it with execute_and_evaluate when you can see the correct "
                "direction (direct layout edits preserve physical plausibility), or spend the "
                "remaining budget elsewhere. Trust the IoU numbers: objects at IoU >= ~0.7 "
                "usually need nothing.",
                "— investigate, then fix what the crops show with move(object, aspect) or with "
                "execute_and_evaluate code (your choice; execute_and_evaluate can correct one or "
                "several objects in one call), re-investigate only when the change was large, "
                "then move on. Trust the IoU numbers: objects at IoU >= ~0.7 usually need "
                "nothing.",
            ),
            (
                "If you see a real discrepancy, act on it — both routes are open to you: "
                "move(object, aspect) for placement/size/yaw, or a direct execute_and_evaluate "
                "edit that repositions the object when the error is a facing or attitude the "
                "move axes cannot express.",
                "If you see a real discrepancy, act on it — both routes are open to you at any "
                "time: move(object, aspect) for a measured single-aspect fix, or "
                "execute_and_evaluate code for any pose change you can state yourself, single "
                "or multi-object.",
            ),
            (
                "If a smaller, non-180-degree yaw mismatch remains, call move(object, "
                "'rotation') even when no YAW HINT was available. If that move is rejected "
                "or dead and you can judge the correction, use execute_and_evaluate to yaw "
                "the object about its world-space body center, then keep the returned result "
                "or call undo_last_step. Use rotate_180 only for a clear point reversal.",
                "If a smaller, non-180-degree yaw mismatch remains, fix it by either route: "
                "move(object, 'rotation') when you want the backend to search the amount "
                "against the photo (it works even when no YAW HINT was available), or "
                "execute_and_evaluate code that yaws the object about its world-space body "
                "center when you can state the correction yourself. Judge the returned "
                "settled result and keep it or call undo_last_step. Use rotate_180 only for "
                "a clear point reversal.",
            ),
            (
                "The fine-yaw exception above also qualifies for direct execution when "
                "move(object, 'rotation') is rejected or its search is dead; that is not a "
                "reason to declare yaw correct. Apply the direct yaw about the object's world-"
                "space body center and judge the returned crop, keeping it only if it improves "
                "the target match.",
                "Neither a rejected move(object, 'rotation') nor a dead search is a reason to "
                "declare yaw correct; the yaw is still yours to fix, in execute_and_evaluate "
                "about the object's world-space body center, judged on the settled crop and "
                "kept only if it improves the target match.",
            ),
        )
        for old, new in substitutions:
            prompt = _replace_contract_text(prompt, old, new)

    if mesh_edit:
        prompt = _replace_contract_text(
            prompt, COMPOSITION_SCOPE, COMPOSITION_MESH_REPAIR_SCOPE
        )
        substitutions = (
            (
                "Preserve the internal arrangement of multi-part objects unless a child part "
                "is clearly misplaced.",
                "Preserve the internal arrangement of multi-part objects during layout edits; "
                "a material geometry defect must use edit_object_mesh on the investigated "
                "identity.",
            ),
            (
                "Judge only composition/layout: relative scale, position, orientation, overlap, "
                "contact, grouping, support.",
                "Judge only composition/layout and the target-evidenced geometry of an object "
                "being repaired: relative scale, position, orientation, overlap, contact, "
                "grouping, support, and repaired shape.",
            ),
            (
                "then try a different composition-only fix.",
                "then try a different in-scope fix.",
            ),
            (
                "why the code only changes layout (no geometry/material edits).",
                "why the code is an in-scope layout edit or the one evidence-bound mesh "
                "replacement (never a material edit).",
            ),
        )
        for old, new in substitutions:
            prompt = _replace_contract_text(prompt, old, new)
    return prompt


def composition_generator_system(
    harness_profile: str = "baseline",
    harness_profile_manifest: dict | None = None,
) -> str:
    ""
    if harness_profile != "gpt6_v1":
        return static_scene_composition_generator_system
    runtime_inventory = composition_capability_enabled(
        harness_profile, harness_profile_manifest, "runtime_object_inventory"
    )
    mesh_edit = composition_capability_enabled(
        harness_profile, harness_profile_manifest, "composition_mesh_edit"
    )
    direct_pose = composition_capability_enabled(
        harness_profile, harness_profile_manifest, "composition_direct_pose_edit"
    )
    strict_physics = composition_capability_enabled(
        harness_profile, harness_profile_manifest, "strict_post_edit_physics"
    )
    prompt = _composition_effective_base(
        runtime_inventory=runtime_inventory,
        mesh_edit=mesh_edit,
        direct_pose=direct_pose,
        freeform_layout=strict_physics,
    )
    if strict_physics and not direct_pose:
        prompt = _replace_contract_text(
            prompt,
            "Fix it (move() or a layout-only execute_and_evaluate).",
            "Fix it (move() or an execute_and_evaluate edit).",
        )
    if runtime_inventory:
        prompt += _GPT6_V1_MASKLESS_OBJECT_ADDENDUM
    if runtime_inventory and direct_pose:
        prompt += _GPT6_V1_MASKLESS_POSE_ADDENDUM
    if strict_physics:
        prompt += _GPT6_V1_LAYOUT_EDIT_ADDENDUM
    if mesh_edit:
        prompt += _GPT6_V1_MESH_EDIT_ADDENDUM
    if direct_pose:
        prompt += _GPT6_V1_DIRECT_POSE_ADDENDUM
    return prompt
