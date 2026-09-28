"""Single source of truth for each static-scene stage's edit boundary.

Every stage's "what it may and may not touch" used to be restated in three places —
the generator system prompt, the verifier system prompt, and the generator's per-round
render-feedback in ``lib/agents/generator.py`` — which silently drifted apart (the texture
feedback still judged objects long after the system prompt was changed to root-surfaces
only). These constants are the ONE place to edit a stage's scope; the prompts and the
render-feedback all embed them.

Pipeline order (see ``lib/agents/root.py``): initializer -> texture -> lighting ->
composition. Physics is settled in PREPROCESSING (before init), so there is no physics stage;
there is no geometry stage, no camera stage, and composition is the FINAL stage — do not
reference stages that do not exist.
"""

from __future__ import annotations

# Shared task statement prepended to every static-scene stage system prompt (one source).
TASK_PREAMBLE = (
    "[Task] You are one stage of a pipeline that reconstructs a 3D scene in Blender from a "
    "SINGLE reference image. Your edits accumulate in a delivered .blend file and its "
    "reference-view render. Isaac/USD conversion is a separate, later operation and is "
    "not implied by completion of this Blender stage."
)


def reconstruction_exclusions(ignore_objects: list[str] | None) -> str:
    """Format the runtime ignore list for every stage's system prompt.

    Args:
        ignore_objects: Object categories explicitly excluded by the run configuration.

    Returns:
        Shared exclusion guidance, or an empty string when no categories are excluded.
    """
    if not ignore_objects:
        return ""
    return (
        "[Reconstruction Exclusions]\n"
        "The following objects are intentionally excluded from reconstruction, even when "
        "visible in the reference image or pseudo-GT views: "
        + ", ".join(ignore_objects)
        + ". Do not reconstruct them as geometry or reproduce them through textures. "
        "Their absence from renders is expected; do not reject an attempt or request "
        "corrections to restore them. Where they occlude the reference scene, do not "
        "mistake their boundaries for underlying surface boundaries. Other scene objects "
        "remain in scope. This instruction does not override your stage's "
        "object-preservation rules."
    )


# Meaning of `end` for ONE stage. Keep this in every generator prompt so stage-specific
# workflow prose cannot quietly redefine budget exhaustion as successful completion.
#
# PER-STAGE since 2026-08-22 (audits/PROMPT_LEAKAGE_AUDIT_2026_08_22.md). One shared
# paragraph used to spell out EVERY stage's gating to every stage, so the initializer was
# told "composition has no verifier" — cross-stage process detail it cannot act on. It also
# closed by explaining that a non-approved stage "remains a quality warning rather than
# making a structurally valid Blender artifact fail": orchestrator bookkeeping that told the
# agent its own failure was tolerated. Each stage now sees only what gates ITS end call.
_STAGE_GATES = {
    "initializer": (
        "A current check_rules_enforced pass is REQUIRED before you may call end, and "
        "your paired verifier must also approve this attempt."
    ),
    "texture": (
        "This stage has no rules gate; your paired verifier reviews what you deliver. "
        "Call end as soon as the completion conditions above are met."
    ),
    "lighting": (
        "This stage has no rules gate; your paired verifier reviews what you deliver. "
        "Call end as soon as the completion conditions above are met."
    ),
    "composition": (
        "A current check_rules_enforced pass is REQUIRED before you may call end. This "
        "stage has no verifier: the layout you leave is the delivered scene."
    ),
}


def completion_contract(stage: str) -> str:
    """Return the `[Completion Contract]` block for ONE stage.

    Args:
        stage: One of ``initializer``, ``texture``, ``lighting``, ``composition``.
    """
    if stage not in _STAGE_GATES:
        raise ValueError(f"No completion contract defined for stage {stage!r}")
    return (
        "[Completion Contract]\n"
        "Calling end voluntarily declares that this stage's stated completion conditions "
        "and any required rules gate are satisfied. " + _STAGE_GATES[stage]
    )


# Exact support predicate shared by the initializer generator, verifier, and rule-tool
# description. Keep the threshold and the photo exception in one place.
# 2026-09-16 (owner): one support contract for BOTH harnesses — each object is checked
# against its exact declared direct support (scene-graph edge), never collapsed to the
# main support. Enforced by script_generators.objects_on_direct_supports_report.
OBJECT_SUPPORT_CONTRACT = (
    "Each current object's XY footprint must be fully inside its exact declared direct support "
    "with a small margin, OR be a photo-supported partial overhang with at least 65% of its "
    "footprint on that direct support. Check a stacked object against its immediate object "
    "support and a root-supported object against that root surface; never collapse the check to "
    "an ancestor. A photo-supported overhang is valid and must not be eliminated by growing the "
    "direct support past the photo's visible edge. A missing, unknown, or cyclic support chain "
    "fails closed."
)

# Initialization normally trusts the physics-settled poses, but the old absolute freeze
# made the resting/penetration gate prescribe an edit that the structure gate then rejected.
# Keep the narrow exception in one shared contract: generator prompt, verifier prompt, and
# per-round short scope must not independently redefine which object corrections are allowed.
INITIALIZER_OBJECT_CORRECTION_CONTRACT = (
    "Imported object poses are physics-settled and are the trusted default, not an invitation "
    "to relayout them. Do not edit an object's transform directly in Blender code. "
    "`nudge_object` is the only initializer object-edit path: use the smallest necessary "
    "translation, never rotation, only when either check_rules_enforced reports a verified "
    "penetration or resting defect involving that object, or a paired current/reference-view "
    "comparison shows a clear, visually significant position mismatch. Correct one affected "
    "object or support stack at a time. Inspect the current render attached automatically "
    "after a successful correction, then immediately rerun check_rules_enforced. "
    "Verified physical defects use a separate bounded repair budget from the two-object "
    "visual correction allowance: at most 200 mm per call, two calls per target, 16 total. "
    "Use measured small-clearance candidates where offered; numerical contacts clearing "
    "within 0.25 mm need no repair. "
    "Never use an object correction to hide an incorrectly sized, placed, or oriented root "
    "surface. Preserve its declared support and required support margin, and do not create a "
    "new collision, penetration, floating state, or unsupported stack. Never rotate, scale, "
    "create, delete, duplicate, rename, reparent, remesh, deform, or change the material or "
    "texture of an imported object. Minor or ambiguous pose refinement belongs to composition."
)

# Opt-in initializer object repair is an effective contract, not an addendum that asks the
# model to reconcile itself with the nudge-only baseline above.  Keep the baseline constant
# byte-stable; profile-aware prompt selectors substitute this block only when the complete
# authenticated transaction capability set is present.
INITIALIZER_OBJECT_REPAIR_CAPABILITIES = (
    "initializer_code_transactions",
    "runtime_object_inventory",
    "mutation_journal",
)


def initializer_object_repair_enabled(
    harness_profile: str | None,
    harness_profile_manifest: dict | None,
) -> bool:
    """Whether the complete authenticated initializer repair contract is available."""

    selected_profile = harness_profile or (harness_profile_manifest or {}).get("name")
    if selected_profile != "gpt6_v1":
        return False
    capabilities = (harness_profile_manifest or {}).get("capabilities")
    # 2026-09-15 owner decision: a missing manifest fails CLOSED.
    return bool(capabilities) and all(
        capabilities.get(capability, False)
        for capability in INITIALIZER_OBJECT_REPAIR_CAPABILITIES
    )


INITIALIZER_OBJECT_REPAIR_CONTRACT = (
    "Imported object poses and meshes remain trusted defaults; do not churn an object merely "
    "because a repair tool is available. A wrong pose is fixed by a rigid pose correction and a "
    "wrong overall size by a uniform rescale; neither ever justifies replacing a mesh. Replace an "
    "imported mesh ONLY when (1) the object cannot stand as reconstructed - a simulation report "
    "or the preprocessing pose history marks it toppled, unsupported, or repair_needed and no "
    "pose correction can make it rest (for example toast slices whose reconstructed shape cannot "
    "sit in the rack and scatter onto the surface) - or (2) the reconstruction is missing a "
    "critical structural part that the target clearly shows (for example a monitor without its "
    "stand or base), or (3) its reconstructed PROPORTIONS are impossible for what it is (a "
    "bread slice or book 5 cm thick, a plate that is a dome) so that no rigid pose and no "
    "uniform rescale can match the photo - then replace it with a WATERTIGHT shape of the right "
    "proportions (every authored part a closed manifold solid, parts overlapping into one "
    "connected solid; the reconstructed fragment soup cannot be edited in place). Cite the report row, name the "
    "missing part, or state the measured proportion. An object that stands, shows all its parts "
    "and has possible proportions is never replaced, however imperfect its texture, size, or "
    "orientation looks. There is no nudge_object in this harness: every object pose change, "
    "including clearing a reported penetration or resting defect, is a rigid translation or pose "
    "correction written as an authenticated execute_and_evaluate object transaction (simulated "
    "within that object's hierarchy). Use that transaction for an evidence-bound addition, "
    "removal, replacement under conditions (1)-(3), or rigid pose / uniform-size correction, "
    "with exact declarations and backend-provided identity bindings. Preserve every unchanged "
    "object, support relation, material, and hierarchy. A replacement keeps the same graph "
    "identity and exact support chain and happens at most once per object; a pose or size edit "
    "changes the canonical object root rigidly or by a uniform scale, never individual mesh "
    "parts or a non-uniform scale. Never use an object edit to hide an incorrectly sized, placed, or "
    "oriented root surface, and never introduce a collision, penetration, floating state, or "
    "unsupported stack. Minor, ambiguous, decorative, or heavily occluded discrepancies remain "
    "unchanged for composition. Trust a mutation only when its transaction reports COMMITTED."
)

# All stage roles share one missing-preseed fallback. Keep this exact wording in one
# place so a prompt cannot simultaneously claim the seed is guaranteed and instruct
# the model to spend a round fetching it unconditionally.
SCENE_INFO_FALLBACK = (
    "If that block is absent because preseeding is disabled or failed, call "
    "get_scene_info once; otherwise do not re-fetch it."
)

VERIFIER_SCENE_INFO_GUIDANCE = (
    "The CURRENT SCENE STATE block contains the bounded scene snapshot when "
    "preseeding succeeds. "
    + SCENE_INFO_FALLBACK
    + " It does not change during your read-only review. Do not approve from "
    "scene-info text alone."
)

# Shared camera/viewpoint note. The camera is fixed to the reference view; the agent picks a
# viewpoint via the tools' (azimuth, elevation) arg rather than editing the camera in code.
VIEWPOINT_NOTE = (
    "[Viewpoints] The camera is FIXED to the reference view (the input image's viewpoint) — NEVER "
    "create, move, delete, or re-point the camera in your Blender code. To see the scene from "
    "another angle, pass an (azimuth, elevation) from the offered set to execute_and_evaluate "
    "(edits, then renders that view) or render_current_scene (renders that view without editing). "
    "(azimuth, elevation) is an ORBIT OFFSET (a DELTA) FROM the reference view, NOT an absolute "
    "world angle: (0,0) IS the reference view — compare its render against the REAL target photo "
    "at the top of this conversation (the photo is not re-attached under the default reference "
    "deduplication); azimuth orbits horizontally away from it and elevation tilts up/down from it, "
    "and those views return a pseudo-GT (a soft target). Inspect more than just (0,0) so the scene is "
    "right in depth, not only head-on. render_current_scene called with NO (azimuth, elevation) "
    "gives a bare render of the source view with no reference attached — the target photo at the "
    "top of this conversation is that view's reference, so compare against it there; pass a "
    "NOVEL viewpoint whenever you want a render/reference pair returned together."
)

# Lighting's replacement for VIEWPOINT_NOTE. That stage's render tools carry no
# (azimuth, elevation) argument (see exec.py execute_and_evaluate_source_view_tool: a novel
# view's pseudo-GT has inpainter-invented illumination, so it cannot settle exposure, shadow
# or colour temperature). Says only what IS true — the camera constraint, which no schema
# enforces, and what comes back. It does NOT explain the missing argument: advertising a
# capability a stage lacks is what TL2 was.
VIEWPOINT_NOTE_SOURCE_ONLY = (
    "[Viewpoint] The camera is FIXED to the reference view (the input image's viewpoint) — "
    "NEVER create, move, delete, or re-point the camera in your Blender code. A successful "
    "execute_and_evaluate call returns the edited render paired with the REAL target photo. "
    "render_current_scene returns a bare current render from the same reference view; compare "
    "it with the target photo attached at the top of this conversation."
)

INITIALIZER_SCOPE = (
    "Build and refine the CURRENT REGISTERED ROOTS and set up basic scene lighting. At stage "
    "entry these are the PREPROCESSED ROOTS named in the per-scene data (typically a supporting "
    "surface and one or more walls; sometimes a floor, ceiling, or distinct cabinet/base). Use "
    "execute_and_evaluate to create or edit any current registered root. If those roots are "
    "insufficient because the reference clearly requires a DISTINCT architectural wall plane "
    "that no current registered root can represent, use build_root_surface sparingly to "
    "introduce it. The backend assigns and registers its identity; the returned RUNTIME-ADDED "
    "ROOT immediately becomes a current registered root and may then be refined with "
    "execute_and_evaluate. Do not use build_root_surface for an already registered wall, an "
    "extension/repositioning of its plane, an opening, a frame/detail, a window/door, or a "
    "duplicate plane. The sole general exception is ONE level floor/ground helper: reuse a "
    "preprocessed floor/ground when present, otherwise you may create "
    "one at a physically plausible height to ground the scene. That helper is allowed even "
    "when it is absent from the per-scene roots or invisible in the reference, for every main-"
    "support form including tabletop. Build EVERY current registered root exactly once, "
    "including every current registered wall; never omit or delete a preprocessed root. Wall "
    "STRUCTURE is in scope only for a current registered wall: "
    "cut-through window/door openings may be carved into its own mesh and wall_N_<part> detail "
    "children may be added. Never introduce an unregistered root with execute_and_evaluate, "
    "assign a new graph ID yourself, or directly edit scene_graph.json or any preprocessing "
    "artifact. remove_root_surface may remove only a runtime-added root; it must never remove a "
    "preprocessed root. The helper exception alone never permits an extra wall, ceiling, "
    "cabinet, base, or other non-floor root; a genuinely missing wall uses build_root_surface. "
    "The camera is the reference view — the "
    "viewpoint of the input image — so never create, move, or delete it. "
    + INITIALIZER_OBJECT_CORRECTION_CONTRACT
    + " Never clear the scene. "
    "Coverage ground truth can itself be wrong: the segmentation model sometimes under- or "
    "over-segments large soft surfaces (curtains, cloth, glass). When a coverage failure fires "
    "you get tinted photo/render pairs per surface — judge the PHOTO side first. If the GT mask "
    "is visibly wrong, bypass([...]) waives that surface's coverage check. It may also waive "
    "visibility for a preprocessed registered wall that the target genuinely does not depict, "
    "but the wall must still be built. A runtime-added wall found unnecessary must instead be "
    "removed with remove_root_surface. Use bypass sparingly, never to dodge a "
    "fixable build error. The main support is bypassable too, but treat that "
    "as a last resort — it anchors every object placement."
)

TEXTURE_SCOPE = (
    "Edit ONLY the materials of the root surfaces the initializer actually built — whichever of "
    "table/counter, floor/ground, walls, ceiling, or cabinet/base exist in the scene (the "
    "non-`obj_*` meshes; use the CURRENT SCENE STATE seeded at stage entry for their names). "
    "`wall_N_<part>` detail children (window frames, glass, mullions, backdrops) are "
    "root-surface parts too — texture them: give glass its transmission; a backdrop child may "
    "carry the photo's window content. "
    "For each root surface you may set material "
    "slots, shader nodes, procedural/image textures, UVs, color, roughness/specular/metallic/"
    "transmission, and bump/normal to match the reference. Every imported object mesh (its name "
    "starts with `obj_`) already ships final baked SAM3D materials — NEVER edit, reassign, "
    "recolor, remove, or judge an object's materials. Do NOT edit geometry, transforms, "
    "hierarchy, cameras, lights, world/background, render/compositor/color-management, "
    "composition, or layout."
)

COMPOSITION_SCOPE = (
    "Objects are ALREADY metrically placed and pose-registered to the reference, so make only "
    "SMALL corrective TWEAKS to the layout — never a relayout. Nudge relative object/group "
    "location, rotation, scale, spacing, contact/grounding, overlap, front/back order, "
    "visibility, and grouped shifts through equal world-space transform deltas to "
    "fix obvious residual errors (a floating or sunken object, a missed contact, an overlap, "
    "a slightly-off size). Every imported `obj_*` pipeline object must remain an "
    "independent Blender root: do NOT parent it under any object or Empty and do not add "
    "a CHILD_OF constraint. Do NOT "
    "move objects far from their current transforms or rearrange the scene. Do NOT edit camera, "
    "mesh/curve geometry, materials/textures/shaders/UVs, or lights/world/exposure/color "
    "management; leave a bad shape, material, or light to its owning stage."
)

# The authenticated mesh-repair route changes exactly one clause of composition's scope.
# Derive it from the baseline so all layout and preservation language stays aligned while the
# exported baseline constant remains byte-identical.
COMPOSITION_MESH_REPAIR_SCOPE = COMPOSITION_SCOPE.replace(
    "Do NOT edit camera, mesh/curve geometry, materials/textures/shaders/UVs, or lights/world/"
    "exposure/color management; leave a bad shape, material, or light to its owning stage.",
    "Do NOT edit camera, materials/textures/shaders/UVs, or lights/world/exposure/color "
    "management. Replace a whole object's mesh only through edit_object_mesh for one "
    "current, successfully investigated identity whose target crop proves a material shape "
    "defect; code that changes an existing mesh re-cooks its collider before the settle. "
    "Preserve every unrelated object and leave material or lighting defects to their owning "
    "stage.",
)

LIGHTING_SCOPE = (
    "Refine ONLY illumination: light objects/data, key/fill/rim/back lights, placement and "
    "direction, type, energy, color temperature, size/angle/softness, shadow strength/softness, "
    "world/ambient lighting, exposure, and color management. Do NOT edit object geometry, "
    "materials/textures/shaders/UVs, object layout/composition, or camera pose/lens/framing — "
    "leave those as-is and make the existing scene readable and target-like through lighting alone."
)

# ---- Shared prompt blocks (dedup 2026-07-14) -------------------------------------- #
# The per-stage generator/verifier prompts repeated these near-verbatim (one noun
# swapped per stage) and drifted. Single-source them here; tool USAGE details live in
# the tool schemas the model already sees — prompts should not re-describe tools.


def internal_render_feedback(judge: str, fix_kind: str) -> str:
    """Per-stage [Internal Render Feedback] block (texture/lighting/composition)."""
    return (
        "[Internal Render Feedback]\n"
        "After every rendered edit, compare the new render against the previous state, "
        "the reference for the rendered view, and any approval_checklist in stage "
        f"context. Judge only {judge}. If the last edit made the match worse or "
        "regressed a done item, call undo_last_step immediately, then try a different "
        f"{fix_kind} fix."
    )


def response_rule(before_execute: str) -> str:
    """Per-stage [Response Rule] block."""
    return (
        "[Response Rule]\n"
        "During normal tool-call rounds, every response must be exactly one tool call. "
        "If and only if you receive [Terminal End-Only Grace], call end only when you "
        "voluntarily certify completion; otherwise emit no tool call. Before "
        "execute_and_evaluate, " + before_execute
    )


# Verifier: first-evidence protocol (the texture and lighting verifiers; composition is
# single-pass with no verifier, and the initializer has its own longer evidence section).
# The no-render clause is not hypothetical — between 07-22 and
# 07-26 the render never reached ANY verifier (audits/VERIFIER_RENDER_2026_07_26.md), and the
# only reason verdicts stayed grounded was that models happened to render first unprompted.
VERIFIER_FIRST_STEP = (
    "[Required First Step]\n"
    "The first verifier user message should include the attached target image and a current "
    "full-scene render from the generator attempt. Compare those "
    "two images first. If NO render is attached, your FIRST tool call must be "
    "render_reference_view (it re-renders the CURRENT scene from the locked photo "
    "camera) — do not start investigating and do not decide without it. Use "
    "investigation tools only to inspect what those two images cannot settle (hidden "
    "placement, bounds, alternate views, light/shadow behavior). "
    + VERIFIER_SCENE_INFO_GUIDANCE
)

# Verifier: a generator that ended WITHOUT a passing rules gate (or was cut off at max
# rounds) is a procedural non-finish — the gate's guarantees were never confirmed.
NON_FINISH_REJECT = (
    "ALWAYS set approved=false if the attempt note says the generator ended WITHOUT a "
    "passing check_rules_enforced or was CUT OFF at max rounds (a procedural "
    "non-finish: the gate's guarantees were never confirmed) — but still evaluate "
    "every aspect and give concrete fixes + a full approval_checklist for the next "
    "attempt."
)


def verifier_decision_tail(scope_word: str) -> str:
    return (
        f"If rejecting, provide an approval_checklist containing ONLY {scope_word} "
        "tasks. Always include regression_check."
    )


# One-line versions of each scope, for places that re-state the boundary every round (e.g. the
# generator's per-render feedback) where the full text above would just burn tokens. Keep these in
# sync with the full scope they summarize.
INITIALIZER_SCOPE_SHORT = (
    "build every current registered root exactly once and edit only those roots + basic "
    "lighting; use execute_and_evaluate to create/edit preprocessed roots and runtime-added "
    "roots; use build_root_surface sparingly only for a distinct missing wall plane, never to "
    "introduce an unregistered root through execute_and_evaluate; permit the allowed single "
    "floor/ground helper and detail children on current registered walls; never edit the scene "
    "graph directly, and remove_root_surface may remove only a runtime-added root; "
    "never move the reference-view camera; object poses are the trusted default and may be "
    "translated only through nudge_object for a verified physical defect or a clear, visually "
    "significant reference-position mismatch; otherwise fix the surface, and leave minor pose refinement "
    "to composition"
)
INITIALIZER_OBJECT_REPAIR_SCOPE_SHORT = (
    "build every current registered root exactly once and edit only those roots + basic "
    "lighting, except for a bounded target-evidenced object repair through the authenticated "
    "transaction path; admit a new object only when its declared support chain reaches the "
    "main support, preserve exact direct supports and valid stacks, never edit the scene graph "
    "directly, and never move the reference-view camera"
)
TEXTURE_SCOPE_SHORT = (
    "edit only the root-surface materials (wall_N_* detail children included); never touch "
    "obj_* objects, geometry, transforms, camera, lights, world, or layout"
)
COMPOSITION_SCOPE_SHORT = (
    "make only small corrective layout tweaks to already-placed objects (no relayout); "
    "do not edit geometry, materials, or lighting"
)
COMPOSITION_OBJECT_REPAIR_SCOPE_SHORT = (
    "make only small corrective layout tweaks to already-placed objects (no relayout), plus "
    "an evidence-bound edit_object_mesh replacement of one successfully investigated identity; "
    "do not edit unrelated geometry, materials, or lighting"
)
LIGHTING_SCOPE_SHORT = "edit only lighting/world/exposure; do not touch geometry, materials, layout, or camera"


def effective_initializer_scope_short(
    harness_profile: str | None,
    harness_profile_manifest: dict | None,
) -> str:
    """Return the initializer caption scope for the actually enabled tool contract."""

    if initializer_object_repair_enabled(harness_profile, harness_profile_manifest):
        return INITIALIZER_OBJECT_REPAIR_SCOPE_SHORT
    return INITIALIZER_SCOPE_SHORT


def composition_capability_enabled(
    harness_profile: str | None,
    harness_profile_manifest: dict | None,
    capability: str,
) -> bool:
    """Resolve one opt-in composition capability without changing baseline behavior."""

    selected_profile = harness_profile or (harness_profile_manifest or {}).get("name")
    if selected_profile != "gpt6_v1":
        return False
    capabilities = (harness_profile_manifest or {}).get("capabilities")
    # 2026-09-15 owner decision: a missing manifest fails CLOSED (no opt-in capability).
    return bool(capabilities) and capabilities.get(capability, False) is True


def effective_composition_scope_short(
    harness_profile: str | None,
    harness_profile_manifest: dict | None,
) -> str:
    """Return the composition caption scope for the actually enabled edit surface."""

    if composition_capability_enabled(
        harness_profile, harness_profile_manifest, "composition_mesh_edit"
    ):
        return COMPOSITION_OBJECT_REPAIR_SCOPE_SHORT
    return COMPOSITION_SCOPE_SHORT
