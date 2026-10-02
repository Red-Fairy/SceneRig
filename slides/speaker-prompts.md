# SceneRig: From Images to Robot-Ready Worlds — October 1, 2026

## 1. SceneRig

Open with a genuine moving-camera render of the Wendy1 Claude Opus 5 reconstruction. The opening camera orbit plays forward and backward in a continuous loop, without a viewpoint reset. This is a reconstructed scene, not the input photograph. The talk is about preserving both object arrangement and physical behavior well enough for robotic interaction. Presentation appearance update: the front white mug uses the existing logo-free website photograph/SAM3D visual variant. Original reconstruction states, camera paths and recorded solver trajectories are retained; this does not denote a new benchmark run. The two completion photographs preserve their saved content outside the lettering patch.

Discussion notes:

## 2. Visual similarity is not enough

Original top-camera recording of everything-in-bin episode74fbca2f, the same scene used for the paper teaser and native Goal example. Source is the original1920×1080 HEVC recording remuxed from /top-camera, not a wrist view or reconstructed render. The full frame is preserved. Skip the first12seconds of initialization and play the remaining recorded actions at3×, ending with a one-second hold. The real robot transfers all3 new objects into the bin; the mayonnaise bottle was already inside. A rendered image can look convincing while containing collisions, floating objects, or inaccurate metric placement. These are separate requirements. Robotics experiments use measured depth and calibrated intrinsics; in-the-wild scenes use monocular estimates.

Discussion notes:

## 3. Reconstruct with a general coding agent

New native persistent Goal experiment on the exact74fbca2f input. GPT-6 Astra/ultra completed the original single-image procedural reconstruction prompt, producing independently grouped bin, tape roll and three bottles. NativeGoal API is the same persistent-goal mechanism behind /goal, rather than bulk codex exec. This newly rendered camera orbit starts24degrees left of the source direction, moves24degrees right, then returns continuously to the left. All object bounds remain visible. It shows the original authored scene before physics; geometry and poses are unchanged. Only static-scene camera motion is reversed. A completed goal and inspected render do not by themselves specify a simulation-ready collision model.

Discussion notes:

## 4. The yogurt bottle slides inside the tape roll

This fresh nativeGoal example passes the specified five-second gravity test: maximum object translation32.985mm, from the yogurt bottle; the other four move less than4.5mm. A rose box follows the yogurt bottle; the dashed light-gray outline marks its starting position. Tracking projects the original bottle body/cap profile through the recorded simulation poses and source camera. All rigid-body poses are actual PhysX steps. The test preserves the tape-roll hole with convex decomposition, leaving rendered geometry and authored poses unchanged. A default single convex hull fills that cavity and falsely ejects the bottle; do not present that collider artifact as reconstruction failure. This is an illustrative physics setup/check, not evidence that this reconstructed scene is unstable. Published bulk-Codex statistics on other scenes are a separate experiment. Video: one second authored pose, five seconds simulated motion, one second final hold.

Discussion notes:

## 5. Reconstructions can look like the input yet lack physical stability

Main paper Table4. All three use GPT-6 Astra; VIGA and Codex metrics are evaluated after5seconds of PhysX settling. A scene passes only when every object moves less than50mm in that test. This is an empirical benchmark criterion, not a guarantee of stability under arbitrary interaction. The paper Codex baseline used bulk codex exec; distinguish it from the new nativeGoal anecdote in the opening. VIGA uses medium reasoning, Codex ultra, and SceneRig has its own structured harness/preprocessing.

Discussion notes:

## 6. Stable scenes also need accurate placement

Exact teaser episode74fbca2f. Start at12seconds to omit operator initialization and play both at3×. The real episode transfers all3 new objects. SimFoundry transfers1/3; the white-blue bottle and ring remain outside the bin. The scene is stable but the recorded manipulation does not reproduce the real outcome. This qualitative comparison motivates accurate geometry and placement; it does not isolate placement from shape as a controlled causal intervention. Website replay-01 preserves the requested red-bin episode; the old Comparey filename has since been overwritten.

Discussion notes:

## 7. Ground object edits in geometry and physics

For routine corrections, the agent selects the object and pose aspect; the placement tool searches a bounded adjustment and simulates the selected candidate. The agent can also directly edit scene code when needed through execute_and_evaluate. Both edit paths return physical simulation feedback to ground subsequent decisions. Placement-tool updates use automatic acceptance checks; direct code edits are reviewed by the agent, which can retain or undo them. Returned visual, geometric and physical evidence guides which object to revisit. Simulate every edit refers to the resulting tool-selected or code-edited scene, not every intermediate search render.

Discussion notes:

## 8. Recover, construct, refine

Three high-level stages from the paper, illustrated with the actual abc_2 Claude Opus 5 run from benchmark_final. The same final-scene camera and intrinsics are used for all three rendered snapshots. Relational/metric initialization creates an object/support structure. Geometry, material, and lighting agents construct the environment. The pose-refinement agent then revisits object placement with geometry and physical feedback. The initial snapshot has nine object assets and no environment geometry or lights; neutral world illumination lights those assets; their original alpha is now composited over white for visibility on the black slide. Later panels retain their archived lighting. These are archived stage states, not baseline substitutions or a fresh reconstruction.

Discussion notes:

## 9. The agent builds a relational scene graph

Illustrative construction of the archived abc_2 scene graph from benchmark_final. Start with the input image, identify four mask-backed root surfaces (table and three walls), add all nine movable objects, then reveal support and confirmed surface relations. The table supports the tray, saucer, napkin and one spoon; the tray supports the plate, fork, knife and other spoon; the saucer supports the cup. Arrows run from supporting to dependent entities. The table is against the back wall; the side walls meet the back wall at corners. Those three surface relations are confirmed hard constraints in the saved graph. The unmasked floor and its rejected under relations, and advisory side-wall contacts, are omitted. Saved masks visually locate each predicted entity; this does not imply segmentation precedes the agent inventory. The24second progressive reveal explains saved predictions, not original streamed reasoning or execution timing. Evidence and hashes are frozen in evidence/scene_graph; root scripts/slides_scene_graph reproduces it.

Discussion notes:

## 10. Use points to select instance masks

The VLM inventories distinct entities with instance descriptions. SAM3 shares a category-level candidate pool. This animation shows three mugs, two markers, and two boxes from the archived Wendy1 run. MolmoPoint localizes each instance description; each saved point lies in exactly one mask in its category pool. All seven selected masks match their saved pool members exactly. The marker crop enlarges the original image, masks and coordinates together; it does not relocalize them. Follow-up recovery escalates from full-image category/synonym prompts to point-centered crops and then point-prompted masks; overlap and inventory checks repair missed or merged instances. Root surfaces skip cropped-mask recovery to avoid losing their extent. Presentation appearance update: the front white mug uses the existing logo-free website photograph/SAM3D visual variant. Original reconstruction states, camera paths and recorded solver trajectories are retained; this does not denote a new benchmark run. The two completion photographs preserve their saved content outside the lettering patch.

Discussion notes:

## 11. Complete the object before reconstructing it

Two actual Wendy1 completion cases. Monitor: the saved edit extends the image192pixels left to recover the border-clipped screen, also removing the foreground coffee machine. Keyboard: saved occlusion completion removes the coffee machine and foreground black mug to recover its hidden extent. This is occlusion completion, not a mask-only vital-part repair. Both cases show the original image, a wipe to the saved completion, a completed-image hold, the saved object mask, background removal, and an actual SAM3D asset orbit. Monitor occupies0–16seconds and keyboard16–32seconds, with each final4seconds showing its reconstructed3D mesh. The monitor retains the entire extended image; keyboard images and masks stay at their full-scene position and scale, with no zoom. The front white mug remains. Coffee-machine removal is confirmed by both the saved redetection metadata and the completed image; it is not an editorial removal. Wipes and mask sweeps illustrate archived predictions, not intermediate model computations. The monitor mask still touches the image boundary, so this does not claim perfect completion. No box completion is shown. Presentation appearance update: the front white mug uses the existing logo-free website photograph/SAM3D visual variant. Original reconstruction states, camera paths and recorded solver trajectories are retained; this does not denote a new benchmark run. The two completion photographs preserve their saved content outside the lettering patch.

Discussion notes:

## 12. Canonicalize the scene

A24second explanation of the actual Wendy1 MoGe-2 geometry and saved support-plane alignment. The opening stage states that monocular depth is estimated with MoGe-2 and shows the point cloud in camera coordinates; the model name disappears after that stage. Then highlight the archived tabletop mask in both the input image and its3Dpoints. Reveal the saved fitted plane and upward normal. The point cloud and plane undergo the exact archived19.236degree rotation and0.279869m vertical translation, putting the fitted tabletop at worldz=0 with its normal along+z. A short observer orbit shows the completed worldframe; the finalview holds. All331776valid sourcepoints retain their metric distances and original noise; their RGB values sample the existing logo-free photograph at the saved pixel coordinates. Tabletop points are not flattened: their residual to the saved plane has1.15mmRMSE. To keep the initial camera-coordinate scene readable, the display uses the fixed proper convention rotationM:(x,y,z)to(-x,-z,-y), with the camera triad transformed consistently; the animation then applies only the saved alignmentRandT, finishing at exactarchived worldcoordinates. The mask/plane reveal and coordinate interpolation are explanatory graphics of saved estimates, not recorded detector iterations or physical motion. This is predicted geometry, not groundtruth. Meshes subsequently align to masked metricpoints; robotexperiments use calibrated depth/intrinsics instead of monocular prediction. Exactsourcehashes, transform/plane checks and animationaudits are in evidence/grounding; the prior interactivecloud data remains archived. Presentation appearance update: the front white mug uses the existing logo-free website photograph/SAM3D visual variant. Original reconstruction states, camera paths and recorded solver trajectories are retained; this does not denote a new benchmark run. The two completion photographs preserve their saved content outside the lettering patch.

Discussion notes:

## 13. Settle objects, then test them together

Fresh PhysX demonstration from all twelve archived accepted Wendy1 preprocessing poses. Plays at1.5times the previous draft speed: all432original frames are preserved at36fps, giving12seconds including the final hold. Each object is released in support order, with earlier objects fixed, then the final joint pass frees all twelve bodies without drift caps. Every dynamic pose is a recorded solver step. The accepted starting poses include the original recovery and ICP results; this is not a replay of raw SAM3D first attempts or the original unrecorded full preprocessing history. The animation ends on the complete scene and holds, rather than looping back halfway through initialization. Wendy1 has a flat object-support structure; the following abc_2 example demonstrates dependencies between objects. Recovery is illustrated separately afterward. Presentation appearance update: the front white mug uses the existing logo-free website photograph/SAM3D visual variant. Original reconstruction states, camera paths and recorded solver trajectories are retained; this does not denote a new benchmark run. The two completion photographs preserve their saved content outside the lettering patch.

Discussion notes:

## 14. Settle supports before their contents

Actual abc_2 support graph from benchmark_final, with all nine movable objects. The table supports the tray, napkin, saucer and one spoon; the tray supports the plate, other spoon, fork and knife; the saucer supports the cup. The animation follows the archived dependency order, using fresh PhysX releases from accepted preprocessing poses, followed by a joint free-settling check of all nine bodies. Accepted poses include the archived recovery and ICP adjustments; this illustrates the accumulated-scene step rather than the original unrecorded preprocessing history. Labels identify each object and its support. Every dynamic pose is a captured solver state, without interpolation. The same presentation timing as the previous clip is retained:20frames per object,72joint-settling frames and48final frames, played at36fps (1.5times24fps), giving8.333seconds total. All nine objects stay within the camera frame, with a gentle viewpoint change.

Discussion notes:

## 15. Recover when an object topples

Fresh PhysX runs of the actual misc_IMG_8219 plush_toy#0 with its saved laptop support and50g VLM mass. The original physical model topples; the full guarded stabilization remains upright. This comparison changes the contact/physical bundle, not only CoM: narrow8mm collider-base flattening, saved lowerCoM, friction0.9, damping1.5, and matched inertia. The archived recovery ladder attempted the raw pose, then canonical upright, before stabilization; those old tilt summaries were68.31degrees,68.33degrees,0.039degrees. The new video shows actual solver trajectories, not interpolation of those archived outcomes. Rendering uses a front camera and broad neutral fill so both eyes and the original blue/lilac texture remain visible. The50g mass is the archived VLM estimate;103.4g in solver_mass_kg is the density reference used to scale saved inertia, not the chosen body mass.

Discussion notes:

## 16. A guarded center-of-mass adjustment

Restored original parameter visualization, based on the actual plush collider, computed uniform-density CoM, and saved physical override. This diagram is not a dynamics video. CoM height above the original collider minimum changes from68.9mm to42.5mm. Successful archived plush recovery uses a bundle:8mm collider-base flattening, CoM adjustment, friction0.9, angular damping1.5, and consistent inertia. The isolated plush CoM comparison did not explain its recovery; do not attribute that bundle result to CoM alone. A contact-footprint guard rejects sliver-like support and rollable objects skip stabilization. The override compensates for uncertain reconstructed geometry and mass distribution, rather than measuring the real object mass distribution.

Discussion notes:

## 17. Geometry stage: build and check rules

The rule descriptions are embedded in the movie. It first reveals the archived initial build (objects, desk, wall and lights), then transitions to the corrected saved scene after desk/wall clearance and support-coverage revisions. Only that final scene is shown while Structure, Pose and Contact animate in order, with active progress indicators and accumulating checkmarks. The final All rules passed visual is supported by original initializer memory message35: ALL RULES PASS (structure, pose, contact), corresponding to renders/32/state.blend. Rule animation and duration are editorial explanations of one tool response, not separate recorded calls. Structure covers object integrity, root-surface validity and support construction; pose covers image coverage, support footprints and applicable yaw constraints; contact covers penetration, resting and applicable relationships. Passing deterministic rules establishes a minimum geometric standard; visual matching and dynamic stability are separate. The initial build failed and is never labeled as passing. Construction reveals are newly rendered from saved geometry; camera motion over the corrected scene plays forward and backward, while check status advances only forward. The16.25second movie keeps construction and refinement at their previous speed, plays only the rule-check sequence at2times speed, and preserves the final3second all-passed ending. Presentation appearance update: the front white mug uses the existing logo-free website photograph/SAM3D visual variant. Original reconstruction states, camera paths and recorded solver trajectories are retained; this does not denote a new benchmark run. The two completion photographs preserve their saved content outside the lettering patch.

Discussion notes:

## 18. Material stage: refine surface appearance

The material agent changes root-surface color, roughness and patterns, while object meshes, object appearance and placement remain fixed. The visual verifier assesses appearance within that scope. Scope separation prevents a material edit from hiding a geometry error. The video shows1second of prior-stage camera motion, a1second matched-viewpoint wipe, then3seconds of the new-stage orbit. The transition illustrates saved stage differences, not the optimization history. This camera-and-wipe sequence plays forward and backward without duplicate endpoint frames; it contains no physical dynamics. Presentation appearance update: the front white mug uses the existing logo-free website photograph/SAM3D visual variant. Original reconstruction states, camera paths and recorded solver trajectories are retained; this does not denote a new benchmark run. The two completion photographs preserve their saved content outside the lettering patch.

Discussion notes:

## 19. Lighting stage: match scene illumination

The lighting agent adjusts illumination and exposure to match shadows, color and contrast. Its verifier focuses on lighting. The subsequent pose agent therefore works with stable background geometry and rendering conditions. The video transitions from the prior material-stage endpoint to the lighting-stage start at the exact same camera, then follows the new-stage orbit. This illustrates saved stage changes, not the light-edit history itself. The camera-and-wipe sequence plays forward and backward continuously; no physical dynamics are reversed. Presentation appearance update: the front white mug uses the existing logo-free website photograph/SAM3D visual variant. Original reconstruction states, camera paths and recorded solver trajectories are retained; this does not denote a new benchmark run. The two completion photographs preserve their saved content outside the lettering patch.

Discussion notes:

## 20. Inspect an object, then choose what to correct

Real consecutive composition calls from the preserved project-page trace, not invented reasoning. The spoon is initially misplaced. Red boxes identify the spoon in both input and render at the two inspect steps. All five consecutive calls are shown: inspect, XY, scale, reinspect, XY. The reinspection uses the after-scale crop, before the final XY correction. Tools search magnitudes and return post-settlement crops and measurements. Silhouette IoU is not monotonically increasing across different aspect scores: scaling improves size and its backend score even though raw IoU decreases. Do not show a fake monotonic improvement curve. Every object must be investigated at least once; nearby changes can trigger revisits. Use the step controls to walk the actual recorded crops.

Discussion notes:

## 21. Simulate code edits, then decide

A genuine direct-code edit and agent-initiated undo from Wendy1 CompositionAgent memory messages81–85. The agent uses execute_and_evaluate to rotate pliers#0 by−20degrees and translate by−30mm inX and−5mm inY. Physical settlement returns pliers leaning on the box and worse image alignment (IoU0.26to0.15). The agent explicitly calls undo_last_step next; this is not automatic placement-tool rejection or a claimed stability-test failure. The movie shows the exact prior state, exact code candidate before release, fresh chronological PhysX steps at5×slow motion, archived returned result, and hard-cut restoration. Fresh and archived endpoints agree within0.041mm and0.253degrees. A tracked rose box identifies the pliers. The end holds; no simulation is reversed. Sources, exact recorded rationale, source hashes and decoded checks are in evidence/code_edit_rejection/.

Discussion notes:

## 22. Search, simulate, and retain the edit

Wendy1 accepted box XY correction, explained with the same stage-caption and tracked-object style as the breakfast progression. Four phases show the prior scene, placement-tool position proposal, actual physical simulation, and retained result. The candidate is the exact archived transform before the2mm release lift; settle motion is subtle and not exaggerated. The tracked Box label and outline identify the edited mesh. The tool searches bounded candidates including no change, tests the selected candidate, and returns the result. Supports and dependent objects propagate together when appropriate. The lower centered camera contains every object throughout, and the final state holds. Chronological PhysX poses are newly captured from the archived edit; this is not a recording of the original run. Evidence is under evidence/placement_accepted_context/. Presentation appearance update: the front white mug uses the existing logo-free website photograph/SAM3D visual variant. Original reconstruction states, camera paths and recorded solver trajectories are retained; this does not denote a new benchmark run. The two completion photographs preserve their saved content outside the lettering patch.

Discussion notes:

## 23. From raw objects to a refined scene

Breakfast-table reconstruction from the archived Claude Opus5 benchmark_final run. The progression starts with all19 raw reconstructed assets, then illustrates support-ordered settlement of every object and joint simulation before initialization, material, lighting, the complete composition edit sequence, and export. Preprocessing settlement plays3times faster than the earlier draft, preserving chronological frames, endpoints and all19releases. This is a fresh PhysX demonstration from accepted preprocessing poses after earlier recovery/ICP, not a recreation of the unrecorded raw-to-accepted recovery history. The z=0support slab matches the solver; later constructed table/floor remain hidden in this insert. Composition covers every22object-edit call (21placement-tool calls and1direct-code edit) across10distinct targets, plus both explicit agent undos, in recorded order. All19objects were investigated; only these10received edits. Repeated adjustments and all3automatic rejections remain visible. Each candidate and retained/restored state is tied to the trace and exact archived snapshot. Routine state changes are hard cuts with continuous camera motion, not interpolated object dynamics. The bagel-placement and milk-carton-scale examples additionally retain actual fresh PhysX trajectories. Automatic rejection, temporary tool acceptance and subsequent agent undo are distinguished. The bagel rejection restores5_flip itself;6_move belongs to the later orange-scale edit and is not its restoration. After the final orange undo, the exported scene includes the archived all19-object joint-settling certification; this endpoint is shown by a saved-state cut labeled After joint settling, not synthesized dynamics. Rose boxes and readable labels identify the edited meshes, and the edit counter follows all22calls. All objects remain inside the lower centered camera. These are newly rendered archived states and selected fresh simulation captures, not original screen recordings; physics is never reversed. Final state holds. Full coverage, source hashes, state mapping and timing are recorded in evidence/breakfast_progress/ and references/SOURCES.md.

Discussion notes:

## 24. Replay the same robot commands

Two selected successful open-loop replays: top is preserved74fbca2f everything-in-bin, the same episode used for the motivation. Both real and SceneRig transfer3new objects; one already in the bin is excluded. SimFoundry only transfers1/3. Bottom is the preserved website replay-07 cup-into-bowl example: real and SceneRig succeed1/1, SimFoundry fails0/1. SimFoundry is not shown on this real/ours comparison slide. Each pair uses identical measured joint and gripper commands with no policy query or trajectory adaptation. The top pair starts at source12seconds after initialization; bottom starts at0. All play at3x. Each real/sim pair shares source time; the shorter row holds its final frame rather than restarting. Selected cases are illustrative; the next slide reports all25demonstrations. Sources and decoded checks: evidence/robotics_examples/.

Discussion notes:

## 25. Placement tools matter for successful replay

Aggregate25successful real demonstrations, five per task. SceneRig20/25, SimFoundry9/25. The code-only ablation without the bounded move tool succeeds11/25, using task counts1,1,2,2,5; the second task is cup-to-bowl, not the old fruits-to-plate name. Direct code poses retain physics settling but the agent chooses edit magnitudes. This ablation supports the value of the bounded placement tool rather than removing every component of the system.

Discussion notes:

## 26. Evaluate the same policy in real and sim

Preserved d40be789 policy-evaluation episode: mustard into the left bin. Both real and simulated executions pass. Unlike open-loop replay, the policy observes and acts independently in each environment; trajectories need not match. Both clips play from their own start at3times speed. The shorter simulated execution holds its endpoint for2.11seconds while the real clip finishes. A selected success case illustrates the protocol. The following slide adds two three-environment examples, then the results slide reports all100episode comparisons across two policies.

Discussion notes:

## 27. Compare real and simulated policy rollouts

Exact requested Compareyπ0.5 examples: bowl_cup_on_plate ep5,80ec87c4 (top), and everything_in_bin ep8,2ebb8901 (bottom). Each row compares independent executions of the same policy in real, SimFoundry and SceneRig environments. Both real and SceneRig episodes succeed; both SimFoundry episodes fail. Bowl/cup: SceneRig2/2 and SimFoundry1/2. Everything: SceneRig5/5 and SimFoundry5/6; these reconstruction-specific detector denominators are not matched object counts, so only binary outcomes are displayed. Real sources are matched using original MCAP task, policy mode, score and timestamps. Each stream starts at its own0 and plays3×, with genuine endpoint holds for shorter streams, not synchronized trajectories. The bowl/cup example is from the sixth task in the60-episode dashboard and is outside the paper’s five-task50-episodeπ0.5 aggregate. It is a qualitative example only; page27 metrics remain the verified two-policy, five-task evaluation. Exact IDs, source hashes, original endpoints and transformations are in evidence/policy_requested_examples/.

Discussion notes:

## 28. Predict real-robot policy outcomes more faithfully

Independently recomputed from paired pi05_episode_outcomes.csv and lingbot_episode_outcomes.csv and checked against frozen manuscript statistics. π0.5: SceneRig36/50agreement(72%), kappa0.3761, task-success-rate r0.7009; SimFoundry24/50(48%), kappa−0.0673, r−0.375. LingBot-VLA2.0: SceneRig33/50(66%), kappa0.2478, r0.9253; SimFoundry30/50(60%), kappa0.0909, r0.875. Pooled:69/100versus54/100, kappa0.3830versus0.0941, r0.9168versus0.8375. Agreement is the fraction of real/sim episode pairs with the same binary outcome, and kappa adjusts this for chance agreement. Pearson r compares task-level real/sim success rates: five task pairs per policy, ten concatenated pairs pooled. Neither episode agreement nor correlation is a simulated task success rate, and correlation alone does not establish calibrated rates. Policies use separate50episode sets and were finetuned on the same demonstration data. Shared checkpoint/calibration/controllers/physics within each matched comparison. Exact per-task counts, confusion matrices, source hashes and reproducible checks are in evidence/policy_metrics_detail/verified_metrics.json and verify.py.

Discussion notes:

## 29. A scene should preserve what the robot can do

The Wendy1 input photograph appears on the left beside its moving-camera reconstruction on the right. After3seconds, the input and both labels fade as the reconstruction shrinks into the top-left tile over2seconds and eleven additional benchmark_final Claude Opus5 reconstructions appear. All twelve camera orbits then play simultaneously in a three-row, four-column grid. Row2,column2 is misc_IMG_8226; row2,column3 is misc_still_life; row3,column1 is misc_online2. Full16:9views are preserved without crops. These are camera animations of archived static reconstructions, not physical simulations; each tile may reverse its camera orbit smoothly. Source scenes, input photograph and full projection/hash audits are in evidence/visual_revision/gallery.json. Close the story: SceneRig recovers individual objects and supports, grounds placement in geometry, and validates corrections with physical simulation. Demonstrated real-to-sim policy evaluation and open-loop trajectory replay open a path to scalable simulated data generation and evaluation; policy training and arbitrary articulated/deformable reconstruction are not demonstrated here. Presentation appearance update: the front white mug uses the existing logo-free website photograph/SAM3D visual variant. Original reconstruction states, camera paths and recorded solver trajectories are retained; this does not denote a new benchmark run. The two completion photographs preserve their saved content outside the lettering patch.

Discussion notes:

