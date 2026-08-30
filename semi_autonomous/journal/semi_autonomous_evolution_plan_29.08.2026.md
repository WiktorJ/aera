# Semi-Autonomous Evolution Plan — 29.08.2026

Goal: collect ~100 successful real pick&place trajectories to fine-tune `pi05_ar4_mk3`
(sim-DR-pretrained). Current pipeline (hand-eye calib + segmentation + depth + IK,
open-loop) yields ~10% success and risks arm damage near the table.

## Diagnosis

- Pipeline is open-loop: segmentation + depth + calibration + kinematic errors sum once,
  with no correction step.
- It relies on absolute accuracy (poor on a DIY stepper arm, ~3–10 mm) instead of
  repeatability (sub-mm).
- Fine-tune data only needs *successful* episodes with visual diversity — the collection
  method doesn't need to be general.

## Plan

### 1. Permanent markers, no more gripper removal

- Glue a ChArUco patch (or 2–3 markers at different orientations, ≥4 cm) on the gripper
  body / wrist top. Hand-eye calibration solves for the marker→link transform, so exact
  placement doesn't matter — only rigidity and visibility. Pass the gripper-body frame as
  the "hand" frame in the calibration config.
- Fix a ChArUco board to the workspace surface. Measure T_tablemarker→base once; after
  that, one detection re-establishes cam→base per session (camera bumps become a non-issue).
- Beware single-ArUco planar pose ambiguity: use ChArUco / multiple markers,
  `IPPE_SQUARE`, reject the solution inconsistent with FK.

### 2. Replace depth estimation with ray–plane intersection

- Segment object in RGB (existing), cast pixel ray, intersect with the table plane from
  the board; grasp z = plane + known object height / 2. Drops the depth camera from pose
  estimation entirely.

### 3. Close the loop with the wrist marker

- At pre-grasp: measure wrist marker, compare to FK prediction, command the residual
  (1–2 iterations). Object and gripper are measured by the same camera, so FK absolute
  error, extrinsic translation error, and intrinsics scale cancel as common mode.
- Measure markers only while the arm is stationary (motion blur / rolling shutter).

### 4. Expected accuracy

- Open-loop: ~6–10 mm (FK-dominated; systematic part absorbable into per-session offsets → 3–5 mm).
- Closed-loop: ~2–4 mm lateral, ~2–3 mm vertical → 3–5× margin on the tightest grasp axis
  for 3 cm objects.
- Cheapest upgrade: 1080p stream + good intrinsics (<0.5 px reprojection); at 640×480 all
  errors roughly double.

### 5. Table protection + sim height alignment

- 3–5 mm dense EVA mat (not thick soft foam): absorbs hard collisions, stays flat enough
  for the plane model (~1 mm), keeps the height shift negligible. Table marker goes ON the mat.
- Add workspace-surface-height DR (±2–3 cm, robot base fixed) to the next sim data batch
  so height is read from the image, not memorized. Current sim has objects only at
  arm-base level.

### 6. Marker contamination of training data

- Markers on robot body + table are static scene features, low risk of becoming a grasp
  cue. Keep them on at deployment → zero collection/deployment shift.
- Optionally add flat marker textures (wrist + table, randomly present/absent) to sim DR.

## Fallbacks / alternatives if accuracy still disappoints

- **Teach-and-replay grid**: paper grid on the mat, jog + save one grasp pose per cell,
  replay with small waypoint noise; near-100% success, no perception, exploits
  repeatability. Highest data-per-hour option.
- **Teleop jogging** (MoveIt Servo + spacemouse/gamepad): natural corrective demos,
  doubles as an intervention tool.
- Funnel-shaped / compliant fingertips: turn ~1 cm residual error into successful grasps.
