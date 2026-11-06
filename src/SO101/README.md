# SO101 Robot - URDF and MuJoCo Description

This repository contains the URDF and MuJoCo (MJCF) files for the SO101 robot.

## Overview

- The robot model files were generated using the [onshape-to-robot](https://github.com/Rhoban/onshape-to-robot) plugin from a CAD model designed in Onshape.
- The generated URDFs were modified to allow meshes with relative paths instead of `package://...`.
- Base collision meshes were removed due to problematic collision behavior during simulation and planning.

## Calibration Methods

The MuJoCo file `scene.xml` supports two differenly calibrated SO101 robot files:

- **New Calibration (Default)**: Each joint's virtual zero is set to the **middle** of its joint range. Use -> `so101_new_calib.xml`. 
- **Old Calibration**: Each joint's virtual zero is set to the configuration where the robot is **fully extended horizontally**. Use -> `so101_old_calib.xml`.

To switch between calibration methods, modify the included robot file in `scene.xml`.

## Leader Arm URDF

`so101_leader_new_calib.urdf` is a URDF for the **SO-101 leader** (teleoperation) arm, sharing the "new calibration" zero convention above. It reuses the follower's base/shoulder/upper-arm/lower-arm/wrist links and meshes; only the gripper subtree differs: the fixed follower gripper body is replaced by `Wrist_Roll_SO101.stl` and `Handle_SO101.stl`, and the moving jaw by `Trigger_SO101.stl` (meshes converted from `STEP/SO101/Leader_Specific/`). The actuated joint keeps the name `gripper` for compatibility with the other five arm joints.

```bash
rerun Simulation/SO101/so101_leader_new_calib.urdf
```

**Calibration caveat:** the five arm joints carry the same calibration as `so101_new_calib.urdf`. The trigger joint's zero, travel limits (provisionally 0–45°) and the printed-part inertia estimates were derived from CAD geometry and a rough average printed-part density, not from a measured leader unit — there is no MJCF/XML counterpart yet.

## Motor Parameters

Motor properties for the STS3215 motors used in the robot are adapted from the [Open Duck Mini project](https://github.com/apirrone/Open_Duck_Mini).

## Gripper Note

In LeRobot, the gripper is represented as a **linear joint**, where:

* `0` = fully closed
* `100` = fully open

This mapping is **not yet reflected** in the current URDF and MuJoCo files. 

### FRIDA fixed brush grip

Each `go_to_cartesian_pose()` call reads the current normalized gripper opening
and reuses that exact value for every command in the resulting trajectory.
IK and seed sampling change only the five arm joints.

No manual gripper angle is required. For collision geometry, FRIDA nominally
maps the observed 0–100 opening onto the URDF gripper limits. This is an estimate,
not a measured servo-to-model alignment.

The planner retries up to ten IK seeds per waypoint, requiring a collision-free
goal and a clear joint-space segment from the original start. Each numerical
attempt is limited to 300 iterations and 15 seconds. Collision checks remain
enabled on the time-parameterized trajectory. These retries do not search for
routes around obstacles when every direct connection is blocked.

Timing uses the quintic profile `s(u) = 10u³ - 15u⁴ + 6u⁵`, with `u = t/T`,
over a natural cubic joint path parameterized by joint-space chord length.
Acceleration is continuous, and velocity and acceleration are zero at the start
and end. Calibrated joint limits, collisions and the 10-degree brush tolerance
are checked before returning the trajectory. If the cubic path fails validation,
the same checked IK waypoints are connected by straight joint-space segments,
each with quintic timing and a smooth stop at its end. The fallback passes the
same validation; an invalid trajectory is never returned for execution.

Both choices use base velocity scale 0.15 and acceleration scale 0.225, with
`SO101MotionPlanner(speed_scale=1.5)` as the default. To compress motion time by
1.5, velocity scales by 1.5 and acceleration by 1.5²: the effective fractions
are 0.225 and 0.50625. Set `speed_scale=1.0` to recover the previous timing.
Effective fractions never exceed 1.0 (the model's full limits). Duration
is chosen from conservative continuous velocity/acceleration bounds and rounded
up to whole 0.02-second command periods. These smoother profiles can take longer
than the former TOPPRA timing. Joint limits and tilt tolerances are unchanged.
These scales are dimensionless fractions of the joint limits, in `(0, 1]`.
The current arm limits are 10 rad/s and 5 rad/s², giving scaled limits of
2.25 rad/s and 2.53125 rad/s² with the default speed multiplier. Actual motion
may be slower; these are joint limits,
not a requested Cartesian brush speed. The YAML `max_jerk` field is still not
enforced; quintic timing provides continuous acceleration, not a specified jerk
bound. The command rate remains 50 Hz. Period rounding means short moves are
approximately, rather than exactly, 1.5 times faster. The experimental runner's
40°/s guard is not a new global planner limit; larger paths still use the limits
above and need their own supervised verification before drawing.

Consecutive successful moves preserve the last commanded arm endpoint when
measured feedback remains within the existing 2° completion tolerance. This
avoids commanding a small step backward to lagging feedback at the next move's
start. Both the measured configuration and its connection to the commanded
start remain subject to joint-limit, collision and brush-tilt validation.
Failures, reconnects/disconnects, or feedback farther than 2° from the previous
command discard this continuity state. Each move still stops at its endpoint;
this does not blend through separate calls or remove planning/settling pauses.

`good_morning_robot()` applies and reads back zero for both `CW_Dead_Zone` and
`CCW_Dead_Zone` on the five arm joints after LeRobot configures the connection.
The gripper's settings are unchanged. These are session writes with EEPROM
locked, matching the experiments. No extra register writes occur while streaming.
Existing P/I/D and servo profile configuration are retained. In particular,
normal startup currently uses wrist-roll P=48 and LeRobot acceleration=254;
the latest supervised comparison used wrist-roll P=16 and acceleration=0.
Those settings were not established as improvements by that comparison.

From `src`, run `python -m pytest -q ../tests/test_motion_planner.py ../tests/test_so101_calibrated_limits.py -k quintic`
to check derivative consistency, fallback stops, and the recorded diagnostic
lift/straight-stroke poses entirely offline.

IK uses joint limits without an internal collision barrier. Numerical search
steps are not executed: each candidate goal and connecting path must pass
collision and brush-tilt validation before time parameterization, and the
resulting trajectory is checked again. These collision checks reject mesh
intersections; they do not enforce the former barrier's 2 mm clearance margin.

### Brush alignment

The planner aims the brush bristles downward along base-frame -Z and accepts
waypoints only when tip position error is at most 2 mm and shaft tilt is at most
10 degrees. Cartesian paths also use a 2 mm model-space tolerance. Two position
tasks track the tip and shaft midpoint; rotation about
the shaft is free. Their separation and local axis come from the URDF's fixed
`brush_tip` and `brush_exposed_visual` frames (nominal separation: 38.5 mm).

A tilted measured start is allowed. During the initial approach it may be outside
the upright cone; once it enters the cone, subsequent motion must stay within
10 degrees. The direct connections and final timed trajectory are checked at
joint increments no larger than 0.01 rad, alongside the existing collision
checks. Failed direct connections trigger another IK seed. The check rejects
tilt excursions after time parameterization before any commands are sent.

The upright requirement reduces the usable workspace. In offline replay, the
original neutral target at robot XYZ `(0.08627, 0.19837, 0.015)` did not yield an
accepted upright solution; the same XY at Z=0.08 m did. This historical target
differs from the current `to_neutral()` target. Failure to find a solution is
not proof of global infeasibility.

`Painter.to_neutral()` uses a vertical lift, horizontal traverse at
`Z_CANVAS + 0.10` m or higher, and vertical descent. The complete transfer must
pass planning before movement; it never retries via a lower route. Before
`Z_CANVAS` is available during fresh calibration, the original direct neutral
move remains. With `--use_cache`, height is available before the first return.

`Painter.approach_stroke()` uses the same high transfer when returning from a
station to the stroke's existing 3 cm hover. Consecutive on-canvas approaches
retain the direct-first route and, if needed, a canvas-center fallback at
`Z_CANVAS + 0.04` m. The stroke start includes camera-coordinate correction;
the center comes from the canvas settings. Descent, painting, and lift-off
retain their paths. All routes retain collision, joint-limit, and brush-tilt
checks. An execution failure never triggers a second route.

From `src`, run `python -m pytest -q ../tests/test_so101_calibrated_limits.py -k station_sequence`
for offline neutral → rag → water → palette → neutral regressions from five
starting poses, including tilted starts. These use station settings captured on
2026-10-20 and validate full timed trajectories, without a robot connection.
`dip_brush_in_water()` now requests one circle of radius 5 mm, centered on
`WATER_POSITION` at the existing dip depth, using 12 segments. It approaches
from the hover position, dips at the center, stirs, then returns to the center
and lifts out. The same planner checks and quintic timing/fallback apply.
Use `-k circular_dipping` in the same command to test the actual method's
approach and stirring requests offline from three starts at two hover heights.
Use `-k dipping` to also run the historical random-pattern regressions,
including the two cubic-overshoot cases and a pattern rejected during IK.

### Nine-point canvas contact-height calibration

After the normal center brush-height calibration exists, preview the 3 x 3
measurement grid without importing hardware drivers or moving anything:

```bash
cd /Users/luke/Projects/Frida/src
python ../scripts/calibrate_canvas_z.py \
  --materials_json ../my_materials.json \
  --cache_dir caches/so101_mvp
```

Add `--execute` when ready to move the SO101 and collect the nine contact heights.
This command only calibrates height; it does not start painting, connect the
camera, collect a stroke library, or train a model. Use a clean, flat sheet and
the same firmly mounted brush used for painting. Close other programs using the
motor port first. The brush should initially be in clear air.

At each point it starts 40 mm above the existing center reference. Lower the
brush manually until its bristles just touch, without compression:

- `w` / `s`: up / down 0.5 mm.
- `W` / `S`: up / down 2 mm.
- Enter: record this point and lift before travelling to the next.
- Esc or Ctrl-C: abort without replacing the previous calibration.

The steps are commanded distances, not guaranteed physical resolution: the
existing 2 mm model-space and 2 degree execution tolerances still apply. Contact
heights are bounded to +/-30 mm of the old reference (`--max-adjustment-mm`).
`--clearance-mm` must exceed that bound. Every request uses normal IK, collision,
joint-limit, tilt, and trajectory checks. A rejected manual planning step leaves
the requested height unchanged; execution failures abort. On exit the port is
closed with holding torque retained; no recovery movement is initiated on error.

The default grid is **20 mm inside each edge** (`--inset-mm 20`), with the middle
point at the canvas center. Nominal offline routes on the current canvas passed
at this inset; a 1 mm inset hit existing near-center collision and far-corner
tilt restrictions. This is not a guarantee for a live start or measured heights.
Rows in the saved matrix run from low to high FRIDA Y; columns run from low to
high FRIDA X. Measurements use commanded robot XY, so camera XY correction is
not applied while collecting them.

Only after all nine points and their lifts succeed does the command back up and
atomically update `brush_tip_to_table_calib.json`, adding `canvas_z_grid`.
`Z_CANVAS`, `Z_MAX_CANVAS`, and the existing pressure range remain unchanged.
With `--use_cache`, `Painter` validates and loads the grid before connecting.
It bilinearly interpolates contact height at the final camera-adjusted/clamped
stroke XY, then offsets every stroke sample, approach/descent, lift, and
canvas-center fallback before motion planning. The outer strips beyond the
measured grid hold the nearest edge correction instead of extrapolating.
Paint, water, rag, neutral targets, and raw manual moves remain uncompensated.
Legacy height files without a grid keep their previous behavior.

Changing canvas dimensions/position requires recollecting the grid; repeat the
calibration after changing the brush mount or robot geometry/offsets as well.
Saved stroke plans fingerprint the entire height file, so old plans will reject
the changed calibration. Prepare a new plan, and recollect/retrain the stroke
library to model the corrected contact depths. Running without `--use_cache`
performs the original center setup and replaces the height cache, discarding
the grid; run the grid command again afterwards.

Offline checks (no serial or camera connection):

```bash
python -m pytest -q ../tests/test_canvas_z.py ../tests/test_canvas_z_planning.py
```

### Saving and resuming painting plans

`paint.py` automatically saves `cache_dir/stroke_plan.json` after the initial
optimization, before the ready window opens. `--save_plan PATH` selects a
different file. The JSON contains the ordered stroke parameters, the exact RGB
palette in slot order, each stroke's palette index, setup/model fingerprints,
and execution progress. Palette slots are printed in the terminal and shown
under TensorBoard Images → `paint_colors/palette`.

The checkpoint is replaced atomically before attempting a stroke and after each
completed stroke. Adaptive replanning saves the updated unpainted strokes while
retaining the completed prefix and original palette order. A failed stroke is
kept pending. Resume with the same canvas and paints using, from `src`:

```bash
python paint.py --robot so101 --use_cache \
  --materials_json ../my_materials.json \
  --cache_dir caches/so101_mvp \
  --load_plan caches/so101_mvp/stroke_plan.json
```

`--dont_plan` is shorthand for loading `cache_dir/stroke_plan.json`. Loading skips
both initial optimization and subsequent adaptive optimization, and automatically
reuses the cached stroke model without retraining it. No target image is needed.
The saved rendering height, stroke order, colors, and palette indices are used;
`--num_strokes`, `--n_colors`, and `--use_colors_from` do not regenerate them.
Progress continues in the loaded file unless `--save_plan` names another file.
A fully completed plan exits without connecting to hardware.

For an SO101 run, `--skip_failed_strokes` logs and checkpoints a stroke when its
Cartesian route cannot be planned. It first attempts to return to neutral; only
after that succeeds does it advance past the stroke. The checkpoint records its
number and planning error. Since a failure can happen after earlier parts of a
stroke ran, inspect the canvas before resuming. Robot command/execution failures
and failures returning to neutral still stop the run, leaving the stroke pending.

Loading checks canvas/stroke settings, palette layout, and the saved calibration
and model files before robot startup. Keep the physical canvas placement and
paint-slot arrangement unchanged. If the previous run stopped during a stroke,
the terminal identifies the stroke that will be retried. It may already be
partially or fully painted if execution stopped before completion was recorded;
inspect the canvas before closing the existing ready window to resume.

`--simulate` now saves/previews the plan and exits without advancing its execution
progress, so a plan prepared offline can subsequently be painted. Physical replay
still requires matching cached physical calibration. This feature cannot recover
plans from runs started before checkpointing was added.

Physical `paint.py` runs now reload paint after **four completed strokes** by
default. Override with `--how_often_to_get_paint N`. Cleaning happens initially,
on a color change, and after **12 completed strokes** since the previous cleaning.
Each cleaning uses **one water/rag cycle** and always reloads immediately
after cleaning, then restarts the four-stroke reload counter. Failed/skipped
strokes do not advance either counter. Each resumed session starts with a fresh
clean and reload. Stroke order, geometry, optimization, and minimum stroke size
are unaffected.

Stroke-library collection uses the same configurable reload interval (default
4), washes every 12 strokes, and saves the care settings in its library metadata.
This baseline comes from the [authors' released collection code](https://github.com/cmubig/Frida/blob/master/src/painter.py#L524-L533).
The [paper's robot setup section](https://ar5iv.labs.arxiv.org/html/2210.00664)
describes cleaning with water and a rag when changing colors, but does not state
a numerical interval. These counts are a practical baseline, not a guarantee of
consistent paint loading or brush force. Explicit CLI intervals still override
the default, including older commands that specify 5.

With calibrated canvas height, SO101 station transfers follow three legs:
**lift vertically at the current XY, travel horizontally, then lower vertically
at the destination XY**. The common travel plane is `Z_CANVAS + 0.10 m`. A higher
starting pose or destination raises that plane; the brush never lowers before
crossing. This applies to entering water, rag, and paint stations, returning to
the canvas after a station visit, and returning to neutral. Consecutive strokes
on the canvas retain their existing short approaches. Contact stroke paths,
water circles, rag wiping, paint dabs, and their depths are unchanged.

All three legs are checked by the Cartesian planner before sending any movement
commands. A failed transfer stops without retrying a lower diagonal route.
Existing joint-limit, collision, brush-tilt, and 2 mm model path-tolerance checks
remain active. The water container is not a collision object, so the 10 cm
travel plane still needs physical clearance over the actual rim. Fresh setup's
initial neutral move, before any canvas-height calibration exists, retains its
previous behavior; cached painting runs have the height available at startup.

Offline transfer validation: `python -m pytest -q ../tests/test_station_transfers.py`.

The Innomaker overhead view records automatically during physical `paint.py`
runs, starting after `Painter` initialization and ending on completion, an error,
or Ctrl-C. Initialization/calibration before that point is not recorded. The
terminal prints a unique session directory under `../outputs/recordings/` (when
running from `src`). It contains:

- `overhead.mp4`: the full overhead view at the camera's calibrated native
  resolution (currently 1920 × 1080), sampled at 10 FPS by default. This is the
  raw camera view, not the cropped/warped canvas.
- `events.jsonl`: stroke start/end/failure/skip, paint-reload and cleaning
  start/end events, with one-based plan stroke numbers and elapsed timestamps.
- `frames.jsonl`: each video's frame index and its real elapsed timestamp on the
  same clock as the events. Use this mapping for timing analysis: if capture or
  encoding cannot keep up, playback can be shorter than the actual session.
- `recording.json`: UTC start time, duration, resolution, FPS, end reason, written
  and dropped frame counts, and any recording errors.

Use `--recording_dir PATH` to select the parent folder, `--recording_fps N` to
adjust sampling, or `--no_recording` to disable it. Simulation and stroke-library
collection do not record. Recording uses the existing camera connection, a
single capture thread, and a bounded background video writer; it does not open a
second camera or add robot movements. Still captures continue to use fresh frames
from the same stream. A video startup failure stops before painting; later video
write failures are reported without changing stroke checkpoints. A forced process
kill or power loss can still leave an unfinished MP4.

Offline validation: `python -m pytest -q ../tests/test_stroke_plan.py ../tests/test_paint_reload.py ../tests/test_camera_recording.py`.
Tests use fake camera/robot entry points and temporary caches; they do not connect
to hardware or modify an active run's files.

### Paint-loading experiment before brush training

`paint_loading.py` measures how a single paint load changes across ten strokes.
It defaults to three repetitions on **three fresh sheets**, twenty-mm straight
strokes, normalized middle compression 0.25, and two wash/wipe cycles before
each load (this older experiment's protocol, not the current painting default).
There are **no reloads or washes within the
ten-stroke sequence**. One load calls the existing three-dab `get_paint(0)` once.

```bash
cd /Users/luke/Projects/Frida/src
python ../scripts/paint_loading.py             # layouts/settings only; no hardware
python ../scripts/paint_loading.py --execute   # prompts before connecting and each sheet
```

Defaults are `../my_materials.json` and `caches/so101_round`; override with
`--materials_json` and `--cache_dir`. This requires the saved brush contact/
compression heights, nine-point height grid, and camera canvas homography.
It does not need a trained model or stroke library. If `cached_H_coord.pkl`
exists, it uses the same start-position correction as production strokes;
otherwise it explicitly reports uncorrected robot command XY. Completing XY
calibration first is useful for consistency with later collection. To do that
without repeating the saved height/camera calibration, run:

```bash
MPLBACKEND=Agg python paint.py --robot so101 --use_cache \
  --materials_json ../my_materials.json --cache_dir caches/so101_round
```

Stop with Ctrl-C at **"Need to create stroke library"**. The XY calibration is
then saved; do not start stroke collection yet. Keep the camera/canvas/brush
mount unchanged between calibration and the experiment.

The loading test calls the existing `BrushStroke.execute` and `Painter` station
methods without invoking `Painter` initialization. It keeps the height-grid
correction, ten-cm station transfers, planner checks, and production pressure
profile: endpoints use 0.2 and the middle uses `--stroke-z`. With a twelve-mm
calibrated compression range, the default commands 2.4 mm at the endpoints and
3 mm in the middle; these are position offsets, **not measured force**. It does
not change the optimizer, material settings, cache, saved plans, or normal
painting/library reload intervals. Matplotlib is forced to `Agg`, with no plot
windows or corner-clicking prompts.

Options: `--length-mm 20`, `--stroke-z 0.25`, `--wash-passes 2` (or 1),
`--repeats 3`, `--recording-fps 10`, `--notes "brush / paint / paper details"`.
The existing random rag/paint offsets repeat with `--seed 101` for each load.
Use consistent paint dilution, water, rag wetness and paper. The stroke order
shifts by three slots between sheets to reduce position/depletion confounding;
use the corresponding numbered layout to identify the marks. Numbers are only
in the preview, not painted. The layout is centered in the measured command-space
height grid, with 20 mm row spacing and a 20 mm gap between columns. When XY
correction exists, its inverse determines the target starts; normal stroke
execution maps them back into the grid. Full-length strokes retain a six-mm
envelope inside both the command-space height grid and target-space canvas,
and must remain separated in both frames. Calibration is never disabled or
clipped to fit. The preview shows target positions, which can look slightly
skewed after inverse mapping. A wider
physical footprint can still overlap; inspect the layout/marks accordingly.

Every invocation creates a unique directory in `outputs/paint-loading/` with:

- `load-01-layout.svg` (etc.): numbered layout for each sheet; preview is geometry
  only and does not run IK.
- `test.json`: settings, calibration, source hashes, positions, completed count,
  and any partial stroke/error. It is an experiment log, not a resumable plan.
- `load-01-before/after-canvas.png` and corresponding `-overhead.png` files:
  fresh frames before loading and after returning to neutral. Photos are wet;
  earlier strokes have had more drying time. Compare under consistent lighting
  and again after drying before attributing light patches to missing paint.
- `recording/`: full Innomaker video, stroke/load/wash events, frame timestamps,
  and recording metadata. No extra neutral moves occur between test strokes.

A motion failure or Ctrl-C stops the test without retrying a partial stroke,
reloading, or commanding a recovery movement. Video is finalized and the port
is closed with holding torque retained. Do not resume a partial depletion
sequence; start a new test with fresh paper. Keep paper edges aligned when
replacing sheets so the saved camera/height calibration remains applicable.

Offline tests: `python -m pytest -q ../tests/test_paint_loading.py
../tests/test_paint_loading_planning.py`. The latter exercises real IK with a fake
executor; neither test connects to hardware.

### Washing-interval experiment

After establishing a reload interval with the loading test, use
`paint_washing.py` to check whether brush condition deteriorates despite fresh
paint. Defaults are **30 strokes with reloads every 5 and no intervening wash**,
followed by a wash and **5 recovery strokes**. It washes at the start and before
recovery only; each wash uses two water/rag cycles by this experiment's default. Each load calls
the existing three-dab `get_paint(0)` exactly once, with the same seeded dab
offsets regardless of whether washing preceded it.

```bash
cd /Users/luke/Projects/Frida/src
python ../scripts/paint_washing.py                 # offline layouts/settings preview
python ../scripts/paint_washing.py --execute       # physical run, with recording
```

The default uses **four fresh sheets**: strokes 1–10, 11–20, 21–30, then recovery
31–35. Paper changes do not reset the wash counter, wash the brush, or split a
load. Do not manually wash/wipe/reload the brush between batches. Keep paper
changes brief and consistent: the log records their duration, and stroke events
include time since the last wash/load, because drying during a pause can also
affect the footprint. The arm parks using the existing high transfer before
each paper prompt; no extra neutral movements occur between strokes in a load.

Use `--reload-every 3` if the loading test establishes three strokes per load,
or change `--batches 6` to set how many loads occur before the recovery wash.
Allowed reload intervals are 2–10, batches 2–20. Whole loads are packed into
at most ten marks per sheet; recovery always uses its own sheet. Defaults retain
the loading test's 20 mm stroke length, middle compression 0.25, endpoint
compression 0.2, round-brush cache, optional saved XY correction, nine-point
height grid, normal station travel, and motion checks. It imports the shared
loading-test setup/recording code and never collects/trains a model or changes
the optimizer or normal cleaning/reload counters.

Every invocation creates `outputs/paint-washing/<session>/` containing numbered
`sheet-01-layout.svg` files, before/after overhead and rectified canvas PNGs,
`test.json`, and the full video/event timeline in `recording/`. Matplotlib uses
`Agg`; no calibration windows open. As in the loading test, preview does not
run IK, and execution requires `--execute` and the setup/paper prompts. Failures
stop without retries or a recovery wash, preserve partial counts, close the
recording, and disconnect with holding torque retained. A partial test cannot
resume without invalidating its wash history.

For the default protocol, compare **2, 12, 22, 32**: each is the second stroke
after loading, at the same canvas slot; 32 follows the recovery wash. Also
compare **7, 17, 27** at their shared slot. `test.json` lists these groups for
other settings. Deterioration that remains after fresh paint but improves after
washing supports a shorter wash interval. If it does not improve, investigate
compression/placement, paint changes, and pause/drying effects. The script does
not automatically infer an interval from wet photos; compare again after drying.

Offline tests: `python -m pytest -q ../tests/test_paint_washing.py
../tests/test_paint_washing_planning.py` (fake hardware, real offline IK in the latter).

### Round-brush repeatability before collecting a library

Use `paint_repeatability.py` when pressure/placement variation makes depletion
tests hard to interpret. It reuses the cached nine-point height grid and optional
XY correction, with no model training, optimizer changes, or calibration writes.

```bash
cd /Users/luke/Projects/Frida/src
python ../scripts/paint_repeatability.py             # numbered layouts/settings, no hardware
python ../scripts/paint_repeatability.py --execute   # physical test with Innomaker recording
```

The default is **three fresh sheets, 12 marks per sheet (36 total)**: left,
center, and right regions, each containing horizontal, vertical, +45°, and -45°
strokes. All are **16 mm straight black strokes**. This length fits four separated
rows in the current measured height grid; it does not change the minimum stroke
length or trained model. The outside columns are near opposite edges of the
measured grid, with a full 6 mm brush envelope inside it and the paper. Layouts
that do not fit are rejected, never shortened or clamped to fit.

For 32 mm strokes, all twelve marks fit on one sheet in three regions:
horizontal across the top, vertical beside two stacked diagonals. Full 6 mm
measurement envelopes are checked in command and target space. For the 67 mm brush:

```bash
python /Users/luke/Projects/Frida/scripts/paint_repeatability.py \
  --cache_dir /Users/luke/Projects/Frida/src/caches/so101_67mm \
  --length-mm 32 --repeats 1
```

This previews one sheet, twelve marks total; add `--execute` to paint and record.
Increase `--repeats` to repeat the same positions on fresh sheets.
SO101 strokes descend from the 30 mm approach hover directly to first contact
using the normal straight-line planner, with no 5 mm waypoint or stop. The
descent ends at rest at first contact, then the contact stroke uses its own
continuous timing curve.
The nine-point Z correction and FRIDA compression profile are retained. Lift
and station travel retain their stops. The complete route is checked before
execution with the existing joint, collision, tilt, and 2 mm path bounds;
continuous-path failures abort rather than falling back to tiny contact stops.
Velocity/acceleration limits and `speed_scale=6.0` are unchanged.
Each sheet starts with a wash/wipe and reload; reload again after four strokes.
Normalized compression stays at 0.25 in the middle and 0.2 at the ends; the
millimeter values depend on the selected cache and are printed before execution.
With multiple sheets, measurements group identical slots across repeats. Directions
occupy different locations, so comparisons do not fully isolate direction
from position or paint conditions. Recording and before/after snapshots remain
enabled. Optional `--split-directions` retains two sheets per repeat, with six
marks and a wash before each sheet.

Analysis version 2 assigns whole newly darkened paint components to the nearest
finite target path (12 mm search allowance), then expands the inspection crop to
include the assigned paint. Each component belongs to at most one mark; fragments
of a broken stroke can share a slot. Ambiguous assignments and canvas-edge
clipping are flagged and excluded from repeat summaries. Target start coordinates
and the 6 mm path-coverage corridor remain unchanged, so a displaced but complete
mark can have low path coverage. These are image measurements, not brush force
or robot tracking measurements. `measurements.json` records the analysis version.

Each fresh sheet repeats exactly the same target positions and directions. The
painting order shifts between sheets so a given mark has different ages within
its paint load. Wash/wipe once before each sheet (12-stroke interval), reload
black in palette slot 0 every four strokes, and keep the paint consistency and
paper placement the same. Rag and paint-dab offsets use a repeatable seed. The
robot parks and prompts before each sheet. No automatic retry follows a failed
stroke; partial results are preserved and holding torque is retained at exit.

Before starting, position the arm less folded with the brush upright in clear
air over the canvas center, roughly 5–10 cm above the paper. Support the arm and
switch off motor power before repositioning; restore power before connecting.
Do not force powered joints. A collision-free stationary pose does not guarantee
that a straight upward lift is possible: a folded start can bring the lower arm
into the shoulder while lifting.

All three paint experiments now read the actual starting joints and preflight
the initial 10 cm lift/traverse/descent before motor configuration or motion.
The check opens only the bus for position reads and closes it without changing
torque. `test.json` saves the starting joint values, estimated tip position and
tilt, and any planning failure. Failure stops with repositioning guidance; it
never substitutes a lower route or relaxes collision/tilt limits. The normal
executor replans from fresh feedback before movement. Offline layout previews
still do not connect to hardware or validate the arm's actual starting pose.

The default normalized middle compression remains **0.25**, with FRIDA's **0.2
at the endpoints**: currently 3.0 mm and 2.4 mm commanded compression. These are
positions, not measured forces. `--stroke-z` and `--length-mm` change only the test;
`--repeats` accepts 1–6 sheets. Use the first sheet's result to decide whether to
continue or cancel at the next paper prompt.

Each `outputs/paint-repeatability/<session>/` contains:

- `test.json`: setup, calibration/source hashes, care settings, target and
  corrected command positions, execution order, and partial/completed counts.
- `sheet-*-layout.svg`: numbered target paths and starting circles.
- Before/after overhead and rectified canvas PNGs, plus full video and timed
  stroke/wash/reload events in `recording/`.
- `comparison.png`: the same position across sheets, with after-image crops and
  the dark-paint masks used for measurement.
- `measurements.json`: visible starting-edge offsets/spread, length, width,
  painted area, and path coverage for each mark, plus ranges across repeats.
- `summary.txt`: a readable table of those repeatability comparisons.

Analysis uses newly darkened pixels within 6 mm of each expected path envelope.
Length and width are the 2nd–98th percentile extents along/across the path;
coverage is the percentage of expected approximately 1 mm path bins with paint.
Starting-edge position includes the brush footprint: compare its **spread**
between repeats, not its absolute offset as though it were the brush center.
Empty marks stay empty, disconnected pieces are retained, and marks touching a
crop boundary are flagged. Inspect the masks: wet-paint glare, thresholds, camera
calibration, and paper movement can bias these measurements. This is not force
measurement. Different directions occupy different rows, so directional and
local height effects are not fully isolated by this first screening test.

Measurements run after the hardware session closes. To analyze saved completed
sheets again (also works after canceling between sheets):

```bash
python ../scripts/paint_repeatability.py --analyze /absolute/path/to/outputs/paint-repeatability/session
```

Compare the same direction left/center/right first, then identical positions
across repeats. Consistent lengths but changing widths/coverage suggest contact
or paint-transfer variation; systematic location shifts point toward calibration
or tracking. These patterns guide the next test rather than proving a cause.
If one direction remains consistently different, try that direction at another
row before attributing the difference to the brush.

Offline checks: `python -m pytest -q ../tests/test_paint_repeatability.py
../tests/test_paint_repeatability_planning.py ../tests/test_paint_experiment_recording.py`.
These include real IK/collision/tilt checks with fake robot execution, synthetic
image measurements, and real recording/PNG/video output from a synthetic camera.

### HELLO WORLD drawing

From `src`, `python ../scripts/paint_hello_world.py` saves `outputs/hello_world.svg` without
connecting to hardware. The preview shows layout only; it does not run IK.
To move the robot with a loaded brush or pen:

```bash
python ../scripts/paint_hello_world.py --execute --air
python ../scripts/paint_hello_world.py --execute --canvas
```

Air mode draws 20 mm above the saved `Z_CANVAS`; canvas mode uses `Z_CANVAS`
as the touch height, with no extra brush compression. `--air-height-mm` and
`--letter-height-mm` change air clearance and letter size (both default to 20).
The two lines occupy 84 × 50 mm, centered horizontally, starting 35 mm inside
the front canvas edge. The script reads `my_materials.json` and the saved
`caches/sharpie_short_strokes/brush_tip_to_table_calib.json`. It uses no camera,
does not fetch paint, and finishes lifted 10 mm above the drawing plane.
Lifts, transfers, descents and strokes all go through the existing SO101
planner, retaining collision checks, calibrated bounds and the 10° brush cone.

For an existing `Painter`, use `from paint_hello_world import paint_hello_world`
and `paint_hello_world(painter, air_test=True)` (or `False` for contact).
Run `python -m pytest -q ../tests/test_paint_hello_world.py
../tests/test_so101_calibrated_limits.py -k 'hello_world or pen_up or oversized or cli_defaults'`
for the offline drawing tests. These replay both modes from a recorded start
without connecting to the robot, checking the full timed paths and lettering.

### Lift and straight-stroke diagnostic

From `src`, `python ../scripts/diagnose_so101_motion.py` plans a 10 mm lift followed by a
20 mm stroke along FRIDA +X, using the saved observation in `SO101/so101_offsets.json`.
The default run is offline: it does not open a motor connection or invent
tracking measurements. `--start-joints FILE` selects another saved observation.

When ready to record hardware, place the tip in clear air and run:

```bash
python ../scripts/diagnose_so101_motion.py --execute
```

This performs the two moves relative to the current tip position, using the
normal SO101 connection, calibration, planner checks and executor. It does not
approach the canvas, descend, or return to the starting pose. `--lift-mm`, signed
`--stroke-mm`, and `--axis x|y` select the displacements. No planner settings,
PID settings beyond normal SO101 startup, or trajectory scheduling are changed.

Files go into a new directory under `outputs/so101_diagnostic/`:

- `planned.csv`: commanded positions and planner velocity/acceleration.
- `commands.csv`: scheduled times, actual send starts, intervals, lateness,
  write duration and joint targets; hardware runs only.
- `feedback.csv`: motor readings, read start/end times and durations, and
  signed error = last sent target minus measured servo degrees.
- `report.json` and `diagnostic.png`: summary, phase timing and plots. Completed
  samples are saved if a move fails. Offline plots clearly mark missing measurements.

Feedback is read after every command by default. That adds bus traffic and can
itself cause delays. `--read-every 5` reduces sampling; `--read-every 0` is a
timing-only control run (the executor's normal settling reads still happen).
Return to the same starting pose before comparing runs: each invocation moves
relative to its own measured start. CSV rows are buffered until motion ends.

Actual intervals should follow the planned intervals (normally 20 ms, with a
possibly shorter final interval). Long gaps followed by short intervals suggest
late commands and catch-up bursts. Compare feedback-read duration with the
command period to identify measurement overhead. Lateness is anchored to each
move's first send; it does not measure delay before that first command.
Abrupt changes in planned acceleration support a high-jerk explanation, but
these plots do not directly measure physical jerk. Tracking error compares the
last position command to a later bus reading; motor samples are not simultaneous.
Use the recorded read interval when interpreting lag. For a video, record the
whole arm and tip with a fixed camera, preferably at 60 fps, and retain the
original file and the corresponding log directory.

Offline tests: `python -m pytest -q ../tests/test_so101_motion_diagnostic.py
../tests/test_so101_execution.py ../tests/test_so101_calibrated_limits.py -k 'diagnostic or execution'`.

### Shoulder-pan dead-zone comparison

`python ../scripts/compare_so101_p_gains.py` only prints the experiment settings; it does
not connect to the arm. When ready to move, from `src`:

```bash
python ../scripts/compare_so101_p_gains.py --execute
```

The filename is retained, but this now runs three blocks with both pan dead
zones set to **1 -> 0 -> 1**, at fixed **P=48**. It no longer sweeps P gains.
Each block repeats the same staircase three times: center -> -8 ticks -> +8
ticks -> center. Every staircase command changes the goal by exactly one tick
(about 0.088 degrees), held for 0.4 seconds, with 50 Hz feedback. Targets stay
anchored to the initial raw position, never to the latest measured position.
`--extent-ticks`, `--hold-s`, `--repeats`, and `--sample-hz` adjust the experiment.
Before each block, it approaches center from the positive side using two setup
commands. `--return-s` controls those setup holds and the final center hold.
Setup is logged separately. Initial residual error is recorded, not assumed zero.

Place the brush in clear air: this directly commands joint steps, without the
planner's IK, collision, or brush-tilt checks. With defaults, allow about +/-1
degree of clear pan travel. Other joints hold their measured positions; I/D,
startup force, offsets, and planner settings are unchanged. Only one program
may use the motor port. The script temporarily disables pan torque to change
P/dead zones, and attempts to restore and verify the original P, both dead zones,
and all torque/lock states on completion, exception, or Ctrl-C. It restores the
original settings even if the original dead zones were not 1. Joints that started
with torque off will relax on exit. If pan settings or its hold target cannot be
verified during cleanup, it attempts to leave pan torque disabled.
Restoration failures are reported explicitly; loss of power or a forced process
kill can prevent restoration. No old position targets are replayed on exit.

Timestamped folders in `outputs/so101_p_gains/` contain `samples.csv` and
`report.json` (schema version 2). The CSV includes block/repeat/phase, previous
goal, command increment, measured starting position and residual error, raw
encoder position, tracking error, current, signed drive output, and voltage.
Each feedback channel has separate read timestamps: these measurements are
sequential, not simultaneous. `Present_Load` is sign-decoded by LeRobot and
reported as raw counts and duty-cycle percent (0.1%/count), not joint torque.
Current is recorded raw and in nominal mA (6.5 mA/count), voltage raw and in V
(0.1 V/count), per the
[ST register documentation](https://docs.waveshare.com/Memory_Map_Explanation/ST_Servo_Memory_Map_Explanation).
Original firmware/resolution/settings, verified block settings, per-command
summaries, and restoration results are saved in the report. Completed commands
and partial samples survive a normal exception or Ctrl-C.

Movement delay is measured separately at one- and two-tick thresholds from each
command's measured baseline, in the direction of the command increment,
confirmed by a second sample. `null` means no such response during that hold,
not zero delay; an ideal one-tick response need not cross the two-tick threshold.
Timing resolution is limited by feedback intervals and serial transactions.
Compare complete staircase traces and initial errors across blocks; consecutive
commands are not independent step-response trials. This experiment does not
measure Cartesian brush accuracy or the planner's speed profile.

### Saved joint offsets

`SO101MotionPlanner` loads `src/SO101/so101_offsets.json` at initialization, resolving
the path beside `motion_planner.py` rather than against the working directory.
Pass `joint_offsets_path=...` to use another file. The viewer's exported
`joint_offsets_deg` must contain finite degree offsets for all five arm joints.
Missing/invalid files raise an error; the planner does not silently substitute
another calibration. Saved observations are only metadata and are never used
as live joint readings.

The same loaded mapping is used in both directions:

```text
model_radians = deg2rad(observed_degrees + offset_degrees)
command_degrees = rad2deg(model_radians) - offset_degrees
```

The gripper's normalized opening is preserved separately. Restart the robot
program after saving new offsets so the next planner instance loads them;
an active instance keeps one calibration for both planning and execution.

### Calibrated joint limits

`SO101MotionPlanner` reads the saved LeRobot calibration JSON at startup. By
default it uses
`~/.cache/huggingface/lerobot/calibration/robots/so_follower/my_arm.json`;
pass `lerobot_calibration_path=...` to override it. `robot.py` passes the active
follower's `calibration_fpath`, so the planner uses that robot's saved ranges.
Loading the planner does not connect to a motor bus or write motor registers.
Missing, malformed, or invalid arm ranges stop initialization.

For the five STS3215 arm joints, the conversion matches LeRobot's degree mode:

```text
midpoint = (range_min + range_max) / 2
model_lower_deg = (range_min - midpoint) * 360 / 4095 + offset_deg
model_upper_deg = (range_max - midpoint) * 360 / 4095 + offset_deg
```

The stored ranges already include the motor homing convention; `homing_offset`
is not added again. These bounds replace the nominal URDF arm position limits
in the in-memory RoboPlan scene before constructing IK and time parameterization.
Start validation, IK position constraints, random seeds, goal validation and
output trajectory validation therefore share the same calibrated ranges.
Inputs are not clamped. The gripper's URDF bounds and normalized command mapping,
velocity/acceleration/jerk limits, collision geometry, SRDF exclusions, and goal
and path collision checks are retained. Source URDF and calibration
files are not modified. Restart after recalibrating motors or saving new offsets.

The separate offset viewer still labels its range column **URDF range**: it
shows the original nominal bounds for reference, not the planner's overrides.

### Execution timeout diagnostics

`SO101.execute_trajectory()` waits up to three seconds after sending the last
sample for all five arm joints to be within two degrees of their requested
servo angles. During this settling period it prints the target, measured angle,
and signed error for joints outside tolerance every half-second. The gripper
remains in normalized 0–100 units and is excluded from this degree check.

A timeout reports each joint's position range over roughly the final half-second:
a near-zero range with a persistent error suggests the joint has stopped short;
a changing position requires checking the error trend for convergence or
oscillation. It then reads the failed joints' goal-position registers, PID gains,
torque settings, limits, velocity, load, current, voltage, temperature and status.
Only `Goal_Position_deg` is converted to servo degrees; other register values
are printed in raw units. Read failures are reported without replacing the
original timeout. These diagnostic reads do not write registers, change torque,
resend commands, or relax the completion tolerance.

Compare the requested target to `Goal_Position_deg` first (allowing for encoder
quantization). If they disagree, investigate command delivery or limits. Also
compare `Goal_Position_raw` with `Min_Position_Limit` and `Max_Position_Limit`:
the planner now uses the saved calibration, while these readbacks expose the
motor's current limits. A matching goal register alone does not establish that
the requested angle is within those limits. For an in-range target, inspect
load, torque limits, status and the settling trend before tuning gains. Zero
offsets align the kinematic model; they should not be adjusted to compensate
for servo tracking error.

### Read-only brush position

From `src`, run `python ../scripts/position_util.py` and enter a label for each reading
(`q` exits). It calls `SO101.get_cartesian_pose()` and prints `frida_xyz_m`:
the brush-tip `[x, y, z]` in FRIDA coordinates, in meters, using the saved offsets.
It uses the guarded read-only bus, without motor configuration, calibration
writes, movement commands, or a viewer. Torque is left unchanged on connection
and exit. A calibration mismatch stops the reader without changing registers.

### Read-only zero-offset viewer

From `src`, in the `frida-lerobot` environment:

```bash
python ../scripts/preview_so101_offsets.py
```

This standalone script reads the six motor positions directly from the bus and
closes the serial connection before opening Viser at `http://127.0.0.1:8081`.
It does not call the robot connection/configuration routine, change torque,
write calibration, or send motion commands. Motor-register writes are blocked.
Stop other programs using the same serial port before starting it.

Adjust the five arm sliders to match the stationary physical links. The convention
is `model angle = observed angle + offset`, in degrees. Initial offsets load from
`src/SO101/so101_offsets.json` when it exists. Before the first save, the initial guess
is elbow -4.6 degrees and the others zero. Out-of-limit angles are
shown without clamping; matching the physical pose is the objective, not merely
making the limit warning disappear. The gripper stays at its nominal model angle
inferred from the observed 0–100 opening.

**Refresh measured pose** opens the bus for another read and closes it again;
offsets are retained. Verify an offset against several observed poses. Existing
torque state is preserved throughout. **Save snapshot and offsets** writes
`src/SO101/so101_offsets.json` in the robot model directory and prints the result. The next planner
instance loads the saved offsets; running planners and the URDF are unchanged.
Exit with Ctrl-C.

Use `--port`, `--calibration`, `--urdf`, `--viewer-port`, and `--output` to override
paths/settings. `--offset shoulder_lift=4.0` overrides an initial slider value;
`--offset-range 60` expands the default +/-30 degree sliders. To replay an exported
snapshot and offsets entirely offline:

```bash
python ../scripts/preview_so101_offsets.py --observation SO101/so101_offsets.json \
    --initial-offsets SO101/so101_offsets.json --output ../outputs/so101_offsets_revised.json
```

`--observation` also accepts a plain JSON object with `shoulder_pan.pos`,
`shoulder_lift.pos`, `elbow_flex.pos`, `wrist_flex.pos`, `wrist_roll.pos` (degrees),
and `gripper.pos` (0–100 opening). `--no-browser` leaves the URL available without
opening a browser automatically.

---

Feel free to open an issue or contribute improvements!
