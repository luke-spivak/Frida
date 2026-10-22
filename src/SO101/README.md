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

From `src`, run `python -m pytest -q tests/test_motion_planner.py tests/test_so101_calibrated_limits.py -k quintic`
to check derivative consistency, fallback stops, and the recorded diagnostic
lift/straight-stroke poses entirely offline.

IK uses joint limits without an internal collision barrier. Numerical search
steps are not executed: each candidate goal and connecting path must pass
collision and brush-tilt validation before time parameterization, and the
resulting trajectory is checked again. These collision checks reject mesh
intersections; they do not enforce the former barrier's 2 mm clearance margin.

### Brush alignment

The planner aims the brush bristles downward along base-frame -Z and accepts
waypoints only when tip position error is at most 1 mm and shaft tilt is at most
10 degrees. Two position tasks track the tip and shaft midpoint; rotation about
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

From `src`, run `python -m pytest -q tests/test_so101_calibrated_limits.py -k station_sequence`
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

### HELLO WORLD drawing

From `src`, `python paint_hello_world.py` saves `outputs/hello_world.svg` without
connecting to hardware. The preview shows layout only; it does not run IK.
To move the robot with a loaded brush or pen:

```bash
python paint_hello_world.py --execute --air
python paint_hello_world.py --execute --canvas
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
Run `python -m pytest -q tests/test_paint_hello_world.py
tests/test_so101_calibrated_limits.py -k 'hello_world or pen_up or oversized or cli_defaults'`
for the offline drawing tests. These replay both modes from a recorded start
without connecting to the robot, checking the full timed paths and lettering.

### Lift and straight-stroke diagnostic

From `src`, `python diagnose_so101_motion.py` plans a 10 mm lift followed by a
20 mm stroke along FRIDA +X, using the saved observation in `so101_offsets.json`.
The default run is offline: it does not open a motor connection or invent
tracking measurements. `--start-joints FILE` selects another saved observation.

When ready to record hardware, place the tip in clear air and run:

```bash
python diagnose_so101_motion.py --execute
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

Offline tests: `python -m pytest -q tests/test_so101_motion_diagnostic.py
tests/test_so101_execution.py tests/test_so101_calibrated_limits.py -k 'diagnostic or execution'`.

### Shoulder-pan dead-zone comparison

`python compare_so101_p_gains.py` only prints the experiment settings; it does
not connect to the arm. When ready to move, from `src`:

```bash
python compare_so101_p_gains.py --execute
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

`SO101MotionPlanner` loads `src/so101_offsets.json` at initialization, resolving
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

From `src`, run `python position_util.py` and enter a label for each reading
(`q` exits). It calls `SO101.get_cartesian_pose()` and prints `frida_xyz_m`:
the brush-tip `[x, y, z]` in FRIDA coordinates, in meters, using the saved offsets.
It uses the guarded read-only bus, without motor configuration, calibration
writes, movement commands, or a viewer. Torque is left unchanged on connection
and exit. A calibration mismatch stops the reader without changing registers.

### Read-only zero-offset viewer

From `src`, in the `frida-lerobot` environment:

```bash
python preview_so101_offsets.py
```

This standalone script reads the six motor positions directly from the bus and
closes the serial connection before opening Viser at `http://127.0.0.1:8081`.
It does not call the robot connection/configuration routine, change torque,
write calibration, or send motion commands. Motor-register writes are blocked.
Stop other programs using the same serial port before starting it.

Adjust the five arm sliders to match the stationary physical links. The convention
is `model angle = observed angle + offset`, in degrees. Initial offsets load from
`src/so101_offsets.json` when it exists. Before the first save, the initial guess
is elbow -4.6 degrees and the others zero. Out-of-limit angles are
shown without clamping; matching the physical pose is the objective, not merely
making the limit warning disappear. The gripper stays at its nominal model angle
inferred from the observed 0–100 opening.

**Refresh measured pose** opens the bus for another read and closes it again;
offsets are retained. Verify an offset against several observed poses. Existing
torque state is preserved throughout. **Save snapshot and offsets** writes
`src/so101_offsets.json` beside the script and prints the result. The next planner
instance loads the saved offsets; running planners and the URDF are unchanged.
Exit with Ctrl-C.

Use `--port`, `--calibration`, `--urdf`, `--viewer-port`, and `--output` to override
paths/settings. `--offset shoulder_lift=4.0` overrides an initial slider value;
`--offset-range 60` expands the default +/-30 degree sliders. To replay an exported
snapshot and offsets entirely offline:

```bash
python preview_so101_offsets.py --observation so101_offsets.json \
    --initial-offsets so101_offsets.json --output so101_offsets_revised.json
```

`--observation` also accepts a plain JSON object with `shoulder_pan.pos`,
`shoulder_lift.pos`, `elbow_flex.pos`, `wrist_flex.pos`, `wrist_roll.pos` (degrees),
and `gripper.pos` (0–100 opening). `--no-browser` leaves the URL available without
opening a browser automatically.

---

Feel free to open an issue or contribute improvements!
