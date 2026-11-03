"""Measure paint depletion with three loads of ten identical SO101 strokes.

Preview is the default. --execute connects to hardware and asks for a fresh
sheet before each load. No training, optimization, or cache writes are performed.
"""

if __package__:
    from ._bootstrap import ROOT, SOURCE_DIR
else:
    from _bootstrap import ROOT, SOURCE_DIR


import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import pickle
from types import SimpleNamespace
from uuid import uuid4

import numpy as np

from canvas_z import canvas_geometry, read_height_calibration


STROKES_PER_LOAD = 10
STARTUP_PROMPT = (
    "This will move the SO101. Start less folded, with the brush upright over the canvas center, "
    "tip roughly 5-10 cm above the paper. Support the arm and switch off motor power before manual "
    "repositioning; restore power afterward. Stations ready and hands clear. "
    "Press Enter for the read-only startup check, then execution (Ctrl-C cancels): "
)


class StartupPreflightError(RuntimeError):
    """Initial read/route check failed before motor setup or movement."""


def add_common_arguments(parser, output_subdir):
    """Shared controls for the loading and washing experiments."""
    parser.add_argument("--execute", action="store_true", help="Connect and paint; default is preview only")
    parser.add_argument("--materials_json", type=Path, default=SOURCE_DIR.parent / "my_materials.json")
    parser.add_argument("--cache_dir", type=Path, default=SOURCE_DIR / "caches/so101_round")
    parser.add_argument("--output-dir", type=Path, default=SOURCE_DIR.parent / "outputs" / output_subdir)
    parser.add_argument("--length-mm", type=float, default=20.)
    parser.add_argument("--stroke-z", type=float, default=.25,
                        help="Normalized compression in the stroke middle; endpoints retain FRIDA's 0.2")
    parser.add_argument("--wash-passes", type=int, choices=(1, 2), default=2,
                        help="Water/rag cycles whenever the protocol washes")
    parser.add_argument("--seed", type=int, default=101,
                        help="Seed for repeatable rag-wiping and paint-dabbing offsets")
    parser.add_argument("--recording-fps", type=float, default=10.)
    parser.add_argument("--notes", default="", help="Brush, paint dilution, paper, or other experiment notes")
    return parser


def parse_common_arguments(parser, argv):
    args = parser.parse_args(argv)
    if not 0 <= args.seed < 2**32:
        parser.error("--seed must fit an unsigned 32-bit integer")
    if not np.isfinite([args.length_mm, args.stroke_z, args.recording_fps]).all():
        parser.error("Length, compression, and recording FPS must be finite")
    if args.length_mm <= 0 or not 0 < args.stroke_z <= .99 or not 0 < args.recording_fps <= 60:
        parser.error("Length must be positive, compression in (0, .99], and FPS in (0, 60]")
    args.materials_json = args.materials_json.resolve()
    args.cache_dir = args.cache_dir.resolve()
    args.output_dir = args.output_dir.resolve()
    return args


def arguments(argv=None):
    parser = add_common_arguments(argparse.ArgumentParser(description=__doc__), "paint-loading")
    parser.add_argument("--repeats", type=int, default=3)
    args = parse_common_arguments(parser, argv)
    if not 1 <= args.repeats <= 10:
        parser.error("--repeats must be between 1 and 10")
    return args


def load_setup(args):
    materials = json.loads(args.materials_json.read_text())
    opt = SimpleNamespace(**{**materials, "robot": "so101", "simulate": False,
                             "brush_length": None, "HOVER_FACTOR": .1})
    position, width, height = canvas_geometry(opt)
    opt.X_CANVAS_MIN, opt.X_CANVAS_MAX = position[0] - width / 2 + .001, position[0] + width / 2 - .001
    opt.Y_CANVAS_MIN, opt.Y_CANVAS_MAX = position[1] + .001, position[1] + height - .001
    params, grid = read_height_calibration(args.cache_dir / "brush_tip_to_table_calib.json", opt)
    if grid is None:
        raise ValueError("Calibrate the nine canvas heights before this test")
    if float(params["Z_MAX_CANVAS"]) >= float(params["Z_CANVAS"]):
        raise ValueError("The maximum-compression height must be below the contact height")
    limits = [opt.MIN_STROKE_LENGTH, opt.MAX_STROKE_LENGTH, opt.MIN_STROKE_Z]
    if not np.isfinite(limits).all() or not 0 < limits[0] <= limits[1] or not 0 <= limits[2] <= .99:
        raise ValueError("Invalid stroke limits in materials")
    if not opt.MIN_STROKE_LENGTH <= args.length_mm / 1000 <= opt.MAX_STROKE_LENGTH:
        raise ValueError("Test length must be inside the material stroke-length limits")
    if args.stroke_z < opt.MIN_STROKE_Z:
        raise ValueError("Test compression must be at least MIN_STROKE_Z")
    for name in ("WATER_POSITION", "RAG_POSTITION", "PALLETTE_POSITION"):
        xy = np.asarray(getattr(opt, name), dtype=float)
        if xy.shape != (2,) or not np.isfinite(xy).all():
            raise ValueError(f"{name} must contain two finite XY coordinates")
        depth = float(params["Z_CANVAS"])
        if name == "PALLETTE_POSITION":
            depth -= .2 * (float(params["Z_CANVAS"]) - float(params["Z_MAX_CANVAS"]))
        setattr(opt, name, [*xy.tolist(), depth])
    if not np.isfinite(opt.PAINT_DIFFERENCE) or opt.PAINT_DIFFERENCE <= 0:
        raise ValueError("PAINT_DIFFERENCE must be positive and finite")
    # Trusted local calibration, as in the production camera code. No GUI or
    # fallback calibration is allowed: this experiment must not paint extra dots.
    with (args.cache_dir / "cached_H_canvas.pkl").open("rb") as stream:
        homography = np.asarray(pickle.load(stream), dtype=float)
    if homography.shape != (3, 3) or not np.isfinite(homography).all() or np.linalg.matrix_rank(homography) != 3:
        raise ValueError("Invalid cached camera homography")
    xy_path = args.cache_dir / "cached_H_coord.pkl"
    xy_correction = None
    if xy_path.exists():
        with xy_path.open("rb") as stream:
            xy_correction = np.asarray(pickle.load(stream), dtype=float)
        if (xy_correction.shape != (3, 3) or not np.isfinite(xy_correction).all()
                or np.linalg.matrix_rank(xy_correction) != 3):
            raise ValueError("Invalid cached XY calibration")
    return opt, params, grid, homography, xy_correction


def _project_xy(matrix, point):
    projected = matrix @ [*point, 1.]
    if not np.isfinite(projected).all() or abs(projected[2]) < 1e-12:
        raise ValueError("XY calibration maps a test mark to an invalid coordinate")
    return projected[:2] / projected[2]


def make_layout(opt, grid, length_m, xy_correction=None):
    """Center two columns/five rows in the measured *command-space* Z grid.

    Invert the XY correction to obtain targets that BrushStroke.execute maps
    back onto these positions. Keep 20 mm row spacing, a 20 mm column gap, and
    the requested stroke length. Check 6 mm envelopes in BOTH command-space
    grid bounds and target-space paper bounds; never clip strokes or calibrations.
    This is a layout check, not an IK or measured brush-footprint guarantee.
    """
    position, width, height = canvas_geometry(opt)
    if not np.isfinite(length_m) or length_m <= 0:
        raise ValueError("Test stroke length must be positive and finite")
    correction = np.eye(3) if xy_correction is None else np.asarray(xy_correction, dtype=float)
    if (correction.shape != (3, 3) or not np.isfinite(correction).all()
            or np.linalg.matrix_rank(correction) != 3):
        raise ValueError("Invalid cached XY calibration")
    inverse = np.linalg.inv(correction)
    center_x, center_y = (grid.x_m[0] + grid.x_m[-1]) / 2, (grid.y_m[0] + grid.y_m[-1]) / 2
    starts = (center_x - length_m - .01, center_x + .01)
    rows = center_y + np.linspace(.04, -.04, 5)
    paper_bounds = (position[0] - width / 2, position[0] + width / 2,
                    position[1], position[1] + height)
    grid_bounds = (grid.x_m[0], grid.x_m[-1], grid.y_m[0], grid.y_m[-1])
    layout = []
    for commanded_y in rows:
        for commanded_x in starts:
            x, y = _project_xy(inverse, (commanded_x, commanded_y))
            # Forward-project again, matching production's transform of the
            # start only. Stroke length/direction are added AFTER this transform.
            cx, cy = _project_xy(correction, (x, y))
            for px, py, bounds, space in ((cx, cy, grid_bounds, "measured height grid"),
                                          (x, y, paper_bounds, "canvas after XY correction")):
                left, right, bottom, top = bounds
                margins = np.array([px - .006 - left, right - px - length_m - .006,
                                    py - .006 - bottom, top - py - .006])
                if margins.min() < -1e-12:
                    raise ValueError(f"Test layout does not fit inside the {space} with 6 mm margins "
                                     f"(stroke {len(layout)+1}, short by {-margins.min()*1000:.2f} mm)")
            for previous in layout:
                for px, py, x_key, y_key in ((cx, cy, "command_x", "command_y"), (x, y, "x", "y")):
                    if (abs(py - previous[y_key]) < .012 - 1e-12
                            and abs(px - previous[x_key]) < length_m + .012 - 1e-12):
                        raise ValueError("XY calibration leaves less than 12 mm separation between test marks")
            layout.append({"slot": len(layout) + 1, "x": float(x), "y": float(y),
                           "command_x": float(cx), "command_y": float(cy)})
    return layout


def sequence_layout(layout, repeat):
    # Shift order on successive sheets so every load doesn't deplete at exactly
    # the same locations. Keep all strokes left-to-right and geometry identical.
    shift = (repeat * 3) % len(layout)
    slots = layout[shift:] + layout[:shift]
    return [{**slot, "stroke": number} for number, slot in enumerate(slots, 1)]


def save_preview(path, opt, layout, length_m):
    position, width, height = canvas_geometry(opt)
    left, bottom = position[0] - width / 2, position[1]
    lines = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width*1000} {height*1000}">',
             '<rect width="100%" height="100%" fill="white" stroke="gray"/>',
             '<text x="5" y="8" font-size="4">Target layout: numbered order, all strokes left to right</text>']
    for mark in layout:
        x, y = (mark["x"] - left) * 1000, (height - mark["y"] + bottom) * 1000
        lines.extend([f'<path d="M {x} {y} h {length_m*1000}" stroke="black" stroke-width="1"/>',
                      f'<text x="{x}" y="{y-4}" font-size="4">{mark["stroke"]}</text>'])
    lines.append('</svg>')
    path.write_text("\n".join(lines))


def save_report(path, report):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


@contextmanager
def repeatable_loading(seed):
    """Production loading uses NumPy's global RNG; restore it after each load."""
    state = np.random.get_state()
    try:
        np.random.seed(seed)
        yield
    finally:
        np.random.set_state(state)


def make_painter(opt, params, grid, robot, xy_correction=None):
    from painter import Painter
    # Bypass Painter.__init__ deliberately: it calibrates, collects a library,
    # and trains. Only these existing motion methods are needed by the test.
    painter = Painter.__new__(Painter)
    painter.opt, painter.robot = opt, robot
    painter.Z_CANVAS = float(params["Z_CANVAS"])
    painter.Z_MAX_CANVAS = float(params["Z_MAX_CANVAS"])
    painter.Z_RANGE = painter.Z_CANVAS - painter.Z_MAX_CANVAS
    painter.canvas_z_calibration = grid
    painter.H_coord = xy_correction
    painter._at_canvas = False
    return painter


def make_stroke(args):
    import torch
    # Execute the production trajectory directly; a neural model is unnecessary.
    return SimpleNamespace(stroke_length=torch.tensor(args.length_mm / 1000),
                           stroke_bend=torch.tensor(0.), stroke_z=torch.tensor(args.stroke_z),
                           stroke_alpha=torch.tensor(0.))


def save_snapshot(recording, homography, opt, directory, name):
    import cv2
    frame = recording.read()
    h, w = frame.shape[:2]
    canvas_width = int(h * opt.CANVAS_WIDTH_M / opt.CANVAS_HEIGHT_M)
    if canvas_width > w:
        raise ValueError("Camera frame is too narrow for the cached canvas calibration")
    canvas = cv2.warpPerspective(frame, homography, (w, h))[:, :canvas_width]
    for suffix, img in (("overhead", frame), ("canvas", canvas)):
        filename = f"{name}-{suffix}.png"
        if not cv2.imwrite(str(directory / filename), img):
            raise OSError(f"Could not save {filename}")
    recording.event("snapshot", snapshot_name=name)


def run_sequences(args, painter, stroke, report, directory, recording, snapshot,
                  execute_stroke, prompt=input):
    """No automatic retries: a failed/partial stroke invalidates that sequence."""
    report_path = directory / "test.json"
    for trial in report["sequences"]:
        repeat = trial["repeat"]
        # Park via the existing high station transfer before the paper prompt.
        painter.to_neutral()
        recording.event("paper_prompt", repeat=repeat)
        prompt(f"\nLoad {repeat}/{args.repeats}: place a FRESH sheet in the same canvas position. "
               "Black paint in slot 0; water and rag ready. Keep paint dilution and rag wetness consistent. "
               "Press Enter to wash, load once, and paint ten strokes (Ctrl-C cancels): ")
        snapshot(f"load-{repeat:02d}-before")
        trial["status"] = "running"
        save_report(report_path, report)
        with repeatable_loading(args.seed):
            recording.event("clean_start", repeat=repeat, passes=args.wash_passes)
            for _ in range(args.wash_passes):
                painter.clean_paint_brush()
            recording.event("clean_end", repeat=repeat)
            recording.event("reload_start", repeat=repeat, palette_slot=0)
            painter.get_paint(0)
            recording.event("reload_end", repeat=repeat)
        for mark in trial["strokes"]:
            trial["in_progress"] = mark["stroke"]
            save_report(report_path, report)
            print(f"Load {repeat}/{args.repeats}, stroke {mark['stroke']}/10 (no reload)", flush=True)
            recording.event("stroke_start", repeat=repeat, **mark)
            result = execute_stroke(stroke, painter, mark["x"], mark["y"], 0.)
            if result is False:
                raise RuntimeError("Stroke was not executed; this sequence is incomplete")
            trial["completed"] += 1
            trial["in_progress"] = None
            save_report(report_path, report)
            recording.event("stroke_end", repeat=repeat, **mark)
        painter.to_neutral()
        snapshot(f"load-{repeat:02d}-after")
        trial["status"] = "completed"
        save_report(report_path, report)
        recording.event("sequence_end", repeat=repeat)


def preflight_start(robot, travel_z, directory, report, recording):
    """Read the real start and check the initial park route before motor setup.

    Open only the bus, never Follower.connect/good_morning_robot: those configure
    motors. No goals, torque changes, or recovery motions are sent by this check.
    The normal executor reads the start and plans again immediately before moving.
    """
    from painter import SO101_NEUTRAL_POSITION
    planner, bus = robot.motion_planner, robot.arm.bus
    state = report["startup"] = {"status": "reading", "travel_z_m": travel_z,
                                 "neutral_frida_m": list(SO101_NEUTRAL_POSITION)}
    save_report(directory / "test.json", report)
    try:
        bus.connect(handshake=False)
        values = {name: float(value) for name, value in bus.sync_read("Present_Position").items()}
        state.update(read_utc=datetime.now(timezone.utc).isoformat(), joint_positions=values)
        save_report(directory / "test.json", report)
        if not np.isfinite(list(values.values())).all() or not 0 <= values["gripper"] <= 100:
            raise ValueError("Invalid measured joint positions")
        # Match SO101.go_to_cartesian_pose: arm degrees are before offsets;
        # gripper observations are 0-100, mapped to its nominal model limits.
        measured = {name: values[name] for name in planner.joint_names}
        limits = planner.scene.getJointInfo("gripper").limits
        measured["gripper"] = float(np.rad2deg(limits.min_position[0] + values["gripper"] / 100 *
                                               (limits.max_position[0] - limits.min_position[0])))
        q = planner.scene.getCurrentJointPositions().copy()
        for name, angle in measured.items():
            q[planner.scene.getJointPositionIndices([name])[0]] = np.deg2rad(angle + planner.joint_offsets_deg.get(name, 0.))
        xyz = planner.scene.forwardKinematics(q, planner.tip_frame, planner.base_frame)[:3, 3]
        state.update(tip_frida_m=[float(-xyz[1]), float(xyz[0]), float(xyz[2])],
                     brush_tilt_deg=float(planner.brush_tilt_deg(q)), status="planning")
        recording.event("startup_pose", **state)
        save_report(directory / "test.json", report)
        print("Startup check (read-only): tip FRIDA XYZ " +
              ", ".join(f"{v*1000:.1f}" for v in state["tip_frida_m"]) +
              f" mm; brush tilt {state['brush_tilt_deg']:.1f} degrees.", flush=True)
        target = robot.frida_to_robot(np.asarray([SO101_NEUTRAL_POSITION], dtype=float))
        planner.plan_xyz_path(target, measured, travel_z=travel_z)
        state["status"] = "passed"
        recording.event("startup_check_passed")
    except (RuntimeError, ValueError, KeyError) as exc:
        state.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        recording.event("startup_check_failed", error=state["error"])
        raise StartupPreflightError(
            "Startup preflight failed; no motion commands or motor-setting writes were sent. "
            "Could not plan the initial lift/traverse/descent from the measured pose. "
            "If this is an IK/collision error, support the arm and switch off motor power before "
            "repositioning it less folded, with the brush upright in clear air over the canvas center "
            "(roughly 5-10 cm above the paper); restore power and rerun. Do not force powered joints. "
            f"Measured start and planner error are saved in {directory / 'test.json'}. Cause: {exc}"
        ) from exc
    finally:
        try:
            if bus.is_connected:
                bus.disconnect(disable_torque=False)
        finally:
            save_report(directory / "test.json", report)


def execute_test(args, opt, params, grid, homography, xy_correction, directory, report,
                 sequence_runner=None):
    # Set before any module imports pyplot. This test has no interactive plots.
    os.environ["MPLBACKEND"] = "Agg"
    import matplotlib
    matplotlib.use("Agg", force=True)
    from brush_stroke import BrushStroke
    from camera.innomaker import WebCam
    from robot import SO101

    previous_cwd = Path.cwd()
    robot, camera, recording = None, None, None
    reason = "error"
    try:
        os.chdir(SOURCE_DIR)  # Existing robot model paths are relative to src.
        # Check camera/recording before opening the motor port. No camera GUI.
        camera = WebCam(SimpleNamespace())
        recording = camera.start_recording(directory / "recording", fps=args.recording_fps)
        recording.event(f"{report.get('experiment', 'loading')}_test_start", report=str(directory / "test.json"))
        save_snapshot(recording, homography, opt, directory, "setup")
        robot = SO101(debug=True)
        painter = make_painter(opt, params, grid, robot, xy_correction)
        preflight_start(robot, float(params["Z_CANVAS"]) + .10, directory, report, recording)
        robot.good_morning_robot()
        report["status"] = "running"
        runner = run_sequences if sequence_runner is None else sequence_runner
        runner(args, painter, make_stroke(args), report, directory, recording,
               lambda name: save_snapshot(recording, homography, opt, directory, name),
               BrushStroke.execute)
        reason = "completed"
    except BaseException as exc:
        reason = "interrupted" if isinstance(exc, (KeyboardInterrupt, EOFError)) else "error"
        report["error"] = f"{type(exc).__name__}: {exc}"
        if recording is not None:
            recording.event("test_exit", reason=reason, error=report["error"])
        raise
    finally:
        report["status"] = reason
        # Never command a recovery movement or drop holding torque on failure.
        # Also retain torque on completion, as in the canvas-height calibrator.
        try:
            if robot is not None and robot.arm.bus.is_connected:
                robot.arm.bus.disconnect(disable_torque=False)
        finally:
            try:
                if camera is not None:
                    camera.close(reason=reason)
            finally:
                os.chdir(previous_cwd)
                save_report(directory / "test.json", report)


def base_report(args, params, xy_correction, source_file=Path(__file__)):
    """Save the physical setup and code versions without touching their sources."""
    sources = [args.materials_json, args.cache_dir / "brush_tip_to_table_calib.json",
               args.cache_dir / "cached_H_canvas.pkl", SOURCE_DIR / "brush_stroke.py",
               SOURCE_DIR / "painter.py", SOURCE_DIR / "robot.py", SOURCE_DIR / "motion_planner.py",
               SOURCE_DIR / "SO101/so101_offsets.json", Path(__file__)]
    if xy_correction is not None:
        sources.append(args.cache_dir / "cached_H_coord.pkl")
    if source_file != Path(__file__):
        sources.append(source_file)
    z_range = float(params["Z_CANVAS"]) - float(params["Z_MAX_CANVAS"])
    return {
        "version": 1, "created_utc": datetime.now(timezone.utc).isoformat(), "status": "preview",
        "settings": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "calibration": params, "materials": json.loads(args.materials_json.read_text()),
        "source_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        "xy_frame": ("Canvas targets with cached XY correction" if xy_correction is not None
                     else "FRIDA command XY; no camera XY correction"),
        "xy_correction": None if xy_correction is None else xy_correction.tolist(),
        "compression_mm": {"middle": args.stroke_z * z_range * 1000, "endpoints": .2 * z_range * 1000},
    }


def main(argv=None):
    args = arguments(argv)
    opt, params, grid, homography, xy_correction = load_setup(args)
    layout = make_layout(opt, grid, args.length_mm / 1000, xy_correction)
    name = datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid4().hex[:8]
    directory = args.output_dir / name
    directory.mkdir(parents=True, exist_ok=False)
    report = base_report(args, params, xy_correction)
    report.update({
        "sequence_protocol": "Fresh sheet; wash, wipe, load slot 0 once; ten left-to-right straight strokes; no reload",
        "sequences": [{"repeat": i + 1, "status": "pending", "completed": 0, "in_progress": None,
                       "strokes": sequence_layout(layout, i)} for i in range(args.repeats)],
    })
    for trial in report["sequences"]:
        save_preview(directory / f"load-{trial['repeat']:02d}-layout.svg", opt,
                     trial["strokes"], args.length_mm / 1000)
    save_report(directory / "test.json", report)
    print(f"{args.repeats} loads x 10 strokes, {args.length_mm:g} mm each; {args.wash_passes} wash/wipe passes per load.")
    print(f"Compression: middle {report['compression_mm']['middle']:.2f} mm; endpoints {report['compression_mm']['endpoints']:.2f} mm (commanded, not force).")
    print("Same positions on fresh sheets; numbered order shifts between sheets to reduce location bias.")
    print(report["xy_frame"] + ". No trained stroke model is needed.")
    print(f"Layouts and settings: {directory}")
    if not args.execute:
        print("Preview only: no robot/camera connection or IK check. Add --execute to paint and record.")
        return 0
    input(STARTUP_PROMPT)
    execute_test(args, opt, params, grid, homography, xy_correction, directory, report)
    print(f"Completed. Compare each sheet using its numbered layout; video and stroke events: {directory / 'recording'}")
    print("After-images contain wet paint of different ages: keep lighting consistent and compare again after drying.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except StartupPreflightError as exc:
        print(f"\nStopped: {exc}")
        raise SystemExit(2)
    except (KeyboardInterrupt, EOFError):
        print("\nStopped. Do not continue a partial depletion sequence; use a fresh sheet and a new test.")
        raise SystemExit(130)
