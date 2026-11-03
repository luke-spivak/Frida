"""Manually collect nine canvas contact heights. Preview only unless --execute.

From src:
    python ../scripts/calibrate_canvas_z.py --materials_json ../my_materials.json \
        --cache_dir caches/so101_mvp
Add --execute to connect to the SO101 and adjust the heights interactively.
No camera, paint loading, stroke library collection, or model training is used.
"""

if __package__:
    from ._bootstrap import ROOT, SOURCE_DIR
else:
    from _bootstrap import ROOT, SOURCE_DIR


import argparse
import json
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np

from canvas_z import DEFAULT_INSET_M, CanvasZCalibration, HEIGHT_FILENAME, grid_axes, read_height_calibration, save_grid


class CalibrationCancelled(Exception):
    pass


def point_order():
    """Center first, then serpentine rows; each point is visited exactly once."""
    return [(1, 1)] + [(row, col) for row in range(3)
                      for col in (range(3) if row % 2 == 0 else range(2, -1, -1))
                      if (row, col) != (1, 1)]


def validate_motion_settings(reference_z, clearance_m, max_adjustment_m):
    if (not np.isfinite([reference_z, clearance_m, max_adjustment_m]).all()
            or not 0 < max_adjustment_m < clearance_m):
        raise ValueError("Clearance must exceed the positive maximum contact-height adjustment")


def collect_contact_grid(opt, reference_z, move, initial_xyz, read_key,
                         inset_m=DEFAULT_INSET_M, clearance_m=0.04, max_adjustment_m=0.03,
                         planning_errors=()):
    """Collect command-space heights through the caller's checked motion API.

    Every point starts in clear air. No automatic descent to a guessed contact
    height. Manual steps bypass any previous grid, but never the normal planner.
    Planning-only rejections permit another key; execution failures abort.
    """
    validate_motion_settings(reference_z, clearance_m, max_adjustment_m)
    xs, ys = grid_axes(opt, inset_m)
    initial_xyz = np.asarray(initial_xyz, dtype=float)
    if initial_xyz.shape != (3,) or not np.isfinite(initial_xyz).all():
        raise ValueError("Expected a finite initial brush-tip XYZ")
    travel_z = reference_z + clearance_m
    # Lift before traversing. Do not first descend if already above travel Z.
    move([[float(initial_xyz[0]), float(initial_xyz[1]), max(float(initial_xyz[2]), travel_z)]])
    contacts = np.full((3, 3), np.nan)
    increments = {"w": 0.0005, "s": -0.0005, "W": 0.002, "S": -0.002}
    for number, (row, col) in enumerate(point_order(), 1):
        x, y = float(xs[col]), float(ys[row])
        target = [x, y, travel_z]
        move([target], fallback_positions=[[float(xs[1]), float(ys[1]), travel_z], target])
        current_z = travel_z
        print(f"\nPoint {number}/9: row {row + 1}/3, column {col + 1}/3 "
              f"(X={x:.6f}, Y={y:.6f} m). Rows run from low Y to high Y.")
        print("Lower until the bristles JUST touch, without compression.\n"
              "w/s: up/down 0.5 mm; W/S: up/down 2 mm; Enter: record; Esc/Ctrl-C: abort.\n"
              "Steps are commanded distances; existing IK/servo tolerances still apply.")
        print(f"Commanded Z: {current_z * 1000:+.2f} mm")
        while True:
            key = read_key()
            if key in ("\x1b", "\x03"):
                raise CalibrationCancelled("Calibration aborted; previous height calibration is unchanged")
            if key in ("\r", "\n"):
                if abs(current_z - reference_z) > max_adjustment_m + 1e-9:
                    print("Still above the contact adjustment range; lower to gentle contact before recording.")
                    continue
                contacts[row, col] = current_z
                break
            if key not in increments:
                continue
            proposed_z = current_z + increments[key]
            if not reference_z - max_adjustment_m - 1e-9 <= proposed_z <= travel_z + 1e-9:
                print("Height adjustment bound reached; move up or abort and inspect the setup.")
                continue
            try:
                move([[x, y, proposed_z]])
            except planning_errors as exc:
                print(f"Step rejected by motion planning; commanded height unchanged: {exc}")
                continue
            current_z = proposed_z
            print(f"Commanded Z: {current_z * 1000:+.2f} mm")
        # Lift at this XY before moving to the next point, including the last.
        move([[x, y, travel_z]])
    return CanvasZCalibration(opt, contacts, inset_m)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--materials_json", type=Path, required=True)
    parser.add_argument("--cache_dir", type=Path, required=True)
    parser.add_argument("--execute", action="store_true", help="Connect and move the SO101; otherwise preview only")
    parser.add_argument("--inset-mm", type=float, default=DEFAULT_INSET_M * 1000,
                        help="Grid margin inside each canvas edge (default: 20 mm)")
    parser.add_argument("--clearance-mm", type=float, default=40.0,
                        help="Travel height above the existing center reference (default: 40 mm)")
    parser.add_argument("--max-adjustment-mm", type=float, default=30.0,
                        help="Maximum contact-height change from the center reference (default: 30 mm)")
    args = parser.parse_args(argv)
    opt = SimpleNamespace(**{**json.loads(args.materials_json.read_text()), "robot": "so101"})
    path = args.cache_dir / HEIGHT_FILENAME
    if not path.is_file():
        parser.error(f"First complete the normal center brush-height calibration; missing {path}")
    original = path.read_bytes()
    # Recalibration must also work after canvas geometry changes or a stale grid.
    params, _ = read_height_calibration(path, opt, load_grid=False)
    reference_z = float(params["Z_CANVAS"])
    inset, clearance, adjustment = args.inset_mm / 1000, args.clearance_mm / 1000, args.max_adjustment_mm / 1000
    validate_motion_settings(reference_z, clearance, adjustment)
    xs, ys = grid_axes(opt, inset)
    print("3 x 3 canvas Z calibration: FRIDA commanded coordinates, after camera XY correction.")
    print(f"Reference Z: {reference_z * 1000:+.2f} mm; travel Z: {(reference_z + clearance) * 1000:+.2f} mm")
    for number, (row, col) in enumerate(point_order(), 1):
        print(f"  {number}: row {row + 1}, column {col + 1}: X={xs[col]:+.6f}, Y={ys[row]:+.6f} m")
    if not args.execute:
        print("Preview only: no hardware imported/connected, no movement or files changed. Add --execute to calibrate.")
        return 0

    input("Secure a clean, flat sheet and the brush. Start with the brush in clear air. "
          "Press Enter to connect and begin manual calibration (Ctrl-C cancels): ")
    # Hardware imports and connection are reachable only with explicit --execute.
    import getch
    from robot import MotionPlanningError, SO101

    robot = SO101(debug=True, show_viewer=False)
    try:
        robot.good_morning_robot()
        grid = collect_contact_grid(
            opt, reference_z, robot.go_to_cartesian_pose, robot.get_cartesian_pose(),
            getch.getch, inset, clearance, adjustment, planning_errors=(MotionPlanningError,),
        )
        if path.read_bytes() != original:
            raise RuntimeError("Height calibration changed during collection; refusing to overwrite it")
        backup = path.with_name(f"{path.stem}.before_z_grid.{time.time_ns()}.json")
        backup.write_bytes(original)
        save_grid(path, params, grid)
        print("\nHeight corrections in mm (rows: low Y to high Y; columns: low X to high X):")
        print(np.round((grid.contact_z_m - reference_z) * 1000, 2))
        print(f"Saved {path.resolve()}\nPrevious calibration: {backup.resolve()}")
        print("Painting with --use_cache will apply the grid. Existing saved plans will reject the changed "
              "height calibration; prepare a new plan. Recollect the stroke library for the corrected contact depths.")
    finally:
        # Do not automatically move after an error or disable torque at contact.
        if robot.arm.bus.is_connected:
            robot.arm.bus.disconnect(disable_torque=False)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (CalibrationCancelled, KeyboardInterrupt, EOFError) as exc:
        print(f"\n{exc or 'Stopped; only a completed grid can replace the height calibration.'}")
        raise SystemExit(130)
