"""Draw HELLO / WORLD with the SO101, using the existing checked motion planner.

Run from src (the brush/pen should already be loaded and clamped):
    python paint_hello_world.py                       # SVG preview only
    python paint_hello_world.py --execute --air       # 20 mm above the canvas
    python paint_hello_world.py --execute --canvas    # at calibrated touch height

Also usable from painter_test.py:
    from paint_hello_world import paint_hello_world
    paint_hello_world(painter, air_test=True)
"""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np


# Single-line block letters. Each polyline is a separate pen-down stroke.
LETTERS = {
    "H": [[(0, 0), (0, 1)], [(1, 0), (1, 1)], [(0, .5), (1, .5)]],
    "E": [[(1, 1), (0, 1), (0, 0), (1, 0)], [(0, .5), (.8, .5)]],
    "L": [[(0, 1), (0, 0), (1, 0)]],
    "O": [[(.2, 0), (.8, 0), (1, .2), (1, .8), (.8, 1), (.2, 1),
           (0, .8), (0, .2), (.2, 0)]],
    "W": [[(0, 1), (.2, 0), (.5, .5), (.8, 0), (1, 1)]],
    "R": [[(0, 0), (0, 1), (.8, 1), (1, .8), (1, .65), (.8, .5), (0, .5)],
          [(.5, .5), (1, 0)]],
    "D": [[(0, 0), (0, 1), (.65, 1), (1, .75), (1, .25), (.65, 0), (0, 0)]],
}
SOURCE_DIR = Path(__file__).resolve().parent


def hello_world_strokes(opt, z, letter_height=0.02):
    """FRIDA XYZ in metres, centered horizontally near the front of the canvas."""
    values = [z, letter_height, opt.CANVAS_WIDTH_M, opt.CANVAS_HEIGHT_M,
              *opt.CANVAS_POSITION]
    if not np.all(np.isfinite(values)) or letter_height <= 0:
        raise ValueError("Canvas geometry and letter height must be finite; height must be positive")
    width, advance = 0.6 * letter_height, 0.9 * letter_height
    text_width = 4 * advance + width
    margin = 0.035  # Leave room at the front edge for the upright brush posture.
    if text_width + 2 * margin > opt.CANVAS_WIDTH_M or 2.5 * letter_height + 2 * margin > opt.CANVAS_HEIGHT_M:
        raise ValueError("HELLO WORLD does not fit inside the canvas with 35 mm margins")
    left = opt.CANVAS_POSITION[0] - text_width / 2
    bottom = opt.CANVAS_POSITION[1] + margin
    strokes = []
    for word, baseline in (("HELLO", bottom + 1.5 * letter_height), ("WORLD", bottom)):
        for column, letter in enumerate(word):
            for polyline in LETTERS[letter]:
                corners = np.array([[left + column * advance + x * width,
                                     baseline + y * letter_height, z] for x, y in polyline])
                # Sample the lettering for preview/callers. The planner merges
                # collinear samples and checks the complete Cartesian segments.
                points = [corners[0]]
                for start, end in zip(corners, corners[1:]):
                    segments = max(1, int(np.ceil(np.linalg.norm(end - start) / 0.003)))
                    points.extend(np.linspace(start, end, segments + 1)[1:])
                strokes.append(np.asarray(points))
    return strokes


def draw_strokes(robot, strokes, lift=0.050):
    """Plan travel, descent, drawing, and lift together for each stroke."""
    if not np.isfinite(lift) or lift <= 0:
        raise ValueError("Lift must be positive and finite")
    current = np.asarray(robot.get_cartesian_pose(), dtype=float)
    travel_z = strokes[0][0, 2] + lift
    robot.go_to_cartesian_pose([[current[0], current[1], max(current[2], travel_z)]])
    for index, stroke in enumerate(strokes, 1):
        print(f"Stroke {index}/{len(strokes)}", flush=True)
        # Each call keeps the existing joint limits, collisions, tilt tolerance,
        # gripper hold and trajectory timing in SO101.go_to_cartesian_pose().
        path = np.vstack((
            [stroke[0, 0], stroke[0, 1], travel_z],
            stroke,
            [stroke[-1, 0], stroke[-1, 1], travel_z],
        ))
        robot.go_to_cartesian_pose(path)


def paint_hello_world(painter, air_test=True, air_height=0.02, letter_height=0.02):
    """Convenience entry point for an already initialized Painter."""
    if not np.isfinite(air_height) or air_height <= 0:
        raise ValueError("Air height must be positive and finite")
    z = painter.Z_CANVAS + (air_height if air_test else 0.0)
    draw_strokes(painter.robot, hello_world_strokes(painter.opt, z, letter_height))


def save_preview(path, opt, strokes):
    """Top-down SVG in canvas millimetres; requires no robot or camera."""
    width, height = opt.CANVAS_WIDTH_M * 1000, opt.CANVAS_HEIGHT_M * 1000
    left = opt.CANVAS_POSITION[0] - opt.CANVAS_WIDTH_M / 2
    bottom = opt.CANVAS_POSITION[1]
    lines = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}">',
             '<rect width="100%" height="100%" fill="white" stroke="gray"/>']
    for stroke in strokes:
        points = " ".join(f"{(x-left)*1000:.3f},{height-(y-bottom)*1000:.3f}" for x, y, _ in stroke)
        lines.append(f'<polyline points="{points}" fill="none" stroke="black" stroke-width="0.6"/>')
    lines.append('</svg>')
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--air", dest="air", action="store_true", help="Draw above the canvas (default)")
    mode.add_argument("--canvas", dest="air", action="store_false", help="Draw at calibrated Z_CANVAS")
    parser.set_defaults(air=True)
    parser.add_argument("--execute", action="store_true", help="Connect and move the robot; otherwise only preview")
    parser.add_argument("--air-height-mm", type=float, default=20.0)
    parser.add_argument("--letter-height-mm", type=float, default=20.0)
    parser.add_argument("--materials", type=Path, default=SOURCE_DIR.parent / "my_materials.json")
    parser.add_argument("--calibration", type=Path,
                        default=SOURCE_DIR / "caches/sharpie_short_strokes/brush_tip_to_table_calib.json")
    parser.add_argument("--preview", type=Path, default=SOURCE_DIR.parent / "outputs/hello_world.svg")
    args = parser.parse_args(argv)
    if not np.isfinite(args.air_height_mm) or args.air_height_mm <= 0:
        parser.error("--air-height-mm must be positive and finite")
    opt = SimpleNamespace(**json.loads(args.materials.read_text()))
    canvas_z = float(json.loads(args.calibration.read_text())["Z_CANVAS"])
    z = canvas_z + (args.air_height_mm / 1000 if args.air else 0.0)
    strokes = hello_world_strokes(opt, z, args.letter_height_mm / 1000)
    save_preview(args.preview, opt, strokes)
    print(f"{'Air' if args.air else 'Canvas'} drawing: {len(strokes)} strokes, FRIDA Z={z:.4f} m")
    print(f"Layout preview: {args.preview.resolve()}")
    if not args.execute:
        print("Preview only. Add --execute to move the robot. This preview does not run IK.")
        return

    # SO101 currently resolves its model paths relative to src. Avoid Painter
    # initialization: lettering does not need a camera or a stroke library.
    import os
    from robot import SO101
    os.chdir(SOURCE_DIR)
    robot = SO101(debug=True)
    try:
        robot.good_morning_robot()
        draw_strokes(robot, strokes)
    finally:
        if robot.arm.bus.is_connected:
            robot.good_night_robot()


if __name__ == "__main__":
    main()
