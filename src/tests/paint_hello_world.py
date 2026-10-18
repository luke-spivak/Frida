"""Paint two rows of block letters using an already initialized FRIDA Painter.

Put this file beside painter_test.py. After your existing Painter(opt) setup:

    from paint_hello_world import paint_hello_world
    paint_hello_world(painter, paint_index=1)

Optional physical air pass (moves the robot, but does not dip or touch canvas):
    paint_hello_world(painter, air_test=True)

Uses your existing calibration, brush orientation, and servo gain setup.
Does not initialize/disconnect the robot, retrain models, or change servo gains.
On a movement error it stops issuing commands and propagates the exception;
your caller retains responsibility for robot shutdown/support.
"""

import math
import time


# Each list is a continuous brush-down polyline; (0, 0) is letter bottom-left.
LETTERS = {
    "H": [[(0, 0), (0, 1)], [(1, 0), (1, 1)], [(0, .5), (1, .5)]],
    "E": [[(1, 1), (0, 1), (0, 0), (1, 0)], [(0, .5), (.85, .5)]],
    "L": [[(0, 1), (0, 0), (1, 0)]],
    "O": [[(.2, 0), (0, .2), (0, .8), (.2, 1), (.8, 1),
           (1, .8), (1, .2), (.8, 0), (.2, 0)]],
    "W": [[(0, 1), (.2, 0), (.5, .55), (.8, 0), (1, 1)]],
    "R": [[(0, 0), (0, 1), (.75, 1), (1, .8), (1, .65),
           (.75, .5), (0, .5)], [(.5, .5), (1, 0)]],
    "D": [[(0, 0), (0, 1), (.6, 1), (1, .75), (1, .25),
           (.6, 0), (0, 0)]],
}


def canvas_to_global_coordinates(x, y, z, opt):
    """Normalized canvas XY -> FRIDA meters; Z is already a global height."""
    return (
        (x - .5) * opt.CANVAS_WIDTH_M + opt.CANVAS_POSITION[0],
        y * opt.CANVAS_HEIGHT_M + opt.CANVAS_POSITION[1],
        z,
    )


def paint_hello_world(
    painter,
    paint_index=0,
    *,
    letter_height_m=.05,
    pressure=1.0,
    step_m=.01,
    waypoint_pause=.10,
    vertical_step_m=.01,
    vertical_pause=.10,
    air_test=False,
):
    """Paint HELLO above WORLD, centered on the calibrated canvas.

    Default footprint: 133 x 84 mm (shrinks to fit smaller canvases).
    pressure: fraction between just-touching Z_CANVAS and deepest calibrated
              Z_MAX_CANVAS, not a force in newtons.
    step_m: maximum spacing of commanded waypoints during brush strokes.
    waypoint_pause: dwell after each waypoint; not a guaranteed tool speed.
    vertical_step_m: maximum commanded Z increment on lowering/lifting (1 mm).
    vertical_pause: dwell after each vertical step (100 ms). Smoothstep easing
                    reduces command increments near the start and end.
    air_test: trace 10 mm above the canvas without loading paint.

    Painter.move_to accepts FRIDA coordinates. The SO101 robot adapter already
    performs its XY swap/sign change: do NOT apply that conversion here again.
    No image homography is applied; coordinates follow the supplied formula.
    Start with brush clear of the canvas/containers, as after Painter setup.
    """
    opt = painter.opt
    width, height = float(opt.CANVAS_WIDTH_M), float(opt.CANVAS_HEIGHT_M)
    numbers = (width, height, letter_height_m, pressure, step_m, waypoint_pause,
               vertical_step_m, vertical_pause,
               float(painter.Z_CANVAS), float(painter.Z_MAX_CANVAS),
               float(opt.HOVER_FACTOR), *map(float, opt.CANVAS_POSITION[:2]))
    if not all(math.isfinite(value) for value in numbers):
        raise ValueError("Dimensions, calibration and parameters must be finite.")
    if min(width, height, letter_height_m, step_m, vertical_step_m, vertical_pause) <= 0:
        raise ValueError("Canvas dimensions, letter height and step must be positive.")
    if not 0 <= pressure <= 1 or waypoint_pause < 0:
        raise ValueError("pressure must be 0..1 and waypoint_pause nonnegative.")
    if not isinstance(paint_index, int) or not 0 <= paint_index < 12:
        raise ValueError("paint_index must be an integer from 0 to 11.")
    if opt.HOVER_FACTOR <= 0 or painter.Z_MAX_CANVAS > painter.Z_CANVAS:
        raise ValueError("Expected positive hover and deepest Z <= contact Z.")

    # 5 letters: 5 * .6h + 4 * .2h = 3.8h; two rows span 2.4h.
    h = min(letter_height_m, .75 * width / 3.8, .65 * height / 2.4)
    letter_w, gap = .6 * h, .2 * h
    left = (width - 3.8 * h) / 2
    bottom = (height - 2.4 * h) / 2
    z_touch = float(painter.Z_CANVAS)
    z_draw = z_touch + pressure * (float(painter.Z_MAX_CANVAS) - z_touch)
    if air_test:
        z_draw = z_touch + .010
    z_hover = max(z_touch + float(opt.HOVER_FACTOR), z_draw + .010)

    def move(x_m, y_m, z, pause=None):
        # x_m/y_m are distances from canvas bottom-left, not robot coordinates.
        painter.move_to(*canvas_to_global_coordinates(x_m / width, y_m / height, z, opt))
        time.sleep(waypoint_pause if pause is None else pause)

    def move_vertical(x_m, y_m, z_start, z_end):
        # Ease commanded motion in/out, with maximum smoothstep slope 1.5.
        # This paces targets; the robot's joint tolerance still limits accuracy.
        count = max(1, math.ceil(1.5 * abs(z_end - z_start) / vertical_step_m))
        for j in range(1, count + 1):
            t = j / count
            fraction = t * t * (3 - 2 * t)
            move(x_m, y_m, z_start + fraction * (z_end - z_start),
                 pause=vertical_pause)

    print(f"{'Air tracing' if air_test else 'Painting'} HELLO / WORLD: "
          f"{3.8*h*1000:.0f} x {2.4*h*1000:.0f} mm")

    for word, row_y in (("HELLO", bottom + 1.4 * h), ("WORLD", bottom)):
        for column, letter in enumerate(word):
            print(f"{word}: letter {column + 1}/5 ({letter})")
            if not air_test:
                painter.get_paint(paint_index)
            x0 = left + column * (letter_w + gap)
            for stroke in LETTERS[letter]:
                points = [(x0 + x * letter_w, row_y + y * h) for x, y in stroke]
                x, y = points[0]
                move(x, y, z_hover)  # Travel with brush raised.
                move_vertical(x, y, z_hover, z_draw)
                for (ax, ay), (bx, by) in zip(points, points[1:]):
                    count = max(1, math.ceil(math.hypot(bx - ax, by - ay) / step_m))
                    for j in range(1, count + 1):
                        t = j / count
                        move(ax + t * (bx - ax), ay + t * (by - ay), z_draw)
                move_vertical(*points[-1], z_draw, z_hover)

    # Leave the brush hovering above the final letter; caller owns shutdown.
    print("Finished; brush is raised above the final letter.")


if __name__ == "__main__":
    print("Import this script from your existing painter_test.py after Painter setup:\n"
          "  from paint_hello_world import paint_hello_world\n"
          "  paint_hello_world(painter, paint_index=1)")
