"""Repeat a small SO101 stroke sheet at identical positions, with recording.

Preview by default. Three fresh sheets, each with left/center/right marks in
four directions. No stroke model, optimizer, or calibration writes are needed.
"""

if __package__:
    from ._bootstrap import ROOT, SOURCE_DIR
else:
    from _bootstrap import ROOT, SOURCE_DIR


import argparse
from datetime import datetime
from pathlib import Path
import time
from uuid import uuid4

import numpy as np

from canvas_z import canvas_geometry
from paint_loading import (STARTUP_PROMPT, StartupPreflightError, _project_xy, add_common_arguments, base_report,
                           execute_test, load_setup, parse_common_arguments,
                           repeatable_loading, save_report)


DIRECTIONS = (("horizontal", 0), ("vertical", 90), ("diagonal_up", 45), ("diagonal_down", -45))
REGIONS = ("left", "center", "right")
RELOAD_EVERY = 4
CLEAN_EVERY = 12
MARGIN = .006


def arguments(argv=None):
    parser = add_common_arguments(argparse.ArgumentParser(description=__doc__), "paint-repeatability")
    # Four separated rows fit in this canvas's 99 mm measured height range.
    parser.set_defaults(length_mm=16., wash_passes=1)
    parser.add_argument("--repeats", type=int, default=3, help="Fresh sheets repeating the same positions (1-6)")
    parser.add_argument("--split-directions", action="store_true",
                        help="Use two sheets per repeat: horizontal/vertical, then both diagonals")
    parser.add_argument("--analyze", type=Path, help="Recompute measurements from an existing run; no hardware")
    args = parse_common_arguments(parser, argv)
    if not 1 <= args.repeats <= 6:
        parser.error("--repeats must be between 1 and 6")
    if args.execute and args.analyze:
        parser.error("--analyze cannot be combined with --execute")
    return args


def bounds(start, delta):
    end = start + delta
    return np.r_[np.minimum(start, end) - MARGIN, np.maximum(start, end) + MARGIN]


def make_layout(opt, grid, length_m, xy_correction=None, directions=DIRECTIONS):
    """Fit separated envelopes in both command/grid and target/paper coordinates.

    Production corrects the start with H_coord, then adds the rotated stroke.
    Invert that transform to lay out commands within the measured height grid.
    Never shorten a stroke, clip a path, or disable calibration to make it fit.
    """
    if not np.isfinite(length_m) or length_m <= 0:
        raise ValueError("Stroke length must be positive and finite")
    correction = np.eye(3) if xy_correction is None else np.asarray(xy_correction, dtype=float)
    if correction.shape != (3, 3) or not np.isfinite(correction).all() or np.linalg.matrix_rank(correction) != 3:
        raise ValueError("Invalid cached XY calibration")
    inverse = np.linalg.inv(correction)
    position, width, height = canvas_geometry(opt)
    limits = [np.array([grid.x_m[0], grid.y_m[0], grid.x_m[-1], grid.y_m[-1]]),
              np.array([position[0] - width / 2, position[1], position[0] + width / 2, position[1] + height])]
    deltas = [length_m * np.array([np.cos(np.deg2rad(a)), np.sin(np.deg2rad(a))]) for _, a in directions]
    spans = [abs(delta[1]) + 2 * MARGIN for delta in deltas]
    gap = (grid.y_m[-1] - grid.y_m[0] - sum(spans)) / (len(spans) + 1)
    if gap < 0:
        if tuple(directions) != DIRECTIONS:
            raise ValueError("Repeatability rows do not fit inside the measured height grid with 6 mm margins")
        # Pack four orientations into each of three side-by-side regions:
        # horizontal across the top; vertical beside two stacked diagonals.
        # Keep the full measurement envelopes, plus 4 mm between them.
        packing_gap = .004
        diagonal_span = length_m / np.sqrt(2) + 2 * MARGIN
        region_width = max(length_m + 2 * MARGIN, 2 * MARGIN + packing_gap + diagonal_span)
        region_height = 2 * MARGIN + 2 * packing_gap + 2 * diagonal_span
        if (3 * region_width + .002 > grid.x_m[-1] - grid.x_m[0]
                or region_height > grid.y_m[-1] - grid.y_m[0]):
            raise ValueError("Repeatability marks do not fit inside the measured height grid with 6 mm margins")
        columns = np.linspace(grid.x_m[0] + region_width / 2 + .001,
                              grid.x_m[-1] - region_width / 2 - .001, 3)
        top = (grid.y_m[0] + grid.y_m[-1] + region_height) / 2
        offsets = [(0., top - MARGIN),
                   (-region_width / 2 + MARGIN, top - 2 * MARGIN - packing_gap - diagonal_span - packing_gap / 2),
                   (region_width / 2 - diagonal_span / 2, top - 2 * MARGIN - packing_gap - diagonal_span / 2),
                   (region_width / 2 - diagonal_span / 2, top - 2 * MARGIN - 2 * packing_gap - 1.5 * diagonal_span)]
        centers = [[(cx + dx, cy) for cx in columns] for dx, cy in offsets]
    else:
        inset = length_m / 2 + MARGIN + .004
        columns = np.linspace(grid.x_m[0] + inset, grid.x_m[-1] - inset, 3)
        centers = []
        top = grid.y_m[-1] - gap
        for span in spans:
            centers.append([(cx, top - span / 2) for cx in columns])
            top -= span + gap
    layout, envelopes = [], [[], []]
    for (direction, angle), delta, region_centers in zip(directions, deltas, centers):
        for region, center in zip(REGIONS, region_centers):
            command_start = np.array(center) - delta / 2
            start = _project_xy(inverse, command_start)
            command_start = _project_xy(correction, start)
            for space, point in enumerate((command_start, start)):
                envelope = bounds(point, delta)
                limit = limits[space]
                if np.any(envelope[:2] < limit[:2] - 1e-12) or np.any(envelope[2:] > limit[2:] + 1e-12):
                    raise ValueError("Repeatability layout does not fit inside the grid/canvas with 6 mm margins")
                if any(np.all(envelope[:2] < old[2:] - 1e-12) and np.all(old[:2] < envelope[2:] - 1e-12)
                       for old in envelopes[space]):
                    raise ValueError("Repeatability marks have less than 12 mm separation after XY correction")
                envelopes[space].append(envelope)
            layout.append({"slot": len(layout) + 1, "region": region, "direction": direction,
                           "angle_deg": angle, "x": float(start[0]), "y": float(start[1]),
                           "command_x": float(command_start[0]), "command_y": float(command_start[1])})
    return layout


def make_protocol(args, layout):
    if len(layout) != CLEAN_EVERY:
        raise ValueError("This protocol requires twelve positions per sheet")
    sheets = []
    groups = [layout[:6], layout[6:]] if args.split_directions else [layout]
    for repeat in range(args.repeats):
        # Shift by five: each fixed position moves to a different position in
        # its four-stroke load, and directions/regions are interleaved.
        for group_index, group in enumerate(groups):
            shift = (repeat * 5) % len(group)
            order = group[shift:] + group[:shift]
            number = len(sheets) + 1
            sheets.append({"sheet": number, "status": "pending", "completed": 0,
                           "strokes": [{**mark, "sheet": number,
                                        "stroke": repeat * len(layout) + group_index * len(group) + i,
                                        "stroke_on_sheet": i, "stroke_in_load": (i - 1) % RELOAD_EVERY + 1}
                                       for i, mark in enumerate(order, 1)]})
    return {"experiment": "repeatability", "sheets": sheets, "completed_strokes": 0, "in_progress": None,
            "paint_care": {"reload_every": RELOAD_EVERY, "clean_every": len(groups[0]), "wash_passes": args.wash_passes},
            "sequence_protocol": "Fresh sheet; wash/wipe; black strokes, reloading every four. Repeat same positions with shifted order."}


def save_preview(path, opt, marks, length_m):
    position, width, height = canvas_geometry(opt)
    left, bottom = position[0] - width / 2, position[1]
    lines = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width*1000} {height*1000}">',
             '<rect width="100%" height="100%" fill="white" stroke="gray"/>',
             '<text x="4" y="7" font-size="3.5">Target positions; circle = start; number = painting order</text>']
    for mark in marks:
        x, y = (mark["x"] - left) * 1000, (height - mark["y"] + bottom) * 1000
        theta = np.deg2rad(mark["angle_deg"])
        dx, dy = length_m * 1000 * np.array([np.cos(theta), -np.sin(theta)])
        lines.extend([f'<path d="M {x} {y} l {dx} {dy}" stroke="black" stroke-width="1"/>',
                      f'<circle cx="{x}" cy="{y}" r="1.5" fill="none" stroke="#b45309" stroke-width=".5"/>',
                      f'<text x="{x-3}" y="{y-5}" font-size="3.5">{mark["stroke"]}</text>'])
    lines.append('</svg>')
    path.write_text("\n".join(lines))


def run_sheets(args, painter, stroke, report, directory, recording, snapshot,
               execute_stroke, prompt=input, clock=time.monotonic):
    for sheet in report["sheets"]:
        number = sheet["sheet"]
        painter.to_neutral()
        recording.event("paper_wait_start", sheet=number)
        start = clock()
        prompt(f"\nSheet {number}/{len(report['sheets'])}: place FRESH paper at the same canvas position. "
               "Black paint in slot 0; water/rag ready. Press Enter when hands are clear (Ctrl-C cancels): ")
        sheet["paper_wait_seconds"] = clock() - start
        recording.event("paper_ready", sheet=number, wait_seconds=sheet["paper_wait_seconds"])
        snapshot(f"sheet-{number:02d}-before")
        sheet["status"] = "running"
        save_report(directory / "test.json", report)
        recording.event("clean_start", sheet=number, passes=args.wash_passes)
        with repeatable_loading(args.seed):
            for _ in range(args.wash_passes):
                painter.clean_paint_brush()
        wash_end = clock()
        recording.event("clean_end", sheet=number)
        for mark in sheet["strokes"]:
            if mark["stroke_in_load"] == 1:
                recording.event("reload_start", stroke=mark["stroke"], palette_slot=0)
                with repeatable_loading(args.seed):
                    painter.get_paint(0)
                load_end = clock()
                recording.event("reload_end", stroke=mark["stroke"])
            report["in_progress"] = mark["stroke"]
            save_report(directory / "test.json", report)
            print(f"Sheet {number}: {mark['stroke_on_sheet']}/{len(sheet['strokes'])}, {mark['region']} {mark['direction']}, "
                  f"{mark['stroke_in_load']}/4 in load", flush=True)
            recording.event("stroke_start", **mark, seconds_since_load=clock() - load_end,
                            seconds_since_wash=clock() - wash_end)
            if execute_stroke(stroke, painter, mark["x"], mark["y"], np.deg2rad(mark["angle_deg"])) is False:
                raise RuntimeError("Stroke was not executed; repeatability test is incomplete")
            sheet["completed"] += 1
            report["completed_strokes"] += 1
            report["in_progress"] = None
            save_report(directory / "test.json", report)
            recording.event("stroke_end", **mark, seconds_since_load=clock() - load_end,
                            seconds_since_wash=clock() - wash_end)
        painter.to_neutral()
        snapshot(f"sheet-{number:02d}-after")
        sheet["status"] = "completed"
        save_report(directory / "test.json", report)


def main(argv=None):
    args = arguments(argv)
    if args.analyze:
        from repeatability_analysis import analyze
        analyze(args.analyze.resolve())
        return 0
    opt, params, grid, h, xy = load_setup(args)
    if args.split_directions:
        layout = []
        for directions in (DIRECTIONS[:2], DIRECTIONS[2:]):
            for mark in make_layout(opt, grid, args.length_mm / 1000, xy, directions):
                layout.append({**mark, "slot": len(layout) + 1})
    else:
        layout = make_layout(opt, grid, args.length_mm / 1000, xy)
    report = base_report(args, params, xy, Path(__file__))
    report.update(make_protocol(args, layout))
    directory = args.output_dir / (datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid4().hex[:8])
    directory.mkdir(parents=True, exist_ok=False)
    for sheet in report["sheets"]:
        save_preview(directory / f"sheet-{sheet['sheet']:02d}-layout.svg", opt, sheet["strokes"], args.length_mm / 1000)
    save_report(directory / "test.json", report)
    per_sheet = len(report['sheets'][0]['strokes'])
    print(f"{len(report['sheets'])} fresh sheets x {per_sheet} strokes = {args.repeats*12} marks; {args.length_mm:g} mm each.")
    print("Left/center/right; horizontal, vertical, and both diagonals. Identical positions on each sheet; shifted order.")
    print(f"Black only. Reload every 4; wash every {per_sheet} (before each sheet), {args.wash_passes} water/rag cycle(s).")
    print(f"Commanded compression: middle {report['compression_mm']['middle']:.2f} mm; endpoints "
          f"{report['compression_mm']['endpoints']:.2f} mm. This does not measure force.")
    print(f"Layouts/settings: {directory}")
    if not args.execute:
        print("Preview only; no robot/camera connection or IK check. Add --execute to paint and record.")
        return 0
    input(STARTUP_PROMPT)
    execute_test(args, opt, params, grid, h, xy, directory, report, sequence_runner=run_sheets)
    from repeatability_analysis import analyze
    analyze(directory)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except StartupPreflightError as exc:
        print(f"\nStopped: {exc}")
        raise SystemExit(2)
    except (KeyboardInterrupt, EOFError):
        print("\nStopped. Keep partial results for inspection; rerun on fresh sheets, not over existing marks.")
        raise SystemExit(130)
