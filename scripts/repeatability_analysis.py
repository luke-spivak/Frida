"""Image-only measurements for paint_repeatability; never connects to hardware."""

if __package__:
    from ._bootstrap import ROOT, SOURCE_DIR
else:
    from _bootstrap import ROOT, SOURCE_DIR


import json
from pathlib import Path

import cv2
import numpy as np

from paint_loading import save_report


def assign_paint(before, after, marks, materials, length_mm):
    """Assign whole paint components once per sheet, before making any crops.

    A 12 mm search allowance accommodates displaced marks. Median distance to
    each finite requested path groups broken pieces without including nearby
    strokes merely because their bounding rectangles overlap. Unclear or joined
    components are flagged rather than treated as reliable measurements.
    """
    if before is None or after is None or before.shape != after.shape or before.ndim != 3:
        raise ValueError("Before/after snapshots must be matching color images")
    if not marks:
        return []
    height, width = after.shape[:2]
    sx, sy = materials["CANVAS_WIDTH_M"] * 1000 / width, materials["CANVAS_HEIGHT_M"] * 1000 / height
    left = (materials["CANVAS_POSITION"][0] - materials["CANVAS_WIDTH_M"] / 2) * 1000
    top = (materials["CANVAS_POSITION"][1] + materials["CANVAS_HEIGHT_M"]) * 1000
    pre = cv2.cvtColor(before, cv2.COLOR_BGR2GRAY).astype(np.int16)
    post = cv2.cvtColor(after, cv2.COLOR_BGR2GRAY).astype(np.int16)
    dark = ((pre-post >= 35) & (post <= 180)).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(dark, connectivity=8)
    starts = np.array([[m["x"], m["y"]] for m in marks]) * 1000
    angles = np.deg2rad([m["angle_deg"] for m in marks])
    tangents = np.column_stack((np.cos(angles), np.sin(angles)))
    owners = np.full(n, -1, dtype=int)
    ambiguous = np.zeros(len(marks), dtype=bool)
    for label in range(1, n):
        x, y, w, h, area = stats[label]
        if area < max(3, .05/(sx*sy)):
            continue
        yy, xx = np.nonzero(labels[y:y+h, x:x+w] == label)
        points = np.column_stack((left+(xx+x+.5)*sx, top-(yy+y+.5)*sy))
        relative = points[:, None, :] - starts
        along = np.einsum('nki,ki->nk', relative, tangents)
        closest = np.clip(along, 0, length_mm)[..., None] * tangents
        distances = np.linalg.norm(relative-closest, axis=-1)
        scores = np.median(distances, axis=0)
        order = np.argsort(scores)
        winner = order[0]
        if scores[winner] > 12:
            continue
        owners[label] = winner
        if np.mean(distances[:, winner] > 12) > .2:
            ambiguous[winner] = True
        # Flag close alternatives and components that may join two marks.
        if len(marks) > 1 and (scores[order[1]]-scores[winner] < 2
                or np.mean(np.argmin(distances, axis=1) != winner) > .2):
            ambiguous[winner] = True
            ambiguous[order[1]] = True
    return [(owners[labels] == i, bool(ambiguous[i])) for i in range(len(marks))]


def measure_mark(before, after, mark, materials, length_mm, assignment=None):
    """Measure newly darkened pixels near a target, not force or TCP motion.

    The visible leading edge includes the brush footprint; it need not coincide
    with the commanded start. Compare its spread across repeats. Camera/XY
    calibration, paper placement, glare, and thresholding affect these estimates.
    """
    if before is None or after is None or before.shape != after.shape or before.ndim != 3:
        raise ValueError("Before/after snapshots must be matching color images")
    height, width = after.shape[:2]
    sx, sy = materials["CANVAS_WIDTH_M"] * 1000 / width, materials["CANVAS_HEIGHT_M"] * 1000 / height
    left = (materials["CANVAS_POSITION"][0] - materials["CANVAS_WIDTH_M"] / 2) * 1000
    top = (materials["CANVAS_POSITION"][1] + materials["CANVAS_HEIGHT_M"]) * 1000
    start = np.array([mark["x"], mark["y"]]) * 1000
    theta = np.deg2rad(mark["angle_deg"])
    tangent = np.array([np.cos(theta), np.sin(theta)])
    normal = np.array([-tangent[1], tangent[0]])
    end = start + length_mm * tangent
    lo, hi = np.minimum(start, end) - 6, np.maximum(start, end) + 6
    x0, x1 = max(0, int(np.floor((lo[0] - left) / sx))), min(width, int(np.ceil((hi[0] - left) / sx)))
    y0, y1 = max(0, int(np.floor((top - hi[1]) / sy))), min(height, int(np.ceil((top - lo[1]) / sy)))
    if x1 <= x0 or y1 <= y0:
        raise ValueError("Mark ROI falls outside the canvas image")
    full_mask, ambiguous = (assign_paint(before, after, [mark], materials, length_mm)[0]
                            if assignment is None else assignment)
    ys, xs = np.nonzero(full_mask)
    if len(xs):
        # Include the full assigned mark plus 1 mm for visual inspection; never
        # move the target reference to hide a genuine placement error.
        px, py = int(np.ceil(1/sx)), int(np.ceil(1/sy))
        x0, x1 = min(x0, max(0, xs.min()-px)), max(x1, min(width, xs.max()+px+1))
        y0, y1 = min(y0, max(0, ys.min()-py)), max(y1, min(height, ys.max()+py+1))
    crop = after[y0:y1, x0:x1].copy()
    mask = full_mask[y0:y1, x0:x1].astype(np.uint8)
    ys, xs = np.nonzero(mask)
    result = {key: mark[key] for key in ("slot", "sheet", "stroke", "stroke_in_load", "region", "direction")}
    result.update({"length_mm": None, "width_mm": None, "start_offset_xy_mm": None,
                   "painted_area_mm2": float(len(xs) * sx * sy), "path_coverage_pct": 0.,
                   "roi_clipped": bool(mask[0].any() or mask[-1].any() or mask[:, 0].any() or mask[:, -1].any()),
                   "assignment_ambiguous": ambiguous,
                   "detected": bool(len(xs))})
    if len(xs):
        relative = np.column_stack([left + (xs + x0 + .5) * sx, top - (ys + y0 + .5) * sy]) - start
        along, across = relative @ tangent, relative @ normal
        low, high = np.percentile(along, [2, 98])
        side_low, side_high = np.percentile(across, [2, 98])
        leading_across = float(np.median(across[along <= low + 1.]))
        result.update({"length_mm": float(high - low), "width_mm": float(side_high - side_low),
                       "start_offset_xy_mm": (low * tangent + leading_across * normal).tolist()})
        # Fraction of the commanded path's ~1 mm bins containing dark paint
        # within 6 mm of its centerline. This is not optical opacity or force.
        bins = max(1, int(np.ceil(length_mm)))
        valid = (along >= 0) & (along < length_mm) & (abs(across) <= 6)
        occupied = np.unique(np.floor(along[valid] / length_mm * bins))
        result["path_coverage_pct"] = float(100 * len(occupied) / bins)
    def pixel(point):
        return tuple(np.rint([(point[0] - left) / sx - x0, (top - point[1]) / sy - y0]).astype(int))
    cv2.line(crop, pixel(start), pixel(end), (220, 180, 0), 1)
    cv2.drawMarker(crop, pixel(start), (0, 0, 220), cv2.MARKER_CROSS, 9, 1)
    return result, crop, mask * 255


def summarize(measurements):
    groups = []
    for slot in sorted({m["slot"] for m in measurements}):
        all_marks = [m for m in measurements if m["slot"] == slot]
        usable = [m for m in all_marks if not m["roi_clipped"] and not m.get("assignment_ambiguous", False)]
        marks = [m for m in usable if m["detected"]]
        item = {"slot": slot, "region": all_marks[0]["region"], "direction": all_marks[0]["direction"],
                "detected_repeats": sum(m["detected"] for m in all_marks),
                "valid_repeats": len(marks), "total_repeats": len(all_marks), "start_spread_mm": None}
        if len(marks) > 1:
            starts = np.array([m["start_offset_xy_mm"] for m in marks])
            item["start_spread_mm"] = float(np.linalg.norm(starts[:, None] - starts[None, :], axis=-1).max())
        for key in ("length_mm", "width_mm", "painted_area_mm2", "path_coverage_pct"):
            values = [m[key] for m in usable if m[key] is not None]
            item[key] = ({"mean": float(np.mean(values)), "min": min(values), "max": max(values)} if values else None)
        groups.append(item)
    return groups


def analyze(directory):
    directory = Path(directory)
    report = json.loads((directory / "test.json").read_text())
    if report.get("experiment") != "repeatability":
        raise ValueError("Expected a repeatability test report")
    sheets = [s for s in report["sheets"] if s["status"] == "completed"]
    if not sheets:
        raise ValueError("No completed sheets with after-images to analyze")
    measurements, panels = [], {}
    for sheet in sheets:
        prefix = directory / f"sheet-{sheet['sheet']:02d}"
        before = cv2.imread(str(prefix) + "-before-canvas.png")
        after = cv2.imread(str(prefix) + "-after-canvas.png")
        assigned = assign_paint(before, after, sheet["strokes"], report["materials"], report["settings"]["length_mm"])
        for mark, assignment in zip(sheet["strokes"], assigned):
            measured, crop, mask = measure_mark(before, after, mark, report["materials"], report["settings"]["length_mm"], assignment)
            measurements.append(measured)
            panels[mark["slot"], mark["sheet"]] = (measured, crop, mask)
    summary = summarize(measurements)
    save_report(directory / "measurements.json", {
        "analysis_version": 2,
        "method": "New dark pixels: before minus after >=35, after <=180. Whole components assigned to finite target paths "
                  "within 12 mm before expanding crops; ambiguous assignments excluded from repeat summaries. "
                  "Length/width: 2nd-98th percentile. "
                  "Coverage: percent of expected 1 mm path bins with paint within 6 mm. Units from cached canvas dimensions.",
        "limitations": "Visible mark edges, not TCP positions or pressure. Inspect masks: wet glare, registration error, "
                        "thresholds, cropped spill, and paper movement can bias measurements. Directions use different rows.",
        "marks": measurements, "same_position_repeats": summary})
    lines = ["Same-position repeats (mm; min-max across sheets). Visible edges, not brush-center/force measurements.",
             "Region Direction        Seen/valid Start spread  Length       Width        Area(mm2)   Path coverage"]
    def value_range(item, key):
        values = item[key]
        return "n/a" if values is None else f"{values['min']:.1f}-{values['max']:.1f}"
    for item in summary:
        spread = "n/a" if item["start_spread_mm"] is None else f"{item['start_spread_mm']:.2f}"
        lines.append(f"{item['region']:<6} {item['direction']:<16} "
                     f"{item['detected_repeats']}/{item['total_repeats']} valid={item['valid_repeats']}  {spread:>8}      "
                     f"{value_range(item, 'length_mm'):<12} {value_range(item, 'width_mm'):<12} "
                     f"{value_range(item, 'painted_area_mm2'):<11} {value_range(item, 'path_coverage_pct')}%")
    lines.append("Inspect comparison.png masks; ROI-edge flags and per-mark values are in measurements.json.")
    (directory / "summary.txt").write_text("\n".join(lines) + "\n")
    cell_w, cell_h = 330, 145
    image = np.full((55 + 12 * cell_h, len(sheets) * cell_w, 3), 255, np.uint8)
    cv2.putText(image, "After / detected paint. Red cross = target start. Cyan = target path.",
                (8, 20), cv2.FONT_HERSHEY_SIMPLEX, .45, (0, 0, 0), 1)
    for column, sheet in enumerate(sheets):
        cv2.putText(image, f"Sheet {sheet['sheet']}", (column * cell_w + 8, 43), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 0, 0), 1)
        for slot in range(1, 13):
            if (slot, sheet["sheet"]) not in panels:
                continue  # Split-direction sheets contain only six of the twelve slots.
            measured, crop, mask = panels[slot, sheet["sheet"]]
            x, y = column * cell_w, 55 + (slot - 1) * cell_h
            label = f"{measured['stroke']}: {measured['region']} {measured['direction']}"
            cv2.putText(image, label, (x + 5, y + 15), cv2.FONT_HERSHEY_SIMPLEX, .4, (0, 0, 0), 1)
            for k, thumbnail in enumerate((crop, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR))):
                scale = min(155 / thumbnail.shape[1], 88 / thumbnail.shape[0])
                resized = cv2.resize(thumbnail, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
                image[y+22:y+22+resized.shape[0], x+5+k*165:x+5+k*165+resized.shape[1]] = resized
            label = (f"L {measured['length_mm']:.1f} W {measured['width_mm']:.1f} mm; path {measured['path_coverage_pct']:.0f}%"
                     if measured["detected"] else "No dark mark detected")
            if measured["roi_clipped"]:
                label += " [ROI edge]"
            if measured.get("assignment_ambiguous"):
                label += " [ambiguous]"
            cv2.putText(image, label, (x + 5, y + 133), cv2.FONT_HERSHEY_SIMPLEX, .37, (0, 0, 0), 1)
    if not cv2.imwrite(str(directory / "comparison.png"), image):
        raise OSError("Could not save comparison.png")
    print(f"Comparison image: {directory / 'comparison.png'}")
    print(f"Measurements: {directory / 'measurements.json'}")
    print(f"Readable comparison: {directory / 'summary.txt'}")
    print("Check the masks before interpreting numbers; wet glare and paper placement affect these estimates.")
    return measurements
