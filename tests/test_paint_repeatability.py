"""Repeatability geometry, protocol, and image metrics; no physical hardware."""

import json
import pickle
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import cv2
import numpy as np
import pytest

from paint_loading import load_setup
from paint_repeatability import arguments, main, make_layout, make_protocol, run_sheets
from repeatability_analysis import analyze, measure_mark, summarize
from test_paint_loading import setup, ROUND_BRUSH_XY


def repeatability_args(setup, *extra):
    return arguments(["--materials_json", str(setup.materials_json), "--cache_dir", str(setup.cache_dir),
                      "--output-dir", str(setup.output_dir), *extra])


def report_for(args):
    opt, _, grid, _, xy = load_setup(args)
    return make_protocol(args, make_layout(opt, grid, args.length_mm / 1000, xy))


def test_preview_offline_without_modifying_calibration(setup, monkeypatch):
    for name in ("robot", "painter", "camera.innomaker", "cv2", "torch", "matplotlib"):
        monkeypatch.setitem(sys.modules, name, None)
    before = {p.name: p.read_bytes() for p in setup.cache_dir.iterdir()}
    assert main(["--materials_json", str(setup.materials_json), "--cache_dir", str(setup.cache_dir),
                 "--output-dir", str(setup.output_dir)]) == 0
    path, = setup.output_dir.glob("*/test.json")
    report = json.loads(path.read_text())
    assert report["paint_care"] == dict(reload_every=4, clean_every=12, wash_passes=1)
    assert len(list(path.parent.glob("*-layout.svg"))) == 3
    assert {p.name: p.read_bytes() for p in setup.cache_dir.iterdir()} == before


def test_long_split_sheets_repeat_positions_record_and_analyze(setup):
    args = repeatability_args(setup, "--length-mm", "32", "--split-directions", "--repeats", "2")
    assert main(["--materials_json", str(args.materials_json), "--cache_dir", str(args.cache_dir),
                 "--output-dir", str(args.output_dir), "--length-mm", "32",
                 "--split-directions", "--repeats", "2"]) == 0
    path, = args.output_dir.glob("*/test.json")
    report = json.loads(path.read_text())
    assert len(report["sheets"]) == 4
    assert all(len(s["strokes"]) == 6 for s in report["sheets"])
    marks = [m for s in report["sheets"] for m in s["strokes"]]
    assert [m["stroke"] for m in marks] == list(range(1, 25))
    for slot in range(1, 13):
        pair = [m for m in marks if m["slot"] == slot]
        assert len(pair) == 2
        assert len({(m["x"], m["y"], m["angle_deg"]) for m in pair}) == 1
        assert pair[0]["stroke_in_load"] != pair[1]["stroke_in_load"]
    painter = SimpleNamespace(to_neutral=Mock(), clean_paint_brush=Mock(), get_paint=Mock())
    execute = Mock(return_value=True)
    def snapshot(name):
        cv2.imwrite(str(path.parent / f"{name}-canvas.png"), np.full((139, 216, 3), 255, np.uint8))
    run_sheets(args, painter, object(), report, path.parent, Mock(), snapshot,
               execute, prompt=lambda _: None)
    assert execute.call_count == 24
    assert painter.clean_paint_brush.call_count == 4
    assert painter.get_paint.call_count == 8
    assert len(analyze(path.parent)) == 24
    assert cv2.imread(str(path.parent / "comparison.png")) is not None


def test_32mm_all_directions_fit_one_sheet_with_67mm_brush_xy(setup, monkeypatch):
    correction = np.array([[.97732774, -.05603982, .02027078],
                           [.11799699, .93978824, .01919113],
                           [.34128479, -.07051059, 1.]])
    (setup.cache_dir / "cached_H_coord.pkl").write_bytes(pickle.dumps(correction))
    for name in ("robot", "painter", "camera.innomaker", "torch"):
        monkeypatch.setitem(sys.modules, name, None)
    assert main(["--materials_json", str(setup.materials_json), "--cache_dir", str(setup.cache_dir),
                 "--output-dir", str(setup.output_dir), "--length-mm", "32", "--repeats", "1"]) == 0
    path, = setup.output_dir.glob("*/test.json")
    report = json.loads(path.read_text())
    assert len(report["sheets"]) == 1
    marks = report["sheets"][0]["strokes"]
    assert len(marks) == 12
    for region in ("left", "center", "right"):
        assert {m["angle_deg"] for m in marks if m["region"] == region} == {0, 90, 45, -45}
    opt, _, grid, _, _ = load_setup(setup)
    limits = [np.array([grid.x_m[0], grid.y_m[0], grid.x_m[-1], grid.y_m[-1]]),
              np.array([opt.CANVAS_POSITION[0] - opt.CANVAS_WIDTH_M / 2, opt.CANVAS_POSITION[1],
                        opt.CANVAS_POSITION[0] + opt.CANVAS_WIDTH_M / 2,
                        opt.CANVAS_POSITION[1] + opt.CANVAS_HEIGHT_M])]
    for space, keys in enumerate((("command_x", "command_y"), ("x", "y"))):
        boxes = []
        for mark in marks:
            start = np.array([mark[k] for k in keys])
            angle = np.deg2rad(mark["angle_deg"])
            end = start + .032 * np.array([np.cos(angle), np.sin(angle)])
            low, high = np.minimum(start, end) - .006, np.maximum(start, end) + .006
            assert np.all(low >= limits[space][:2]) and np.all(high <= limits[space][2:])
            assert all(np.any(high <= old_low) or np.any(old_high <= low) for old_low, old_high in boxes)
            boxes.append((low, high))


@pytest.mark.parametrize("xy", [None, ROUND_BRUSH_XY])
def test_same_positions_all_directions_and_varied_load_age(setup, xy):
    args = repeatability_args(setup)
    opt, _, grid, _, _ = load_setup(args)
    layout = make_layout(opt, grid, .016, xy)
    report = make_protocol(args, layout)
    assert len(layout) == 12
    for slot in range(1, 13):
        marks = [m for s in report["sheets"] for m in s["strokes"] if m["slot"] == slot]
        assert len({(m["x"], m["y"], m["angle_deg"]) for m in marks}) == 1
        assert len({m["stroke_in_load"] for m in marks}) == 3
    assert {m["angle_deg"] for m in layout} == {0, 90, 45, -45}
    assert {m["region"] for m in layout} == {"left", "center", "right"}
    for mark in layout:
        start = np.array([mark["x"], mark["y"], 1.])
        command = start if xy is None else xy @ start
        np.testing.assert_allclose(command[:2] / command[2], [mark["command_x"], mark["command_y"]])
        theta = np.deg2rad(mark["angle_deg"])
        for f in np.linspace(0, 1, 20):
            point = command[:2] / command[2] + f * .016 * np.array([np.cos(theta), np.sin(theta)])
            assert grid.x_m[0] + .006 <= point[0] <= grid.x_m[-1] - .006
            assert grid.y_m[0] + .006 <= point[1] <= grid.y_m[-1] - .006


def test_layout_rejects_nonfitting_length_or_correction(setup):
    opt, _, grid, _, _ = load_setup(setup)
    with pytest.raises(ValueError, match="fit"):
        make_layout(opt, grid, .06)
    with pytest.raises(ValueError, match="fit"):
        make_layout(opt, grid, .016, np.array([[1, 0, .2], [0, 1, 0], [0, 0, 1.]]))
    with pytest.raises(ValueError, match="separation"):
        make_layout(opt, grid, .016, np.array([[1, 0, 0], [0, 2, -.34], [0, 0, 1.]]))


def test_care_counters_and_rotations_with_no_retries(setup, tmp_path):
    args = repeatability_args(setup)
    report = report_for(args)
    done, washes, loads, offsets = [], [], [], []
    painter = SimpleNamespace(to_neutral=Mock(),
                              clean_paint_brush=lambda: washes.append(len(done)),
                              get_paint=lambda slot: (loads.append(len(done)), offsets.append(np.random.randn(6))))
    recording, snapshot = Mock(), Mock()
    run_sheets(args, painter, object(), report, tmp_path, recording, snapshot,
               lambda *a: done.append(a), prompt=lambda _: None)
    assert washes == [0, 12, 24]
    assert loads == list(range(0, 36, 4))
    assert report["completed_strokes"] == 36 and report["in_progress"] is None
    assert {round(np.rad2deg(a[-1])) for a in done} == {0, 90, 45, -45}
    assert snapshot.call_count == 6
    for offset in offsets:
        np.testing.assert_array_equal(offset, offsets[0])
    # Explicit failure does not mark completion or proceed to another stroke.
    report = report_for(args)
    execute = Mock(side_effect=[None, False])
    with pytest.raises(RuntimeError, match="incomplete"):
        run_sheets(args, painter, object(), report, tmp_path, recording, snapshot, execute, prompt=lambda _: None)
    assert execute.call_count == 2
    assert report["completed_strokes"] == 1 and report["in_progress"] == 2


@pytest.mark.parametrize("angle", [0, 90, 45, -45])
def test_visible_mark_metrics_are_orientation_consistent_and_detect_gaps(angle):
    materials = {"CANVAS_POSITION": [0., 0.], "CANVAS_WIDTH_M": .1, "CANVAS_HEIGHT_M": .1}
    mark = dict(slot=1, sheet=1, stroke=1, stroke_in_load=1, region="center", direction="test",
                x=-.01, y=.04, angle_deg=angle)
    before = np.full((1000, 1000, 3), 255, np.uint8)
    yy, xx = np.mgrid[:1000, :1000]
    theta = np.deg2rad(angle)
    x, y = (xx + .5) / 10 - 50 + 10, 100 - (yy + .5) / 10 - 40
    along, across = x * np.cos(theta) + y * np.sin(theta), -x * np.sin(theta) + y * np.cos(theta)
    mask = (along >= 0) & (along <= 16) & (abs(across) <= 1.5)
    after = before.copy()
    after[mask] = 0
    full, _, _ = measure_mark(before, after, mark, materials, 16)
    assert full["length_mm"] == pytest.approx(15.36, abs=.2)
    assert full["width_mm"] == pytest.approx(2.88, abs=.2)
    assert full["painted_area_mm2"] == pytest.approx(48., abs=1.)
    assert full["path_coverage_pct"] == 100.
    assert np.linalg.norm(full["start_offset_xy_mm"]) < .6
    after[(along >= 5) & (along < 9)] = 255
    broken, _, _ = measure_mark(before, after, mark, materials, 16)
    assert broken["path_coverage_pct"] == 75.
    assert broken["painted_area_mm2"] < full["painted_area_mm2"]
    empty, _, _ = measure_mark(before, before, mark, materials, 16)
    assert not empty["detected"] and empty["length_mm"] is None
    assert empty["path_coverage_pct"] == 0
    comparison = summarize([full, broken, empty])[0]
    assert comparison["detected_repeats"] == 2
    assert comparison["path_coverage_pct"]["min"] == 0
