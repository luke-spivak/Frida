"""Loading protocol and fail-closed geometry, using synthetic data/fake hardware."""

import json
import pickle
import sys
from types import SimpleNamespace
from unittest.mock import Mock, create_autospec

import numpy as np
import pytest

from canvas_z import CanvasZCalibration
from paint_loading import (arguments, load_setup, main, make_layout, make_painter,
                           make_stroke, repeatable_loading, run_sequences,
                           save_snapshot, sequence_layout)


# Saved round-brush XY correction that pushed the original top row outside the
# grid's 6 mm margin. A fixture constant avoids depending on live user caches.
ROUND_BRUSH_XY = np.array([
    [1.1109017405857147, -.047286361540405795, .018087216186335867],
    [.11502974578061907, 1.167645988288315, -.017119661618037844],
    [.23662873132875792, .24959630819822945, 1.],
])


@pytest.fixture
def setup(tmp_path):
    materials = dict(CANVAS_POSITION=[-.011450742868149252, .2744202253733551],
                     CANVAS_WIDTH_M=.216, CANVAS_HEIGHT_M=.139, MIN_STROKE_LENGTH=.005,
                     MAX_STROKE_LENGTH=.06, MIN_STROKE_Z=.1, MAX_BEND=.02, MAX_ALPHA=0,
                     WATER_POSITION=[-.2129, .1931], RAG_POSTITION=[-.2601, .1122],
                     PALLETTE_POSITION=[-.2136, .27739], PAINT_DIFFERENCE=.03976)
    opt = SimpleNamespace(**materials, robot="so101")
    grid = CanvasZCalibration(opt, [[-.0015, -.0025, -.0025], [.0025, .0005, .0025],
                                  [.0075, .0055, .0055]])
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "brush_tip_to_table_calib.json").write_text(json.dumps({
        "Z_CANVAS": -.003, "Z_MAX_CANVAS": -.015, "canvas_z_grid": grid.to_dict()}))
    (cache / "cached_H_canvas.pkl").write_bytes(pickle.dumps(np.eye(3)))
    path = tmp_path / "materials.json"
    path.write_text(json.dumps(materials))
    args = arguments(["--materials_json", str(path), "--cache_dir", str(cache),
                      "--output-dir", str(tmp_path / "output")])
    return args


def test_preview_requires_no_hardware_camera_torch_or_matplotlib(setup, monkeypatch):
    for name in ("robot", "painter", "torch", "matplotlib", "cv2", "camera.innomaker"):
        monkeypatch.setitem(sys.modules, name, None)
    before = {p.name: p.read_bytes() for p in setup.cache_dir.iterdir()}
    assert main(["--materials_json", str(setup.materials_json), "--cache_dir", str(setup.cache_dir),
                 "--output-dir", str(setup.output_dir)]) == 0
    report_path, = setup.output_dir.glob("*/test.json")
    report = json.loads(report_path.read_text())
    assert report["status"] == "preview"
    assert report["compression_mm"] == pytest.approx({"middle": 3., "endpoints": 2.4})
    assert len(report["sequences"]) == 3
    assert len(list(report_path.parent.glob("*-layout.svg"))) == 3
    assert {p.name: p.read_bytes() for p in setup.cache_dir.iterdir()} == before


def test_layout_orders_reuse_positions_without_overlap(setup):
    opt, _, grid, _, _ = load_setup(setup)
    slots = make_layout(opt, grid, .02)
    assert len(slots) == 10
    sequences = [sequence_layout(slots, i) for i in range(3)]
    assert [sequence[0]["slot"] for sequence in sequences] == [1, 4, 7]
    for sequence in sequences:
        assert [m["stroke"] for m in sequence] == list(range(1, 11))
        assert sorted(m["slot"] for m in sequence) == list(range(1, 11))
    for mark in slots:
        assert mark["command_x"] == mark["x"]
        assert mark["command_y"] == mark["y"]


def test_xy_is_optional_loaded_when_present_and_checked_after_correction(setup):
    matrix = np.array([[1., 0, .002], [0, 1, -.001], [0, 0, 1]])
    (setup.cache_dir / "cached_H_coord.pkl").write_bytes(pickle.dumps(matrix))
    opt, _, grid, _, xy = load_setup(setup)
    marks = make_layout(opt, grid, .02, xy)
    assert all(m["command_x"] == pytest.approx(m["x"] + .002) for m in marks)
    assert all(m["command_y"] == pytest.approx(m["y"] - .001) for m in marks)
    matrix[0, 2] = .3
    with pytest.raises(ValueError, match="does not fit"):
        make_layout(opt, grid, .02, matrix)
    with pytest.raises(ValueError, match="does not fit"):
        make_layout(opt, grid, .02, np.diag([1, .5, 1]) + np.array([[0, 0, 0], [0, 0, .172], [0, 0, 0]]))


def test_round_brush_correction_keeps_full_strokes_inside_grid_and_paper(setup):
    opt, _, grid, _, _ = load_setup(setup)
    original = make_layout(opt, grid, .02)
    corrected = make_layout(opt, grid, .02, ROUND_BRUSH_XY)
    assert len(corrected) == 10
    for plain, mark in zip(original, corrected):
        target = np.array([mark["x"], mark["y"], 1.])
        projected = ROUND_BRUSH_XY @ target
        command = projected[:2] / projected[2]
        np.testing.assert_allclose(command, [plain["command_x"], plain["command_y"]], atol=1e-12)
        np.testing.assert_allclose(command, [mark["command_x"], mark["command_y"]], atol=1e-12)
        assert grid.x_m[0] <= command[0] - .006
        assert command[0] + .02 + .006 <= grid.x_m[-1]
        assert command[1] - .006 - grid.y_m[0] >= .00349
        assert grid.y_m[-1] - command[1] - .006 >= .00349
        assert opt.X_CANVAS_MIN < mark["x"] - .006
        assert mark["x"] + .026 < opt.X_CANVAS_MAX
        assert opt.Y_CANVAS_MIN < mark["y"] - .006
        assert mark["y"] + .006 < opt.Y_CANVAS_MAX
    # Calibration is still used: target XY differs from commanded XY.
    assert abs(corrected[0]["y"] - corrected[0]["command_y"]) > .008


def test_too_small_grid_is_not_bypassed_or_strokes_shortened(setup):
    opt, _, _, _, _ = load_setup(setup)
    small_grid = CanvasZCalibration(opt, np.zeros((3, 3)), inset_m=.03)
    with pytest.raises(ValueError, match="measured height grid.*short by"):
        make_layout(opt, small_grid, .02, ROUND_BRUSH_XY)


def test_correction_cannot_pack_strokes_too_close_on_paper(setup):
    opt, _, grid, _, _ = load_setup(setup)
    # Even a fitting command grid is rejected if its inverse makes the target
    # rows overlap. Preserve both footprint checks during layout fitting.
    center = opt.CANVAS_POSITION[1] + opt.CANVAS_HEIGHT_M / 2
    correction = np.array([[1., 0, 0], [0, 2., -center], [0, 0, 1.]])
    with pytest.raises(ValueError, match="separation"):
        make_layout(opt, grid, .02, correction)


@pytest.mark.parametrize("change,message", [
    ({"Z_MAX_CANVAS": 0}, "below"), ({"Z_CANVAS": float("nan")}, "finite"),
    ({"canvas_z_grid": None}, "Invalid canvas Z"),
])
def test_invalid_calibration_rejected_before_hardware(setup, change, message):
    path = setup.cache_dir / "brush_tip_to_table_calib.json"
    data = json.loads(path.read_text())
    data.update(change)
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match=message):
        load_setup(setup)


@pytest.mark.parametrize("flag,value", [("--length-mm", "nan"), ("--stroke-z", "inf"),
                                        ("--stroke-z", "1"), ("--repeats", "0"),
                                        ("--recording-fps", "0"), ("--seed", "-1")])
def test_invalid_cli_values_fail(flag, value):
    with pytest.raises(SystemExit):
        arguments([flag, value])


def report_for(args):
    opt, _, grid, _, xy = load_setup(args)
    layout = make_layout(opt, grid, args.length_mm / 1000, xy)
    return {"sequences": [{"repeat": i + 1, "status": "pending", "completed": 0,
                            "in_progress": None, "strokes": sequence_layout(layout, i)}
                           for i in range(args.repeats)]}


def test_three_loads_each_have_two_washes_one_reload_and_ten_strokes(setup, tmp_path):
    calls, random_loading = [], []
    painter = SimpleNamespace(
        to_neutral=lambda: calls.append("park"),
        clean_paint_brush=lambda: (calls.append("wash"), random_loading.append(np.random.randn(2))),
        get_paint=lambda slot: (calls.append(("load", slot)), random_loading.append(np.random.randn(2))),
    )
    report = report_for(setup)
    recording = Mock()
    run_sequences(setup, painter, object(), report, tmp_path, recording,
                  lambda name: calls.append(("snapshot", name)),
                  lambda *a: calls.append("stroke"), lambda message: calls.append("paper"))
    assert calls.count("wash") == 6
    assert calls.count(("load", 0)) == 3
    assert calls.count("stroke") == 30
    assert calls.count("paper") == 3
    for i, call in enumerate(calls):
        if call == ("load", 0):
            assert calls[i-2:i] == ["wash", "wash"]
            assert calls[i+1:i+11] == ["stroke"] * 10
    np.testing.assert_allclose(random_loading[:3], random_loading[3:6])
    np.testing.assert_allclose(random_loading[:3], random_loading[6:])
    assert all(trial["completed"] == 10 and trial["status"] == "completed" for trial in report["sequences"])
    assert sum(c.args[0] == "stroke_end" for c in recording.event.call_args_list) == 30
    assert json.loads((tmp_path / "test.json").read_text()) == report


@pytest.mark.parametrize("failure", [TimeoutError("servo failed"), KeyboardInterrupt(), False])
def test_failed_stroke_never_retries_reloads_or_records_completion(setup, tmp_path, failure):
    painter, recording = Mock(), Mock()
    report = report_for(setup)
    attempts = []

    def stroke(*args):
        attempts.append(args)
        if len(attempts) == 3:
            if failure is False:
                return False
            raise failure

    with pytest.raises((TimeoutError, KeyboardInterrupt, RuntimeError)):
        run_sequences(setup, painter, object(), report, tmp_path, recording, Mock(), stroke, Mock())
    assert len(attempts) == 3
    assert painter.get_paint.call_count == 1
    assert painter.clean_paint_brush.call_count == 2
    assert painter.to_neutral.call_count == 1  # No attempted recovery motion.
    saved = json.loads((tmp_path / "test.json").read_text())
    assert saved["sequences"][0]["completed"] == 2
    assert saved["sequences"][0]["in_progress"] == 3
    assert saved["sequences"][1]["status"] == "pending"


def test_random_seed_restored_even_on_loading_failure():
    state = np.random.get_state()
    with pytest.raises(RuntimeError):
        with repeatable_loading(101):
            np.random.randn(20)
            raise RuntimeError()
    restored = np.random.get_state()
    assert state[0] == restored[0]
    np.testing.assert_array_equal(state[1], restored[1])
    assert state[2:] == restored[2:]


def test_snapshot_saves_raw_and_rectified_pngs(setup, tmp_path):
    import cv2
    from camera.recording import CameraRecording
    opt, _, _, _, _ = load_setup(setup)
    frame = np.full((100, 200, 3), 255, np.uint8)
    frame[30:40, 40:60] = 0
    recording = create_autospec(CameraRecording, instance=True)
    recording.read.return_value = frame
    save_snapshot(recording, np.eye(3), opt, tmp_path, "load-01-after")
    raw = cv2.imread(str(tmp_path / "load-01-after-overhead.png"))
    canvas = cv2.imread(str(tmp_path / "load-01-after-canvas.png"))
    np.testing.assert_array_equal(raw, frame)
    np.testing.assert_array_equal(canvas, frame[:, :155])


def test_production_stroke_uses_grid_xy_and_depth_without_model_or_initializer(setup, monkeypatch):
    from brush_stroke import BrushStroke
    from painter import Painter
    monkeypatch.setattr(Painter, "__init__", lambda *a: pytest.fail("Must not initialize Painter"))
    opt, params, grid, _, _ = load_setup(setup)
    robot = Mock()
    xy = np.array([[1., 0, .002], [0, 1, -.001], [0, 0, 1]])
    painter = make_painter(opt, params, grid, robot, xy)
    stroke = make_stroke(setup)
    mark = make_layout(opt, grid, .02, xy)[0]
    BrushStroke.execute(stroke, painter, mark["x"], mark["y"], 0.)
    requests = robot.go_to_cartesian_pose.call_args_list
    # Four production stages: approach, descend, sampled contact/lift, final lift.
    assert len(requests) == 4
    assert requests[0].kwargs == {"travel_z": .097}
    points = np.asarray(requests[2].args[0])
    assert points[0, 0] == pytest.approx(mark["command_x"])
    assert points[0, 1] == pytest.approx(mark["command_y"])
    assert points[0, 2] == pytest.approx(grid.contact_z(*points[0, :2]) - .2 * .012)
    assert points[4, 2] == pytest.approx(grid.contact_z(*points[4, :2]) - .25 * .012)
    assert points[-1, 0] == pytest.approx(mark["command_x"] + .02)


@pytest.mark.parametrize("stage", ["camera", "connect", "stroke", "success"])
def test_live_lifecycle_closes_video_and_retains_torque_on_error(setup, tmp_path, monkeypatch, stage):
    import paint_loading
    from camera.innomaker import WebCam
    from robot import SO101
    from pathlib import Path
    camera, robot, recording = Mock(spec=WebCam), Mock(spec=SO101), Mock()
    robot.arm = SimpleNamespace(bus=Mock(is_connected=False))
    camera.start_recording.return_value = recording
    if stage == "camera":
        camera.start_recording.side_effect = RuntimeError("camera failed")

    def connect():
        robot.arm.bus.is_connected = True
        if stage == "connect":
            raise RuntimeError("connection setup failed")

    robot.good_morning_robot.side_effect = connect
    robot_factory = Mock(return_value=robot)
    monkeypatch.setattr("camera.innomaker.WebCam", Mock(return_value=camera))
    monkeypatch.setattr("robot.SO101", robot_factory)
    monkeypatch.setattr(paint_loading, "save_snapshot", Mock())
    # Startup's real read-only bus/planner path has its own regression tests.
    monkeypatch.setattr(paint_loading, "preflight_start", Mock())
    monkeypatch.setattr(paint_loading, "run_sequences", Mock(
        side_effect=RuntimeError("stroke failed") if stage == "stroke" else None))
    args = setup
    opt, params, grid, h, xy = load_setup(args)
    report = report_for(args)
    previous_cwd = Path.cwd()
    if stage == "success":
        paint_loading.execute_test(args, opt, params, grid, h, xy, tmp_path, report)
    else:
        with pytest.raises(RuntimeError):
            paint_loading.execute_test(args, opt, params, grid, h, xy, tmp_path, report)
    assert Path.cwd() == previous_cwd
    assert report["status"] == ("completed" if stage == "success" else "error")
    camera.close.assert_called_once_with(reason=report["status"])
    if stage == "camera":
        robot_factory.assert_not_called()
    else:
        robot.arm.bus.disconnect.assert_called_once_with(disable_torque=False)
    robot.good_night_robot.assert_not_called()
    robot.go_to_cartesian_pose.assert_not_called()
