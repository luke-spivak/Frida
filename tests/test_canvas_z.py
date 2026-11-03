"""Z compensation, manual collection, and cache integration without hardware."""

import copy
import json
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

import canvas_z
from canvas_z import CanvasZCalibration, HEIGHT_FILENAME, grid_axes, read_height_calibration, save_grid
from calibrate_canvas_z import CalibrationCancelled, collect_contact_grid, main, point_order


@pytest.fixture
def opt():
    return SimpleNamespace(robot="so101", simulate=False,
                           CANVAS_POSITION=[-0.004074, 0.2546],
                           CANVAS_WIDTH_M=0.279, CANVAS_HEIGHT_M=0.216)


@pytest.fixture
def height_cache(tmp_path, opt):
    opt.cache_dir = str(tmp_path)
    params = {"Z_CANVAS": 0.0015, "Z_MAX_CANVAS": -0.009, "extra": "preserve"}
    path = tmp_path / HEIGHT_FILENAME
    path.write_text(json.dumps(params))
    return path, params


def surface(x, y):
    # Includes a bilinear term to catch using a single plane or wrong axis order.
    return 0.0015 + .02*x + .03*(y-.36) + .1*x*(y-.36)


def surface_grid(opt):
    xs, ys = grid_axes(opt)
    return CanvasZCalibration(opt, [[surface(x, y) for x in xs] for y in ys])


def test_axes_use_actual_canvas_bounds_and_center(opt):
    xs, ys = grid_axes(opt)
    np.testing.assert_allclose(xs, [-.123574, -.004074, .115426])
    np.testing.assert_allclose(ys, [.2746, .3626, .4506])
    assert len(set(point_order())) == 9
    assert point_order()[0] == (1, 1)


def test_bilinear_interpolation_matches_nodes_and_unmeasured_points(opt):
    grid = surface_grid(opt)
    for x in np.linspace(grid.x_m[0], grid.x_m[-1], 11):
        for y in np.linspace(grid.y_m[0], grid.y_m[-1], 13):
            assert grid.contact_z(x, y) == pytest.approx(surface(x, y), abs=1e-12)
    # Continuous at both interior cell boundaries.
    x, y = grid.x_m[1], grid.y_m[1]
    assert abs(grid.contact_z(x-1e-9, y)-grid.contact_z(x+1e-9, y)) < 1e-9
    assert abs(grid.contact_z(x, y-1e-9)-grid.contact_z(x, y+1e-9)) < 1e-9


def test_edges_hold_nearest_measured_height_without_extrapolation(opt):
    grid = surface_grid(opt)
    assert grid.contact_z(-10, 10) == grid.contact_z(grid.x_m[0], grid.y_m[-1])
    assert grid.contact_z(10, .36) == grid.contact_z(grid.x_m[-1], .36)
    with pytest.raises(ValueError, match="finite XY"):
        grid.contact_z(np.nan, .36)


@pytest.mark.parametrize("heights", [np.zeros((2, 3)), [[float('nan')]*3]*3, [[float('inf')]*3]*3])
def test_incomplete_or_nonfinite_grids_rejected(opt, heights):
    with pytest.raises(ValueError, match="nine finite"):
        CanvasZCalibration(opt, heights)


@pytest.mark.parametrize("inset", [-.001, .108, float("nan")])
def test_invalid_grid_inset_rejected(opt, inset):
    with pytest.raises(ValueError, match="inset"):
        grid_axes(opt, inset)


def test_legacy_cache_and_atomic_grid_round_trip(opt, height_cache):
    path, params = height_cache
    assert read_height_calibration(path, opt) == (params, None)
    grid = surface_grid(opt)
    save_grid(path, params, grid)
    saved, restored = read_height_calibration(path, opt)
    for name, value in params.items():
        assert saved[name] == value
    assert restored.to_dict() == grid.to_dict()
    assert restored.contact_z(.02, .32) == pytest.approx(surface(.02, .32))


def test_atomic_save_failure_preserves_previous_grid(opt, height_cache, monkeypatch):
    path, params = height_cache
    original = path.read_bytes()
    monkeypatch.setattr(canvas_z.os, "replace", Mock(side_effect=OSError("disk failure")))
    with pytest.raises(OSError, match="disk failure"):
        save_grid(path, params, surface_grid(opt))
    assert path.read_bytes() == original
    assert not list(path.parent.glob(f".{path.name}.*"))


@pytest.mark.parametrize("change", [
    lambda d: d.update(version=2),
    lambda d: d.update(frame="camera_xy"),
    lambda d: d.update(robot="franka"),
    lambda d: d.update(canvas_width_m=.30),
    lambda d: d.update(canvas_position_m=[0, .25]),
    lambda d: d.update(contact_z_m=[[0]*3]*2),
])
def test_stale_and_invalid_grids_rejected(opt, change):
    data = surface_grid(opt).to_dict()
    change(data)
    with pytest.raises(ValueError):
        CanvasZCalibration.from_dict(data, opt)


def keys_for_grid(contacts, reference):
    keys = []
    for row, col in point_order():
        keys.extend(["S"] * 20)  # 40 mm hover down to the reference.
        ticks = int(round((contacts[row, col] - reference) / .0005))
        keys.extend(["w" if ticks > 0 else "s"] * abs(ticks))
        keys.append("\r")
    return keys


def test_manual_collection_records_nine_commanded_heights_with_lifts(opt):
    reference = .0015
    contacts = reference + np.arange(9).reshape(3, 3) * .0005
    keys = iter(keys_for_grid(contacts, reference))
    requests = []
    def move(positions, **kwargs):
        requests.append((copy.deepcopy(positions), copy.deepcopy(kwargs)))
    grid = collect_contact_grid(opt, reference, move, [-.01, .36, .02], lambda: next(keys))
    np.testing.assert_allclose(grid.contact_z_m, contacts, atol=1e-12)
    np.testing.assert_allclose(requests[0][0], [[-.01, .36, .0415]])
    transfers = [(i, p, kw) for i, (p, kw) in enumerate(requests) if kw]
    assert len(transfers) == 9
    for (index, positions, kwargs), (row, col) in zip(transfers, point_order()):
        x, y = grid.x_m[col], grid.y_m[row]
        np.testing.assert_allclose(positions, [[x, y, .0415]])
        np.testing.assert_allclose(kwargs["fallback_positions"],
                                   [[grid.x_m[1], grid.y_m[1], .0415], [x, y, .0415]])
        if index > 1:
            assert requests[index-1][0][0][2] == pytest.approx(.0415)
    assert requests[-1][0][0][2] == pytest.approx(.0415)


def test_planning_rejection_retries_same_height_and_abort_never_saves(opt, height_cache):
    path, _ = height_cache
    original = path.read_bytes()
    class PlanningRejected(Exception):
        pass
    move = Mock(side_effect=[None, None, PlanningRejected("tilt limit"), None])
    keys = iter(["s", "s", "\x1b"])
    with pytest.raises(CalibrationCancelled):
        collect_contact_grid(opt, .0015, move, [0, .36, .04], lambda: next(keys),
                             planning_errors=(PlanningRejected,))
    assert move.call_args_list[2].args == move.call_args_list[3].args
    assert path.read_bytes() == original


def test_execution_failure_is_not_treated_as_recoverable_planning_failure(opt):
    move = Mock(side_effect=[None, None, TimeoutError("servo tracking failed")])
    with pytest.raises(TimeoutError, match="servo tracking"):
        collect_contact_grid(opt, .0015, move, [0, .36, .04], lambda: "s")
    assert move.call_count == 3


def test_enter_in_hover_and_lower_bound_are_not_accepted(opt, capsys):
    keys = iter(["\r"] + ["S"]*36 + ["\x1b"])
    requests = []
    with pytest.raises(CalibrationCancelled):
        collect_contact_grid(opt, .0015, lambda p, **kw: requests.extend(p),
                             [0, .36, .04], lambda: next(keys))
    assert min(p[2] for p in requests) >= .0015-.03-1e-9
    output = capsys.readouterr().out
    assert "Still above" in output
    assert "bound reached" in output


def cli_args(tmp_path, opt):
    materials = tmp_path / "materials.json"
    materials.write_text(json.dumps(vars(opt)))
    return ["--materials_json", str(materials), "--cache_dir", str(tmp_path)]


def test_default_cli_preview_cannot_import_hardware(tmp_path, opt, height_cache, monkeypatch, capsys):
    args = cli_args(tmp_path, opt)
    path, _ = height_cache
    before = path.read_bytes()
    monkeypatch.setitem(sys.modules, "robot", None)
    monkeypatch.setitem(sys.modules, "getch", None)
    assert main(args) == 0
    assert "Preview only" in capsys.readouterr().out
    assert before == path.read_bytes()


@pytest.mark.parametrize("abort", [False, True])
def test_cli_fake_robot_success_and_cancel(tmp_path, opt, height_cache, monkeypatch, abort):
    args = cli_args(tmp_path, opt) + ["--execute"]
    path, params = height_cache
    before = path.read_bytes()
    contacts = np.full((3, 3), params["Z_CANVAS"] + .001)
    keys = iter(["\x1b"] if abort else keys_for_grid(contacts, params["Z_CANVAS"]))
    fake = SimpleNamespace(good_morning_robot=Mock(), go_to_cartesian_pose=Mock(),
                           get_cartesian_pose=lambda: [0, .36, .04],
                           arm=SimpleNamespace(bus=SimpleNamespace(is_connected=True, disconnect=Mock())))
    robot_module = ModuleType("robot")
    robot_module.SO101 = Mock(return_value=fake)
    robot_module.MotionPlanningError = type("MotionPlanningError", (Exception,), {})
    monkeypatch.setitem(sys.modules, "robot", robot_module)
    monkeypatch.setitem(sys.modules, "getch", SimpleNamespace(getch=lambda: next(keys)))
    monkeypatch.setattr("builtins.input", lambda _: "")
    if abort:
        with pytest.raises(CalibrationCancelled):
            main(args)
        assert path.read_bytes() == before
        assert not list(tmp_path.glob("*.before_z_grid.*.json"))
    else:
        assert main(args) == 0
        saved, grid = read_height_calibration(path, opt)
        np.testing.assert_allclose(grid.contact_z_m, contacts)
        assert saved["Z_CANVAS"] == params["Z_CANVAS"]
        assert saved["Z_MAX_CANVAS"] == params["Z_MAX_CANVAS"]
        assert next(tmp_path.glob("*.before_z_grid.*.json")).read_bytes() == before
    fake.arm.bus.disconnect.assert_called_once_with(disable_torque=False)


def test_painter_rejects_stale_grid_before_robot_construction(opt, height_cache, monkeypatch):
    import painter as painter_module
    path, params = height_cache
    save_grid(path, params, surface_grid(opt))
    opt.use_cache = True
    opt.CANVAS_WIDTH_M += .01
    constructor = Mock(side_effect=AssertionError("must not create hardware"))
    monkeypatch.setattr(painter_module, "SO101", constructor)
    with pytest.raises(ValueError, match="Canvas settings changed"):
        painter_module.Painter(opt)
    constructor.assert_not_called()


def test_painter_loads_grid_before_first_neutral(opt, height_cache, monkeypatch):
    import painter as painter_module
    path, params = height_cache
    grid = surface_grid(opt)
    save_grid(path, params, grid)
    opt.use_cache, opt.writer = True, Mock()
    monkeypatch.setattr(painter_module, "SO101", Mock(return_value=Mock(spec=["good_morning_robot"])))
    monkeypatch.setattr(painter_module, "WebCam", Mock(return_value=object()))
    class StopBeforeMotion(Exception):
        pass
    def check_then_stop(p):
        assert p.canvas_z_calibration.to_dict() == grid.to_dict()
        assert p.canvas_z(.02, .33, p.Z_CANVAS+.04) == pytest.approx(surface(.02, .33)+.04)
        raise StopBeforeMotion
    monkeypatch.setattr(painter_module.Painter, "to_neutral", check_then_stop)
    with pytest.raises(StopBeforeMotion):
        painter_module.Painter(opt)


@pytest.fixture
def painter(opt):
    from painter import Painter
    p = Painter.__new__(Painter)
    p.opt, p.Z_CANVAS, p.Z_MAX_CANVAS = opt, .0015, -.009
    p.robot = SimpleNamespace(go_to_cartesian_pose=Mock(return_value=True))
    p.canvas_z_calibration = surface_grid(opt)
    return p




def test_stations_and_raw_moves_do_not_receive_canvas_correction(painter):
    p = painter
    p.opt.WATER_POSITION = [-.2129, .1931, p.Z_CANVAS]
    p.opt.HOVER_FACTOR = .05
    p.dip_brush_in_water()
    positions = p.robot.go_to_cartesian_pose.call_args.args[0]
    assert min(pt[2] for pt in positions) == p.Z_CANVAS
    p.move_to(0, .36, .02)  # Raw/manual moves stay uncompensated, even on canvas.
    assert p.robot.go_to_cartesian_pose.call_args.args[0] == [0, .36, .02]


