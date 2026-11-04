"""Contact-path regressions with real IK and fake hardware only."""

import json
import pickle
from unittest.mock import Mock

import numpy as np
import pytest

from brush_stroke import BrushStroke
from motion_planner import SO101MotionPlanner
from paint_loading import load_setup, make_painter, make_stroke
from paint_repeatability import arguments, make_layout, make_protocol, run_sheets
from robot import MotionPlanningError
from test_paint_loading import setup
from test_so101_calibrated_limits import calibrated_planner
from test_canvas_z_planning import offline_robot


def test_all_twelve_32mm_contacts_flow_without_internal_stops(setup, offline_robot, monkeypatch, tmp_path):
    # Recorded 67 mm brush calibration, copied as fixtures rather than reading
    # or changing the user's cache. Include its non-planar grid and XY correction.
    path = setup.cache_dir / "brush_tip_to_table_calib.json"
    calibration = json.loads(path.read_text())
    calibration["Z_MAX_CANVAS"] = -.0105
    calibration["canvas_z_grid"]["contact_z_m"] = [
        [-.0015, -.0035, -.0025], [.0025, .0005, .0005], [.0055, .0065, .0055]]
    path.write_text(json.dumps(calibration))
    (setup.cache_dir / "cached_H_coord.pkl").write_bytes(pickle.dumps(np.array([
        [.97732774, -.05603982, .02027078], [.11799699, .93978824, .01919113],
        [.34128479, -.07051059, 1.]])))
    args = arguments(["--cache_dir", str(setup.cache_dir), "--materials_json", str(setup.materials_json),
                      "--length-mm", "32", "--repeats", "1"])
    opt, params, grid, _, xy = load_setup(args)
    painter = make_painter(opt, params, grid, offline_robot, xy)
    report = make_protocol(args, make_layout(opt, grid, .032, xy))
    planner = offline_robot.motion_planner
    native = planner._plan_continuous_stroke
    contacts = []

    def capture(points, q):
        trajectory = native(points, q)
        contacts.append(trajectory)
        v, a = np.asarray(trajectory.velocities), np.asarray(trajectory.accelerations)
        np.testing.assert_allclose(v[[0, -1]], 0, atol=1e-12)
        np.testing.assert_allclose(a[[0, -1]], 0, atol=1e-12)
        assert np.all(np.linalg.norm(v[1:-1], axis=1) > 1e-8)
        limits = [planner.scene.getJointInfo(name).limits for name in planner.joint_names]
        vmax = min(.15 * planner.speed_scale, 1.) * np.array([l.max_velocity[0] for l in limits])
        amax = min(.225 * planner.speed_scale**2, 1.) * np.array([l.max_acceleration[0] for l in limits])
        assert np.all(abs(v) <= vmax + 1e-10)
        assert np.all(abs(a) <= amax + 1e-10)
        return trajectory

    monkeypatch.setattr(planner, "_plan_continuous_stroke", capture)
    run_sheets(args, painter, make_stroke(args), report, tmp_path, Mock(), Mock(),
               BrushStroke.execute, prompt=lambda _: None)
    assert len(contacts) == report["completed_strokes"] == 12


def test_failed_continuous_path_sends_no_commands(offline_robot, monkeypatch):
    robot = offline_robot
    x, y, z = robot.get_cartesian_pose()
    monkeypatch.setattr(robot.motion_planner, "_plan_continuous_stroke",
                        Mock(side_effect=RuntimeError("continuous path rejected")))
    with pytest.raises(MotionPlanningError, match="continuous path rejected"):
        robot.go_to_cartesian_pose([[x, y, z], [x+.016, y, z], [x+.032, y, z],
                                    [x+.032, y, z+.02]], continuous_range=(0, 2))
    robot.execute_trajectory.assert_not_called()


@pytest.mark.parametrize("span", [(-1, 1), (0, 3), (2, 1), (1, 1), (0., 2), (False, 2), (0,)])
def test_bad_contact_ranges_fail_before_model_access(span):
    planner = SO101MotionPlanner.__new__(SO101MotionPlanner)
    with pytest.raises(ValueError, match="continuous_range"):
        planner.plan_xyz_path([[0, 0, 0], [.01, 0, 0], [.02, 0, 0]], {}, continuous_range=span)


@pytest.mark.parametrize("bad", ["excursion", "reversal", "between_samples"])
def test_stroke_validator_checks_assigned_progress_and_between_commands(calibrated_planner, monkeypatch, bad):
    from types import SimpleNamespace
    planner = calibrated_planner
    def fk(q, *args):
        pose = np.eye(4)
        pose[:3, 3] = [q[0], .003 if bad == "excursion" else 0., 0.]
        if bad == "between_samples":
            pose[1, 3] = .003 * np.sin(np.pi*q[0]/.01)
        return pose
    monkeypatch.setattr(planner, "scene", SimpleNamespace(forwardKinematics=fk))
    rows = np.zeros((3, len(planner.arm_indices)))
    rows[:, 0] = [0, .01, 0] if bad == "reversal" else [0, .01, .02]
    with pytest.raises(RuntimeError, match="assigned path"):
        planner._validate_stroke_path(SimpleNamespace(positions=rows), np.zeros(6),
                                      np.array([[0, 0, 0], [.02, 0, 0]]),
                                      [0, .02], [0, .02], [0, .5, 1], [0, 1])
