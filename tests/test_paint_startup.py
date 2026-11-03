"""Actual startup collision regression; real planner, fake read-only bus."""

import json
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from paint_loading import execute_test, load_setup, preflight_start
from test_paint_loading import setup
from test_so101_calibrated_limits import calibrated_planner
from test_canvas_z_planning import offline_robot


# Read-only snapshot immediately after the 2026-11-02 repeatability failure.
# It starts collision-free but lifting folds lower_arm_link into shoulder_link.
FOLDED_START = {
    "shoulder_pan": -79.20879120879121, "shoulder_lift": -12.43956043956044,
    "elbow_flex": 94.24175824175825, "wrist_flex": -73.58241758241758,
    "wrist_roll": 92.87912087912088, "gripper": 8.131720430107526,
}


def fake_bus(robot, values):
    # No write/configure/torque interfaces: adding any of them to preflight fails.
    bus = Mock(spec_set=["connect", "sync_read", "disconnect", "is_connected"])
    bus.is_connected = False
    bus.connect.side_effect = lambda **kwargs: setattr(bus, "is_connected", True)
    bus.disconnect.side_effect = lambda **kwargs: setattr(bus, "is_connected", False)
    bus.sync_read.return_value = values
    robot.arm.bus = bus
    return bus


def test_folded_start_rejected_with_saved_measurement_and_no_motion(offline_robot, tmp_path):
    robot = offline_robot
    bus = fake_bus(robot, FOLDED_START)
    report, recording = {}, Mock()
    with pytest.raises(RuntimeError, match="Startup preflight failed.*less folded"):
        preflight_start(robot, .097, tmp_path, report, recording)
    assert report["startup"]["status"] == "failed"
    assert report["startup"]["joint_positions"] == FOLDED_START
    assert report["startup"]["brush_tilt_deg"] < 2
    np.testing.assert_allclose(report["startup"]["tip_frida_m"], [-.22046255, .09327027, .01359747], atol=1e-7)
    assert "No collision-free connection" in report["startup"]["error"]
    assert json.loads((tmp_path / "test.json").read_text()) == report
    robot.execute_trajectory.assert_not_called()
    bus.connect.assert_called_once_with(handshake=False)
    bus.sync_read.assert_called_once_with("Present_Position")
    bus.disconnect.assert_called_once_with(disable_torque=False)
    assert not bus.is_connected


def test_clear_start_passes_preflight_and_actual_unchanged_transfer(offline_robot, tmp_path):
    robot = offline_robot
    values = {k.removesuffix(".pos"): v for k, v in robot.arm.get_observation().items()}
    bus = fake_bus(robot, values)
    report = {}
    preflight_start(robot, .097, tmp_path, report, Mock())
    assert report["startup"]["status"] == "passed"
    robot.execute_trajectory.assert_not_called()
    assert not bus.is_connected
    # The execution path independently replans from feedback; it still uses the
    # normal lift/traverse/descent, collision checks, limits, and tilt cone.
    from painter import SO101_NEUTRAL_POSITION
    robot.go_to_cartesian_pose([SO101_NEUTRAL_POSITION], travel_z=.097)
    robot.execute_trajectory.assert_called_once()


def test_preflight_failure_prevents_motor_setup_and_sequence(setup, offline_robot, tmp_path, monkeypatch):
    from camera.innomaker import WebCam
    robot = offline_robot
    bus = fake_bus(robot, FOLDED_START)
    robot.good_morning_robot = Mock()
    camera = Mock(spec=WebCam)
    monkeypatch.setattr("camera.innomaker.WebCam", lambda *args: camera)
    monkeypatch.setattr("robot.SO101", lambda **kwargs: robot)
    monkeypatch.setattr("paint_loading.save_snapshot", Mock())
    opt, params, grid, h, xy = load_setup(setup)
    report, runner = {}, Mock()
    with pytest.raises(RuntimeError, match="Startup preflight failed"):
        execute_test(setup, opt, params, grid, h, xy, tmp_path, report, sequence_runner=runner)
    runner.assert_not_called()
    robot.good_morning_robot.assert_not_called()
    robot.execute_trajectory.assert_not_called()
    assert report["status"] == "error" and report["startup"]["status"] == "failed"
    bus.disconnect.assert_called_once_with(disable_torque=False)
    camera.close.assert_called_once_with(reason="error")
