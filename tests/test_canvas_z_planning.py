"""Actual IK/path validation with synthetic contact heights; no hardware I/O."""

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from canvas_z import CanvasZCalibration
from calibrate_canvas_z import collect_contact_grid, point_order
from test_so101_calibrated_limits import calibrated_planner, configuration


@pytest.fixture
def offline_robot(calibrated_planner):
    from robot import SO101
    p = calibrated_planner
    p.scene.setRngSeed(101)
    robot = SO101.__new__(SO101)
    robot.motion_planner, robot.arm_joint_names = p, p.joint_names
    robot._last_arm_command_deg = None
    observation = {
        "shoulder_pan.pos": -8.439560440, "shoulder_lift.pos": -1.450549451,
        "elbow_flex.pos": 85.626373627, "wrist_flex.pos": -83.428571429,
        "wrist_roll.pos": 93.230769231, "gripper.pos": 8.938172043010754,
    }
    # No connect, bus, or send_action interface exists on this fake.
    robot.arm = Mock(spec=["get_observation"])
    robot.arm.get_observation.side_effect = lambda: observation.copy()
    gripper = p.scene.getJointPositionIndices(["gripper"])[0]
    limits = p.scene.getJointInfo("gripper").limits
    grip_angle = limits.min_position[0] + observation["gripper.pos"] / 100 * (
        limits.max_position[0] - limits.min_position[0])

    def execute_offline(times, commands):
        assert len(times) == len(commands)
        assert np.all(np.diff(times) > 0)
        for command in commands:
            q = configuration(p, {name: command[f"{name}.pos"] for name in p.joint_names})
            q[gripper] = grip_angle
            assert p.scene.isValidConfiguration(q)
            assert not p.scene.hasCollisions(q)
            assert p.brush_tilt_deg(q) <= 10.0
        observation.update(commands[-1])
        robot._last_arm_command_deg = {name: observation[f"{name}.pos"] for name in p.joint_names}

    robot.execute_trajectory = Mock(side_effect=execute_offline)
    return robot


def settings():
    return SimpleNamespace(
        robot="so101", simulate=False, CANVAS_POSITION=[-.004074, .2546],
        CANVAS_WIDTH_M=.279, CANVAS_HEIGHT_M=.216, brush_length=None,
        X_CANVAS_MIN=-.142574, X_CANVAS_MAX=.134426,
        Y_CANVAS_MIN=.2556, Y_CANVAS_MAX=.4696,
    )


def test_all_nine_manual_contact_routes_pass_real_planner(offline_robot):
    # Replay keyboard adjustments and real direct/fallback planning, using a
    # reconstructed start and synthetic heights. No physical measurements.
    reference = .0015
    contacts = reference + np.array([[0, .001, .002], [.001, .002, .003], [.002, .003, .004]])
    keys = []
    for row, col in point_order():
        keys += ["S"] * 20
        keys += ["w"] * int(round((contacts[row, col]-reference)/.0005))
        keys += ["\r"]
    keys = iter(keys)
    grid = collect_contact_grid(settings(), reference, offline_robot.go_to_cartesian_pose,
                                offline_robot.get_cartesian_pose(), lambda: next(keys))
    np.testing.assert_allclose(grid.contact_z_m, contacts, atol=1e-12)
    p = offline_robot.motion_planner
    assert p.ik_position_tolerance == p.cartesian_path_tolerance == .002
    assert p.brush_tilt_tolerance_deg == 10.0
    assert offline_robot.execute_trajectory.call_count > 9


def test_corrected_stroke_passes_real_planner(offline_robot):
    import torch
    from painter import Painter
    from brush_stroke import BrushStroke
    p = Painter.__new__(Painter)
    p.opt, p.robot = settings(), offline_robot
    p.Z_CANVAS, p.Z_MAX_CANVAS = .0015, -.009
    p.H_coord = None
    p.canvas_z_calibration = CanvasZCalibration(p.opt, .0015 + np.array(
        [[0, .001, .002], [.001, .002, .003], [.002, .003, .004]]))
    stroke = SimpleNamespace(stroke_length=torch.tensor(.02), stroke_bend=torch.tensor(.003),
                             stroke_z=torch.tensor(.4), stroke_alpha=torch.tensor(0.))
    BrushStroke.execute(stroke, p, -.01, .36, 0.)
    assert offline_robot.execute_trajectory.call_count == 3  # Approach, descent + stroke, lift.
