"""Exercise calibrated observation -> model -> command conversion without hardware."""

import json
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest

import robot as robot_module
from robot import SO101
from motion_planner import SO101MotionPlanner


@pytest.mark.parametrize("show_viewer", [True, False])
def test_robot_creation_does_not_connect(monkeypatch, show_viewer):
    follower = Mock()
    follower.bus.motors = {"shoulder_pan": None, "gripper": None}
    monkeypatch.setattr(robot_module, "SO101Follower", Mock(return_value=follower))
    planner_constructor = Mock()
    monkeypatch.setattr(robot_module, "SO101MotionPlanner", planner_constructor)
    SO101(debug=False, show_viewer=show_viewer)
    follower.connect.assert_not_called()
    planner_constructor.assert_called_once_with(
        show_viewer=show_viewer, lerobot_calibration_path=follower.calibration_fpath,
    )


def test_loaded_offsets_are_applied_once_and_inverted_for_every_command(tmp_path, monkeypatch, lerobot_calibration_path):
    offsets = {
        "shoulder_pan": 3.0, "shoulder_lift": -1.5, "elbow_flex": -4.6,
        "wrist_flex": 1.2, "wrist_roll": -2.4,
    }
    path = tmp_path / "offsets.json"
    path.write_text(json.dumps({"joint_offsets_deg": offsets}))
    model_dir = Path(__file__).resolve().parents[1] / "SO101"
    planner = SO101MotionPlanner(
        urdf_path=model_dir / "so101_brush_tip_nominal.urdf",
        srdf_path=model_dir / "so101_arm.srdf",
        joint_limits_path=model_dir / "so101_config.yaml",
        joint_offsets_path=path,
        lerobot_calibration_path=lerobot_calibration_path,
    )
    # Use synthetic IK/tilt here to isolate the real input conversion, trajectory
    # parameterizer, collision checks, and robot command construction.
    delta = np.zeros(len(planner.scene.getCurrentJointPositions()))
    delta[planner.arm_indices] = np.deg2rad(1.0)
    solve = Mock(side_effect=lambda a, b, seed: planner._quintic_trajectory([
        seed[planner.arm_indices], (seed + delta)[planner.arm_indices]]))
    monkeypatch.setattr(planner, "_plan_cartesian_segment", solve)
    monkeypatch.setattr(planner, "_validate_brush_tilt_path", Mock())

    robot = object.__new__(SO101)
    robot.arm_joint_names = list(offsets)
    robot.motion_planner = planner
    robot.arm = Mock()
    observation = dict(zip(
        (f"{name}.pos" for name in offsets), [11.0, 2.0, -5.0, 3.0, 7.0],
    ))
    robot.arm.get_observation.side_effect = [
        {**observation, "gripper.pos": 37.0},
        {**observation, "gripper.pos": 38.0},
    ]
    robot.execute_trajectory = Mock()

    robot.go_to_cartesian_pose([0.1, 0.2, 0.1])
    robot.go_to_cartesian_pose([0.2, 0.2, 0.1])

    for solve_call, execute_call, grip in zip(
        solve.call_args_list, robot.execute_trajectory.call_args_list, [37.0, 38.0],
    ):
        seed = solve_call.args[2]
        times, commands = execute_call.args
        assert len(commands) == len(times) > 1
        for name in offsets:
            index = planner.scene.getJointPositionIndices([name])[0]
            assert np.rad2deg(seed[index]) == pytest.approx(observation[f"{name}.pos"] + offsets[name])
            assert commands[0][f"{name}.pos"] == pytest.approx(observation[f"{name}.pos"])
            assert commands[-1][f"{name}.pos"] == pytest.approx(observation[f"{name}.pos"] + 1.0)
        assert all(command["gripper.pos"] == grip for command in commands)
        gripper_index = planner.scene.getJointPositionIndices(["gripper"])[0]
        limits = planner.scene.getJointInfo("gripper").limits
        expected_gripper = limits.min_position[0] + grip / 100.0 * (limits.max_position[0] - limits.min_position[0])
        assert seed[gripper_index] == pytest.approx(expected_gripper)
    robot.arm.connect.assert_not_called()
    robot.arm.send_action.assert_not_called()
