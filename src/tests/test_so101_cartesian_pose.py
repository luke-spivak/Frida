"""Measured brush FK with mocked observations; never connects to a robot."""

import json
from pathlib import Path
from unittest.mock import Mock, call

import numpy as np
import pinocchio as pin
import pytest

from motion_planner import SO101MotionPlanner
from robot import SO101


@pytest.fixture(scope="module")
def fk_planner(tmp_path_factory, lerobot_calibration_path):
    root = Path(__file__).resolve().parents[1] / "SO101"
    offsets = tmp_path_factory.mktemp("fk") / "offsets.json"
    offsets.write_text(json.dumps({
        "shoulder_pan": 3.0, "shoulder_lift": -1.5, "elbow_flex": -5.2,
        "wrist_flex": 1.2, "wrist_roll": -2.4,
    }))
    return SO101MotionPlanner(
        urdf_path=root / "so101_brush_tip_nominal.urdf",
        srdf_path=root / "so101_arm.srdf",
        joint_limits_path=root / "so101_config.yaml",
        joint_offsets_path=offsets, lerobot_calibration_path=lerobot_calibration_path,
        show_viewer=False,
    )


@pytest.mark.parametrize("angles,opening", [
    ([11.0, 2.0, -5.0, 3.0, 7.0], 0.0),
    ([-24.0, -103.780219780, 95.032967033, 22.857142857, 96.747252747], 37.0),
    # Reading a pose must still work outside planning limits and upright tolerance.
    ([8.0, -16.0, -33.0, 101.0, 104.0], 100.0),
])
def test_measured_brush_tip_in_frida_matches_independent_fk(fk_planner, angles, opening):
    planner = fk_planner
    robot = object.__new__(SO101)
    robot.motion_planner = planner
    robot.arm = Mock()
    robot.arm_joint_names = list(reversed(planner.joint_names))
    observation = dict(zip((f"{name}.pos" for name in planner.joint_names), angles))
    observation["gripper.pos"] = opening
    robot.arm.get_observation.return_value = observation

    # FK must use the new observation, not stale planning state, and leave it alone.
    stale = np.full(len(planner.scene.getCurrentJointPositions()), 0.123)
    planner.scene.setJointPositions(stale)
    position = robot.get_cartesian_pose()

    # Separate Pinocchio model is the reference; include the calibrated offsets
    # and actual jaw opening, which has no effect on this rigidly mounted brush.
    model = pin.buildModelFromUrdf(str(planner.urdf))
    data = model.createData()
    q = pin.neutral(model)
    for name, angle in zip(planner.joint_names, angles):
        q[model.joints[model.getJointId(name)].idx_q] = np.deg2rad(
            angle + planner.joint_offsets_deg[name]
        )
    grip = model.joints[model.getJointId("gripper")].idx_q
    q[grip] = model.lowerPositionLimit[grip] + opening / 100 * (
        model.upperPositionLimit[grip] - model.lowerPositionLimit[grip]
    )
    pin.framesForwardKinematics(model, data, q)
    expected = data.oMf[model.getFrameId("brush_tip")].translation

    assert isinstance(position, list) and len(position) == 3
    assert all(isinstance(value, float) for value in position)
    np.testing.assert_allclose(position, [-expected[1], expected[0], expected[2]], atol=1e-12)
    np.testing.assert_allclose(robot.frida_to_robot(np.array(position)), expected, atol=1e-12)
    np.testing.assert_array_equal(planner.scene.getCurrentJointPositions(), stale)
    assert robot.arm.mock_calls == [call.get_observation()]


def test_nonfinite_observation_is_rejected(fk_planner):
    robot = object.__new__(SO101)
    robot.motion_planner = fk_planner
    robot.arm = Mock()
    observation = {f"{name}.pos": 0.0 for name in fk_planner.joint_names}
    observation["elbow_flex.pos"] = float("nan")
    robot.arm.get_observation.return_value = observation
    with pytest.raises(ValueError, match="Invalid angle for elbow_flex"):
        robot.get_cartesian_pose()
    assert robot.arm.mock_calls == [call.get_observation()]
