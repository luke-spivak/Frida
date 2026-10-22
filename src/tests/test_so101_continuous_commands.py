"""Production command continuity and dead zones, with no hardware access."""

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

import robot as robot_module
from robot import SO101

JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]


@pytest.fixture
def rig(monkeypatch):
    clock = SimpleNamespace(now=0.0)
    monkeypatch.setattr(robot_module, "time", SimpleNamespace(
        perf_counter=lambda: clock.now,
        sleep=lambda seconds: setattr(clock, "now", clock.now + seconds),
    ))
    observation = {**{n+".pos": 0.0 for n in JOINTS}, "gripper.pos": 9.0}
    offsets = dict(zip(JOINTS, [3., -1.5, -5.2, 0., 0.]))
    robot = SO101.__new__(SO101)
    robot.arm_joint_names = JOINTS
    robot._last_arm_command_deg = None

    def send(action):
        observation.update({n+".pos": action[n+".pos"]-.75 for n in JOINTS})
        return action

    def plan(points, current, *, previous_command_deg=None):
        start = previous_command_deg if previous_command_deg is not None else current
        q = np.array([start[n]+offsets[n] for n in JOINTS])
        return SimpleNamespace(times=[0., .02], joint_names=JOINTS,
                               positions=np.deg2rad([q, q+1.0]))

    robot.arm = SimpleNamespace(
        get_observation=Mock(side_effect=lambda: observation.copy()),
        send_action=Mock(side_effect=send),
    )
    robot.motion_planner = SimpleNamespace(
        plan_xyz_path=Mock(side_effect=plan), joint_offsets_deg=offsets,
        scene=SimpleNamespace(getJointInfo=lambda _: SimpleNamespace(
            limits=SimpleNamespace(min_position=[-.1], max_position=[1.7]))),
    )
    return robot, observation


def test_next_move_keeps_previous_goal_despite_tracking_error(rig):
    robot, observation = rig
    robot.go_to_cartesian_pose([0, .2, .1])
    previous = robot.arm.send_action.call_args.args[0].copy()
    assert observation["elbow_flex.pos"] == pytest.approx(previous["elbow_flex.pos"]-.75)
    robot.go_to_cartesian_pose([.02, .2, .1])
    next_start = robot.arm.send_action.call_args_list[2].args[0]
    assert next_start == pytest.approx(previous)
    assert all(call.args[0]["gripper.pos"] == 9 for call in robot.arm.send_action.call_args_list)
    # The planner receives real measurements as well as the prior setpoint.
    call = robot.motion_planner.plan_xyz_path.call_args
    assert call.args[1]["elbow_flex"] == pytest.approx(.25)
    assert call.kwargs["previous_command_deg"]["elbow_flex"] == pytest.approx(1.)


def test_manual_reposition_discards_cached_goal(rig):
    robot, observation = rig
    robot.go_to_cartesian_pose([0, .2, .1])
    observation["shoulder_pan.pos"] = 5.
    robot.go_to_cartesian_pose([.02, .2, .1])
    assert not robot.motion_planner.plan_xyz_path.call_args.kwargs
    assert robot.arm.send_action.call_args_list[2].args[0]["shoulder_pan.pos"] == pytest.approx(5.)


@pytest.mark.parametrize("failure", ["read", "plan", "send"])
def test_failure_invalidates_previous_goal(rig, failure):
    robot, _ = rig
    robot.go_to_cartesian_pose([0, .2, .1])
    target = {"read": robot.arm.get_observation,
              "plan": robot.motion_planner.plan_xyz_path,
              "send": robot.arm.send_action}[failure]
    target.side_effect = RuntimeError("injected failure")
    with pytest.raises(RuntimeError, match="injected failure"):
        robot.go_to_cartesian_pose([.02, .2, .1])
    assert robot._last_arm_command_deg is None


def test_noop_retains_valid_previous_goal(rig):
    robot, _ = rig
    robot.go_to_cartesian_pose([0, .2, .1])
    previous = robot._last_arm_command_deg.copy()
    robot.motion_planner.plan_xyz_path.side_effect = None
    robot.motion_planner.plan_xyz_path.return_value = None
    robot.go_to_cartesian_pose([0, .2, .1])
    assert robot._last_arm_command_deg == previous
    assert robot.arm.send_action.call_count == 2


@pytest.mark.parametrize("readback", [0, 1])
def test_startup_sets_and_verifies_arm_dead_zones_only(readback):
    robot = SO101.__new__(SO101)
    robot.arm_joint_names = JOINTS
    robot._last_arm_command_deg = dict.fromkeys(JOINTS, 1.)
    bus = Mock()
    bus.torque_disabled.return_value = nullcontext()
    bus.read.return_value = readback
    robot.arm = Mock(bus=bus)
    if readback:
        with pytest.raises(RuntimeError, match="Failed to set"):
            robot.good_morning_robot()
    else:
        robot.good_morning_robot()
        writes = [call for call in bus.write.call_args_list if "Dead_Zone" in call.args[0]]
        assert len(writes) == 10
        assert {(c.args[0], c.args[1], c.args[2]) for c in writes} == {
            (reg, n, 0) for reg in ("CW_Dead_Zone", "CCW_Dead_Zone") for n in JOINTS
        }
        assert all(c.kwargs == {"normalize": False} for c in writes)
        assert bus.read.call_count == 10
    assert robot._last_arm_command_deg is None
    robot.arm.send_action.assert_not_called()


def test_disconnect_invalidates_previous_goal(rig):
    robot, _ = rig
    robot._last_arm_command_deg = dict.fromkeys(JOINTS, 1.)
    robot.arm.disconnect = Mock()
    robot.good_night_robot()
    assert robot._last_arm_command_deg is None
