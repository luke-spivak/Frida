"""Exercise trajectory completion/diagnostics with a fake clock and motor bus."""

from types import SimpleNamespace
from unittest.mock import Mock
import re

import pytest

import robot as robot_module
from robot import SO101


JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]


@pytest.fixture
def executor(monkeypatch):
    clock = SimpleNamespace(now=0.0)

    def sleep(seconds):
        clock.now += seconds

    monkeypatch.setattr(robot_module, "time", SimpleNamespace(
        perf_counter=lambda: clock.now, sleep=sleep,
    ))
    robot = object.__new__(SO101)
    robot.arm_joint_names = JOINTS
    robot.arm = Mock(spec=["send_action", "get_observation", "bus"])
    # Any attempt to write/configure the diagnostic bus will fail the tests.
    robot.arm.bus = Mock(spec=["sync_read"])
    robot.arm.bus.sync_read.side_effect = (
        lambda register, motors, **kwargs: {name: 0 for name in motors}
    )
    target = {f"{name}.pos": 0.0 for name in JOINTS}
    target["gripper.pos"] = 9.0
    robot.arm.get_observation.return_value = target.copy()
    return robot, clock, target


def test_success_preserves_command_timing_and_does_not_read_extra_registers(executor):
    robot, clock, target = executor
    sends = []
    robot.arm.send_action.side_effect = lambda action: sends.append((clock.now, action.copy()))
    # The gripper is deliberately excluded from the degree-based completion test.
    robot.arm.get_observation.return_value = {**target, "gripper.pos": 100.0}

    robot.execute_trajectory([0.0, 0.04, 0.08], [target.copy() for _ in range(3)])

    assert [t for t, _ in sends] == pytest.approx([0.0, 0.04, 0.08])
    assert all(action == target for _, action in sends)
    robot.arm.bus.sync_read.assert_not_called()


@pytest.mark.parametrize("errors", [
    [0.41, 1.12, 2.74, 0.21, 0.32],
    [0.31, 0.23, 2.18, 1.67, 0.81],
    [0.42, 0.23, 2.46, 0.08, 0.41],
])
def test_reported_failures_still_time_out_and_diagnose_only_elbow(executor, errors, capsys):
    robot, clock, target = executor
    robot.arm.get_observation.return_value = {
        **target, **{f"{name}.pos": -error for name, error in zip(JOINTS, errors)},
    }

    with pytest.raises(TimeoutError) as caught:
        robot.execute_trajectory([0.0], [target])

    message = str(caught.value)
    assert "tolerance=2.00°" in message
    assert f"actual={-errors[2]:+.3f}°, error={errors[2]:+.3f}°" in message
    assert "recent_range=0.000°" in message
    assert message.count("OUTSIDE tolerance") == 1
    assert "Goal_Position_deg=0.000" in message
    assert clock.now == pytest.approx(3.0, abs=0.021)
    # Diagnostics never resend a target, connect, or change a motor register.
    robot.arm.send_action.assert_called_once_with(target)
    assert robot.arm.bus.mock_calls
    for call in robot.arm.bus.mock_calls:
        assert call[0] == "sync_read"
        assert call.kwargs["motors"] == ["elbow_flex"]
        assert call.kwargs["num_retry"] == 0
    goal_calls = [c for c in robot.arm.bus.sync_read.call_args_list if c.args == ("Goal_Position",)]
    assert [c.kwargs["normalize"] for c in goal_calls] == [True, False]
    assert "[SO101 settling" in capsys.readouterr().out


def test_recent_range_distinguishes_continued_motion_from_stationary_error(executor):
    robot, clock, target = executor
    robot.arm.get_observation.side_effect = lambda: {
        **target, "elbow_flex.pos": -4.0 + clock.now,
    }
    with pytest.raises(TimeoutError) as caught:
        robot.execute_trajectory([0.0], [target], timeout=1.0)

    elbow_line = next(line for line in str(caught.value).splitlines() if "elbow_flex: target=" in line)
    recent_range = float(re.search(r"recent_range=([0-9.]+)", elbow_line).group(1))
    assert 0.48 <= recent_range <= 0.54


def test_diagnostic_read_failure_does_not_mask_timeout_or_stop_other_reads(executor):
    robot, _, target = executor
    robot.arm.get_observation.return_value = {**target, "elbow_flex.pos": -2.46}

    def read(register, motors, **kwargs):
        if register == "Goal_Position":
            raise ConnectionError("no status packet")
        return {name: 32 if register == "P_Coefficient" else 0 for name in motors}

    robot.arm.bus.sync_read.side_effect = read
    with pytest.raises(TimeoutError) as caught:
        robot.execute_trajectory([0.0], [target], timeout=0.1)

    assert "Goal_Position_deg: unavailable (ConnectionError: no status packet)" in str(caught.value)
    assert "P_Coefficient=32.000" in str(caught.value)
    robot.arm.send_action.assert_called_once_with(target)


def test_convergence_within_deadline_returns_without_diagnostic_reads(executor):
    robot, clock, target = executor
    robot.arm.get_observation.side_effect = lambda: {
        **target, "elbow_flex.pos": -3.0 + clock.now,
    }
    robot.execute_trajectory([0.0], [target], timeout=3.0)
    assert clock.now == pytest.approx(1.0, abs=0.021)
    robot.arm.bus.sync_read.assert_not_called()
