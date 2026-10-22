"""Diagnostic instrumentation with the real executor, a fake clock and fake I/O."""

import json
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

import diagnose_so101_motion as diagnostic
import robot as robot_module
from robot import SO101

JOINTS = ['shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll']


@pytest.fixture
def rig(monkeypatch):
    clock = SimpleNamespace(now=0.0, read_cost=.004, write_cost=.001)
    def sleep(seconds):
        clock.now += seconds
    fake_time = SimpleNamespace(perf_counter=lambda: clock.now, sleep=sleep)
    monkeypatch.setattr(diagnostic, 'time', fake_time)
    monkeypatch.setattr(robot_module, 'time', fake_time)
    observation = {**{n+'.pos': 0.0 for n in JOINTS}, 'gripper.pos': 9.0}
    def read():
        clock.now += clock.read_cost
        return observation.copy()
    def send(action):
        clock.now += clock.write_cost
        observation.update({n+'.pos': action[n+'.pos']-1 for n in JOINTS})
        return action
    robot = SO101.__new__(SO101)
    robot.arm_joint_names = JOINTS
    robot.arm = SimpleNamespace(get_observation=Mock(side_effect=read), send_action=Mock(side_effect=send))
    trajectory = SimpleNamespace(
        times=[0., .02, .04], joint_names=JOINTS,
        positions=np.deg2rad(np.array([[0]*5, [.5]*5, [1]*5])),
        velocities=np.array([[0]*5, [.2]*5, [0]*5]),
        accelerations=np.array([[1]*5, [0]*5, [-1]*5]),
    )
    robot.motion_planner = SimpleNamespace(
        plan_xyz_path=Mock(return_value=trajectory), joint_offsets_deg=dict.fromkeys(JOINTS, 0),
        scene=SimpleNamespace(getJointInfo=lambda name: SimpleNamespace(limits=SimpleNamespace(
            min_position=[-.174533], max_position=[1.74533]))),
    )
    return robot, clock, observation


@pytest.mark.parametrize('read_every,tracking_count', [(0, 0), (1, 3), (2, 2)])
def test_timing_and_error_measurements(rig, read_every, tracking_count):
    robot, clock, _ = rig
    original_send, original_read = robot.arm.send_action, robot.arm.get_observation
    log = diagnostic.MotionLog(robot, read_every)
    log.move('lift', [[0, .3, .1]])
    assert len(log.commands) == 3
    assert log.events[0]['start_observation']['elbow_flex.pos'] == 0
    assert [r['interval_ms'] for r in log.commands[1:]] == pytest.approx([20, 20])
    assert [r['write_ms'] for r in log.commands] == pytest.approx([1]*3)
    assert max(abs(r['lateness_ms']) for r in log.commands) < 1e-9
    tracking = [r for r in log.feedback if r['kind'] == 'tracking']
    assert len(tracking) == tracking_count
    assert len(log.feedback) == tracking_count + 1  # Normal executor settling read.
    assert all(r['read_ms'] == pytest.approx(4) for r in log.feedback)
    assert all(r['elbow_flex_error_deg'] == pytest.approx(1) for r in log.feedback)
    assert robot.arm.send_action is original_send
    assert robot.arm.get_observation is original_read
    assert all(c.args[0]['gripper.pos'] == 9 for c in original_send.call_args_list)
    assert log.planned[0]['elbow_flex_acceleration_deg_s2'] == pytest.approx(np.rad2deg(1))


def test_slow_measurement_is_visible_in_lateness(rig):
    robot, clock, _ = rig
    clock.read_cost = .030
    log = diagnostic.MotionLog(robot, read_every=1)
    log.move('lift', [[0, .3, .1]])
    result = log.summary()['lift']
    assert result['interval_ms_max'] == pytest.approx(31)
    assert result['max_lateness_ms'] == pytest.approx(22)
    assert result['read_ms_p95'] == pytest.approx(30)
    assert result['tracking_error_deg']['elbow_flex']['rms'] == pytest.approx(1)


def test_delayed_send_then_catchup_burst_is_visible(rig):
    robot, clock, _ = rig
    original = robot.arm.send_action.side_effect
    costs = iter([.050, .001, .001])
    def slow_once(action):
        clock.write_cost = next(costs)
        return original(action)
    robot.arm.send_action.side_effect = slow_once
    log = diagnostic.MotionLog(robot, read_every=0)
    log.move('lift', [[0, .3, .1]])
    assert [r['interval_ms'] for r in log.commands[1:]] == pytest.approx([50, 1])
    assert [r['lateness_ms'] for r in log.commands] == pytest.approx([0, 30, 11])


def test_offline_logs_only_plan_and_propagates_endpoint(rig):
    robot, _, observation = rig
    log = diagnostic.MotionLog(robot, offline_observation=observation)
    log.move('lift', [[0, .3, .1]])
    assert len(log.planned) == 3
    assert not log.commands and not log.feedback
    assert observation['elbow_flex.pos'] == pytest.approx(1)
    robot.arm.send_action.assert_not_called()


def test_failure_restores_methods_and_keeps_partial_log(rig, tmp_path, monkeypatch):
    robot, _, _ = rig
    original_plan, original_execute = robot.motion_planner.plan_xyz_path, robot.execute_trajectory
    send, read = robot.arm.send_action, robot.arm.get_observation
    original_effect = send.side_effect
    def fail_on_second(action):
        if send.call_count == 2:
            raise ConnectionError('injected failure')
        return original_effect(action)
    send.side_effect = fail_on_second
    log = diagnostic.MotionLog(robot)
    with pytest.raises(ConnectionError, match='injected'):
        log.move('lift', [[0, .3, .1]])
    assert robot.motion_planner.plan_xyz_path is original_plan
    assert robot.execute_trajectory == original_execute
    assert robot.arm.send_action is send and robot.arm.get_observation is read
    assert len(log.commands) == 1
    monkeypatch.setattr(log, 'plot', Mock())
    log.save(tmp_path, {'mode': 'test'})
    report = json.loads((tmp_path / 'report.json').read_text())
    assert report['events'][0]['status'] == 'failed'
    assert 'injected failure' in report['events'][0]['error']
    assert (tmp_path / 'commands.csv').is_file()
