"""Offline planner regressions: no viewer, serial connection, or robot motion."""

from pathlib import Path
import json
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

pytest.importorskip("roboplan")
import motion_planner
from motion_planner import SO101MotionPlanner


@pytest.fixture(scope="module")
def planner(tmp_path_factory, lerobot_calibration_path):
    model_dir = Path(__file__).resolve().parents[1] / "SO101"
    offsets_path = tmp_path_factory.mktemp("planner") / "offsets.json"
    offsets_path.write_text(json.dumps({
        "shoulder_pan": 0.0, "shoulder_lift": 0.0, "elbow_flex": -4.6,
        "wrist_flex": 0.0, "wrist_roll": 0.0,
    }))
    return SO101MotionPlanner(
        urdf_path=model_dir / "so101_brush_tip_nominal.urdf",
        srdf_path=model_dir / "so101_arm.srdf",
        joint_limits_path=model_dir / "so101_config.yaml",
        show_viewer=False,
        joint_offsets_path=offsets_path,
        lerobot_calibration_path=lerobot_calibration_path,
    )


@pytest.fixture
def start(planner, monkeypatch):
    # These unit tests isolate collisions/retries. Real alignment is tested below.
    monkeypatch.setattr(planner, "_validate_brush_tilt_path", Mock())
    q = np.zeros(len(planner.scene.getCurrentJointPositions()))
    q[planner.scene.getJointPositionIndices(["gripper"])[0]] = 0.25
    planner.scene.setJointPositions(q)
    return q


def test_retries_blocked_connection_from_original_start(planner, start, monkeypatch):
    blocked_goal = start.copy()
    blocked_goal[0] = 0.1
    accepted_goal = start.copy()
    accepted_goal[0] = 0.2
    alternate_seed = start.copy()
    alternate_seed[1] = -0.3
    solve = Mock(side_effect=[blocked_goal, accepted_goal])
    sample = Mock(return_value=alternate_seed)
    path_check = Mock(side_effect=[True, False])
    monkeypatch.setattr(planner, "_solve_xyz_attempt", solve)
    monkeypatch.setattr(planner, "_sample_ik_seed", sample)
    monkeypatch.setattr(motion_planner, "hasCollisionsAlongPath", path_check)

    result = planner._solve_xyz(np.array([0.1, 0.2, 0.1]), start)

    np.testing.assert_array_equal(result, accepted_goal)
    np.testing.assert_array_equal(solve.call_args_list[1].args[1], alternate_seed)
    for call in path_check.call_args_list:
        np.testing.assert_array_equal(call.args[1], start)
    np.testing.assert_array_equal(planner.scene.getCurrentJointPositions(), start)


@pytest.mark.parametrize("failure", ["solver", "invalid", "collision"])
def test_retries_failed_ik_candidates(planner, start, monkeypatch, failure):
    bad_goal = start.copy()
    if failure == "solver":
        bad_goal = RuntimeError("Numerical attempt failed")
    elif failure == "invalid":
        bad_goal[0] = np.nan
    else:
        bad_goal[2] = 1.5
        assert planner.scene.isValidConfiguration(bad_goal)
        assert planner.scene.hasCollisions(bad_goal)
    solve = Mock(side_effect=[bad_goal, start.copy()])
    path_check = Mock(return_value=False)
    monkeypatch.setattr(planner, "_solve_xyz_attempt", solve)
    monkeypatch.setattr(planner, "_sample_ik_seed", Mock(return_value=start.copy()))
    monkeypatch.setattr(motion_planner, "hasCollisionsAlongPath", path_check)

    result = planner._solve_xyz(np.array([0.1, 0.2, 0.1]), start)

    assert solve.call_count == 2
    assert path_check.call_count == 1
    np.testing.assert_array_equal(result, start)
    np.testing.assert_array_equal(planner.scene.getCurrentJointPositions(), start)


def test_exhausted_paths_raise_and_restore_scene(planner, start, monkeypatch):
    monkeypatch.setattr(planner, "ik_max_attempts", 2)
    monkeypatch.setattr(planner, "_sample_ik_seed", Mock(return_value=start.copy()))

    def solve(*args):
        goal = start.copy()
        goal[0] = 0.2
        planner.scene.setJointPositions(goal)
        return goal

    monkeypatch.setattr(planner, "_solve_xyz_attempt", solve)
    monkeypatch.setattr(motion_planner, "hasCollisionsAlongPath", Mock(return_value=True))
    with pytest.raises(RuntimeError, match="after 2 attempts.*connecting path collides"):
        planner._solve_xyz(np.array([0.1, 0.2, 0.1]), start)
    np.testing.assert_array_equal(planner.scene.getCurrentJointPositions(), start)


def test_cannot_replace_colliding_start_with_random_seed(planner, start, monkeypatch):
    start[2] = 1.5
    assert planner.scene.isValidConfiguration(start)
    assert planner.scene.hasCollisions(start)
    sample = Mock()
    monkeypatch.setattr(planner, "_sample_ik_seed", sample)
    with pytest.raises(RuntimeError, match="start configuration collides"):
        planner._solve_xyz(np.array([0.1, 0.2, 0.1]), start)
    sample.assert_not_called()


def test_start_limit_error_reports_inputs_offsets_and_all_joints(planner, start, monkeypatch):
    measured_degrees = {
        name: -planner.joint_offsets_deg.get(name, 0.0)
        for name in planner.scene.getJointNames()
    }
    # Validate against the calibrated ranges, not the old nominal URDF bounds.
    measured_degrees.update(elbow_flex=-98.0, wrist_flex=101.0)
    solve = Mock()
    monkeypatch.setattr(planner, "_solve_xyz", solve)

    with pytest.raises(ValueError, match="IK start violates joint limits") as error:
        planner.plan_xyz_path([0.1, 0.2, 0.1], measured_degrees)

    lines = str(error.value).splitlines()
    elbow = next(line for line in lines if "elbow_flex:" in line)
    assert "input=-98.000000000 deg" in elbow
    assert "offset=-4.600000000 deg" in elbow
    assert "model=-102.600000000 deg" in elbow
    assert "BELOW lower" in elbow
    wrist = next(line for line in lines if "wrist_flex:" in line)
    assert "ABOVE upper" in wrist
    for name in planner.scene.getJointNames():
        assert f"{name}:" in str(error.value)
    solve.assert_not_called()
    np.testing.assert_array_equal(planner.scene.getCurrentJointPositions(), start)


def test_out_of_bounds_start_is_not_clamped_or_replaced(planner, start, monkeypatch):
    start[planner.scene.getJointPositionIndices(["wrist_roll"])[0]] = np.pi + 0.01
    original = start.copy()
    sample = Mock()
    solve = Mock()
    monkeypatch.setattr(planner, "_sample_ik_seed", sample)
    monkeypatch.setattr(planner, "_solve_xyz_attempt", solve)

    with pytest.raises(ValueError, match="wrist_roll:.*ABOVE upper"):
        planner._solve_xyz(np.array([0.1, 0.2, 0.1]), start)

    sample.assert_not_called()
    solve.assert_not_called()
    np.testing.assert_array_equal(start, original)


def test_continuous_start_uses_command_but_checks_measured_connection(planner, start, monkeypatch):
    measured = {
        name: float(np.rad2deg(start[planner.scene.getJointPositionIndices([name])[0]]))
              - planner.joint_offsets_deg.get(name, 0.)
        for name in planner.scene.getJointNames()
    }
    previous = {name: measured[name]+.5 for name in planner.joint_names}
    expected = start.copy()
    expected[planner.arm_indices] += np.deg2rad(.5)
    segment = Mock(side_effect=lambda a, b, q: planner._quintic_trajectory(
        [q[planner.arm_indices], q[planner.arm_indices] + .01]))
    monkeypatch.setattr(planner, "_plan_cartesian_segment", segment)
    check = Mock(wraps=motion_planner.hasCollisionsAlongPath)
    monkeypatch.setattr(motion_planner, "hasCollisionsAlongPath", check)

    trajectory = planner.plan_xyz_path([.1, .2, .1], measured, previous_command_deg=previous)

    np.testing.assert_allclose(segment.call_args.args[2], expected)
    np.testing.assert_allclose(trajectory.positions[0], expected[planner.arm_indices])
    assert any(np.allclose(call.args[1], start) and np.allclose(call.args[2], expected)
               for call in check.call_args_list)
    assert expected[-1] == start[-1]  # fixed measured gripper


def test_continuous_start_cannot_skip_colliding_measured_connection(planner, start, monkeypatch):
    measured = {
        name: float(np.rad2deg(start[planner.scene.getJointPositionIndices([name])[0]]))
              - planner.joint_offsets_deg.get(name, 0.)
        for name in planner.scene.getJointNames()
    }
    previous = {name: measured[name]+.5 for name in planner.joint_names}
    def solve(xyz, q, **kwargs):
        result = q.copy()
        result[planner.arm_indices] += .01
        return result
    monkeypatch.setattr(planner, "_solve_xyz", solve)
    # Reject the measured pose -> previous command connection independently
    # of the safe planned segment. No fake IK geometry is needed here.
    def segment(a, b, q):
        return planner._quintic_trajectory([q[planner.arm_indices], q[planner.arm_indices] + .01])
    monkeypatch.setattr(planner, "_plan_cartesian_segment", segment)
    monkeypatch.setattr(motion_planner, "hasCollisionsAlongPath", Mock(return_value=True))
    with pytest.raises(RuntimeError, match="intersects an obstacle"):
        planner.plan_xyz_path([.1, .2, .1], measured, previous_command_deg=previous)


def test_previous_command_must_still_be_near_feedback(planner, start):
    measured = {name: -planner.joint_offsets_deg.get(name, 0.) for name in planner.scene.getJointNames()}
    previous = {name: measured[name] for name in planner.joint_names}
    previous[planner.joint_names[0]] += 2.01
    with pytest.raises(ValueError, match="more than 2 degrees"):
        planner.plan_xyz_path([.1, .2, .1], measured, previous_command_deg=previous)


def test_unexpected_gripper_change_is_rejected(planner, start, monkeypatch):
    bad_goal = start.copy()
    bad_goal[-1] += 0.1
    monkeypatch.setattr(planner, "_solve_xyz_attempt", Mock(return_value=bad_goal))
    with pytest.raises(RuntimeError, match="outside the arm group"):
        planner._solve_xyz(np.array([0.1, 0.2, 0.1]), start)
    np.testing.assert_array_equal(planner.scene.getCurrentJointPositions(), start)


def test_sampler_preserves_gripper_and_joint_limits(planner, start):
    planner.scene.setRngSeed(101)
    for _ in range(5):
        seed = planner._sample_ik_seed(start)
        assert seed is not None
        assert seed[-1] == start[-1]
        assert planner.scene.isValidConfiguration(seed)
        assert not planner.scene.hasCollisions(seed)


def test_real_ik_and_connection_from_tilted_start(planner, monkeypatch):
    # Tilted pose with room to follow a straight lift. The old folded pose
    # near the shoulder limit needed a curved approach, now forbidden here.
    start = np.zeros(len(planner.scene.getCurrentJointPositions()))
    start[planner.arm_indices] = np.deg2rad([-21.0, -85.0, 69.0, 22.0, 60.0])
    target = planner.scene.forwardKinematics(start, planner.tip_frame, planner.base_frame)[:3, 3] + [0, 0, .04]
    assert planner.brush_tilt_deg(start) > 10.0
    planner.scene.setJointPositions(start)
    planner.scene.setRngSeed(101)
    monkeypatch.setattr(planner, "ik_max_attempts", 3)

    measured_degrees = {
        name: np.rad2deg(start[planner.scene.getJointPositionIndices([name])[0]])
              - planner.joint_offsets_deg.get(name, 0.0)
        for name in planner.scene.getJointNames()
    }
    trajectory = planner.plan_xyz_path(target, measured_degrees)
    assert trajectory is not None
    assert "gripper" not in trajectory.joint_names
    assert np.all(np.diff(trajectory.times) > 0)
    goal = start.copy()
    goal[planner.scene.getJointPositionIndices(trajectory.joint_names)] = trajectory.positions[-1]

    achieved = planner.scene.forwardKinematics(goal, planner.tip_frame, planner.base_frame)
    assert np.linalg.norm(achieved[:3, 3] - target) <= planner.ik_position_tolerance
    assert planner.brush_tilt_deg(goal) <= 10.0
    assert goal[-1] == start[-1]
    assert planner.scene.isValidConfiguration(goal)
    assert not planner.scene.hasCollisions(goal)
    assert not motion_planner.hasCollisionsAlongPath(
        planner.scene, start, goal, max_step_size=0.01
    )
    entered_cone = False
    for row in trajectory.positions:
        q = start.copy()
        q[planner.arm_indices] = row
        tilt = planner.brush_tilt_deg(q)
        if entered_cone:
            assert tilt <= 10.0
        entered_cone |= tilt <= 10.0
    assert entered_cone
    # The public API round-trips through degrees; allow only floating-point noise.
    np.testing.assert_allclose(planner.scene.getCurrentJointPositions(), start, rtol=0, atol=1e-12)


@pytest.mark.parametrize("tilt", [0.0, 10.0, 11.0, 90.0, 180.0])
def test_tilt_measures_shaft_direction_and_ignores_spin(planner, monkeypatch, tilt):
    assert planner.brush_axis_length == pytest.approx(0.0385)
    upright, _ = Rotation.align_vectors([planner.brush_direction], [planner.brush_axis_local])
    pose = np.eye(4)
    # Rotate about the shaft, then tilt it around a horizontal axis.
    pose[:3, :3] = (
        Rotation.from_euler("x", tilt, degrees=True)
        * Rotation.from_rotvec(planner.brush_direction * 1.2) * upright
    ).as_matrix()
    monkeypatch.setattr(planner, "scene", SimpleNamespace(forwardKinematics=Mock(return_value=pose)))
    assert planner.brush_tilt_deg(np.zeros(6)) == pytest.approx(tilt, abs=1e-5)


def test_tilted_start_can_enter_cone_but_cannot_leave_it(planner, monkeypatch):
    monkeypatch.setattr(planner, "brush_tilt_deg", lambda q: float(np.rad2deg(q[0])))
    def poses(*degrees):
        return [np.array([np.deg2rad(value)]) for value in degrees]

    planner._validate_brush_tilt_path(poses(60, 30, 5, 9))
    planner._validate_brush_tilt_path(poses(0, 10))
    with pytest.raises(RuntimeError, match="leaves.*upright cone"):
        planner._validate_brush_tilt_path(poses(60, 5, 11, 0))
    with pytest.raises(RuntimeError, match="never reaches"):
        planner._validate_brush_tilt_path(poses(60, 11))


def test_tilt_is_checked_between_upright_endpoints(planner, monkeypatch):
    monkeypatch.setattr(planner, "brush_tilt_deg", lambda q: 12 * np.sin(np.pi * q[0]))
    with pytest.raises(RuntimeError, match="leaves.*upright cone"):
        planner._validate_brush_tilt_path([np.array([0.0]), np.array([1.0])])


def test_matching_xyz_does_not_accept_a_tilted_goal(planner, monkeypatch):
    q = np.zeros(len(planner.scene.getCurrentJointPositions()))
    xyz = planner.scene.forwardKinematics(q, planner.tip_frame, planner.base_frame)[:3, 3]
    assert planner.brush_tilt_deg(q) > 10.0
    solver = SimpleNamespace(num_variables=planner.ik.num_variables,
                             solveIk=Mock())
    monkeypatch.setattr(planner, "ik", solver)
    # A zero-step solver must fail rather than returning immediately on XYZ alone.
    with pytest.raises(RuntimeError, match="stalled"):
        planner._solve_xyz_attempt(xyz, q, [])
    solver.solveIk.assert_called_once()


def test_upright_goal_with_tilted_connection_is_retried(planner, monkeypatch):
    start = np.zeros(len(planner.scene.getCurrentJointPositions()))
    first, second = start.copy(), start.copy()
    first[0], second[0] = 0.1, 0.2
    solve = Mock(side_effect=[first, second])
    check = Mock(side_effect=[RuntimeError("Brush path leaves upright cone"), None])
    monkeypatch.setattr(planner, "_solve_xyz_attempt", solve)
    monkeypatch.setattr(planner, "_sample_ik_seed", Mock(return_value=start.copy()))
    monkeypatch.setattr(planner, "_validate_brush_tilt_path", check)
    path_check = Mock(return_value=False)
    monkeypatch.setattr(motion_planner, "hasCollisionsAlongPath", path_check)
    result = planner._solve_xyz(np.array([0.1, 0.2, 0.1]), start)
    np.testing.assert_array_equal(result, second)
    assert solve.call_count == path_check.call_count == check.call_count == 2
    for call in check.call_args_list:
        np.testing.assert_array_equal(call.args[0][0], start)


def test_parameterized_tilt_overshoot_is_rejected(planner, monkeypatch):
    start = np.zeros(len(planner.scene.getCurrentJointPositions()))
    start[planner.arm_indices] = np.deg2rad([-76.60162, -98.14478, 84.26316, 13.882, 92.78917])
    goal = start.copy()
    goal[0] += 0.01
    bad = goal.copy()
    bad[planner.scene.getJointPositionIndices(["wrist_roll"])[0]] += np.deg2rad(30)
    assert planner.brush_tilt_deg(start) < 1.0
    assert planner.brush_tilt_deg(goal) < 1.0
    assert planner.brush_tilt_deg(bad) > 10.0
    assert not planner.scene.hasCollisions(bad)
    measured = {
        name: np.rad2deg(start[planner.scene.getJointPositionIndices([name])[0]])
              - planner.joint_offsets_deg.get(name, 0.0)
        for name in planner.scene.getJointNames()
    }
    monkeypatch.setattr(planner, "_solve_xyz", Mock(return_value=goal))
    monkeypatch.setattr(planner, "_quintic_trajectory", Mock(return_value=SimpleNamespace(
        times=[0.0, 0.5, 1.0], joint_names=planner.joint_names,
        positions=[q[planner.arm_indices] for q in (start, bad, goal)],
    )))
    with pytest.raises(RuntimeError, match="leaves.*upright cone"):
        planner.plan_xyz_path([0.1, 0.2, 0.1], measured)


@pytest.fixture
def parameterized_move(planner, monkeypatch):
    start = np.zeros(len(planner.scene.getCurrentJointPositions()))
    start[planner.arm_indices] = np.deg2rad([-76.60162, -98.14478, 84.26316, 13.882, 92.78917])
    goal = start.copy()
    goal[0] += 0.01
    measured = {
        name: np.rad2deg(start[planner.scene.getJointPositionIndices([name])[0]])
              - planner.joint_offsets_deg.get(name, 0.0)
        for name in planner.scene.getJointNames()
    }
    solve = Mock(return_value=goal)
    monkeypatch.setattr(planner, "_solve_xyz", solve)
    monkeypatch.setattr(planner, "cartesian_step", 1.0)
    planner.scene.setJointPositions(start)

    def trajectory(*poses):
        return SimpleNamespace(
            times=np.linspace(0.0, 1.0, len(poses)), joint_names=planner.joint_names,
            positions=[q[planner.arm_indices].copy() for q in poses],
        )

    return start, goal, measured, solve, trajectory


@pytest.mark.parametrize("failure, message", [
    ("limits", "trajectory violates calibrated joint limits"),
    ("collision", "trajectory intersects an obstacle"),
    ("tilt", "leaves.*upright cone"),
])
@pytest.mark.parametrize("fallback_valid", [True, False])
def test_cubic_fallback_validates_both_trajectories_without_repeating_ik(
    planner, parameterized_move, monkeypatch, failure, message, fallback_valid,
):
    start, goal, measured, solve, trajectory = parameterized_move
    start_xyz = planner.scene.forwardKinematics(start, planner.tip_frame, planner.base_frame)[:3, 3]
    end_xyz = planner.scene.forwardKinematics(goal, planner.tip_frame, planner.base_frame)[:3, 3]
    bad = goal.copy()
    if failure == "limits":
        name = "wrist_flex"
        bad[planner.scene.getJointPositionIndices([name])[0]] = (
            planner.scene.getJointInfo(name).limits.min_position[0] - 0.01
        )
    elif failure == "collision":
        bad[:] = 0.0
        bad[planner.scene.getJointPositionIndices(["elbow_flex"])[0]] = 1.5
        assert planner.scene.isValidConfiguration(bad)
        assert planner.scene.hasCollisions(bad)
    else:
        bad[planner.scene.getJointPositionIndices(["wrist_roll"])[0]] += np.deg2rad(30)
        assert planner.scene.isValidConfiguration(bad)
        assert not planner.scene.hasCollisions(bad)
        assert planner.brush_tilt_deg(bad) > 10.0

    cubic = trajectory(start, bad, goal)
    fallback = trajectory(start, goal) if fallback_valid else trajectory(start, bad, goal)
    generate = Mock(side_effect=[cubic, fallback])
    monkeypatch.setattr(planner, "_quintic_trajectory", generate)
    if fallback_valid:
        assert planner._plan_cartesian_segment(start_xyz, end_xyz, start) is fallback
    else:
        with pytest.raises(RuntimeError, match=message):
            planner._plan_cartesian_segment(start_xyz, end_xyz, start)

    solve.assert_called_once()
    assert [call.kwargs['stop_at_waypoints'] for call in generate.call_args_list] == [False, True]
    assert generate.call_args_list[0].args[0] is generate.call_args_list[1].args[0]
    np.testing.assert_allclose(planner.scene.getCurrentJointPositions(), start, atol=1e-12)


def test_valid_cubic_is_returned_without_waypoint_stops(planner, parameterized_move, monkeypatch):
    start, goal, measured, solve, trajectory = parameterized_move
    cubic = trajectory(start, goal)
    generate = Mock(return_value=cubic)
    monkeypatch.setattr(planner, "_quintic_trajectory", generate)
    start_xyz = planner.scene.forwardKinematics(start, planner.tip_frame, planner.base_frame)[:3, 3]
    end_xyz = planner.scene.forwardKinematics(goal, planner.tip_frame, planner.base_frame)[:3, 3]
    assert planner._plan_cartesian_segment(start_xyz, end_xyz, start) is cubic
    generate.assert_called_once()
    assert generate.call_args.kwargs['stop_at_waypoints'] is False
    solve.assert_called_once()


@pytest.mark.parametrize("stop_at_waypoints", [False, True])
def test_quintic_derivatives_limits_and_command_spacing(planner, stop_at_waypoints):
    # Unequal segment lengths and a reversal exercise the spatial curvature
    # term in acceleration; a straight two-point path cannot catch its omission.
    waypoints = np.array([
        [0, 0, 0, 0, 0], [.10, -.04, .02, 0, 0],
        [.11, -.045, .021, 0, 0], [.16, .05, -.03, 0, 0],
    ])
    # Fine sampling resolves the finite jerk changes at fallback stops when
    # estimating acceleration numerically across a join.
    trajectory = planner._quintic_trajectory(waypoints, stop_at_waypoints, dt=.0002)
    t, q, v, a = map(np.asarray, (
        trajectory.times, trajectory.positions, trajectory.velocities, trajectory.accelerations,
    ))
    np.testing.assert_allclose(np.diff(t), .0002, atol=1e-12)
    np.testing.assert_allclose(q[[0, -1]], waypoints[[0, -1]], atol=1e-12)
    np.testing.assert_array_equal(v[[0, -1]], 0)
    np.testing.assert_array_equal(a[[0, -1]], 0)
    assert np.max(abs(v)) <= 10 * .15 * planner.speed_scale + 1e-10
    assert np.max(abs(a)) <= 5 * .225 * planner.speed_scale**2 + 1e-10
    # Independently differentiate the emitted positions/velocities. Verify the
    # reported derivatives, including the interior joins, are physically consistent.
    np.testing.assert_allclose(np.gradient(q, t, axis=0)[1:-1], v[1:-1], atol=2e-5)
    np.testing.assert_allclose(np.gradient(v, t, axis=0)[1:-1], a[1:-1], atol=.004 * planner.speed_scale**3)
    if stop_at_waypoints:
        for waypoint in waypoints[1:-1]:
            hits = np.flatnonzero(np.linalg.norm(q - waypoint, axis=1) < 1e-12)
            assert len(hits) == 1
            index = hits[0]
            np.testing.assert_array_equal(v[index], 0)
            np.testing.assert_array_equal(a[index], 0)
            assert np.max(abs(a[index-1:index+2])) < .05 * planner.speed_scale**3
    else:
        assert np.all(np.linalg.norm(v[1:-1], axis=1) > 0)


def test_quintic_point_to_point_has_smooth_acceleration_reversal(planner):
    waypoints = np.zeros((2, 5))
    waypoints[1, 0] = .2
    trajectory = planner._quintic_trajectory(waypoints, dt=.001)
    a = np.asarray(trajectory.accelerations)[:, 0]
    middle = len(a) // 2
    assert a[1] > 0 and a[-2] < 0
    assert np.interp(trajectory.times[-1] / 2, trajectory.times, a) == pytest.approx(0, abs=1e-10)
    # The old time-optimal triangle jumped from +amax to -amax at its midpoint.
    assert np.max(abs(np.diff(a))) < .03 * planner.speed_scale**3


@pytest.mark.parametrize("stop_at_waypoints", [False, True])
def test_speed_scale_compresses_time_without_changing_path(planner, monkeypatch, stop_at_waypoints):
    waypoints = np.array([[0]*5, [.1, -.04, .02, 0, 0], [.16, .05, -.03, 0, 0]])
    monkeypatch.setattr(planner, "speed_scale", 1.0)
    slow = planner._quintic_trajectory(waypoints, stop_at_waypoints)
    monkeypatch.setattr(planner, "speed_scale", 1.5)
    fast = planner._quintic_trajectory(waypoints, stop_at_waypoints)
    pieces = len(waypoints)-1 if stop_at_waypoints else 1
    assert abs(fast.times[-1] - slow.times[-1]/1.5) < .02 * pieces
    np.testing.assert_allclose(np.diff(fast.times), .02, atol=1e-12)
    np.testing.assert_allclose(np.asarray(fast.positions)[[0, -1]], waypoints[[0, -1]])
    np.testing.assert_array_equal(np.asarray(fast.velocities)[[0, -1]], 0)
    np.testing.assert_array_equal(np.asarray(fast.accelerations)[[0, -1]], 0)


@pytest.mark.parametrize("speed", [0, -1, np.nan, np.inf])
def test_invalid_speed_scale_rejected_before_loading_model(speed):
    with pytest.raises(ValueError, match="speed_scale"):
        SO101MotionPlanner(speed_scale=speed)


@pytest.mark.parametrize("options", [
    {"velocity_scale": np.inf}, {"acceleration_scale": np.inf},
    {"velocity_scale": 0}, {"acceleration_scale": np.nan},
])
def test_invalid_motion_scales_are_not_hidden_by_limit_caps(planner, options):
    with pytest.raises(ValueError, match="positive finite"):
        planner._quintic_trajectory(np.array([[0]*5, [.1]*5]), **options)


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), "3.0", True, None])
def test_offset_loader_rejects_invalid_angles(tmp_path, bad_value):
    path = tmp_path / "offsets.json"
    path.write_text(json.dumps({"joint_offsets_deg": {"shoulder_pan": bad_value}}))
    with pytest.raises(ValueError, match="shoulder_pan.*finite number"):
        motion_planner.load_joint_offsets(path, ["shoulder_pan"])


@pytest.mark.parametrize("saved", [{}, {"elbow_flex": 0.0}, {"shoulder_pan": 3.0, "gripper": 1.0}])
def test_offset_loader_requires_exact_arm_joint_names(tmp_path, saved):
    path = tmp_path / "offsets.json"
    path.write_text(json.dumps({"joint_offsets_deg": saved}))
    with pytest.raises(ValueError, match="missing joints=.*unknown joints="):
        motion_planner.load_joint_offsets(path, ["shoulder_pan"])


def test_offset_loader_fails_clearly_when_file_is_missing_or_malformed(tmp_path):
    path = tmp_path / "missing.json"
    with pytest.raises(FileNotFoundError, match="Save offsets with"):
        motion_planner.load_joint_offsets(path, ["shoulder_pan"])
    path.write_text("{bad JSON")
    with pytest.raises(ValueError, match="Invalid joint offsets JSON"):
        motion_planner.load_joint_offsets(path, ["shoulder_pan"])


def test_default_offset_file_is_independent_of_cwd_and_loaded_once(tmp_path, monkeypatch, lerobot_calibration_path):
    assert motion_planner.DEFAULT_JOINT_OFFSETS_PATH == (
        Path(motion_planner.__file__).resolve().with_name("so101_offsets.json")
    )
    model_dir = Path(__file__).resolve().parents[1] / "SO101"
    path = tmp_path / "offsets.json"
    offsets = {
        "shoulder_pan": 3.0, "shoulder_lift": -1.5, "elbow_flex": -4.6,
        "wrist_flex": 0.0, "wrist_roll": 0.0,
    }
    path.write_text(json.dumps({"joint_offsets_deg": offsets, "observation": {"unused.pos": 999}}))
    monkeypatch.setattr(motion_planner, "DEFAULT_JOINT_OFFSETS_PATH", path)
    monkeypatch.setattr(motion_planner, "DEFAULT_LEROBOT_CALIBRATION_PATH", lerobot_calibration_path)
    monkeypatch.chdir(tmp_path)
    instance = SO101MotionPlanner(
        urdf_path=model_dir / "so101_brush_tip_nominal.urdf",
        srdf_path=model_dir / "so101_arm.srdf",
        joint_limits_path=model_dir / "so101_config.yaml",
    )
    assert instance.joint_offsets_deg == offsets
    path.write_text(json.dumps({name: 99.0 for name in offsets}))
    assert instance.joint_offsets_deg == offsets  # Planning and commands retain one calibration.

