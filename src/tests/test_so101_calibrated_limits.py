"""Calibrated bounds in the real RoboPlan scene and IK; no robot connection."""

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
from roboplan.optimal_ik import ConfigurationTask

import motion_planner
from motion_planner import SO101MotionPlanner, load_calibrated_joint_limits


OFFSETS = {
    "shoulder_pan": 3.0, "shoulder_lift": -1.5, "elbow_flex": -5.2,
    "wrist_flex": 0.0, "wrist_roll": 0.0,
}
EXPECTED = {
    "shoulder_pan": (-117.35164835164835, 123.35164835164835),
    "shoulder_lift": (-106.86263736263736, 103.86263736263736),
    "elbow_flex": (-101.81538461538462, 91.41538461538461),
    "wrist_flex": (-99.6923076923077, 99.6923076923077),
    "wrist_roll": (-180.0, 180.0),
}


@pytest.fixture(scope="module")
def calibrated_planner(tmp_path_factory, lerobot_calibration_path):
    offsets = tmp_path_factory.mktemp("calibrated-planner") / "offsets.json"
    offsets.write_text(json.dumps(OFFSETS))
    root = Path(__file__).resolve().parents[1] / "SO101"
    return SO101MotionPlanner(
        urdf_path=root / "so101_brush_tip_nominal.urdf",
        srdf_path=root / "so101_arm.srdf",
        joint_limits_path=root / "so101_config.yaml",
        joint_offsets_path=offsets, lerobot_calibration_path=lerobot_calibration_path,
    )


def configuration(planner, measured):
    q = np.zeros(len(planner.scene.getCurrentJointPositions()))
    for name, value in measured.items():
        index = planner.scene.getJointPositionIndices([name])[0]
        q[index] = np.deg2rad(value + planner.joint_offsets_deg.get(name, 0.0))
    return q


def test_scene_sampling_and_validation_share_calibrated_bounds(calibrated_planner):
    p = calibrated_planner
    for name, expected in EXPECTED.items():
        limits = p.scene.getJointInfo(name).limits
        assert np.rad2deg([limits.min_position[0], limits.max_position[0]]) == pytest.approx(expected)
        assert limits.max_velocity[0] == 10.0
        assert limits.max_acceleration[0] == 5.0
        assert limits.max_jerk[0] == 5.0
        index = p.scene.getJointPositionIndices([name])[0]
        for bound, direction in zip(expected, [-1, 1]):
            q = np.zeros(len(p.scene.getCurrentJointPositions()))
            q[index] = np.deg2rad(bound + direction * 0.001)
            assert not p.scene.isValidConfiguration(q)

    grip = p.scene.getJointInfo("gripper").limits
    assert grip.min_position[0] == pytest.approx(-0.174533)
    assert grip.max_position[0] == pytest.approx(1.74533)
    p.scene.setRngSeed(101)
    samples = np.array([p.scene.randomPositions() for _ in range(300)])
    for name, (lower, upper) in EXPECTED.items():
        values = np.rad2deg(samples[:, p.scene.getJointPositionIndices([name])[0]])
        assert np.all((values >= lower) & (values <= upper))
    # Sampling can actually use the newly available shoulder range.
    shoulder = samples[:, p.scene.getJointPositionIndices(["shoulder_lift"])[0]]
    assert np.any(shoulder < np.deg2rad(-100))


@pytest.mark.parametrize("measured", [
    {"shoulder_pan": -24.0, "shoulder_lift": -103.780219780,
     "elbow_flex": 95.032967033, "wrist_flex": 22.857142857,
     "wrist_roll": 96.747252747, "gripper": -0.168010825},
    {"shoulder_pan": 8.791208791, "shoulder_lift": -16.571428571,
     "elbow_flex": -33.758241758, "wrist_flex": 95.736263736,
     "wrist_roll": 104.747252747, "gripper": -0.094086062},
])
def test_reported_starts_are_accepted_without_clamping(calibrated_planner, measured):
    p = calibrated_planner
    q = configuration(p, measured)
    assert p.scene.isValidConfiguration(q)
    assert not p.scene.hasCollisions(q)
    # Start validity is independent of the new goal-orientation requirement.
    # Keeping XYZ fixed can now require reorienting the brush, not a no-op.
    p.scene.setJointPositions(q)
    p._validate_start_joint_limits(q, measured)
    np.testing.assert_allclose(p.scene.getCurrentJointPositions(), q, atol=1e-12)


def test_real_ik_constraint_uses_expanded_and_restricted_bounds(calibrated_planner):
    p = calibrated_planner
    q = np.zeros(len(p.scene.getCurrentJointPositions()))
    p.scene.setJointPositions(q)
    target = q.copy()
    shoulder = p.scene.getJointPositionIndices(["shoulder_lift"])[0]
    elbow = p.scene.getJointPositionIndices(["elbow_flex"])[0]
    target[shoulder] = np.deg2rad(-105.0)  # Allowed now, forbidden by nominal URDF.
    target[elbow] = np.deg2rad(95.0)  # Forbidden now, allowed by nominal URDF.
    task = ConfigurationTask(p.ik, target[p.arm_indices], np.ones(p.ik.num_variables))
    # Isolate the real position constraint here; goal and path collision
    # rejection are covered separately.
    for _ in range(20):
        delta = np.zeros(p.ik.num_variables)
        p.ik.solveIk(p.scene, [task], p.ik_limits, [], delta, 1e-9)
        q = p.scene.integrate(q, p.scene.toFullJointVelocities(p.group_name, delta))
        p.scene.setJointPositions(q)
    assert np.rad2deg(q[shoulder]) == pytest.approx(-105.0, abs=1e-3)
    assert np.rad2deg(q[elbow]) == pytest.approx(EXPECTED["elbow_flex"][1], abs=1e-3)


def test_upright_ik_from_reported_shoulder_pose(calibrated_planner, monkeypatch):
    p = calibrated_planner
    measured = {
        "shoulder_pan": -24.0, "shoulder_lift": -103.780219780,
        "elbow_flex": 95.032967033, "wrist_flex": 22.857142857,
        "wrist_roll": 96.747252747, "gripper": -0.168010825,
    }
    start = configuration(p, measured)
    target = np.array([0.08626855950381895, 0.19836798817407578, 0.08])
    p.scene.setRngSeed(101)
    monkeypatch.setattr(p, "ik_max_attempts", 3)
    # Goal reachability and straight-path reachability are different. Preserve
    # the calibration/IK regression without assuming an arbitrary joint-space
    # connection is an acceptable Cartesian line.
    goal = p._solve_xyz(target, start)
    assert p.scene.isValidConfiguration(goal)
    assert p.brush_tilt_deg(goal) <= 10.0
    achieved = p.scene.forwardKinematics(goal, p.tip_frame, p.base_frame)[:3, 3]
    assert np.linalg.norm(achieved - target) <= p.ik_position_tolerance
    with pytest.raises(RuntimeError, match="No collision-free connection|Cartesian segment"):
        p.plan_xyz_path(target, measured)
    np.testing.assert_allclose(p.scene.getCurrentJointPositions(), start, atol=1e-12)


@pytest.mark.parametrize("arm_model_deg", [
    pytest.param([-7.813186813, -105.719780220, 90.184615385, 13.626373626, 90.417582418],
                 id="saved-offset-snapshot"),
    pytest.param([-21.0, -105.280219780, 89.832967033, 22.857142857, 96.747252747],
                 id="reported-folded"),
    pytest.param([-21.0, -105.0, 89.0, 22.0, 60.0], id="tilted-33-degrees"),
    pytest.param([0.0] * 5, id="model-zero-tilted-93-degrees"),
    pytest.param([-43.897898, 1.087683, 58.847912, -57.168766, 92.894642],
                 id="previous-palette-solution"),
])
def test_station_sequence_after_ik_setup_from_several_starts(calibrated_planner, arm_model_deg):
    """Full timed trajectories with captured station settings; never import robot.py."""
    p = calibrated_planner
    p.scene.setRngSeed(101)
    q = np.zeros(len(p.scene.getCurrentJointPositions()))
    q[p.arm_indices] = np.deg2rad(arm_model_deg)
    gripper = p.scene.getJointPositionIndices(["gripper"])[0]
    q[gripper] = np.deg2rad(-0.168010825)
    held_gripper = q[gripper]
    # FRIDA coordinates captured on 2026-10-20. Hover heights follow Painter:
    # Z_CANVAS=-0.02, Z_MAX_CANVAS=0.025, HOVER_FACTOR=0.05.
    neutral = [-0.2260835802956364, 0.1044092100287706, 0.025]
    stations = [
        neutral,
        [-0.2260835802956364, 0.1044092100287706, 0.03],  # rag
        [-0.2129, 0.1931, 0.03],  # water
        [-0.2036, 0.27739, 0.021],  # palette
        neutral,
    ]
    # Initialize at neutral via IK only; this does not claim the straight
    # entry from an arbitrary folded pose is feasible or execute that entry.
    q = p._solve_xyz(np.array([neutral[1], -neutral[0], neutral[2]]), q)
    for x, y, z in stations[1:]:
        target = np.array([y, -x, z])
        measured = {
            name: float(np.rad2deg(q[p.scene.getJointPositionIndices([name])[0]])
                        - p.joint_offsets_deg.get(name, 0.0))
            for name in p.scene.getJointNames()
        }
        trajectory = p.plan_xyz_path(target, measured)
        assert trajectory is not None
        assert "gripper" not in trajectory.joint_names
        assert np.all(np.diff(trajectory.times) > 0)
        output_indices = p.scene.getJointPositionIndices(trajectory.joint_names)
        previous = q.copy()
        aligned = p.brush_tilt_deg(previous) <= 10.0
        for row in trajectory.positions:
            current = q.copy()
            current[output_indices] = row
            assert current[gripper] == held_gripper
            assert p.scene.isValidConfiguration(current)
            # Independently check at finer resolution than production (0.01 rad).
            assert not motion_planner.hasCollisionsAlongPath(
                p.scene, previous, current, max_step_size=0.005,
            )
            steps = max(1, int(np.ceil(np.max(np.abs(current - previous)) / 0.005)))
            for fraction in np.linspace(0.0, 1.0, steps + 1)[1:]:
                tilt = p.brush_tilt_deg(previous + fraction * (current - previous))
                if aligned:
                    assert tilt <= 10.0
                aligned |= tilt <= 10.0
            previous = current
        assert aligned
        achieved = p.scene.forwardKinematics(previous, p.tip_frame, p.base_frame)[:3, 3]
        assert np.linalg.norm(achieved - target) <= 0.001
        assert p.brush_tilt_deg(previous) <= 10.0
        np.testing.assert_allclose(p.scene.getCurrentJointPositions(), q, atol=1e-12)
        q = previous


@pytest.fixture(scope="module")
def water_hover(calibrated_planner):
    """Solve an offline hover fixture without assuming a straight entry path."""
    p = calibrated_planner
    measured = {
        "shoulder_pan": -24.0, "shoulder_lift": -103.780219780,
        "elbow_flex": 95.032967033, "wrist_flex": 22.857142857,
        "wrist_roll": 96.747252747, "gripper": -0.168010825,
    }
    q = configuration(p, measured)
    # Robot coordinates; updated Z_CANVAS=-0.013 and HOVER_FACTOR=0.05.
    for target in ([0.1044092100287706, 0.2260835802956364, 0.025],
                   [0.1931, 0.2129, -0.013 + 0.05]):
        # Construct the fixture with IK only, independently of the approach.
        q = p._solve_xyz(np.asarray(target), q)
        measured = {
            name: float(np.rad2deg(q[p.scene.getJointPositionIndices([name])[0]])
                        - p.joint_offsets_deg.get(name, 0.0))
            for name in p.scene.getJointNames()
        }
    return q, measured


def dipping_positions(seed):
    # Preserve the former random stirring patterns as spline regressions.
    water = np.array([-0.2129, 0.1931, -0.013])
    hover = water + [0.0, 0.0, 0.05]
    jitter = np.clip(np.random.RandomState(seed).randn(5, 2) * 0.01, -0.02, 0.02)
    frida = np.array([hover, water] + [water + [dx, dy, 0.0] for dx, dy in jitter] + [hover])
    return np.column_stack((frida[:, 1], -frida[:, 0], frida[:, 2]))


@pytest.mark.parametrize("hover_height", [0.05, 0.1])
@pytest.mark.parametrize("arm_model_deg", [
    pytest.param([-21.0, -105.280219780, 89.832967033, 22.857142857, 96.747252747],
                 id="reported-folded"),
    pytest.param([0.0] * 5, id="tilted-model-zero"),
    pytest.param([-43.897898, 1.087683, 58.847912, -57.168766, 92.894642],
                 id="previous-palette"),
])
def test_circular_dipping_offline(calibrated_planner, arm_model_deg, hover_height):
    from painter import Painter

    # Skip Painter.__init__: it connects the robot and camera. Capture the real
    # method's motion requests, then plan them without an execution interface.
    painter = Painter.__new__(Painter)
    water = np.array([-0.2129, 0.1931, -0.012])  # Captured current water setting.
    hover = water + [0.0, 0.0, hover_height]
    painter.opt = SimpleNamespace(WATER_POSITION=water, HOVER_FACTOR=hover_height)
    painter.move_to = Mock()
    painter.move_to_trajectories = Mock()
    painter.dip_brush_in_water()
    painter.move_to.assert_called_once_with(*hover)
    painter.move_to_trajectories.assert_called_once()
    positions, orientations = painter.move_to_trajectories.call_args.args
    positions = np.asarray(positions)
    np.testing.assert_allclose(positions[[0, -1]], [hover, hover])
    np.testing.assert_allclose(positions[[1, -2]], [water, water])
    circle = positions[2:-2]
    np.testing.assert_allclose(circle[:, 2], water[2])
    np.testing.assert_allclose(np.linalg.norm(circle[:, :2] - water[:2], axis=1), 0.005)
    np.testing.assert_allclose(circle[0], circle[-1])
    angles = np.unwrap(np.arctan2(circle[:, 1] - water[1], circle[:, 0] - water[0]))
    assert angles[-1] - angles[0] == pytest.approx(2 * np.pi)
    assert np.all(np.diff(angles) > 0)
    assert orientations == [None] * len(positions)

    p = calibrated_planner
    p.scene.setRngSeed(101)
    q = np.zeros(len(p.scene.getCurrentJointPositions()))
    q[p.arm_indices] = np.deg2rad(arm_model_deg)
    gripper = p.scene.getJointPositionIndices(["gripper"])[0]
    q[gripper] = np.deg2rad(-0.168010825)
    # Test the dipping path from a solved hover pose; arbitrary folded-to-hover
    # transits are covered separately and need not admit a straight line.
    h = np.asarray(painter.move_to.call_args.args)
    q = p._solve_xyz(np.array([h[1], -h[0], h[2]]), q)
    for frida in (positions,):
        targets = np.column_stack((frida[:, 1], -frida[:, 0], frida[:, 2]))
        measured = {
            name: float(np.rad2deg(q[p.scene.getJointPositionIndices([name])[0]])
                        - p.joint_offsets_deg.get(name, 0.0))
            for name in p.scene.getJointNames()
        }
        trajectory = p.plan_xyz_path(targets, measured)
        assert trajectory is not None
        assert "gripper" not in trajectory.joint_names
        assert np.all(np.diff(trajectory.times) > 0)
        previous = q.copy()
        aligned = p.brush_tilt_deg(previous) <= 10.0
        for row in trajectory.positions:
            current = q.copy()
            current[p.scene.getJointPositionIndices(trajectory.joint_names)] = row
            assert p.scene.isValidConfiguration(current)
            assert not motion_planner.hasCollisionsAlongPath(
                p.scene, previous, current, max_step_size=0.005,
            )
            steps = max(1, int(np.ceil(np.max(np.abs(current - previous)) / 0.005)))
            for fraction in np.linspace(0.0, 1.0, steps + 1)[1:]:
                tilt = p.brush_tilt_deg(previous + fraction * (current - previous))
                if aligned:
                    assert tilt <= 10.0
                aligned |= tilt <= 10.0
            previous = current
        assert aligned
        achieved = p.scene.forwardKinematics(previous, p.tip_frame, p.base_frame)[:3, 3]
        assert np.linalg.norm(achieved - targets[-1]) <= 0.001
        assert previous[gripper] == q[gripper]
        q = previous


@pytest.mark.parametrize("seed", range(10))
def test_dipping_patterns_preserve_limits_collisions_and_tilt(
    calibrated_planner, water_hover, seed, monkeypatch,
):
    p = calibrated_planner
    start, measured = water_hover
    p.scene.setRngSeed(101)
    targets = dipping_positions(seed)
    # Seeds 1 and 6 previously passed IK but cubic interpolation overshot the
    # wrist_flex lower bound by 0.54 and 4.65 degrees, respectively.
    generate = Mock(wraps=p._quintic_trajectory)
    monkeypatch.setattr(p, "_quintic_trajectory", generate)
    trajectory = p.plan_xyz_path(targets, measured)
    assert trajectory is not None
    assert "gripper" not in trajectory.joint_names
    assert np.all(np.diff(trajectory.times) > 0)
    previous = start.copy()
    output_indices = p.scene.getJointPositionIndices(trajectory.joint_names)
    for row in trajectory.positions:
        current = start.copy()
        current[output_indices] = row
        assert p.scene.isValidConfiguration(current)
        assert not motion_planner.hasCollisionsAlongPath(
            p.scene, previous, current, max_step_size=0.005,
        )
        steps = max(1, int(np.ceil(np.max(np.abs(current - previous)) / 0.005)))
        for fraction in np.linspace(0.0, 1.0, steps + 1):
            assert p.brush_tilt_deg(previous + fraction * (current - previous)) <= 10.0
        previous = current
    achieved = p.scene.forwardKinematics(previous, p.tip_frame, p.base_frame)[:3, 3]
    assert np.linalg.norm(achieved - targets[-1]) <= 0.001
    np.testing.assert_allclose(p.scene.getCurrentJointPositions(), start, atol=1e-12)
    modes = [call.kwargs['stop_at_waypoints'] for call in generate.call_args_list]
    # Timing is now per Cartesian leg. Each leg starts with cubic timing;
    # any fallback remains subject to the same geometric and safety checks.
    assert modes and modes[0] is False


def test_hover_trajectory_is_one_and_half_times_faster(
    calibrated_planner, water_hover, monkeypatch,
):
    p = calibrated_planner
    monkeypatch.setattr(p, "speed_scale", 1.0)  # Avoid the user's model-limit cap.
    _, measured = water_hover
    native = p._quintic_trajectory
    generate = Mock(wraps=native)
    monkeypatch.setattr(p, "_quintic_trajectory", generate)
    trajectory = p.plan_xyz_path([0.1044092100287706, 0.2260835802956364, 0.025], measured)
    baseline = native(
        *generate.call_args.args, **generate.call_args.kwargs,
        velocity_scale=0.1, acceleration_scale=0.1,
    )
    # Durations are rounded up to whole 20 ms command periods.
    assert trajectory.times[-1] == pytest.approx(baseline.times[-1] / 1.5, abs=0.02)


def test_dipping_ik_failure_still_prevents_trajectory_generation(
    calibrated_planner, water_hover, monkeypatch,
):
    p = calibrated_planner
    _, measured = water_hover
    generate = Mock()
    monkeypatch.setattr(p, "_quintic_trajectory", generate)
    monkeypatch.setattr(p, "_solve_xyz", Mock(side_effect=RuntimeError("No collision-free connection")))
    # Deterministic IK failure: the former random seed 0 now succeeds with
    # dense Cartesian IK and is included in the real pattern regressions above.
    with pytest.raises(RuntimeError, match="No collision-free connection"):
        p.plan_xyz_path(dipping_positions(0), measured)
    generate.assert_not_called()


def test_original_low_neutral_target_is_not_accepted_with_excess_tilt(calibrated_planner, monkeypatch):
    p = calibrated_planner
    measured = {
        "shoulder_pan": -24.0, "shoulder_lift": -103.780219780,
        "elbow_flex": 95.032967033, "wrist_flex": 22.857142857,
        "wrist_roll": 96.747252747, "gripper": -0.168010825,
    }
    start = configuration(p, measured)
    # A bounded real solve verifies rejection, not a proof of global infeasibility.
    monkeypatch.setattr(p, "ik_max_attempts", 1)
    monkeypatch.setattr(p, "ik_max_iters", 80)
    monkeypatch.setattr(p, "ik_attempt_timeout", 30.0)
    with pytest.raises(RuntimeError, match="No collision-free connection"):
        p.plan_xyz_path([0.08626855950381895, 0.19836798817407578, 0.015], measured)
    np.testing.assert_allclose(p.scene.getCurrentJointPositions(), start, atol=1e-12)


def test_ik_retries_goal_outside_calibration_even_when_inside_urdf(calibrated_planner, monkeypatch):
    p = calibrated_planner
    monkeypatch.setattr(p, "_validate_brush_tilt_path", Mock())
    start = np.zeros(len(p.scene.getCurrentJointPositions()))
    bad = start.copy()
    bad[p.scene.getJointPositionIndices(["elbow_flex"])[0]] = np.deg2rad(95)
    solve = Mock(side_effect=[bad, start.copy()])
    monkeypatch.setattr(p, "_solve_xyz_attempt", solve)
    monkeypatch.setattr(p, "_sample_ik_seed", Mock(return_value=start.copy()))
    path_check = Mock(return_value=False)
    monkeypatch.setattr(motion_planner, "hasCollisionsAlongPath", path_check)
    np.testing.assert_array_equal(p._solve_xyz(np.array([0.1, 0.2, 0.1]), start), start)
    assert solve.call_count == 2
    assert path_check.call_count == 1


def test_out_of_range_parameterized_sample_is_rejected(calibrated_planner, monkeypatch):
    p = calibrated_planner
    measured = {name: -p.joint_offsets_deg.get(name, 0.0) for name in p.scene.getJointNames()}
    start = configuration(p, measured)
    goal = start.copy()
    goal[0] += 0.01
    bad = goal.copy()
    bad[p.scene.getJointPositionIndices(["elbow_flex"])[0]] = np.deg2rad(95)
    monkeypatch.setattr(p, "_solve_xyz", Mock(return_value=goal))
    monkeypatch.setattr(p, "_quintic_trajectory", Mock(return_value=SimpleNamespace(
        times=[0.0, 1.0], joint_names=p.joint_names,
        positions=[start[p.arm_indices], bad[p.arm_indices]],
    )))
    with pytest.raises(RuntimeError, match="trajectory violates calibrated joint limits"):
        p.plan_xyz_path([0.1, 0.2, 0.1], measured)


@pytest.mark.parametrize("lower,upper", [
    (100, 100), (200, 100), (-1, 100), (0, 4096),
    (True, 100), (0, None), (0, 100.5), (0, float("nan")),
])
def test_invalid_calibration_ranges_fail_clearly(tmp_path, lower, upper):
    path = tmp_path / "calibration.json"
    path.write_text(json.dumps({"elbow_flex": {"range_min": lower, "range_max": upper}}))
    with pytest.raises(ValueError, match="Invalid encoder range for elbow_flex"):
        load_calibrated_joint_limits(path, {"elbow_flex": -5.2})


def test_missing_and_invalid_calibration_files_fail_clearly(tmp_path):
    path = tmp_path / "calibration.json"
    with pytest.raises(FileNotFoundError, match="LeRobot calibration file not found"):
        load_calibrated_joint_limits(path, OFFSETS)
    for text, message in [("{bad", "Invalid LeRobot calibration JSON"),
                          ("[]", "Expected a LeRobot joint calibration mapping"),
                          ("{}", "Missing or invalid calibration for shoulder_pan")]:
        path.write_text(text)
        with pytest.raises(ValueError, match=message):
            load_calibrated_joint_limits(path, OFFSETS)


@pytest.mark.parametrize("air_test", [True, False], ids=["air", "canvas"])
def test_hello_world_offline(calibrated_planner, air_test):
    """Replay all real SO101 planning calls, replacing only hardware I/O."""
    from paint_hello_world import paint_hello_world
    from robot import SO101

    p = calibrated_planner
    p.scene.setRngSeed(101)
    robot = SO101.__new__(SO101)
    robot.motion_planner = p
    robot.arm_joint_names = p.joint_names
    robot.arm = Mock()
    observation = {
        "shoulder_pan.pos": -8.439560440, "shoulder_lift.pos": -1.450549451,
        "elbow_flex.pos": 85.626373627, "wrist_flex.pos": -83.428571429,
        "wrist_roll.pos": 93.230769231, "gripper.pos": 8.938172043010754,
    }
    robot.arm.get_observation.side_effect = lambda: observation.copy()
    gripper = p.scene.getJointPositionIndices(["gripper"])[0]
    grip_limits = p.scene.getJointInfo("gripper").limits
    grip_angle = grip_limits.min_position[0] + observation["gripper.pos"] / 100 * (
        grip_limits.max_position[0] - grip_limits.min_position[0])
    robot.go_to_cartesian_pose = Mock(wraps=robot.go_to_cartesian_pose)

    def execute_offline(times, commands):
        assert len(times) == len(commands) > 0
        assert np.all(np.diff(times) > 0)
        previous = configuration(p, {name: observation[f"{name}.pos"] for name in p.joint_names})
        previous[gripper] = grip_angle
        requested = np.atleast_2d(robot.go_to_cartesian_pose.call_args.args[0])
        # The requested path includes the initial travel to the first point.
        pose = p.scene.forwardKinematics(previous, p.tip_frame, p.base_frame)
        requested = np.vstack(([-pose[1, 3], pose[0, 3], pose[2, 3]], requested))
        for command in commands:
            assert command["gripper.pos"] == observation["gripper.pos"]
            current = configuration(p, {name: command[f"{name}.pos"] for name in p.joint_names})
            current[gripper] = grip_angle
            assert p.scene.isValidConfiguration(current)
            assert p.brush_tilt_deg(current) <= 10.0
            assert not motion_planner.hasCollisionsAlongPath(
                p.scene, previous, current, max_step_size=0.005,
            )
            pose = p.scene.forwardKinematics(current, p.tip_frame, p.base_frame)
            xyz = np.array([-pose[1, 3], pose[0, 3], pose[2, 3]])
            if len(requested) > 2:
                # Also check the actual interpolated lettering, not just IK
                # endpoints: the tip must stay within 2 mm of the intended line.
                segments = np.diff(requested, axis=0)
                lengths2 = np.sum(segments * segments, axis=1)
                keep = lengths2 > 1e-20
                segments, origins = segments[keep], requested[:-1][keep]
                fractions = np.clip(np.sum((xyz - origins) * segments, axis=1)
                                    / lengths2[keep], 0, 1)
                closest = origins + fractions[:, None] * segments
                assert np.min(np.linalg.norm(closest - xyz, axis=1)) <= 0.002
            previous = current
        assert np.linalg.norm(xyz - requested[-1]) <= 0.001
        observation.update(commands[-1])

    robot.execute_trajectory = Mock(side_effect=execute_offline)
    opt = SimpleNamespace(CANVAS_POSITION=[-0.004074, 0.2546],
                          CANVAS_WIDTH_M=0.279, CANVAS_HEIGHT_M=0.216)
    painter = SimpleNamespace(opt=opt, Z_CANVAS=-0.012, robot=robot)
    paint_hello_world(painter, air_test=air_test)
    assert robot.execute_trajectory.call_count == 15
    robot.arm.connect.assert_not_called()
    robot.arm.send_action.assert_not_called()
    assert robot.get_cartesian_pose()[2] == pytest.approx(-0.012 + (0.02 if air_test else 0) + 0.05, abs=0.001)


@pytest.mark.parametrize("phase, measured_arm", [
    pytest.param("lift", [-12.835164835, .659340659, 85.274725275, -75.780219780, 92.0],
                 id="latest-lift"),
    pytest.param("stroke", [-12.835164835, -.395604396, 87.736263736, -78.769230769, 92.0],
                 id="latest-stroke"),
    pytest.param("lift", [-5.450549451, -30.021978022, 89.054945055, -52.659340659, 89.978021978],
                 id="feedback-run-lift"),
    pytest.param("stroke", [-6.065934066, -38.109890110, 89.054945055, -42.373626374, 92.0],
                 id="feedback-run-stroke"),
])
def test_quintic_replays_recorded_diagnostic_poses(calibrated_planner, phase, measured_arm):
    # Captured October 20 diagnostic observations; use only the offline model.
    p = calibrated_planner
    measured = dict(zip(p.joint_names, measured_arm))
    measured['gripper'] = -.168010825
    start = configuration(p, measured)
    xyz = p.scene.forwardKinematics(start, p.tip_frame, p.base_frame)[:3, 3].copy()
    displacement = np.array([0, 0, .01] if phase == 'lift' else [0, -.02, 0])
    targets = ([xyz + displacement] if phase == 'lift'
               else np.linspace(xyz, xyz + displacement, 8))
    p.scene.setRngSeed(101)
    trajectory = p.plan_xyz_path(targets, measured)
    assert trajectory is not None
    np.testing.assert_allclose(np.diff(trajectory.times), .02, atol=1e-12)
    np.testing.assert_array_equal(np.asarray(trajectory.velocities)[[0, -1]], 0)
    np.testing.assert_array_equal(np.asarray(trajectory.accelerations)[[0, -1]], 0)
    assert np.max(abs(np.asarray(trajectory.velocities))) <= 1.5 * p.speed_scale + 1e-10
    assert np.max(abs(np.asarray(trajectory.accelerations))) <= 1.125 * p.speed_scale**2 + 1e-10
    assert np.all(np.linalg.norm(np.asarray(trajectory.velocities)[1:-1], axis=1) > 0)
    previous = start.copy()
    for row in trajectory.positions:
        q = start.copy()
        q[p.arm_indices] = row
        assert p.scene.isValidConfiguration(q)
        assert not motion_planner.hasCollisionsAlongPath(p.scene, previous, q, max_step_size=.005)
        assert p.brush_tilt_deg(q) <= 10.0
        tip = p.scene.forwardKinematics(q, p.tip_frame, p.base_frame)[:3, 3]
        if phase == 'stroke':
            fraction = np.clip(np.dot(tip - xyz, displacement) / np.dot(displacement, displacement), 0, 1)
            assert np.linalg.norm(tip - (xyz + fraction * displacement)) <= .002
        previous = q
    assert np.linalg.norm(tip - targets[-1]) <= .001
    np.testing.assert_allclose(p.scene.getCurrentJointPositions(), start, atol=1e-12)


def test_lift_and_stroke_diagnostic_offline(calibrated_planner):
    from diagnose_so101_motion import MotionLog, run_diagnostic
    from robot import SO101

    robot = SO101.__new__(SO101)
    robot.motion_planner = calibrated_planner
    robot.arm_joint_names = calibrated_planner.joint_names
    observation = {
        "shoulder_pan.pos": -8.439560440, "shoulder_lift.pos": -1.450549451,
        "elbow_flex.pos": 85.626373627, "wrist_flex.pos": -83.428571429,
        "wrist_roll.pos": 93.230769231, "gripper.pos": 8.938172043010754,
    }
    robot.arm = Mock()
    robot.arm.get_observation.side_effect = lambda: observation.copy()
    robot.arm.send_action.side_effect = AssertionError('No hardware commands in offline diagnostic')
    log = MotionLog(robot, offline_observation=observation)
    # Establish a representative HELLO WORLD air pose entirely in the model.
    initial = robot.get_cartesian_pose()
    log.move('setup', [[initial[0], initial[1], .058],
                       [-.046074, .3196, .058], [-.046074, .3196, .008]])
    log.planned.clear()
    log.events.clear()
    start = np.asarray(robot.get_cartesian_pose())
    run_diagnostic(robot, log)
    assert [event['phase'] for event in log.events] == ['lift', 'stroke']
    assert all(event['status'] == 'complete' for event in log.events)
    assert all(event['duration_s'] > 0 for event in log.events)
    assert set(row['phase'] for row in log.planned) == {'lift', 'stroke'}
    assert not log.commands and not log.feedback
    np.testing.assert_allclose(robot.get_cartesian_pose(), start + [.04, 0, .02], atol=.002)
    robot.arm.connect.assert_not_called()
    robot.arm.send_action.assert_not_called()
