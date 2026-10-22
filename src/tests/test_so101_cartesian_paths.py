"""Offline geometry regressions. No robot/serial/camera objects are created."""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from motion_planner import SO101MotionPlanner
from paint_hello_world import draw_strokes, hello_world_strokes


@pytest.fixture
def planner(tmp_path, lerobot_calibration_path):
    root = Path(__file__).resolve().parents[1]
    offsets = tmp_path / "offsets.json"
    offsets.write_text(json.dumps(dict(zip(
        ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"],
        [3.0, -1.5, -5.2, 0.0, 0.0],
    ))))
    p = SO101MotionPlanner(
        urdf_path=root / "SO101/so101_brush_tip_nominal.urdf",
        srdf_path=root / "SO101/so101_arm.srdf",
        joint_limits_path=root / "SO101/so101_config.yaml",
        joint_offsets_path=offsets, lerobot_calibration_path=lerobot_calibration_path,
    )
    p.scene.setRngSeed(101)
    return p


def saved_start(p):
    # Model coordinates from the last upright saved snapshot; no live reads.
    q = np.zeros(len(p.scene.getCurrentJointPositions()))
    q[p.arm_indices] = np.deg2rad([-5.43956044, -2.95054945, 80.42637363,
                                  -83.42857143, 93.23076923])
    q[p.scene.getJointPositionIndices(["gripper"])[0]] = np.deg2rad(-.168010825)
    return q


def servo_degrees(p, q):
    return {n: float(np.rad2deg(q[p.scene.getJointPositionIndices([n])[0]]))
            - p.joint_offsets_deg.get(n, 0.) for n in p.scene.getJointNames()}


def test_collinear_points_merge_but_corners_and_reversals_remain():
    points = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 1], [0, 0, 2],
                       [0, 1, 2], [0, 2, 2], [0, 1, 2]])
    np.testing.assert_array_equal(SO101MotionPlanner._cartesian_corners(points),
                                  points[[0, 3, 5, 6]])


@pytest.mark.parametrize("bad", ["overshoot", "backtrack", "wrong_endpoint", "between_samples"])
def test_cartesian_validator_rejects_unsafe_geometry(planner, monkeypatch, bad):
    # A small synthetic FK isolates the guard from IK and collision checking.
    def fk(q, *args):
        pose = np.eye(4)
        pose[:3, 3] = [q[0], 0, 0]
        if bad == "between_samples":
            pose[1, 3] = .001 * np.sin(np.pi * q[0] / .01)
        return pose
    monkeypatch.setattr(planner, "scene", SimpleNamespace(forwardKinematics=fk))
    samples = {"overshoot": [0, .012, .01], "backtrack": [0, .008, .003, .01],
               "wrong_endpoint": [0, .008], "between_samples": [0, .01]}[bad]
    rows = np.zeros((len(samples), 5))
    rows[:, 0] = samples
    tr = SimpleNamespace(positions=rows)
    with pytest.raises(RuntimeError, match="Cartesian segment"):
        planner._validate_cartesian_segment(tr, np.zeros(6), np.zeros(3), np.array([.01, 0, 0]))


def check_leg(p, start, end, q, tr):
    """Independently sample between commands, and measure the assigned line."""
    rows = np.asarray(tr.positions)
    samples = np.concatenate([a + np.linspace(0, 1, 9)[:, None] * (b-a)
                              for a, b in zip(rows, rows[1:])])
    xyz, tilts = [], []
    for row in samples:
        full = q.copy()
        full[p.arm_indices] = row
        xyz.append(p.scene.forwardKinematics(full, p.tip_frame, p.base_frame)[:3, 3])
        tilts.append(p.brush_tilt_deg(full))
    xyz = np.asarray(xyz)
    direction = end - start
    projection = np.clip((xyz-start) @ direction / np.dot(direction, direction), 0, 1)
    error = np.linalg.norm(xyz - (start + projection[:, None]*direction), axis=1)
    assert error.max() <= p.cartesian_path_tolerance
    assert max(tilts) <= 10.0
    np.testing.assert_allclose(np.asarray(tr.velocities)[[0, -1]], 0, atol=1e-12)
    np.testing.assert_allclose(np.asarray(tr.accelerations)[[0, -1]], 0, atol=1e-12)
    # No artificial stops at the 2 mm IK samples along a straight leg.
    moving = np.linalg.norm(rows[-1]-rows[0]) > 1e-6
    if moving:
        assert np.all(np.linalg.norm(np.asarray(tr.velocities)[1:-1], axis=1) > 1e-10)
    return float(error.max())


@pytest.mark.parametrize("air", [True, False], ids=["air", "canvas"])
def test_full_hello_world_vertical_lifts_and_cartesian_lines(planner, monkeypatch, air):
    p = planner
    q = saved_start(p)
    p.scene.setJointPositions(q)
    # Same current canvas layout and 50 mm lift as the user's Hello World.
    opt = SimpleNamespace(CANVAS_POSITION=[-.004074, .2546], CANVAS_WIDTH_M=.279, CANVAS_HEIGHT_M=.216)
    strokes = hello_world_strokes(opt, -.012 + (.02 if air else 0))
    native = p._plan_cartesian_segment
    errors = []

    def segment(start, end, current):
        tr = native(start, end, current)
        errors.append(check_leg(p, start, end, current, tr))
        return tr
    monkeypatch.setattr(p, "_plan_cartesian_segment", segment)

    def get_pose():
        xyz = p.scene.forwardKinematics(q, p.tip_frame, p.base_frame)[:3, 3]
        return np.array([-xyz[1], xyz[0], xyz[2]])

    calls = []
    def go(points):
        xyz = np.asarray(points)
        robot_xyz = np.column_stack((xyz[:, 1], -xyz[:, 0], xyz[:, 2]))
        tr = p.plan_xyz_path(robot_xyz, servo_degrees(p, q))
        assert tr is not None
        np.testing.assert_allclose(np.diff(tr.times), .02, atol=1e-12)
        assert "gripper" not in tr.joint_names
        q[p.arm_indices] = tr.positions[-1]
        calls.append(tr)

    draw_strokes(SimpleNamespace(get_cartesian_pose=get_pose, go_to_cartesian_pose=go), strokes)
    assert len(calls) == len(strokes) + 1
    assert len(errors) > len(strokes) * 3


def test_small_request_and_same_position_do_not_break_timing(planner):
    p = planner
    q = saved_start(p)
    xyz = p.scene.forwardKinematics(q, p.tip_frame, p.base_frame)[:3, 3]
    measured = servo_degrees(p, q)
    assert p.plan_xyz_path(xyz, measured) is None
    tr = p.plan_xyz_path(xyz + [0, 0, .00001], measured)
    assert np.isfinite(tr.positions).all()
    assert np.isfinite(tr.velocities).all()
    assert np.isfinite(tr.accelerations).all()


def test_tilted_start_can_align_during_a_straight_lift(planner):
    p = planner
    q = saved_start(p)
    q[p.scene.getJointPositionIndices(["wrist_roll"])[0]] = np.deg2rad(70)
    assert p.brush_tilt_deg(q) > 20
    start = p.scene.forwardKinematics(q, p.tip_frame, p.base_frame)[:3, 3]
    tr = p.plan_xyz_path(start + [0, 0, .04], servo_degrees(p, q))
    aligned = False
    for row in tr.positions:
        q[p.arm_indices] = row
        tilt = p.brush_tilt_deg(q)
        if aligned:
            assert tilt <= 10
        aligned |= tilt <= 10
        xyz = p.scene.forwardKinematics(q, p.tip_frame, p.base_frame)[:3, 3]
        assert np.linalg.norm(xyz[:2] - start[:2]) < p.cartesian_path_tolerance
    assert aligned
