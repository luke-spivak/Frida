"""Station transfers with real offline IK and fake hardware interfaces only."""

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from painter import Painter
from robot import MotionPlanningError
from test_so101_calibrated_limits import calibrated_planner
from test_canvas_z_planning import offline_robot


@pytest.fixture
def painter(offline_robot):
    painter = Painter.__new__(Painter)  # Never construct/connect hardware.
    painter.robot = offline_robot
    painter.Z_CANVAS = .0015
    painter.opt = SimpleNamespace(
        robot="so101", simulate=False, HOVER_FACTOR=.1,
        CANVAS_POSITION=[-.011450742868149252, .2744202253733551],
        CANVAS_WIDTH_M=.216, CANVAS_HEIGHT_M=.139,
        WATER_POSITION=[-.2129, .1931, .0015],
        RAG_POSTITION=[-.2601, .1122, .0015],
        PALLETTE_POSITION=[-.2136, .27739, -.0006], PAINT_DIFFERENCE=.03976,
    )
    return painter


def test_current_station_transfers_use_vertical_horizontal_vertical_legs(painter, monkeypatch):
    robot, planner = painter.robot, painter.robot.motion_planner
    native = planner._plan_cartesian_segment
    legs = []

    def check_leg(start, end, q):
        # Test the assigned geometry, then run all production IK/path checks.
        legs.append((start.copy(), end.copy()))
        return native(start, end, q)

    monkeypatch.setattr(planner, "_plan_cartesian_segment", check_leg)
    targets = [(-.01, .34, .0415), (-.2129, .1931, .1015),
               (-.2601, .1122, .1015), (-.2136, .27739, .0994),
               (-.01, .34, .0315), (-.2601, .1122, .025)]
    for target in targets:
        legs.clear()
        painter.travel_to(*target)
        assert legs
        for start, end in legs:
            if np.linalg.norm(end[:2] - start[:2]) > 1e-9:
                assert start[2] == pytest.approx(end[2])
                assert start[2] >= .1015 - 1e-9
            else:
                np.testing.assert_allclose(start[:2], end[:2], atol=1e-12)
        np.testing.assert_allclose(robot.get_cartesian_pose(), target,
                                   atol=planner.cartesian_path_tolerance)
    assert robot.execute_trajectory.call_count == len(targets)


def test_stations_and_neutral_use_transfer_but_local_canvas_keeps_old_approach(painter):
    robot = painter.robot
    robot.go_to_cartesian_pose = Mock()
    painter.move_to_trajectories = Mock()
    painter.clean_paint_brush()
    requests = robot.go_to_cartesian_pose.call_args_list
    assert len(requests) == 2  # Water then rag, no old diagonal 9 cm approach.
    np.testing.assert_allclose(requests[0].args[0], [[-.2129, .1931, .1015]])
    np.testing.assert_allclose(requests[1].args[0], [[-.2601, .1122, .1015]])
    assert all(call.kwargs == {"travel_z": .1015} for call in requests)
    painter.get_paint(0)
    assert robot.go_to_cartesian_pose.call_args.kwargs == {"travel_z": .1015}
    np.testing.assert_allclose(robot.go_to_cartesian_pose.call_args.args[0],
                               [[-.2136, .27739, .0994]])
    painter.approach_stroke(-.01, .34)
    assert robot.go_to_cartesian_pose.call_args.kwargs == {"travel_z": .1015}
    assert painter._at_canvas
    painter.approach_stroke(.01, .35)
    assert "travel_z" not in robot.go_to_cartesian_pose.call_args.kwargs
    assert "fallback_positions" in robot.go_to_cartesian_pose.call_args.kwargs
    painter.to_neutral()
    assert robot.go_to_cartesian_pose.call_args.kwargs == {"travel_z": .1015}
    assert not painter._at_canvas


@pytest.mark.parametrize("paint_index", [0, 7])
def test_paint_dabs_lift_before_sideways_motion_and_resample(painter, monkeypatch, paint_index):
    painter.travel_to = Mock()
    painter.move_to_trajectories = Mock()
    samples = [np.array(p) for p in [(0, 0), (1, -1), (10, -10),
                                     (-1, 1), (-10, 10), (.5, .5)]]
    monkeypatch.setattr(np.random, "randn", Mock(side_effect=samples))
    center = np.array(painter.opt.PALLETTE_POSITION[:2]) + painter.opt.PAINT_DIFFERENCE * np.array(
        [paint_index // 6, paint_index % 6])
    depth = painter.opt.PALLETTE_POSITION[2]
    loads = []
    for _ in range(2):
        painter.get_paint(paint_index)
        positions, orientations = painter.move_to_trajectories.call_args.args
        positions = np.array(positions)
        contacts = positions[np.isclose(positions[:, 2], depth)]
        assert len(contacts) == 3
        assert np.all(np.abs(contacts[:, :2] - center) <= .015 + 1e-12)
        assert orientations == [None] * len(positions)
        for start, end in zip(positions[:-1], positions[1:]):
            if not np.allclose(start[:2], end[:2]):
                assert min(start[2], end[2]) >= depth + .02 - 1e-12
        for index in np.flatnonzero(np.isclose(positions[:, 2], depth)):
            for neighbor in (positions[index - 1], positions[index + 1]):
                np.testing.assert_allclose(neighbor[:2], positions[index, :2])
                assert neighbor[2] == pytest.approx(depth + .02)
        loads.append(contacts)
    assert not np.array_equal(loads[0], loads[1])


def test_failed_second_leg_sends_no_movement_and_never_tries_lower_route(painter, monkeypatch):
    planner, robot = painter.robot.motion_planner, painter.robot
    native = planner._plan_cartesian_segment
    calls = []

    def fail_traverse(start, end, q):
        calls.append((start.copy(), end.copy()))
        if len(calls) == 2:
            raise RuntimeError("horizontal transfer blocked")
        return native(start, end, q)

    monkeypatch.setattr(planner, "_plan_cartesian_segment", fail_traverse)
    painter._at_canvas = True
    with pytest.raises(MotionPlanningError, match="horizontal transfer blocked"):
        painter.travel_to(-.2129, .1931, .1015)
    assert len(calls) == 2
    robot.execute_trajectory.assert_not_called()
    assert not painter._at_canvas


def test_transfer_cannot_be_combined_with_lower_fallback(painter):
    with pytest.raises(ValueError, match="cannot use fallback"):
        painter.robot.go_to_cartesian_pose([[0, .34, .03]], travel_z=.1015,
                                          fallback_positions=[[0, .34, .03]])
    painter.robot.arm.get_observation.assert_not_called()
    painter.robot.execute_trajectory.assert_not_called()


def test_transfer_preserves_higher_start_until_destination(painter, monkeypatch):
    robot, planner = painter.robot, painter.robot.motion_planner
    start = robot.get_cartesian_pose()
    target = [start[0] + .005, start[1], start[2] - .01]
    native = planner._plan_cartesian_segment
    legs = []

    def capture(a, b, q):
        legs.append((a.copy(), b.copy()))
        return native(a, b, q)

    monkeypatch.setattr(planner, "_plan_cartesian_segment", capture)
    robot.go_to_cartesian_pose([target], travel_z=start[2] - .02)
    assert len(legs) == 2
    assert legs[0][0][2] == pytest.approx(start[2])
    assert legs[0][1][2] == pytest.approx(start[2])
    np.testing.assert_allclose(legs[1][0][:2], legs[1][1][:2])


@pytest.mark.parametrize("targets,height", [([[0, .3, .03]], np.nan),
                                            ([[0, .3, .03]], np.inf),
                                            ([[0, .3, .03], [0, .4, .03]], .1)])
def test_invalid_transfer_sends_no_movement(painter, targets, height):
    with pytest.raises(MotionPlanningError, match="one destination and a finite height"):
        painter.robot.go_to_cartesian_pose(targets, travel_z=height)
    painter.robot.execute_trajectory.assert_not_called()


def test_simulated_transfers_do_not_read_or_move_robot(painter):
    painter.opt.simulate = True
    painter.travel_to(0, .34, .03)
    painter.robot.arm.get_observation.assert_not_called()
    painter.robot.execute_trajectory.assert_not_called()
