"""Letter layout and pen-up sequencing, without a robot connection."""

import json
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from paint_hello_world import hello_world_strokes, main, paint_hello_world


@pytest.mark.parametrize("air_test", [False, True])
def test_air_height_and_pen_up_transfers(air_test):
    opt = SimpleNamespace(CANVAS_POSITION=[-0.004074, 0.2546],
                          CANVAS_WIDTH_M=0.279, CANVAS_HEIGHT_M=0.216)
    robot = Mock()
    robot.get_cartesian_pose.return_value = [-0.26, 0.11, 0.025]
    painter = SimpleNamespace(opt=opt, Z_CANVAS=-0.012, robot=robot)
    paint_hello_world(painter, air_test=air_test)
    z = -0.012 + (0.02 if air_test else 0.0)
    strokes = hello_world_strokes(opt, z)
    all_points = np.vstack(strokes)
    assert np.ptp(all_points[:, 0]) == pytest.approx(0.084)
    assert np.ptp(all_points[:, 1]) == pytest.approx(0.05)
    assert np.min(all_points[:, 1]) == pytest.approx(opt.CANVAS_POSITION[1] + 0.035)
    assert np.all(all_points[:, 2] == z)
    calls = robot.go_to_cartesian_pose.call_args_list
    assert len(calls) == 1 + len(strokes)
    np.testing.assert_allclose(calls[0].args[0], [[-0.26, 0.11, max(0.025, z + 0.05)]])
    for index, stroke in enumerate(strokes):
        path = np.asarray(calls[index + 1].args[0])
        np.testing.assert_allclose(path[0], [*stroke[0, :2], z + 0.05])
        np.testing.assert_allclose(path[1:-1], stroke)
        np.testing.assert_allclose(path[-1], [*stroke[-1, :2], z + 0.05])


def test_oversized_letters_rejected():
    opt = SimpleNamespace(CANVAS_POSITION=[0, 0], CANVAS_WIDTH_M=0.1, CANVAS_HEIGHT_M=0.1)
    with pytest.raises(ValueError, match="does not fit"):
        hello_world_strokes(opt, 0, letter_height=0.1)


def test_cli_defaults_to_preview_without_hardware(tmp_path, monkeypatch):
    materials = tmp_path / "materials.json"
    calibration = tmp_path / "calibration.json"
    preview = tmp_path / "hello.svg"
    materials.write_text(json.dumps(dict(CANVAS_POSITION=[0, 0.25],
                                        CANVAS_WIDTH_M=0.279, CANVAS_HEIGHT_M=0.216)))
    calibration.write_text(json.dumps({"Z_CANVAS": -0.012}))
    # Importing the hardware entry point in preview mode would fail this test.
    monkeypatch.setitem(sys.modules, "robot", None)
    main(["--materials", str(materials), "--calibration", str(calibration), "--preview", str(preview)])
    assert '<polyline' in preview.read_text()
