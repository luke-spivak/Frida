"""Default loading-test paths through real offline IK, never hardware."""

from unittest.mock import Mock
import pickle

import pytest

from brush_stroke import BrushStroke
from paint_loading import load_setup, make_painter, make_stroke, run_sequences
from test_paint_loading import setup, report_for, ROUND_BRUSH_XY
from test_so101_calibrated_limits import calibrated_planner
from test_canvas_z_planning import offline_robot


@pytest.mark.parametrize("use_xy", [False, True])
def test_default_sheet_and_wash_load_routes_pass_checked_planner(setup, offline_robot, tmp_path, use_xy):
    # Synthetic calibration matching the dimensions/contact range of the round
    # brush setup; neither this test nor its fixtures open a motor/camera port.
    setup.repeats = 1
    if use_xy:
        (setup.cache_dir / "cached_H_coord.pkl").write_bytes(pickle.dumps(ROUND_BRUSH_XY))
    opt, params, grid, _, xy = load_setup(setup)
    painter = make_painter(opt, params, grid, offline_robot, xy)
    report = report_for(setup)
    run_sequences(setup, painter, make_stroke(setup), report, tmp_path, Mock(),
                  Mock(), BrushStroke.execute, Mock())
    assert report["sequences"][0]["completed"] == 10
    assert report["sequences"][0]["status"] == "completed"
    # Real planning is exercised for station transfers and all four production
    # phases per stroke. The fake executor verifies collision/tilt limits.
    assert offline_robot.execute_trajectory.call_count >= 50
