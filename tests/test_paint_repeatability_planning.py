"""Real IK and production brush execution for every direction; fake hardware."""

import pickle
from unittest.mock import Mock

import pytest

from brush_stroke import BrushStroke
from paint_loading import load_setup, make_painter, make_stroke
from paint_repeatability import run_sheets
from test_paint_repeatability import repeatability_args, report_for
from test_paint_loading import setup, ROUND_BRUSH_XY
from test_so101_calibrated_limits import calibrated_planner
from test_canvas_z_planning import offline_robot


@pytest.mark.parametrize("use_xy", [False, True])
def test_all_marks_and_care_routes_pass_real_planner(setup, offline_robot, tmp_path, use_xy):
    args = repeatability_args(setup)
    if use_xy:
        (args.cache_dir / "cached_H_coord.pkl").write_bytes(pickle.dumps(ROUND_BRUSH_XY))
    opt, params, grid, _, xy = load_setup(args)
    painter = make_painter(opt, params, grid, offline_robot, xy)
    report = report_for(args)
    # With real XY, include all repeats to cover changed ordering/transfers.
    if not use_xy:
        report["sheets"] = report["sheets"][:1]
    run_sheets(args, painter, make_stroke(args), report, tmp_path, Mock(), Mock(),
               BrushStroke.execute, prompt=lambda _: None)
    assert report["completed_strokes"] == (36 if use_xy else 12)
    assert offline_robot.execute_trajectory.call_count >= report["completed_strokes"] * 4
