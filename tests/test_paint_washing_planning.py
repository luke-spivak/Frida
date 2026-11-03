"""Washing experiment with real offline planning and a fake robot executor."""

from unittest.mock import Mock
import pickle

import pytest

from brush_stroke import BrushStroke
from paint_loading import load_setup, make_painter, make_stroke
from paint_washing import run_batches
from test_paint_loading import setup, ROUND_BRUSH_XY
from test_paint_washing import washing_args, report_for
from test_so101_calibrated_limits import calibrated_planner
from test_canvas_z_planning import offline_robot


@pytest.mark.parametrize("use_xy", [False, True])
def test_default_washing_and_recovery_paths_pass_real_planner(setup, offline_robot, tmp_path, use_xy):
    if use_xy:
        (setup.cache_dir / "cached_H_coord.pkl").write_bytes(pickle.dumps(ROUND_BRUSH_XY))
    args = washing_args(setup)
    opt, params, grid, _, xy = load_setup(args)
    painter = make_painter(opt, params, grid, offline_robot, xy)
    report = report_for(args)
    run_batches(args, painter, make_stroke(args), report, tmp_path, Mock(), Mock(), BrushStroke.execute, Mock())
    assert report["completed_strokes"] == 35
    assert report["strokes_since_wash"] == 5
    assert all(s["status"] == "completed" for s in report["sheets"])
    assert offline_robot.execute_trajectory.call_count >= 140
