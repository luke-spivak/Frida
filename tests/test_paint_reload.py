"""Exercise paint.py's execution loop with fake strokes and no hardware imports."""

import ast
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest


@pytest.fixture(scope="module")
def execution_loop():
    # Run the actual loop, excluding startup that connects the robot/camera and
    # trains models. Every dependency used by this block is supplied below.
    path = Path(__file__).resolve().parents[1] / "src/paint.py"
    main = next(node for node in ast.parse(path.read_text()).body
                if isinstance(node, ast.If))
    counters = {"consecutive_paints", "consecutive_strokes_no_clean",
                "curr_color", "strokes_per_adaptation"}
    session = next(node for node in main.body if isinstance(node, ast.With))
    body = [node for node in session.body if isinstance(node, ast.While) or (
        isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id in counters
            for target in node.targets))]
    assert len(body) == 5
    return compile(ast.Module(body=body, type_ignores=[]), str(path), "exec")


def tensor_value(value):
    tensor = Mock()
    tensor.detach.return_value = tensor
    tensor.cpu.return_value = tensor
    tensor.numpy.return_value = np.asarray(value)
    tensor.item.return_value = value
    return tensor


@pytest.fixture
def run_painting(execution_loop):
    def run(colors, interval=4, adaptations=1, ink=False, fail_at=None, skip_failed=False):
        completed, dips = [], []
        motion_error = type("MotionPlanningError", (RuntimeError,), {})

        class Painting(list):
            @property
            def brush_strokes(self):
                return self

            def pop(self):
                return super().pop(0)

        class Checkpoint:
            completed = 0
            begin_stroke = Mock()
            update_remaining = Mock()
            save = Mock()

            @property
            def next_palette_index(self):
                return colors[self.completed]

            def finish_stroke(self, path):
                self.completed += 1

            def skip_stroke(self, path, reason):
                self.completed += 1

        def stroke(index, color):
            def execute(*args):
                if index == fail_at:
                    error = motion_error if skip_failed else RuntimeError
                    raise error("offline stroke failure")
                completed.append(index)

            return SimpleNamespace(
                color_transform=tensor_value([color]), execute=execute,
                transformation=SimpleNamespace(
                    xt=tensor_value(0.0), yt=tensor_value(0.0), a=tensor_value(0.0)),
            )

        painting = Painting(stroke(i, color) for i, color in enumerate(colors, 1))
        opt = SimpleNamespace(ink=ink, how_often_to_get_paint=interval,
                              num_adaptations=adaptations, optim_iter=1,
                              skip_failed_strokes=skip_failed)
        checkpoint = Checkpoint()
        camera = Mock()
        camera.get_canvas_tensor.return_value.to.return_value = np.zeros((1, 3, 2, 2))
        painter = SimpleNamespace(
            opt=opt, camera=camera, clean_paint_brush=Mock(), to_neutral=Mock(),
            get_paint=lambda color: dips.append((checkpoint.completed + 1, color)),
        )
        palette = tensor_value(np.unique(colors))
        optimize = Mock(return_value=(painting, palette))
        state = dict(painter=painter, opt=opt, painting=painting, color_palette=palette,
                     MotionPlanningError=motion_error,
                     checkpoint=checkpoint, plan_path=None, loaded_plan=None, recording=Mock(),
                     device="cpu", h_render=2, w_render=2, optimize_painting=optimize,
                     nearest_color=lambda color, palette: (int(color[0]), None),
                     canvas_to_global_coordinates=lambda x, y, z, opt: (x, y, z))
        if fail_at is None or skip_failed:
            exec(execution_loop, state)
        else:
            with pytest.raises(RuntimeError, match="offline stroke failure"):
                exec(execution_loop, state)
        return SimpleNamespace(state=state, completed=completed, dips=dips,
                               painter=painter, optimize=optimize)

    return run


@pytest.mark.parametrize("interval, expected", [
    (0, list(range(1, 11))),
    (1, list(range(1, 11))),
    (4, [1, 5, 9]),
    (5, [1, 6]),
    (10, [1]),
])
def test_single_color_reload_interval(run_painting, interval, expected):
    result = run_painting([0] * 10, interval=interval)
    assert result.dips == [(stroke, 0) for stroke in expected]
    assert result.completed == list(range(1, 11))


def test_reload_counter_survives_replanning(run_painting):
    result = run_painting([0] * 10, adaptations=2)
    assert result.dips == [(1, 0), (5, 0), (9, 0)]
    assert result.optimize.call_count == 1


def test_color_change_reloads_and_restarts_interval(run_painting):
    result = run_painting([0, 0, 1, 1, 1, 1, 1, 0])
    assert result.dips == [(1, 0), (3, 1), (7, 1), (8, 0)]
    assert result.painter.clean_paint_brush.call_count == 3


def test_failed_stroke_does_not_advance_counter(run_painting):
    result = run_painting([0] * 10, fail_at=4)
    assert result.completed == [1, 2, 3]
    assert result.dips == [(1, 0)]
    assert result.state["consecutive_paints"] == 3
    assert result.state["consecutive_strokes_no_clean"] == 3
    result.optimize.assert_not_called()


def test_ink_never_loads_paint(run_painting):
    result = run_painting([0] * 10, interval=1, ink=True)
    assert result.completed == list(range(1, 11))
    assert result.dips == []
    result.painter.clean_paint_brush.assert_not_called()


def test_clean_after_twelve_completed_strokes_and_reload_immediately(run_painting):
    result = run_painting([0] * 26, interval=5, adaptations=2)
    assert result.dips == [(i, 0) for i in [1, 6, 11, 13, 18, 23, 25]]
    # Match the authors' released collection routine: one water/rag pass.
    assert result.painter.clean_paint_brush.call_count == 3
    assert result.state["consecutive_strokes_no_clean"] == 2
    starts = [call.kwargs["stroke"] for call in result.state["recording"].event.call_args_list
              if call.args[0] == "clean_start"]
    assert starts == [1, 13, 25]


def test_color_change_restarts_cleaning_interval(run_painting):
    result = run_painting([0] * 10 + [1] * 13, interval=5)
    starts = [call.kwargs["stroke"] for call in result.state["recording"].event.call_args_list
              if call.args[0] == "clean_start"]
    assert starts == [1, 11, 23]


def test_skipped_strokes_do_not_count_as_completed_for_reload_or_cleaning(run_painting):
    result = run_painting([0] * 14, interval=5, fail_at=4, skip_failed=True)
    assert result.dips == [(i, 0) for i in [1, 7, 12, 14]]
    events = result.state["recording"].event.call_args_list
    assert [call.kwargs["stroke"] for call in events if call.args[0] == "clean_start"] == [1, 14]
    assert [call.kwargs["stroke"] for call in events if call.args[0] == "stroke_skipped"] == [4]
    assert result.state["consecutive_strokes_no_clean"] == 1


def test_painting_defaults_and_recording_controls():
    import argparse
    from options import Options
    parser = Options().initialize(argparse.ArgumentParser())
    opt = parser.parse_args([])
    assert opt.how_often_to_get_paint == 4
    assert not opt.no_recording and opt.recording_fps == 10
    opt = parser.parse_args(["--no_recording", "--recording_dir", "/tmp/video", "--recording_fps", "15"])
    assert opt.no_recording and opt.recording_dir == "/tmp/video" and opt.recording_fps == 15


def test_default_baseline_reloads_four_and_washes_twelve(run_painting):
    result = run_painting([0] * 36)
    assert result.dips == [(i, 0) for i in range(1, 37, 4)]
    assert result.painter.clean_paint_brush.call_count == 3


def test_collection_care_uses_configurable_reload_and_one_wash_per_twelve():
    # Exercise the actual collection block without startup, model, camera, or
    # hardware. This catches divergence between collection and physical replay.
    path = Path(__file__).resolve().parents[1] / "src/painter.py"
    function = next(n for n in ast.walk(ast.parse(path.read_text()))
                    if isinstance(n, ast.FunctionDef) and n.name == "paint_extended_stroke_library")
    block = next(n for n in ast.walk(function) if isinstance(n, ast.If)
                 and ast.unparse(n.test) == "not self.opt.ink")
    code = compile(ast.Module(body=[block], type_ignores=[]), str(path), "exec")
    for interval in (4, 5):
        washed, loaded = [], []
        painter = SimpleNamespace(opt=SimpleNamespace(ink=False, how_often_to_get_paint=interval),
                                  clean_paint_brush=lambda: washed.append(index),
                                  get_paint=lambda slot: loaded.append(index))
        state = dict(self=painter, strokes_without_cleaning=9999, strokes_without_getting_new_paint=999)
        for index in range(1, 37):
            exec(code, state)
        assert washed == [1, 13, 25]
        assert loaded == [start + offset for start in (1, 13, 25) for offset in range(0, 12, interval)]
