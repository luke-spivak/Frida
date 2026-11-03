"""Saved plans and paint.py recovery, with no robot/camera connections."""

import ast
import copy
import datetime
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from brush_stroke import BrushStroke
from painting import Painting
import stroke_plan
from stroke_plan import CACHE_FILES, DEFAULT_FILENAME, StrokePlan
from paint_recording import record_painting


def nearest_color(color, palette):
    index = int(np.argmin(np.sum((palette - color) ** 2, axis=1)))
    return index, palette[index]


@pytest.fixture
def setup(tmp_path, monkeypatch):
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "stroke_library").mkdir()
    for name in CACHE_FILES:
        (cache / name).write_bytes(name.encode())
    opt = SimpleNamespace(
        robot="so101", ink=False, use_cache=True, simulate=False,
        CANVAS_WIDTH_M=0.279, CANVAS_HEIGHT_M=0.216,
        CANVAS_POSITION=[-0.004074, 0.2546], PALLETTE_POSITION=[-0.2136, 0.27739],
        PAINT_DIFFERENCE=0.03976, MIN_STROKE_LENGTH=0.005, MAX_STROKE_LENGTH=0.06,
        MIN_STROKE_Z=0.1, MAX_BEND=0.02, MAX_ALPHA=0.0,
        cache_dir=str(cache), render_height=64, dont_retrain_stroke_model=False,
        save_plan=None, load_plan=None, dont_plan=False, use_colors_from=None,
        num_strokes=3, num_adaptations=1, init_optim_iter=30, optim_iter=10,
        how_often_to_get_paint=4, tensorboard_dir=str(tmp_path / "logs"),
        n_colors=3, gather_options=Mock(), no_recording=False,
        recording_dir=str(tmp_path / "recordings"), recording_fps=10,
    )
    # A deterministic synthetic renderer tests the real Painting/BrushStroke
    # transformations and composition without loading/training any model.
    monkeypatch.setattr("painting.get_param2img", lambda *args, **kwargs:
                        lambda params, h, w: torch.full((len(params), h, w), 0.45, device=params.device))
    palette = torch.tensor([[0.8, 0.1, 0.2], [0.1, 0.7, 0.3], [0.2, 0.3, 0.9]])
    strokes = []
    for i, index in enumerate([2, 0, 1]):
        strokes.append(BrushStroke(
            opt, stroke_length=torch.tensor([0.02 + i * 0.01]),
            stroke_z=torch.tensor([0.4 + i * 0.2]), stroke_bend=torch.tensor([0.003 * i]),
            stroke_alpha=torch.tensor([0.0]), color=palette[index].clone(),
            a=torch.tensor([i * 0.7]), xt=torch.tensor([-0.7 + 0.6 * i]),
            yt=torch.tensor([0.4 - 0.5 * i]),
        ))
    background = torch.ones(1, 3, 64, 82)
    painting = Painting(opt, background_img=background, brush_strokes=strokes)
    plan = StrokePlan.capture(painting, palette, opt, nearest_color)
    return SimpleNamespace(opt=opt, palette=palette, painting=painting, plan=plan,
                           background=background, path=cache / DEFAULT_FILENAME)


def test_round_trip_preserves_parameters_render_and_palette_slots(setup):
    s = setup
    s.plan.save(s.path)
    loaded = StrokePlan.load(s.path)
    loaded.check_setup(s.opt)
    restored, palette = loaded.restore(s.opt, s.background, "cpu")
    assert torch.equal(palette, s.palette)
    assert [record["palette_index"] for record in loaded.data["strokes"]] == [2, 0, 1]
    before, after = s.painting.state_dict(), restored.state_dict()
    assert before.keys() == after.keys()
    for name in before:
        assert torch.equal(before[name], after[name]), name
    with torch.no_grad():
        assert torch.equal(s.painting(64, 82), restored(64, 82))


def test_progress_keeps_failed_stroke_pending(setup):
    s = setup
    s.plan.begin_stroke(s.path)
    s.plan.finish_stroke(s.path)
    s.plan.begin_stroke(s.path)
    loaded = StrokePlan.load(s.path)
    assert loaded.completed == 1
    assert loaded.remaining == 2
    assert loaded.data["in_progress"] is True
    restored, palette = loaded.restore(s.opt, s.background, "cpu")
    assert len(restored) == 2
    assert torch.equal(restored.brush_strokes[0].transformation.xt,
                       s.painting.brush_strokes[1].transformation.xt)
    assert torch.equal(palette, s.palette)
    assert loaded.next_palette_index == 0


def test_adaptation_preserves_completed_prefix_and_palette(setup):
    s = setup
    s.plan.begin_stroke(s.path)
    s.plan.finish_stroke(s.path)
    prefix = copy.deepcopy(s.plan.data["strokes"][0])
    remaining = SimpleNamespace(brush_strokes=[s.painting.brush_strokes[2], s.painting.brush_strokes[1]])
    s.plan.update_remaining(remaining, nearest_color)
    s.plan.save(s.path)
    loaded = StrokePlan.load(s.path)
    assert loaded.data["strokes"][0] == prefix
    assert [record["palette_index"] for record in loaded.data["strokes"]] == [2, 1, 0]
    assert loaded.data["palette"] == s.palette.tolist()


def test_failed_atomic_write_preserves_previous_checkpoint(setup, monkeypatch):
    s = setup
    s.plan.save(s.path)
    original = s.path.read_bytes()
    monkeypatch.setattr(stroke_plan.os, "replace", Mock(side_effect=OSError("disk failure")))
    with pytest.raises(OSError, match="disk failure"):
        s.plan.begin_stroke(s.path)
    assert s.path.read_bytes() == original
    assert list(s.path.parent.glob(f".{s.path.name}.*")) == []


@pytest.mark.parametrize("mutation", [
    lambda d: d.update(version=999),
    lambda d: d.update(completed_strokes=4),
    lambda d: d.update(completed_strokes=3, in_progress=True),
    lambda d: d["strokes"][0].update(xt=float("nan")),
    lambda d: d["strokes"][0].update(length=-1),
    lambda d: d["strokes"][0].update(palette_index=99),
    lambda d: d["strokes"][0].update(palette_index=0),
    lambda d: d["palette"][0].__setitem__(0, 2.0),
])
def test_invalid_plans_rejected(setup, mutation):
    data = copy.deepcopy(setup.plan.data)
    mutation(data)
    with pytest.raises(ValueError):
        StrokePlan(data)


@pytest.mark.parametrize("change, message", [
    (lambda s: setattr(s.opt, "CANVAS_WIDTH_M", 0.5), "setup differs"),
    (lambda s: s.opt.PALLETTE_POSITION.__setitem__(0, 0.1), "setup differs"),
    (lambda s: (s.path.parent / "cached_H_coord.pkl").write_bytes(b"changed"), "changed"),
    (lambda s: (s.path.parent / "param2img.pt").unlink(), "missing"),
    (lambda s: setattr(s.opt, "use_cache", False), "requires --use_cache"),
])
def test_setup_mismatch_rejected(setup, change, message):
    change(setup)
    with pytest.raises(ValueError, match=message):
        setup.plan.check_setup(setup.opt)


def test_new_z_grid_invalidates_existing_plan_fingerprint(setup):
    import json
    from canvas_z import CanvasZCalibration, HEIGHT_FILENAME, save_grid
    s = setup
    path = s.path.parent / HEIGHT_FILENAME
    params = {"Z_CANVAS": .0015, "Z_MAX_CANVAS": -.009}
    path.write_text(json.dumps(params))
    s.plan.data["cache_fingerprints"] = stroke_plan.cache_fingerprints(s.opt.cache_dir)
    s.plan.check_setup(s.opt)
    grid = CanvasZCalibration(s.opt, np.full((3, 3), .002))
    save_grid(path, params, grid)
    with pytest.raises(ValueError, match="calibration/model changed: brush_tip_to_table_calib.json"):
        s.plan.check_setup(s.opt)


def test_ink_round_trip_has_no_palette(setup):
    s = setup
    s.opt.ink = True
    for stroke in s.painting.brush_strokes:
        with torch.no_grad():
            stroke.color_transform.zero_()
    plan = StrokePlan.capture(s.painting, None, s.opt, nearest_color)
    plan.save(s.path)
    restored, palette = StrokePlan.load(s.path).restore(s.opt, s.background, "cpu")
    assert palette is None
    assert len(restored) == 3
    assert all(record["palette_index"] is None for record in plan.data["strokes"])
    assert all(torch.count_nonzero(stroke.color_transform) == 0 for stroke in restored.brush_strokes)


@pytest.fixture
def run_script(setup, monkeypatch):
    """Execute the actual main block, replacing every hardware/UI entry point."""
    s = setup
    path = Path(__file__).resolve().parents[1] / "src/paint.py"
    main = next(node for node in ast.parse(path.read_text()).body if isinstance(node, ast.If))
    code = compile(ast.Module(body=main.body, type_ignores=[]), str(path), "exec")
    camera = Mock()
    camera.get_canvas_tensor.return_value = torch.ones(1, 4, 64, 82) * 255
    camera.get_canvas.return_value = np.ones((64, 82, 3), dtype=np.uint8) * 255
    painter = SimpleNamespace(opt=s.opt, camera=camera, to_neutral=Mock(),
                              clean_paint_brush=Mock(), get_paint=Mock(), robot=Mock())
    painter_factory = Mock(return_value=painter)
    random_init = Mock(return_value=s.painting)
    optimize = Mock(side_effect=lambda opt, painting, **kwargs: (painting, s.palette))
    load_objectives = Mock()
    module = ModuleType("painting_optimization")
    module.optimize_painting = optimize
    module.load_objectives_data = load_objectives
    monkeypatch.setitem(sys.modules, "painting_optimization", module)
    # Importing these hardware modules in this fixture would fail immediately.
    monkeypatch.setitem(sys.modules, "robot", None)
    monkeypatch.setitem(sys.modules, "painter", None)
    execute = Mock()
    monkeypatch.setattr(BrushStroke, "execute", execute)

    def ready(*args, **kwargs):
        # Every preview/pause must already have a recoverable plan on disk.
        assert StrokePlan.load(s.opt.save_plan or s.opt.load_plan or s.path).remaining > 0

    state = dict(
        MotionPlanningError=type("MotionPlanningError", (RuntimeError,), {}),
        Options=lambda: s.opt, Path=Path, DEFAULT_FILENAME=DEFAULT_FILENAME, StrokePlan=StrokePlan,
        datetime=datetime, sys=sys, np=np, torch=torch, device="cpu", Painter=painter_factory,
        record_painting=record_painting,
        TensorBoard=lambda path: Mock(), input=Mock(return_value=""), cv2=Mock(),
        random_init_painting=random_init, nearest_color=nearest_color,
        save_colors=lambda colors: np.zeros((2, 2, 3)), show_img=Mock(side_effect=ready),
        format_img=lambda t: t.detach().cpu().numpy()[0].transpose(1, 2, 0),
        canvas_to_global_coordinates=lambda x, y, z, opt: (x, y, z),
    )
    return SimpleNamespace(run=lambda: exec(code, state), state=state, painter=painter,
                           factory=painter_factory, execute=execute, optimize=optimize,
                           random_init=random_init, load_objectives=load_objectives)


def test_fresh_run_saves_before_execution_and_checkpoints_each_stroke(setup, run_script):
    def execute(*args):
        saved = StrokePlan.load(setup.path)
        assert saved.completed == run_script.execute.call_count - 1
        assert saved.data["in_progress"]
    run_script.execute.side_effect = execute
    run_script.run()
    saved = StrokePlan.load(setup.path)
    assert saved.completed == 3 and saved.remaining == 0
    assert not saved.data["in_progress"]
    assert [call.args[0] for call in run_script.painter.get_paint.call_args_list] == [2, 0, 1]
    run_script.optimize.assert_called_once()


@pytest.mark.parametrize("alias", [False, True], ids=["load-plan", "dont-plan"])
def test_crash_then_resume_skips_optimization_and_preserves_slots(setup, run_script, monkeypatch, alias):
    run_script.execute.side_effect = [None, RuntimeError("offline failure")]
    with pytest.raises(RuntimeError, match="offline failure"):
        run_script.run()
    saved = StrokePlan.load(setup.path)
    assert saved.completed == 1 and saved.remaining == 2 and saved.data["in_progress"]

    setup.opt.load_plan = None if alias else str(setup.path)
    setup.opt.dont_plan = alias
    setup.opt.use_colors_from = "/nonexistent/do-not-read.png"
    setup.opt.num_adaptations = 20  # Loading never schedules optimization.
    run_script.execute.reset_mock(side_effect=True)
    run_script.painter.get_paint.reset_mock()
    run_script.random_init.side_effect = AssertionError("must not initialize a new plan")
    run_script.optimize.side_effect = AssertionError("must not optimize a loaded plan")
    run_script.load_objectives.side_effect = AssertionError("must not load objectives")
    monkeypatch.setitem(sys.modules, "painting_optimization", None)
    run_script.run()
    assert run_script.execute.call_count == 2
    assert [call.args[0] for call in run_script.painter.get_paint.call_args_list] == [0, 1]
    assert setup.opt.dont_retrain_stroke_model is True
    assert StrokePlan.load(setup.path).remaining == 0
    np.testing.assert_array_equal(np.load(setup.path.parent / "colors_updated.npy"),
                                  (setup.palette.numpy() * 255).astype(np.uint8))


def test_invalid_setup_stops_before_robot_startup(setup, run_script):
    setup.plan.save(setup.path)
    setup.opt.load_plan = str(setup.path)
    setup.opt.CANVAS_HEIGHT_M = 0.1
    with pytest.raises(ValueError, match="setup differs"):
        run_script.run()
    run_script.factory.assert_not_called()
    run_script.execute.assert_not_called()


def test_completed_plan_exits_without_connecting(setup, run_script):
    setup.plan.data["completed_strokes"] = 3
    setup.plan.save(setup.path)
    setup.opt.load_plan = str(setup.path)
    with pytest.raises(SystemExit) as result:
        run_script.run()
    assert result.value.code == 0
    run_script.factory.assert_not_called()


def test_simulation_saves_without_marking_strokes_completed(setup, run_script):
    setup.opt.simulate = True
    with pytest.raises(SystemExit) as result:
        run_script.run()
    assert result.value.code == 0
    saved = StrokePlan.load(setup.path)
    assert saved.completed == 0 and saved.remaining == 3
    assert not saved.data["in_progress"]
    run_script.execute.assert_not_called()


def test_adapted_plan_is_saved_before_next_stroke(setup, run_script):
    setup.opt.num_adaptations = 3
    observed = []
    def optimize(opt, painting, **kwargs):
        if len(painting) == 2:
            painting.brush_strokes = torch.nn.ModuleList(list(reversed(painting.brush_strokes)))
        return painting, setup.palette
    def execute(*args):
        saved = StrokePlan.load(setup.path)
        observed.append(saved.next_palette_index)
    run_script.optimize.side_effect = optimize
    run_script.execute.side_effect = execute
    run_script.run()
    assert observed == [2, 1, 0]
    assert StrokePlan.load(setup.path).data["palette"] == setup.palette.tolist()


def test_save_plan_override_keeps_source_checkpoint(setup, run_script):
    setup.plan.save(setup.path)
    original = setup.path.read_bytes()
    destination = setup.path.parent / "resumed.json"
    setup.opt.load_plan = str(setup.path)
    setup.opt.save_plan = str(destination)
    run_script.run()
    assert setup.path.read_bytes() == original
    assert StrokePlan.load(destination).remaining == 0


def test_checkpoint_failure_prevents_stroke_execution(setup, run_script, monkeypatch):
    monkeypatch.setattr(StrokePlan, "begin_stroke", Mock(side_effect=OSError("disk full")))
    with pytest.raises(OSError, match="disk full"):
        run_script.run()
    run_script.execute.assert_not_called()
    assert StrokePlan.load(setup.path).remaining == 3


def test_recording_events_match_saved_plan_stroke_numbers(setup, run_script):
    setup.plan.save(setup.path)
    setup.opt.load_plan = str(setup.path)
    run_script.run()
    events = run_script.painter.camera.start_recording.return_value.event.call_args_list
    assert [call.kwargs["stroke"] for call in events if call.args[0] == "stroke_start"] == [1, 2, 3]
    assert [call.kwargs["stroke"] for call in events if call.args[0] == "stroke_end"] == [1, 2, 3]
    run_script.painter.camera.close.assert_called_once_with(reason="completed")
    run_script.optimize.assert_not_called()
    assert StrokePlan.load(setup.path).data["strokes"] == setup.plan.data["strokes"]


@pytest.mark.parametrize("error,reason", [(RuntimeError, "error"), (KeyboardInterrupt, "interrupted")])
def test_recording_closes_and_failed_stroke_stays_pending(setup, run_script, error, reason):
    run_script.execute.side_effect = [None, error("stop")]
    with pytest.raises(error):
        run_script.run()
    saved = StrokePlan.load(setup.path)
    assert saved.completed == 1 and saved.data["in_progress"]
    run_script.painter.camera.close.assert_called_once_with(reason=reason)
    events = run_script.painter.camera.start_recording.return_value.event.call_args_list
    assert [call.kwargs["stroke"] for call in events if call.args[0] == "stroke_end"] == [1]
    assert [call.kwargs["stroke"] for call in events if call.args[0] == "stroke_failed"] == [2]


def test_recording_startup_failure_prevents_paint_execution(run_script):
    run_script.painter.camera.start_recording.side_effect = RuntimeError("no video codec")
    with pytest.raises(RuntimeError, match="no video codec"):
        run_script.run()
    run_script.execute.assert_not_called()
    run_script.painter.get_paint.assert_not_called()
    run_script.painter.camera.close.assert_called_once_with(reason="error")


def test_crash_during_adaptation_retains_previous_unpainted_plan(setup, run_script):
    setup.opt.num_adaptations = 3
    run_script.optimize.side_effect = [
        (setup.painting, setup.palette), RuntimeError("optimization interrupted"),
    ]
    with pytest.raises(RuntimeError, match="optimization interrupted"):
        run_script.run()
    saved = StrokePlan.load(setup.path)
    assert saved.completed == 1 and saved.remaining == 2
    assert not saved.data["in_progress"]
    assert [record["palette_index"] for record in saved.data["strokes"]] == [2, 0, 1]


def test_missing_plan_stops_before_robot_startup(setup, run_script):
    setup.opt.load_plan = str(setup.path)
    with pytest.raises(FileNotFoundError):
        run_script.run()
    run_script.factory.assert_not_called()


def test_plan_cli_options():
    import argparse
    from options import Options
    parser = Options().initialize(argparse.ArgumentParser())
    opt = parser.parse_args(["--dont_plan", "--load_plan", "saved.json", "--save_plan", "copy.json"])
    assert opt.dont_plan and opt.load_plan == "saved.json" and opt.save_plan == "copy.json"


def test_palette_preview_preserves_slot_order_above_twelve_colors():
    path = Path(__file__).resolve().parents[1] / "src/paint_utils3.py"
    function = next(node for node in ast.parse(path.read_text()).body
                    if isinstance(node, ast.FunctionDef) and node.name == "save_colors")
    namespace = {"np": np}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), namespace)
    palette = torch.linspace(0.0, 1.0, 42).reshape(14, 3)
    preview = namespace["save_colors"](palette)
    assert preview.shape == (384, 768, 3)
    for index, color in enumerate(palette.numpy()):
        np.testing.assert_array_equal(preview[(index // 6) * 128, (index % 6) * 128], color)
