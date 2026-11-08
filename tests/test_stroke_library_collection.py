"""Collection recovery and append-only batches with fake motion and camera."""

import gzip
import json
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

import painter as painter_module
from painter import Painter
from robot import MotionPlanningError
from stroke_library import inspect_stroke_library


@pytest.fixture
def collection(tmp_path, monkeypatch):
    opt = SimpleNamespace(
        cache_dir=str(tmp_path), resume_stroke_library=False, num_papers=1,
        CANVAS_WIDTH_PIX=200, CANVAS_HEIGHT_PIX=100,
        CANVAS_WIDTH_M=.2, CANVAS_HEIGHT_M=.05, CANVAS_POSITION=[0., .3],
        MAX_BEND=.04, MIN_STROKE_Z=.1, MIN_STROKE_LENGTH=.005,
        MAX_STROKE_LENGTH=.04, MAX_ALPHA=0., ink=False, how_often_to_get_paint=4,
        robot='so101', simulate=False, use_cache=True, dont_retrain_stroke_model=True,
        WATER_POSITION=[0., 0.], RAG_POSTITION=[0., 0.], PALLETTE_POSITION=[0., 0.],
        writer=Mock(),
    )
    frame = np.full((100, 200, 3), 255, dtype=np.uint8)
    p = Painter.__new__(Painter)
    p.opt, p.writer = opt, opt.writer
    p.camera = Mock()
    p.camera.get_canvas.side_effect = lambda: frame.copy()
    p.to_neutral, p.clean_paint_brush, p.get_paint = Mock(), Mock(), Mock()
    state = SimpleNamespace(painter=p, frame=frame, calls=[], fail_at={}, mark_before_reject=False)

    class Stroke:
        def __init__(self, _):
            for name, value in [('stroke_length', .04), ('stroke_bend', 0.),
                                ('stroke_z', .25), ('stroke_alpha', 0.)]:
                setattr(self, name, SimpleNamespace(item=lambda value=value: value))

        def execute(self, painter, x, y, angle):
            state.calls.append((x, y))
            error = state.fail_at.get(len(state.calls))
            if error is None or state.mark_before_reject:
                px = round(x * 200)
                frame[45:61, px:px+16] = 0
            if error is not None:
                raise error
            return True

    monkeypatch.setattr(painter_module, 'BrushStroke', Stroke)
    monkeypatch.setattr(painter_module, 'canvas_to_global_coordinates', lambda x, y, z, opt: (x, y, z))
    monkeypatch.setitem(sys.modules, 'matplotlib.pyplot', Mock())
    monkeypatch.setattr('builtins.input', lambda _: frame.fill(255))
    return state


def saved_batch(cache, count=10):
    directory = cache / 'stroke_library'
    directory.mkdir()
    np.save(directory / f'stroke_parameters{count:05d}.npy', np.full((count, 4), .25))
    with gzip.open(directory / f'stroke_intensities{count:05d}.npy', 'wb') as stream:
        np.save(stream, np.zeros((count, 100, 200), dtype=np.uint8))
    return {p.name: p.read_bytes() for p in directory.iterdir()}


@pytest.mark.parametrize('mark_before_reject', [False, True])
def test_planning_rejection_advances_slot_and_excludes_rejected_mark(collection, tmp_path, capsys, mark_before_reject):
    c = collection
    c.fail_at[1] = MotionPlanningError('Continuous stroke leaves its assigned path (2.003 mm)')
    c.mark_before_reject = mark_before_reject
    c.painter.paint_extended_stroke_library()
    assert len(c.calls) == 4
    assert np.all(np.diff([x for x, _ in c.calls]) > 0)
    assert inspect_stroke_library(tmp_path / 'stroke_library') == (3, 3)
    with gzip.open(tmp_path / 'stroke_library/stroke_intensities00003.npy', 'rb') as stream:
        images = np.load(stream)
    # Each accepted sample contains only its own equal-size new mark, even if
    # the rejected call painted before its final lift was rejected.
    assert len(set(np.count_nonzero(image) for image in images)) == 1
    assert np.count_nonzero(images[0]) > 0
    assert 'Skipping rejected library stroke' in capsys.readouterr().out


def test_all_rejected_strokes_finish_without_looping_at_one_position(collection, tmp_path):
    collection.fail_at = {i: MotionPlanningError('unreachable') for i in range(1, 5)}
    collection.painter.paint_extended_stroke_library()
    assert len(collection.calls) == 4
    assert inspect_stroke_library(tmp_path / 'stroke_library') == (0, 0)


@pytest.mark.parametrize('error', [TimeoutError('servo timeout'), RuntimeError('camera or execution failure'), KeyboardInterrupt()])
def test_execution_failures_stop_but_flush_completed_samples(collection, tmp_path, error):
    collection.fail_at[3] = error
    with pytest.raises(type(error)):
        collection.painter.paint_extended_stroke_library()
    assert len(collection.calls) == 3
    assert inspect_stroke_library(tmp_path / 'stroke_library') == (2, 2)


def test_resume_keeps_first_ten_bytes_and_appends_partial_batch(collection, tmp_path):
    original = saved_batch(tmp_path)
    collection.painter.opt.resume_stroke_library = True
    collection.painter.paint_extended_stroke_library()
    directory = tmp_path / 'stroke_library'
    assert inspect_stroke_library(directory) == (14, 14)
    for name, data in original.items():
        assert (directory / name).read_bytes() == data
    assert (tmp_path / 'stroke_settings_during_library.json').is_file()


def test_full_batch_and_exit_remainder_are_saved_once(collection, tmp_path):
    collection.painter.opt.num_papers = 3
    collection.painter.paint_extended_stroke_library()
    assert len(collection.calls) == 12
    directory = tmp_path / 'stroke_library'
    assert inspect_stroke_library(directory) == (12, 12)
    assert sorted(p.name for p in directory.glob('stroke_parameters*.npy')) == [
        'stroke_parameters00010.npy', 'stroke_parameters00012.npy']


def test_existing_samples_require_explicit_resume_before_motion(collection, tmp_path):
    original = saved_batch(tmp_path)
    with pytest.raises(ValueError, match='resume_stroke_library'):
        collection.painter.paint_extended_stroke_library()
    collection.painter.to_neutral.assert_not_called()
    assert original == {p.name: p.read_bytes() for p in (tmp_path / 'stroke_library').iterdir()}


def test_incomplete_resume_is_rejected_before_hardware(collection, tmp_path, monkeypatch):
    saved_batch(tmp_path)
    (tmp_path / 'stroke_library/stroke_intensities00010.npy').unlink()
    collection.painter.opt.resume_stroke_library = True
    constructor = Mock(side_effect=AssertionError('hardware must not connect'))
    monkeypatch.setattr(painter_module, 'SO101', constructor)
    with pytest.raises(ValueError, match='incomplete batch'):
        Painter(collection.painter.opt)
    constructor.assert_not_called()


@pytest.mark.parametrize('resume', [False, True])
def test_startup_reenters_collection_only_with_resume_flag(collection, tmp_path, monkeypatch, resume):
    saved_batch(tmp_path)
    opt = collection.painter.opt
    opt.resume_stroke_library = resume
    (tmp_path / 'brush_tip_to_table_calib.json').write_text(json.dumps({'Z_CANVAS': 0., 'Z_MAX_CANVAS': -.01}))
    names = ('MAX_BEND', 'MIN_STROKE_Z', 'MIN_STROKE_LENGTH', 'MAX_STROKE_LENGTH',
             'MAX_ALPHA', 'CANVAS_WIDTH_M', 'CANVAS_HEIGHT_M')
    (tmp_path / 'stroke_settings_during_library.json').write_text(json.dumps({k: getattr(opt, k) for k in names}))
    monkeypatch.setattr(painter_module, 'SO101', Mock(return_value=Mock()))
    monkeypatch.setattr(painter_module, 'WebCam', Mock(return_value=collection.painter.camera))
    monkeypatch.setattr(Painter, 'to_neutral', Mock())
    monkeypatch.setattr(Painter, 'coordinate_calibration', Mock())
    collect = Mock()
    monkeypatch.setattr(Painter, 'paint_extended_stroke_library', collect)
    prompt = Mock()
    monkeypatch.setattr('builtins.input', prompt)
    Painter(opt)
    assert collect.call_count == int(resume)
    if resume:
        assert '10 saved strokes' in prompt.call_args.args[0]
        assert 'FRESH' in prompt.call_args.args[0]


@pytest.mark.parametrize('flags', [[], ['--use_cache', '--simulate'], ['--use_cache', '--dont_plan'],
                                   ['--use_cache', '--load_plan', 'plan.json'], ['--use_cache', '--num_papers', '0']])
def test_resume_rejects_incompatible_cli_modes(collection, tmp_path, monkeypatch, flags):
    from options import Options
    materials = tmp_path / 'materials.json'
    materials.write_text('{}')
    monkeypatch.setattr(sys, 'argv', ['paint.py', '--materials_json', str(materials), '--resume_stroke_library', *flags])
    with pytest.raises(SystemExit):
        Options().gather_options()
