"""Wash-age tracking and recovery protocol; all hardware interfaces are fake."""

import itertools
import json
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

import paint_loading
from paint_loading import load_setup, make_layout
from paint_washing import arguments, main, make_protocol, run_batches
from test_paint_loading import setup


def washing_args(setup, *extra):
    return arguments(["--materials_json", str(setup.materials_json), "--cache_dir", str(setup.cache_dir),
                      "--output-dir", str(setup.output_dir), *extra])


def report_for(args):
    opt, _, grid, _, xy = load_setup(args)
    return make_protocol(args, make_layout(opt, grid, args.length_mm / 1000, xy))


def test_preview_is_offline_and_preserves_calibration(setup, monkeypatch):
    for name in ("robot", "painter", "camera.innomaker", "cv2", "torch", "matplotlib"):
        monkeypatch.setitem(sys.modules, name, None)
    before = {p.name: p.read_bytes() for p in setup.cache_dir.iterdir()}
    main(["--materials_json", str(setup.materials_json), "--cache_dir", str(setup.cache_dir),
          "--output-dir", str(setup.output_dir)])
    path, = setup.output_dir.glob("*/test.json")
    report = json.loads(path.read_text())
    assert report["status"] == "preview"
    assert report["experiment"] == "washing"
    assert len(list(path.parent.glob("*-layout.svg"))) == 4
    assert report["completed_strokes"] == 0
    assert report["compare_second_strokes_by_slot"]["2"][-1]["phase"] == "recovery"
    assert {p.name: p.read_bytes() for p in setup.cache_dir.iterdir()} == before


def test_default_protocol_compares_matching_positions_before_and_after_wash(setup):
    report = report_for(washing_args(setup))
    batches = [b for sheet in report["sheets"] for b in sheet["batches"]]
    marks = [s for b in batches for s in b["strokes"]]
    assert [len(sheet["batches"]) for sheet in report["sheets"]] == [2, 2, 2, 1]
    assert [b["batch"] for b in batches if b["wash_before"]] == [1, 7]
    assert [s["stroke"] for s in marks] == list(range(1, 36))
    assert [s["stroke_since_wash"] for s in marks] == list(range(1, 31)) + list(range(1, 6))
    assert [(m["x"], m["y"]) for m in batches[0]["strokes"]] == [(m["x"], m["y"]) for m in batches[-1]["strokes"]]
    assert [m["stroke"] for m in report["compare_second_strokes_by_slot"]["2"]] == [2, 12, 22, 32]
    assert [m["stroke"] for m in report["compare_second_strokes_by_slot"]["7"]] == [7, 17, 27]


@pytest.mark.parametrize("interval", [2, 3, 4, 6, 10])
def test_custom_reload_interval_never_splits_loads_or_reuses_marks_on_sheet(setup, interval):
    args = washing_args(setup, "--reload-every", str(interval), "--batches", "4")
    report = report_for(args)
    for sheet in report["sheets"]:
        marks = [m for b in sheet["batches"] for m in b["strokes"]]
        assert len(marks) <= 10
        assert len({m["slot"] for m in marks}) == len(marks)
        assert all(len(b["strokes"]) == interval for b in sheet["batches"])
    last = report["sheets"][-1]["batches"]
    assert len(last) == 1 and last[0]["phase"] == "recovery"


@pytest.mark.parametrize("flag,value", [("--reload-every", "0"), ("--reload-every", "11"),
                                        ("--batches", "1"), ("--batches", "21"),
                                        ("--stroke-z", "nan"), ("--wash-passes", "0")])
def test_invalid_flags_fail_before_connect(flag, value):
    with pytest.raises(SystemExit):
        arguments([flag, value])


def test_only_initial_and_recovery_wash_fixed_reloads_across_sheet_changes(setup, tmp_path):
    args = washing_args(setup)
    report = report_for(args)
    strokes, washed_at, loaded_at, dabs, prompts = [], [], [], [], []
    painter = SimpleNamespace(
        to_neutral=Mock(),
        clean_paint_brush=lambda: (washed_at.append(len(strokes)), np.random.randn(10)),
        get_paint=lambda slot: (loaded_at.append(len(strokes)), dabs.append(np.random.randn(6))),
    )
    recording, snapshot = Mock(), Mock()
    ticks = itertools.count()
    run_batches(args, painter, object(), report, tmp_path, recording, snapshot,
                lambda *a: strokes.append(a), lambda s: prompts.append(s),
                clock=lambda: float(next(ticks)))
    assert washed_at == [0, 0, 30, 30]
    assert loaded_at == [0, 5, 10, 15, 20, 25, 30]
    assert len(strokes) == 35 and len(prompts) == 4
    assert snapshot.call_count == 8 and painter.to_neutral.call_count == 8
    for dab in dabs:
        np.testing.assert_array_equal(dab, dabs[0])
    assert report["completed_strokes"] == 35 and report["strokes_since_wash"] == 5
    assert report["in_progress"] is None
    assert all(s["status"] == "completed" and s["paper_wait_seconds"] == 1. for s in report["sheets"])
    start_events = [c.kwargs for c in recording.event.call_args_list if c.args[0] == "stroke_start"]
    assert len(start_events) == 35
    assert all(e["seconds_since_wash"] >= 0 and e["seconds_since_load"] >= 0 for e in start_events)
    assert start_events[30]["stroke_since_wash"] == 1
    assert start_events[30]["seconds_since_wash"] < start_events[29]["seconds_since_wash"]
    assert json.loads((tmp_path / "test.json").read_text()) == report


@pytest.mark.parametrize("failure", [False, TimeoutError("stalled"), KeyboardInterrupt()])
def test_partial_stroke_stops_without_retry_wash_or_recovery(setup, tmp_path, failure):
    args = washing_args(setup)
    report = report_for(args)
    painter, recording = Mock(), Mock()
    attempts = []

    def stroke(*a):
        attempts.append(a)
        if len(attempts) == 12:
            if failure is False:
                return False
            raise failure

    with pytest.raises((RuntimeError, TimeoutError, KeyboardInterrupt)):
        run_batches(args, painter, object(), report, tmp_path, recording, Mock(), stroke, Mock())
    assert len(attempts) == 12
    assert painter.clean_paint_brush.call_count == 2
    assert painter.get_paint.call_count == 3
    saved = json.loads((tmp_path / "test.json").read_text())
    assert saved["completed_strokes"] == saved["strokes_since_wash"] == 11
    assert saved["in_progress"] == 12
    assert saved["sheets"][-1]["status"] == "pending"
    assert painter.to_neutral.call_count == 3  # No extra failure-recovery move.


def test_failure_during_recovery_wash_never_loads_or_resets_age(setup, tmp_path):
    args = washing_args(setup)
    report = report_for(args)
    painter = Mock()
    painter.clean_paint_brush.side_effect = [None, None, TimeoutError("water move failed")]
    with pytest.raises(TimeoutError):
        run_batches(args, painter, object(), report, tmp_path, Mock(), Mock(), lambda *a: None, Mock())
    assert painter.get_paint.call_count == 6
    assert report["completed_strokes"] == report["strokes_since_wash"] == 30
    assert report["sheets"][-1]["batches"][0]["completed"] == 0


def test_custom_runner_uses_shared_recording_cleanup_on_interrupt(setup, tmp_path, monkeypatch):
    from camera.innomaker import WebCam
    from robot import SO101
    camera, robot, recording = Mock(spec=WebCam), Mock(spec=SO101), Mock()
    robot.arm = SimpleNamespace(bus=Mock(is_connected=True))
    camera.start_recording.return_value = recording
    monkeypatch.setattr("camera.innomaker.WebCam", Mock(return_value=camera))
    monkeypatch.setattr("robot.SO101", Mock(return_value=robot))
    monkeypatch.setattr(paint_loading, "save_snapshot", Mock())
    monkeypatch.setattr(paint_loading, "preflight_start", Mock())
    args = washing_args(setup)
    opt, params, grid, h, xy = load_setup(args)
    report = {**report_for(args), "experiment": "washing"}
    runner = Mock(side_effect=KeyboardInterrupt())
    with pytest.raises(KeyboardInterrupt):
        paint_loading.execute_test(args, opt, params, grid, h, xy, tmp_path, report, sequence_runner=runner)
    runner.assert_called_once()
    assert report["status"] == "interrupted"
    assert recording.event.call_args_list[0].args[0] == "washing_test_start"
    camera.close.assert_called_once_with(reason="interrupted")
    robot.arm.bus.disconnect.assert_called_once_with(disable_torque=False)
