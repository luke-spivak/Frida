"""Full experiment flows with real recording/snapshots and synthetic hardware."""

from functools import partial
import json
from types import SimpleNamespace
from unittest.mock import Mock

import cv2
import pytest

from paint_loading import base_report, execute_test, load_setup, run_sequences
from paint_washing import run_batches
from paint_repeatability import run_sheets
from repeatability_analysis import analyze
from test_camera_recording import FakeCamera
from test_paint_loading import setup, report_for as loading_report
from test_paint_washing import washing_args, report_for as washing_report
from test_paint_repeatability import repeatability_args, report_for as repeatability_report
from test_so101_calibrated_limits import calibrated_planner
from test_canvas_z_planning import offline_robot


class SyntheticCamera(FakeCamera):
    def read(self):
        success, frame = super().read()
        # Wide enough for the real canvas aspect ratio used by the snapshotter.
        return success, cv2.resize(frame, (80, 48)) if success else None


@pytest.mark.parametrize("experiment,total,snapshots", [("loading", 30, 7), ("washing", 35, 9),
                                                      ("repeatability", 36, 7)])
def test_complete_experiment_uses_real_recorder_event_signature_and_snapshots(
        setup, tmp_path, monkeypatch, experiment, total, snapshots, offline_robot):
    from robot import SO101

    # Never instantiate the physical SO101 or open an actual camera. Retain the
    # production Painter methods, BrushStroke.execute, WebCam, CameraRecording,
    # PNG snapshots, video encoder, and JSON event/report writing end to end.
    camera = SyntheticCamera()
    monkeypatch.setattr("camera.innomaker.WebCam.camera_init", lambda self: camera)
    robot = Mock(spec=SO101)
    robot.arm = SimpleNamespace(bus=Mock(is_connected=False))
    robot.motion_planner = offline_robot.motion_planner
    robot.frida_to_robot.side_effect = lambda positions: SO101.frida_to_robot(robot, positions)
    robot.arm.bus.sync_read.return_value = {name.removesuffix(".pos"): value
                                           for name, value in offline_robot.arm.get_observation().items()}
    robot.arm.bus.connect.side_effect = lambda **kwargs: setattr(robot.arm.bus, "is_connected", True)
    robot.arm.bus.disconnect.side_effect = lambda **kwargs: setattr(robot.arm.bus, "is_connected", False)
    robot.good_morning_robot.side_effect = lambda: setattr(robot.arm.bus, "is_connected", True)
    monkeypatch.setattr("robot.SO101", lambda **kwargs: robot)
    args = {"loading": lambda a: a, "washing": washing_args, "repeatability": repeatability_args}[experiment](setup)
    opt, params, grid, h, xy = load_setup(args)
    report = base_report(args, params, xy)
    report["experiment"] = experiment
    report.update({"loading": loading_report, "washing": washing_report, "repeatability": repeatability_report}[experiment](args))
    runner = {"loading": run_sequences, "washing": run_batches, "repeatability": run_sheets}[experiment]
    execute_test(args, opt, params, grid, h, xy, tmp_path, report,
                 sequence_runner=partial(runner, prompt=lambda message: None))

    events = [json.loads(line) for line in (tmp_path / "recording/events.jsonl").read_text().splitlines()]
    assert events[1]["event"] == f"{experiment}_test_start"
    assert sum(e["event"] == "stroke_end" for e in events) == total
    snapshot_events = [e for e in events if e["event"] == "snapshot"]
    assert len(snapshot_events) == snapshots
    assert snapshot_events[0]["snapshot_name"] == "setup"
    for event in snapshot_events:
        for suffix in ("overhead", "canvas"):
            path = tmp_path / f"{event['snapshot_name']}-{suffix}.png"
            assert cv2.imread(str(path)) is not None
    assert events[-1]["event"] == "recording_stop"
    assert events[-1]["reason"] == "completed"
    metadata = json.loads((tmp_path / "recording/recording.json").read_text())
    assert metadata["errors"] == [] and metadata["written_frames"] > 0
    assert (tmp_path / "recording/overhead.mp4").stat().st_size > 0
    assert json.loads((tmp_path / "test.json").read_text())["status"] == "completed"
    robot.good_morning_robot.assert_called_once()
    assert robot.arm.bus.disconnect.call_count == 2  # Read-only preflight, then actual session.
    assert all(call.kwargs == {"disable_torque": False} for call in robot.arm.bus.disconnect.call_args_list)
    robot.arm.bus.write.assert_not_called()
    assert report["startup"]["status"] == "passed"
    assert camera.released
    if experiment == "repeatability":
        measurements = analyze(tmp_path)
        assert len(measurements) == 36
        assert cv2.imread(str(tmp_path / "comparison.png")) is not None
        assert len(json.loads((tmp_path / "measurements.json").read_text())["same_position_repeats"]) == 12
        assert "Start spread" in (tmp_path / "summary.txt").read_text()
