"""Synthetic frames only: never opens a camera or connects to a robot."""

import json
import threading
import time
from types import SimpleNamespace
from unittest.mock import Mock

import cv2
import numpy as np
import pytest

from camera.recording import CameraRecording
from paint_recording import record_painting


class FakeCamera:
    def __init__(self):
        self.readers = set()
        self.sequence = 0
        self.released = False
        self.failed = False

    def read(self):
        self.readers.add(threading.current_thread().name)
        time.sleep(0.003)
        self.sequence += 1
        return (False, None) if self.failed else (
            True, np.full((48, 64, 3), self.sequence % 255, dtype=np.uint8))

    def release(self):
        self.released = True


class FakeWriter:
    def __init__(self, opened=True):
        self.opened = opened
        self.released = False
        self.count = 0
        self.block = None

    def isOpened(self):
        return self.opened

    def write(self, frame):
        if self.block is not None:
            assert self.block.wait(2), "test writer was not unblocked"
        self.count += 1

    def release(self):
        self.released = True


def wait_until(predicate):
    deadline = time.monotonic() + 3
    while not predicate():
        assert time.monotonic() < deadline, "background worker did not progress"
        time.sleep(0.005)


def json_lines(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_single_capture_owner_fresh_frames_events_and_shutdown(tmp_path, monkeypatch):
    camera, writer = FakeCamera(), FakeWriter()
    monkeypatch.setattr(cv2, "VideoWriter", lambda *args: writer)
    recording = CameraRecording(camera, tmp_path / "run", fps=100).start()
    try:
        first, second = recording.read(), recording.read()
        assert np.all(second > first)
        recording.event("stroke_start", stroke=7)
        recording.event("stroke_end", stroke=7)
    finally:
        recording.close("interrupted")
    recording.close()  # Idempotent; preserves the original end reason.
    assert camera.readers == {"innomaker-capture"}
    assert camera.released and writer.released
    assert all(not thread.is_alive() for thread in recording._threads)
    events = json_lines(recording.directory / "events.jsonl")
    assert [item["event"] for item in events] == [
        "recording_start", "stroke_start", "stroke_end", "recording_stop"]
    assert [item["elapsed_seconds"] for item in events] == sorted(item["elapsed_seconds"] for item in events)
    metadata = json.loads((recording.directory / "recording.json").read_text())
    assert metadata["resolution"] == [64, 48] and metadata["reason"] == "interrupted"
    frames = json_lines(recording.directory / "frames.jsonl")
    assert len(frames) == metadata["written_frames"] == writer.count > 0
    assert [item["frame"] for item in frames] == list(range(len(frames)))
    assert all(item["video_seconds"] == item["frame"] / 100 for item in frames)


def test_slow_writer_has_bounded_queue_and_does_not_block_live_frames(tmp_path, monkeypatch):
    camera, writer = FakeCamera(), FakeWriter()
    writer.block = threading.Event()
    monkeypatch.setattr(cv2, "VideoWriter", lambda *args: writer)
    recording = CameraRecording(camera, tmp_path / "run", fps=300).start()
    try:
        wait_until(lambda: recording.dropped_frames > 0)
        assert recording._frames.qsize() <= 8
        first, second = recording.read(), recording.read()
        assert np.all(second > first)  # Preview/canvas still gets fresh frames.
        assert writer.count == 0
    finally:
        writer.block.set()
        recording.close()
    assert writer.count > 0
    assert json.loads((recording.directory / "recording.json").read_text())["dropped_frames"] > 0


def test_writer_startup_failure_releases_camera(tmp_path, monkeypatch):
    camera, writer = FakeCamera(), FakeWriter(opened=False)
    monkeypatch.setattr(cv2, "VideoWriter", lambda *args: writer)
    recording = CameraRecording(camera, tmp_path / "run")
    with pytest.raises(RuntimeError, match="Cannot open overhead.mp4"):
        recording.start()
    assert camera.released and writer.released
    assert all(not thread.is_alive() for thread in recording._threads)


def test_writer_runtime_failure_keeps_camera_stream_available(tmp_path, monkeypatch):
    camera, writer = FakeCamera(), FakeWriter()
    writer.write = Mock(side_effect=OSError("disk full"))
    monkeypatch.setattr(cv2, "VideoWriter", lambda *args: writer)
    with pytest.warns(RuntimeWarning, match="disk full"):
        recording = CameraRecording(camera, tmp_path / "run").start()
        wait_until(lambda: writer.released)
    try:
        first, second = recording.read(), recording.read()
        assert np.all(second > first)
        recording.event("stroke_end", stroke=1)
    finally:
        recording.close()
    metadata = json.loads((recording.directory / "recording.json").read_text())
    assert metadata["errors"] == ["disk full"]


def test_event_disk_failure_does_not_raise_into_stroke_execution(tmp_path):
    recording = CameraRecording(FakeCamera(), tmp_path / "run")
    recording._events.close()
    stream = Mock(closed=False)
    stream.write.side_effect = OSError("disk full")
    stream.close.side_effect = OSError("flush failed")
    recording._events = stream
    with pytest.warns(RuntimeWarning):
        recording.event("stroke_end", stroke=1)
    assert recording.errors == ["disk full", "flush failed"]
    stream.closed = True
    stream.close.side_effect = None
    recording.close()


def test_camera_failure_does_not_return_stale_canvas(tmp_path, monkeypatch):
    camera, writer = FakeCamera(), FakeWriter()
    monkeypatch.setattr(cv2, "VideoWriter", lambda *args: writer)
    recording = CameraRecording(camera, tmp_path / "run").start()
    try:
        with pytest.warns(RuntimeWarning, match="stopped returning frames"):
            camera.failed = True
            wait_until(lambda: recording._capture_error is not None)
        with pytest.raises(RuntimeError, match="stopped returning frames"):
            recording.read()
    finally:
        recording.close("error")
    assert camera.released and writer.released


def test_real_mp4_encode_decode_with_synthetic_camera(tmp_path):
    # VideoCapture below reads ONLY the generated file, never a device index.
    recording = CameraRecording(FakeCamera(), tmp_path / "run", fps=30).start()
    try:
        wait_until(lambda: recording.written_frames >= 3)
    finally:
        recording.close()
    video = cv2.VideoCapture(str(recording.directory / "overhead.mp4"))
    try:
        frames = []
        while True:
            success, frame = video.read()
            if not success:
                break
            frames.append(frame)
        assert len(frames) == recording.written_frames >= 3
        assert all(frame.shape == (48, 64, 3) for frame in frames)
    finally:
        video.release()


@pytest.mark.parametrize("simulate,disabled", [(True, False), (False, True)])
def test_disabled_and_simulated_sessions_never_start_recording(tmp_path, simulate, disabled):
    camera = Mock()
    opt = SimpleNamespace(simulate=simulate, no_recording=disabled)
    with record_painting(SimpleNamespace(camera=camera), opt, tmp_path / "plan.json") as recording:
        recording.event("stroke_start", stroke=1)
    camera.start_recording.assert_not_called()
    if simulate:
        camera.close.assert_not_called()
    else:
        camera.close.assert_called_once_with(reason="completed")


@pytest.mark.parametrize("error,reason", [(None, "completed"), (KeyboardInterrupt, "interrupted"),
                                        (RuntimeError, "error")])
def test_session_closes_on_completion_error_and_ctrl_c(tmp_path, error, reason):
    camera = Mock()
    opt = SimpleNamespace(simulate=False, no_recording=False, recording_dir=tmp_path,
                          recording_fps=10, how_often_to_get_paint=5)
    def run():
        with record_painting(SimpleNamespace(camera=camera), opt, tmp_path / "plan.json"):
            if error is not None:
                raise error()
    if error:
        with pytest.raises(error):
            run()
    else:
        run()
    camera.close.assert_called_once_with(reason=reason)


def test_innomaker_stills_use_recording_stream_and_preserve_channel_argument(tmp_path, monkeypatch):
    from camera.innomaker import WebCam
    camera = FakeCamera()
    monkeypatch.setattr(WebCam, "camera_init", lambda self: camera)
    monkeypatch.chdir(tmp_path)
    webcam = WebCam(SimpleNamespace())
    recording = webcam.start_recording(tmp_path / "run", fps=30)
    try:
        _, bgr = webcam.get_rgb_image(channels="bgr")
        _, rgb = webcam.get_rgb_image()
        assert bgr.shape == rgb.shape == (48, 64, 3)
        assert camera.readers == {"innomaker-capture"}
        capture = Mock(return_value=("frame", bgr))
        monkeypatch.setattr(webcam, "capture_image", capture)
        webcam.get_rgb_image(channels="bgr")
        capture.assert_called_once_with(channels="bgr")
    finally:
        webcam.close()
    assert camera.released and recording._closed
