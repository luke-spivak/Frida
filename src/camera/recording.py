"""One camera reader, a bounded video writer, and a shared monotonic timeline.

The camera's native resolution is retained: changing it would invalidate the
canvas homography. Video is sampled at the requested FPS; frames.jsonl maps
video frame indices to real capture times, including gaps if the writer drops
behind. No robot operations happen in either background thread.
"""

from datetime import datetime, timezone
import json
import math
from pathlib import Path
from queue import Empty, Full, Queue
import threading
import time
import warnings

import cv2


class CameraRecording:
    def __init__(self, camera, directory, fps=10):
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError("recording_fps must be positive and finite")
        self.camera = camera
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=False)
        self.fps = fps
        self.started = time.monotonic()
        self.started_utc = datetime.now(timezone.utc).isoformat()
        self._condition = threading.Condition()
        self._stop = threading.Event()
        self._frames = Queue(maxsize=8)
        self._latest = None
        self._sequence = 0
        self._closed = False
        self._capture_error = None
        self._writer = None
        self._threads = []
        self.errors = []
        self.dropped_frames = 0
        self.written_frames = 0
        self.resolution = None
        self._events = (self.directory / "events.jsonl").open("w", buffering=1)

    def start(self):
        # The capture thread is the only owner of camera.read/release from here.
        capture = threading.Thread(target=self._capture, name="innomaker-capture", daemon=True)
        self._threads.append(capture)
        capture.start()
        try:
            frame = self.read()
            height, width = frame.shape[:2]
            self.resolution = [width, height]
            self._writer = cv2.VideoWriter(
                str(self.directory / "overhead.mp4"),
                cv2.VideoWriter_fourcc(*"mp4v"), self.fps, (width, height))
            if not self._writer.isOpened():
                raise RuntimeError("Cannot open overhead.mp4 for recording")
            writer = threading.Thread(target=self._write, name="innomaker-video", daemon=True)
            self._threads.append(writer)
            writer.start()
            self.event("recording_start", started_utc=self.started_utc,
                       resolution=self.resolution, fps=self.fps)
        except BaseException:
            self.close("startup_error")
            raise
        return self

    def _capture(self):
        next_frame = self.started
        try:
            while not self._stop.is_set():
                success, frame = self.camera.read()
                captured = time.monotonic()
                if self._stop.is_set():
                    break
                if not success or frame is None:
                    raise RuntimeError("Innomaker camera stopped returning frames")
                with self._condition:
                    self._sequence += 1
                    self._latest = frame.copy()
                    self._condition.notify_all()
                if captured >= next_frame:
                    # Advance the sampling clock without adding drift each frame.
                    next_frame += (int((captured - next_frame) * self.fps) + 1) / self.fps
                    try:
                        self._frames.put_nowait((self._sequence, captured - self.started, frame.copy()))
                    except Full:
                        self.dropped_frames += 1
        except Exception as exc:
            self._capture_error = str(exc)
            self._warn(exc)
            with self._condition:
                self._condition.notify_all()
        finally:
            try:
                self.camera.release()
            except Exception as exc:
                self._warn(exc)

    def _write(self):
        try:
            with (self.directory / "frames.jsonl").open("w", buffering=1) as timestamps:
                while not self._stop.is_set() or not self._frames.empty():
                    try:
                        sequence, elapsed, frame = self._frames.get(timeout=0.1)
                    except Empty:
                        continue
                    if [frame.shape[1], frame.shape[0]] != self.resolution:
                        raise RuntimeError("Camera resolution changed during recording")
                    self._writer.write(frame)
                    timestamps.write(json.dumps({
                        "frame": self.written_frames, "video_seconds": self.written_frames / self.fps,
                        "elapsed_seconds": elapsed, "capture_sequence": sequence,
                    }) + "\n")
                    self.written_frames += 1
        except Exception as exc:
            self._warn(exc)
        finally:
            try:
                self._writer.release()
            except Exception as exc:
                self._warn(exc)

    def read(self, timeout=5):
        """Wait for a fresh frame, avoiding a stale/arm-obscured canvas snapshot."""
        with self._condition:
            sequence = self._sequence
            ready = self._condition.wait_for(
                lambda: self._sequence > sequence or self._capture_error or self._stop.is_set(), timeout)
            if not ready or self._capture_error or self._stop.is_set():
                raise RuntimeError(self._capture_error or "Timed out or stopped waiting for Innomaker frame")
            return self._latest.copy()

    def _warn(self, error):
        message = str(error)
        self.errors.append(message)
        warnings.warn(f"Overhead recording: {message}", RuntimeWarning, stacklevel=2)

    def event(self, name, **details):
        # Event I/O failures must not turn a completed physical stroke into a
        # pending checkpoint. Video failures are likewise reported separately.
        if self._events.closed:
            return
        try:
            self._events.write(json.dumps({
                "event": name, "elapsed_seconds": time.monotonic() - self.started, **details,
            }) + "\n")
        except OSError as exc:
            self._warn(exc)
            self._close_events()

    def _close_events(self):
        try:
            self._events.close()
        except OSError as exc:
            self._warn(exc)

    def close(self, reason="completed"):
        if self._closed:
            return
        self._closed = True
        self.event("recording_stop", reason=reason)
        self._stop.set()
        with self._condition:
            self._condition.notify_all()
        for thread in self._threads:
            thread.join(timeout=5)
            if thread.is_alive():
                self._warn(f"{thread.name} did not stop within five seconds")
        # A failed startup may have opened the writer but not started its thread.
        if self._writer is not None and len(self._threads) < 2:
            try:
                self._writer.release()
            except Exception as exc:
                self._warn(exc)
        self._close_events()
        metadata = {
            "started_utc": self.started_utc, "duration_seconds": time.monotonic() - self.started,
            "fps": self.fps, "resolution": self.resolution, "reason": reason,
            "written_frames": self.written_frames, "dropped_frames": self.dropped_frames,
            "errors": self.errors,
            "timing": "frames.jsonl maps video frames to elapsed_seconds in events.jsonl",
        }
        try:
            (self.directory / "recording.json").write_text(json.dumps(metadata, indent=2) + "\n")
        except OSError as exc:
            self._warn(exc)
