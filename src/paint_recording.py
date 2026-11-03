"""Recording lifecycle for paint.py only; library collection is unchanged."""

from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from uuid import uuid4


class NoRecording:
    def event(self, name, **details):
        pass


@contextmanager
def record_painting(painter, opt, plan_path):
    reason = "completed"
    recording = NoRecording()
    try:
        if not opt.simulate and not opt.no_recording:
            name = datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid4().hex[:8]
            directory = Path(opt.recording_dir) / name
            recording = painter.camera.start_recording(directory, fps=opt.recording_fps)
            recording.event("session_start", plan_path=str(plan_path.resolve()),
                            reload_every=opt.how_often_to_get_paint, clean_every=12, wash_passes=1)
            print(f"Overhead recording: {directory.resolve()}", flush=True)
        yield recording
    except BaseException as exc:
        if isinstance(exc, KeyboardInterrupt):
            reason = "interrupted"
        elif not (isinstance(exc, SystemExit) and exc.code in (None, 0)):
            reason = "error"
        recording.event("session_exit", reason=reason, error=str(exc))
        raise
    finally:
        if not opt.simulate:
            painter.camera.close(reason=reason)
