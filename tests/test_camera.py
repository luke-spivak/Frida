"""Retained camera backends with synthetic frames and cached calibration only."""

import json
from pathlib import Path
import pickle
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import cv2
import numpy as np
import pytest
import torch

from camera.innomaker import WebCam
from camera.simulated import SimulatedWebCam


@pytest.mark.parametrize("width,height", [(0.3, 0.2), (0.2, 0.3)])
def test_simulated_canvas_and_resized_tensor_preserve_values(width, height):
    camera = SimulatedWebCam(SimpleNamespace(CANVAS_WIDTH_M=width, CANVAS_HEIGHT_M=height))
    canvas = camera.get_canvas()
    assert canvas.shape == (1024, int(1024 * (width / height)), 3)
    assert canvas.dtype == np.float32
    assert np.all(canvas == 255)

    # Callers can update the simulated canvas; calibration must preserve it.
    canvas[:] = [10, 20, 30]
    camera.calibrate_canvas(use_cache=True)
    assert camera.get_canvas() is canvas
    tensor = camera.get_canvas_tensor(h=8, w=12)
    expected = torch.tensor([10., 20., 30., 1.]).reshape(1, 4, 1, 1).expand(1, 4, 8, 12)
    torch.testing.assert_close(tensor, expected)


def test_painter_simulation_starts_with_cached_setup_without_hardware(tmp_path, monkeypatch):
    import painter

    settings = dict(CANVAS_WIDTH_M=0.3, CANVAS_HEIGHT_M=0.2, MAX_BEND=0.02,
                    MIN_STROKE_Z=0.1, MIN_STROKE_LENGTH=0.005,
                    MAX_STROKE_LENGTH=0.06, MAX_ALPHA=0.0)
    (tmp_path / "stroke_settings_during_library.json").write_text(json.dumps(settings))
    (tmp_path / "brush_tip_to_table_calib.json").write_text(json.dumps(
        dict(Z_CANVAS=0.01, Z_MAX_CANVAS=0.005)))
    (tmp_path / "cached_H_coord.pkl").write_bytes(pickle.dumps(np.eye(3)))
    opt = SimpleNamespace(**settings, cache_dir=str(tmp_path), simulate=True,
                          robot="so101", use_cache=True, dont_retrain_stroke_model=True,
                          writer=Mock(), WATER_POSITION=[0, 0], RAG_POSTITION=[0, 0],
                          PALLETTE_POSITION=[0, 0])

    forbidden = Mock(side_effect=AssertionError("Simulation must not open hardware or prompt"))
    monkeypatch.setattr(cv2, "VideoCapture", forbidden)
    monkeypatch.setattr(painter, "SO101", forbidden)
    monkeypatch.setattr("builtins.input", forbidden)
    instance = painter.Painter(opt)

    assert isinstance(instance.camera, SimulatedWebCam)
    assert isinstance(instance.robot, painter.SimulatedRobot)
    assert (opt.CANVAS_HEIGHT_PIX, opt.CANVAS_WIDTH_PIX) == instance.camera.get_canvas().shape[:2]
    assert instance.camera.get_canvas_tensor(h=8, w=12).shape == (1, 4, 8, 12)
    forbidden.assert_not_called()


def test_painter_import_and_simulated_camera_need_no_retired_dependencies(tmp_path):
    # Use a fresh process so an earlier import cannot hide a dependency.
    subprocess.run([sys.executable, "-c", """
import sys
from types import SimpleNamespace

# None blocks imports while allowing LeRobot's optional find_spec probes to
# report these packages as unavailable, just as in an environment without them.
for name in ('gphoto2', 'pyrealsense2', 'pyrealsense2_macosx', 'rospy', 'xarm',
             'accelerate', 'audioread', 'diffusers', 'kornia', 'librosa',
             'pycocoevalcap', 'pycocotools', 'segment_anything', 'soundfile',
             'soxr', 'timm', 'tokenizers', 'torchaudio', 'transformers', 'whisper'):
    sys.modules[name] = None
sys.path[:0] = sys.argv[1:]
import painter
from camera.simulated import SimulatedWebCam
camera = SimulatedWebCam(SimpleNamespace(CANVAS_WIDTH_M=0.3, CANVAS_HEIGHT_M=0.2))
assert camera.get_canvas_tensor(h=8, w=12).shape == (1, 4, 8, 12)
""", *sys.path], cwd=tmp_path, check=True, timeout=60)


@pytest.fixture
def innomaker_capture(tmp_path, monkeypatch):
    capture = Mock(spec=["read", "release"])
    capture.read.return_value = (True, np.full((48, 80, 3), 100, dtype=np.uint8))
    monkeypatch.setattr(cv2, "VideoCapture", Mock(return_value=capture))
    monkeypatch.setattr("camera.innomaker.OUTPUT_DIR", tmp_path)
    return capture


def test_innomaker_capture_channels_and_close(innomaker_capture):
    frame = np.full((48, 80, 3), [30, 80, 150], dtype=np.uint8)
    innomaker_capture.read.return_value = True, frame
    camera = WebCam(SimpleNamespace())
    try:
        cv2.VideoCapture.assert_called_once_with(0)
        path, bgr = camera.capture_image(channels="bgr")
        assert Path(path).is_file()
        np.testing.assert_array_equal(bgr, frame)
        path, rgb = camera.get_rgb_image()
        np.testing.assert_array_equal(rgb, cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB))
    finally:
        camera.close()
    camera.close()
    innomaker_capture.release.assert_called_once()


@pytest.mark.parametrize("calib_colors", [False, True])
def test_innomaker_loads_cached_calibration_and_crops_canvas(
        tmp_path, monkeypatch, innomaker_capture, calib_colors):
    homography = tmp_path / "cached_H_canvas.pkl"
    homography.write_bytes(pickle.dumps(np.eye(3)))
    matrix = np.eye(3, 4)
    matrix[:, 3] = 10  # A saved correction must brighten the synthetic frame.
    (tmp_path / "cached_color_calibration.pkl").write_bytes(pickle.dumps(
        dict(color_tmat=matrix, greyval=np.ones(3))))
    before = {path: path.read_bytes() for path in tmp_path.glob("*.pkl")}
    forbidden = Mock(side_effect=AssertionError("Cached calibration must not prompt or find corners"))
    monkeypatch.setattr("builtins.input", forbidden)
    monkeypatch.setattr("camera.innomaker.find_corners", forbidden)
    camera = WebCam(SimpleNamespace(cache_dir=str(tmp_path), calib_colors=calib_colors,
                                   CANVAS_WIDTH_M=0.2, CANVAS_HEIGHT_M=0.2))
    try:
        canvas = camera.get_canvas(use_cache=True, max_height=24)
        assert canvas.shape == (24, 24, 3)
        assert np.all(canvas == (110 if calib_colors else 100))
        tensor = camera.get_canvas_tensor(h=12, w=12)
        assert tensor.shape == (1, 4, 12, 12)
        assert torch.all(tensor[:, :3] == (110 if calib_colors else 100))
        assert torch.all(tensor[:, 3] == 1)
        forbidden.assert_not_called()
        assert {path: path.read_bytes() for path in before} == before
    finally:
        camera.close()
    innomaker_capture.release.assert_called_once()
