"""Read-only snapshot and pose-mapping regressions; never open a real port."""

import json
from unittest.mock import Mock, call

import numpy as np
import pinocchio as pin
import pytest

import preview_so101_offsets as preview


@pytest.fixture
def observation():
    return {
        "shoulder_pan.pos": 11.340659341,
        "shoulder_lift.pos": -103.604395604,
        "elbow_flex.pos": 94.857142857,
        "wrist_flex.pos": 10.109890110,
        "wrist_roll.pos": 86.285714286,
        "gripper.pos": 8.870967742,
    }


@pytest.fixture
def calibration_file(tmp_path):
    values = {
        name: dict(id=index, drive_mode=0, homing_offset=0, range_min=0, range_max=4095)
        for index, name in enumerate(preview.ALL_JOINTS, 1)
    }
    path = tmp_path / "calibration.json"
    path.write_text(json.dumps(values))
    return path


@pytest.mark.parametrize("outcome", ["success", "read_error", "calibration_mismatch", "connect_error"])
def test_snapshot_never_configures_motors_or_changes_torque(
    monkeypatch, calibration_file, observation, outcome,
):
    bus = Mock()
    bus.is_connected = True  # Also models a failed handshake after the port opened.
    bus.is_calibrated = outcome != "calibration_mismatch"
    bus.sync_read.return_value = {name: observation[f"{name}.pos"] for name in preview.ALL_JOINTS}
    if outcome == "connect_error":
        bus.connect.side_effect = ConnectionError("handshake failed")
    elif outcome == "read_error":
        bus.sync_read.side_effect = ConnectionError("read failed")
    factory = Mock(return_value=bus)
    monkeypatch.setattr(preview, "ReadOnlyFeetechBus", factory)

    if outcome == "success":
        assert preview.read_motor_snapshot("unused-port", calibration_file) == observation
    else:
        with pytest.raises((ConnectionError, RuntimeError)):
            preview.read_motor_snapshot("unused-port", calibration_file)

    expected = [call.connect()]
    if outcome in ("success", "read_error"):
        expected.append(call.sync_read("Present_Position", normalize=True, num_retry=2))
    expected.append(call.disconnect(disable_torque=False))
    assert bus.method_calls == expected
    motors = factory.call_args.args[1]
    assert motors["shoulder_lift"].norm_mode is preview.MotorNormMode.DEGREES
    assert motors["gripper"].norm_mode is preview.MotorNormMode.RANGE_0_100


@pytest.mark.parametrize("operation", ["write", "sync_write", "enable_torque", "disable_torque", "configure_motors", "write_calibration"])
def test_register_write_guard_blocks_actual_bus_methods(calibration_file, operation):
    saved = json.loads(calibration_file.read_text())
    calibration = {name: preview.MotorCalibration(**saved[name]) for name in preview.ALL_JOINTS}
    motors = {
        name: preview.Motor(calibration[name].id, "sts3215", preview.MotorNormMode.DEGREES)
        for name in preview.ALL_JOINTS
    }
    bus = preview.ReadOnlyFeetechBus("unused-port", motors, calibration=calibration)
    # Simulate an open port only; opening one here would defeat the test.
    bus.port_handler.is_open = True
    bus.packet_handler = Mock()
    bus.sync_writer = Mock()
    operations = {
        "write": lambda: bus.write("Goal_Position", "shoulder_pan", 10.0),
        "sync_write": lambda: bus.sync_write("Goal_Position", {"shoulder_pan": 10.0}),
        "enable_torque": bus.enable_torque,
        "disable_torque": bus.disable_torque,
        "configure_motors": bus.configure_motors,
        "write_calibration": lambda: bus.write_calibration(calibration),
    }
    with pytest.raises(RuntimeError, match="register writes are disabled"):
        operations[operation]()
    assert bus.packet_handler.mock_calls == []
    assert bus.sync_writer.mock_calls == []


def test_adjustment_changes_only_model_and_keeps_out_of_limit_pose(observation):
    model = pin.buildModelFromUrdf(str(preview.DEFAULT_URDF))
    original = observation.copy()
    before = preview.model_configuration(model, observation, preview.DEFAULT_OFFSETS)
    lift = preview.joint_index(model, "shoulder_lift")
    assert before[lift] < model.lowerPositionLimit[lift]
    assert np.rad2deg(before[lift]) == pytest.approx(-103.604395604)
    assert "**Outside URDF limits:** shoulder_lift." in preview.pose_table(
        model, observation, preview.DEFAULT_OFFSETS, before,
    )

    offsets = {**preview.DEFAULT_OFFSETS, "shoulder_lift": 4.0}
    after = preview.model_configuration(model, observation, offsets)
    assert np.rad2deg(after[lift]) == pytest.approx(-99.604395604)
    unchanged = [i for i in range(model.nq) if i != lift]
    np.testing.assert_array_equal(before[unchanged], after[unchanged])
    assert observation == original
    assert np.rad2deg(after[preview.joint_index(model, "gripper")]) == pytest.approx(-0.241935588)


def test_saved_observation_and_offsets_can_be_reloaded(tmp_path, observation):
    saved = tmp_path / "saved.json"
    offsets = {**preview.DEFAULT_OFFSETS, "shoulder_lift": 4.0}
    saved.write_text(json.dumps({"observation": observation, "joint_offsets_deg": offsets}))
    assert preview.load_observation(saved) == observation
    args, loaded_offsets = preview.parse_args([
        "--observation", str(saved), "--initial-offsets", str(saved),
        "--output", str(tmp_path / "next.json"),
    ])
    assert args.observation == saved
    assert loaded_offsets == offsets


def test_export_cannot_overwrite_motor_calibration(calibration_file):
    with pytest.raises(SystemExit):
        preview.parse_args(["--calibration", str(calibration_file), "--output", str(calibration_file)])


def test_viewer_reuses_saved_offsets_and_saves_beside_script(tmp_path, monkeypatch):
    path = tmp_path / "so101_offsets.json"
    offsets = {**preview.DEFAULT_OFFSETS, "shoulder_pan": 3.0, "shoulder_lift": -1.5}
    path.write_text(json.dumps({"joint_offsets_deg": offsets}))
    monkeypatch.setattr(preview, "DEFAULT_OFFSETS_PATH", path)
    args, loaded = preview.parse_args([])
    assert args.output == path
    assert args.initial_offsets == path
    assert loaded == offsets
